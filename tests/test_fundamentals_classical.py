"""
Unit tests for Step 6 classical learning rules (src/fundamentals/classical_learning.py)
and the early-stopping monitor added to train().
"""

import os
import tempfile
import unittest

import numpy as np

from src.fundamentals import mlp as M
from src.fundamentals.classical_learning import (
    RBM, SOM, Adaline, CompetitiveLayer, Hopfield, Perceptron, bars_and_stripes, cluster_purity,
    corrupt, hebbian_rule, hopfield_capacity, oja_rule, sanger_rule,
)


def _correlated_2d(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[3.0, 1.5], [1.5, 1.0]])
    x = rng.multivariate_normal([0, 0], cov, size=n)
    return x, np.linalg.eigh(cov)[1][:, -1]


class TestHebbianFamily(unittest.TestCase):
    def test_hebb_grows_but_aligns(self):
        x, pc1 = _correlated_2d()
        w, _ = hebbian_rule(x, np.array([0.3, -0.8]), lr=0.002)
        self.assertGreater(np.linalg.norm(w), 1e3)
        self.assertGreater(abs(w @ pc1) / np.linalg.norm(w), 0.999)

    def test_oja_converges_to_unit_pc1(self):
        x, pc1 = _correlated_2d()
        w, _ = oja_rule(x, np.array([0.3, -0.8]), lr=0.002)
        self.assertAlmostEqual(np.linalg.norm(w), 1.0, delta=0.02)
        self.assertGreater(abs(w @ pc1), 0.995)

    def test_sanger_recovers_ordered_components(self):
        rng = np.random.default_rng(1)
        basis = np.linalg.qr(rng.normal(size=(6, 6)))[0]
        x = (rng.normal(size=(2000, 6)) * np.array([3.0, 2.0, 1.0, 0.3, 0.2, 0.1])) @ basis.T
        w, ortho = sanger_rule(x, 3, lr=0.005, epochs=30, decay=0.1)
        for i in range(3):
            self.assertGreater(abs(w[i] @ basis[:, i]) / np.linalg.norm(w[i]), 0.98, msg=f'component {i}')
        self.assertLess(ortho[-1], 0.1)


class TestCompetitive(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        centers = np.array([[0, 0], [5, 5], [0, 5], [5, 0]], dtype=float)
        self.labels = np.repeat(np.arange(4), 100)
        self.x = centers[self.labels] + rng.normal(0, 0.4, size=(400, 2))

    def test_wta_finds_clusters(self):
        # With exactly k=4 prototypes, plain WTA can lock two prototypes into one cluster
        # (seed 0 does: purity 0.75); an over-complete layer avoids that local minimum.
        layer = CompetitiveLayer(8, 2, seed=0).init_from_samples(self.x)
        layer.fit(self.x, lr=0.05, epochs=10)
        self.assertGreater(cluster_purity(layer.predict(self.x), self.labels), 0.95)

    def test_conscience_revives_dead_units(self):
        plain = CompetitiveLayer(4, 2, seed=0).init_clustered(np.array([-10.0, -10.0]))
        plain.fit(self.x, lr=0.05, epochs=10)
        fair = CompetitiveLayer(4, 2, seed=0).init_clustered(np.array([-10.0, -10.0]))
        # bias must be ~k x squared distance to the data (4 x ~250): C=300 is too weak here.
        fair.fit(self.x, lr=0.05, epochs=10, conscience=True, bias_strength=1000.0, freq_rate=1e-2)
        self.assertLess(np.sum(plain.win_counts(self.x) > 0), 4)
        self.assertEqual(np.sum(fair.win_counts(self.x) > 0), 4)

    def test_som_orders_and_quantises(self):
        som = SOM(4, 4, 2, seed=0)
        hist = som.fit(self.x, epochs=10)
        self.assertLess(hist['quantization_error'][-1], hist['quantization_error'][0])
        self.assertLess(hist['topographic_error'][-1], 0.2)

    def test_purity_bounds(self):
        self.assertEqual(cluster_purity(self.labels, self.labels), 1.0)
        self.assertAlmostEqual(cluster_purity(np.zeros(400, int), self.labels), 0.25)


class TestErrorCorrection(unittest.TestCase):
    def test_perceptron_converges_on_separable(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2))])
        t = np.r_[-np.ones(50), np.ones(50)]
        res = Perceptron(2).fit(x, t, epochs=100)
        self.assertTrue(res['converged'])

    def test_perceptron_cannot_learn_xor(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        res = Perceptron(2).fit(x, np.array([-1, 1, 1, -1]), epochs=200)
        self.assertFalse(res['converged'])
        self.assertTrue(all(e > 0 for e in res['errors_per_epoch']))

    def test_perceptron_rejects_01_targets(self):
        with self.assertRaises(ValueError):
            Perceptron(2).fit(np.zeros((2, 2)), np.array([0, 1]))

    def test_lms_reaches_least_squares_solution(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 3))
        t = x @ np.array([0.5, -1.0, 2.0]) + 0.3 + rng.normal(0, 0.05, 300)
        ada = Adaline(3)
        ada.fit(x, t, lr=0.01, epochs=60)
        coef = np.linalg.lstsq(np.c_[x, np.ones(300)], t, rcond=None)[0]
        np.testing.assert_allclose(np.r_[ada.w, ada.b], coef, atol=0.02)


class TestHopfield(unittest.TestCase):
    def test_energy_never_increases_and_recalls(self):
        rng = np.random.default_rng(0)
        patterns = rng.choice([-1.0, 1.0], size=(3, 100))
        net = Hopfield(100).store(patterns)
        out, energies = net.recall(corrupt(patterns[0], 0.1, rng))
        self.assertTrue(np.all(np.diff(energies) <= 1e-9))
        np.testing.assert_array_equal(out, patterns[0])

    def test_pseudoinverse_stores_correlated_patterns(self):
        rng = np.random.default_rng(1)
        base = rng.choice([-1.0, 1.0], size=100)
        patterns = np.stack([np.where(rng.random(100) < 0.15, -base, base) for _ in range(6)])  # highly correlated
        net = Hopfield(100).store(patterns, rule='pseudoinverse')
        for p in patterns:
            out, _ = net.recall(p)
            np.testing.assert_array_equal(out, p)

    def test_capacity_drops_with_load(self):
        cap = hopfield_capacity(100, [3, 30], trials=3, seed=0)
        self.assertGreater(cap[3], 0.9)
        self.assertLess(cap[30], 0.2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Hopfield(4).store(np.array([[1, 0, 1, 1]]))
        with self.assertRaises(ValueError):
            Hopfield(4).store(np.ones((1, 4)), rule='magic')


class TestRBM(unittest.TestCase):
    def test_bars_and_stripes(self):
        data = bars_and_stripes(3)
        self.assertEqual(len(data), 14)
        self.assertEqual(len({tuple(d) for d in data}), 14)

    def test_exact_distribution_normalised(self):
        rbm = RBM(5, 3, seed=0)
        rbm.W += np.random.default_rng(0).normal(0, 1, size=rbm.W.shape)
        _, p = rbm.model_distribution()
        self.assertAlmostEqual(p.sum(), 1.0, places=12)
        # log-likelihood of every state equals log of its enumerated probability
        states, p = rbm.model_distribution()
        self.assertAlmostEqual(rbm.log_likelihood(states[3:4]), np.log(p[3]), places=10)

    def test_free_energy_matches_brute_force_marginal(self):
        rbm = RBM(3, 2, seed=1)
        rbm.W += 0.7
        rbm.b += 0.2
        rbm.c -= 0.3
        v = np.array([1.0, 0.0, 1.0])
        total = 0.0
        for h in [np.array(b, dtype=float) for b in [(0, 0), (0, 1), (1, 0), (1, 1)]]:
            total += np.exp(rbm.b @ v + rbm.c @ h + v @ rbm.W @ h)
        self.assertAlmostEqual(-rbm.free_energy(v)[0], np.log(total), places=12)

    def test_training_improves_likelihood(self):
        data = bars_and_stripes(3)
        for method in ('cd', 'exact'):
            rbm = RBM(9, 8, seed=0)
            start = rbm.log_likelihood(data)
            # the likelihood plateaus for ~200 full-batch epochs before the hidden units specialise
            hist = rbm.fit(data, epochs=1000, method=method, lr=0.5, batch_size=14)
            self.assertGreater(hist['log_likelihood'][-1], start + 1.0, msg=method)

    def test_enumeration_guard(self):
        with self.assertRaises(ValueError):
            RBM(25, 2).log_partition()


class TestEarlyStoppingMonitor(unittest.TestCase):
    def test_monitor_val_metric_picks_best_accuracy_epoch(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, 3))
        y = (x[:, 0] > 0).astype(int)
        model = M.MLP([3, 8, 2], seed=0)
        hist = M.train(model, M.SoftmaxCrossEntropyLoss(), M.Momentum(0.2), x, y, epochs=20, batch_size=10,
                       x_val=x[:20], y_val=y[:20], restore_best=True, monitor='val_metric')
        best = int(np.argmax(hist['val_metric'])) + 1  # argmax returns the first (earliest) maximum
        self.assertEqual(hist['best_epoch'], best)
        self.assertAlmostEqual(M.accuracy(model.forward(x[:20], training=False), y[:20]), max(hist['val_metric']))

    def test_invalid_monitor(self):
        with self.assertRaises(ValueError):
            M.train(M.MLP([2, 2]), M.SoftmaxCrossEntropyLoss(), M.GradientDescent(0.1), np.zeros((4, 2)),
                    np.zeros(4, int), epochs=1, monitor='train_loss')


class TestClassicalDemoSmoke(unittest.TestCase):
    def test_quick_demo_writes_figures(self):
        from src.fundamentals import classical_demo
        with tempfile.TemporaryDirectory() as tmp:
            summary = classical_demo.main(['--quick', '--out', tmp])
            for name in ['fig_hebbian.png', 'fig_competitive.png', 'fig_error_correction.png',
                         'fig_hopfield.png', 'fig_boltzmann.png', 'summary.json']:
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)), msg=name)
        self.assertTrue(summary['hopfield']['energy_monotone_non_increasing'])


if __name__ == '__main__':
    unittest.main()
