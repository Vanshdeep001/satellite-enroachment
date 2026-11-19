"""
Unit tests for from-scratch neurons, MLP backpropagation, optimizers and tracking
(src/fundamentals/neurons.py, src/fundamentals/mlp.py).
"""

import itertools
import os
import tempfile
import unittest

import numpy as np

from src.fundamentals import mlp as M
from src.fundamentals.gradcheck import check_model_gradients
from src.fundamentals.neurons import (
    McCullochPittsNeuron, SigmoidNeuron, mp_and, mp_and_not, mp_not, mp_or, mp_xor,
)

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([0, 1, 1, 0])


class TestMcCullochPitts(unittest.TestCase):
    def test_logic_gates(self):
        for a, b in itertools.product((0, 1), repeat=2):
            self.assertEqual(mp_and()([a, b]), a & b)
            self.assertEqual(mp_or()([a, b]), a | b)
            self.assertEqual(mp_and_not()([a, b]), int(a and not b))
            self.assertEqual(mp_xor([a, b]), a ^ b)
        self.assertEqual(mp_not()([0]), 1)
        self.assertEqual(mp_not()([1]), 0)

    def test_absolute_inhibition(self):
        neuron = McCullochPittsNeuron(3, threshold=1, inhibitory=[2])
        self.assertEqual(neuron([1, 1, 1]), 0)
        self.assertEqual(neuron([1, 0, 0]), 1)

    def test_truth_table_size(self):
        self.assertEqual(len(mp_and(3).truth_table()), 8)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            mp_and()([1, 2])
        with self.assertRaises(ValueError):
            mp_and()([1])
        with self.assertRaises(ValueError):
            McCullochPittsNeuron(2, threshold=1, inhibitory=[5])


class TestSigmoidNeuron(unittest.TestCase):
    def test_learns_and_gate(self):
        neuron = SigmoidNeuron(2, seed=0)
        history = neuron.fit(XOR_X, np.array([0, 0, 0, 1]), lr=2.0, epochs=2000)
        np.testing.assert_array_equal(neuron.predict(XOR_X), [0, 0, 0, 1])
        self.assertLess(history['loss'][-1], history['loss'][0])

    def test_gradient_matches_numeric(self):
        neuron = SigmoidNeuron(2, seed=1)
        y = np.array([0, 1, 1, 1])
        dw, db = neuron.gradients(XOR_X, y)
        eps = 1e-6
        for i in range(2):
            neuron.w[i] += eps
            plus = neuron.loss(XOR_X, y)
            neuron.w[i] -= 2 * eps
            minus = neuron.loss(XOR_X, y)
            neuron.w[i] += eps
            self.assertAlmostEqual(dw[i], (plus - minus) / (2 * eps), places=6)


class TestActivationsAndLosses(unittest.TestCase):
    def test_activation_derivatives(self):
        z = np.linspace(-3, 3, 13) + 0.05  # avoid the ReLU kink at 0
        eps = 1e-6
        for name in ['sigmoid', 'tanh', 'relu', 'identity']:
            fn, grad = M.ACTIVATIONS[name]
            numeric = (fn(z + eps) - fn(z - eps)) / (2 * eps)
            np.testing.assert_allclose(grad(z), numeric, atol=1e-6, err_msg=name)

    def test_sigmoid_is_stable(self):
        out = M.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
        self.assertFalse(np.any(np.isnan(out)))

    def test_softmax_ce_value(self):
        loss = M.SoftmaxCrossEntropyLoss()
        value = loss.forward(np.zeros((2, 4)), np.array([0, 3]))
        self.assertAlmostEqual(value, np.log(4.0))

    def test_bce_matches_naive_formula(self):
        z = np.array([[-2.0], [0.5], [3.0]])
        t = np.array([0, 1, 1])
        p = 1 / (1 + np.exp(-z.ravel()))
        naive = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
        self.assertAlmostEqual(M.BCEWithLogitsLoss().forward(z, t), naive, places=12)

    def test_unknown_activation(self):
        with self.assertRaises(ValueError):
            M.Activation('swishy')


class TestMLPBackprop(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(6, 3))
        self.y_cls = rng.integers(0, 4, size=6)
        self.y_bin = rng.integers(0, 2, size=6)
        self.y_reg = rng.normal(size=(6, 2))

    def test_gradcheck_all_losses_and_activations(self):
        for activation in ['sigmoid', 'tanh', 'relu']:
            for loss, out_dim, y in [
                (M.SoftmaxCrossEntropyLoss(), 4, self.y_cls),
                (M.BCEWithLogitsLoss(), 1, self.y_bin),
                (M.MSELoss(), 2, self.y_reg),
            ]:
                model = M.MLP([3, 5, 4, out_dim], activation=activation, seed=1)
                result = check_model_gradients(model, loss, self.x, y)
                self.assertTrue(
                    result['passed'],
                    msg=f"{activation}/{type(loss).__name__}: {result['max_relative_error']:.2e}",
                )

    def test_parameter_count(self):
        model = M.MLP([2, 8, 3])
        self.assertEqual(model.num_parameters(), 2 * 8 + 8 + 8 * 3 + 3)

    def test_input_shape_validation(self):
        with self.assertRaises(ValueError):
            M.MLP([3, 2]).forward(np.zeros((4, 5)))

    def test_seed_reproducibility(self):
        a, b = M.MLP([2, 4, 1], seed=42), M.MLP([2, 4, 1], seed=42)
        for (_, va, _), (_, vb, _) in zip(a.named_parameters(), b.named_parameters()):
            np.testing.assert_array_equal(va, vb)

    def test_save_load_roundtrip(self):
        model = M.MLP([2, 4, 1], seed=3)
        other = M.MLP([2, 4, 1], seed=99)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mlp.npz')
            model.save(path)
            other.load(path)
        np.testing.assert_allclose(other.forward(XOR_X), model.forward(XOR_X))

    def test_learns_xor(self):
        model = M.MLP([2, 8, 1], activation='tanh', seed=0)
        history = M.train(model, M.BCEWithLogitsLoss(), M.Momentum(lr=0.5), XOR_X, XOR_Y, epochs=500)
        self.assertEqual(history['train_metric'][-1], 1.0)
        self.assertLess(history['train_loss'][-1], 0.1)


class TestOptimizers(unittest.TestCase):
    def _quadratic_steps(self, optimizer, steps=50):
        """Minimise 0.5 * ||w||^2 (gradient = w) and return the final norm."""
        w = np.array([3.0, -2.0])
        for _ in range(steps):
            optimizer.step([('w.W', w, w.copy())])
        return np.linalg.norm(w)

    def test_all_optimizers_converge_on_quadratic(self):
        for opt in [M.GradientDescent(0.1), M.Momentum(0.1, 0.9), M.Nesterov(0.1, 0.9)]:
            self.assertLess(self._quadratic_steps(opt, steps=200), 1e-3, msg=opt.name)

    def test_nesterov_first_step(self):
        # From v=0: v1 = -lr g, theta1 = theta0 + (1 + beta) v1.
        w = np.array([1.0])
        M.Nesterov(lr=0.1, beta=0.9).step([('p.W', w, np.array([2.0]))])
        self.assertAlmostEqual(w[0], 1.0 - 1.9 * 0.2)

    def test_weight_decay_only_on_weights(self):
        w, b = np.array([1.0]), np.array([1.0])
        M.GradientDescent(lr=0.1, weight_decay=0.5).step([('l.W', w, np.zeros(1)), ('l.b', b, np.zeros(1))])
        self.assertAlmostEqual(w[0], 0.95)
        self.assertAlmostEqual(b[0], 1.0)

    def test_invalid_lr(self):
        with self.assertRaises(ValueError):
            M.GradientDescent(lr=0.0)

    def test_batch_size_controls_updates(self):
        # Batch GD makes one update per epoch; SGD with batch 1 makes N updates.
        calls = {'n': 0}

        class Counting(M.GradientDescent):
            def step(self, params):
                calls['n'] += 1
                super().step(params)

        model = M.MLP([2, 3, 1], seed=0)
        M.train(model, M.BCEWithLogitsLoss(), Counting(0.1), XOR_X, XOR_Y, epochs=2)
        self.assertEqual(calls['n'], 2)
        calls['n'] = 0
        M.train(model, M.BCEWithLogitsLoss(), Counting(0.1), XOR_X, XOR_Y, epochs=2, batch_size=1)
        self.assertEqual(calls['n'], 8)


class TestTracker(unittest.TestCase):
    def test_tracker_separates_hyper_and_learned(self):
        model = M.MLP([2, 4, 1], seed=0)
        opt = M.Momentum(lr=0.3)
        hyper = {'layer_sizes': model.layer_sizes, 'batch_size': None, 'epochs': 3, **opt.hyperparameters()}
        tracker = M.ExperimentTracker(hyper)
        M.train(model, M.BCEWithLogitsLoss(), opt, XOR_X, XOR_Y, epochs=3, tracker=tracker)

        self.assertEqual(len(tracker.epochs), 3)
        self.assertIn('layer0.W', tracker.epochs[-1]['parameters'])
        summary = tracker.summary(model)
        self.assertEqual(summary['num_learned_parameters'], model.num_parameters())
        self.assertEqual(summary['hyperparameters']['optimizer'], 'momentum')

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'run.json')
            tracker.save(path, model)
            self.assertTrue(os.path.getsize(path) > 0)


if __name__ == '__main__':
    unittest.main()
