"""
Unit tests for Step 5 additions: BatchNorm1d, Dropout, augmentation / early stopping /
divergence handling in train(), the synthetic land-cover dataset and the ablation runner.
"""

import os
import tempfile
import unittest

import numpy as np

from src.fundamentals import mlp as M
from src.fundamentals.gradcheck import check_model_gradients, relative_error
from src.fundamentals.landcover_toy import (
    NUM_CLASSES, ChannelStandardizer, dihedral_augment, flatten, make_landcover_patches, make_splits,
)


class TestBatchNorm(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(3.0, 2.0, size=(8, 5))
        self.g = rng.normal(size=(8, 5))

    def test_training_output_is_normalized(self):
        bn = M.BatchNorm1d(5)
        out = bn.forward(self.x, training=True)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)

    def _numeric_input_grad(self, bn, training):
        def loss(xv):
            return float(np.sum(bn.forward(xv, training=training) * self.g))
        eps, num = 1e-6, np.zeros_like(self.x)
        for idx in np.ndindex(*self.x.shape):
            xp, xm = self.x.copy(), self.x.copy()
            xp[idx] += eps
            xm[idx] -= eps
            num[idx] = (loss(xp) - loss(xm)) / (2 * eps)
        return num

    def test_backward_training_mode(self):
        bn = M.BatchNorm1d(5)
        bn.gamma[...] = np.linspace(0.5, 2.0, 5)
        num = self._numeric_input_grad(bn, training=True)
        bn.forward(self.x, training=True)
        self.assertLess(relative_error(bn.backward(self.g), num), 1e-7)

    def test_backward_inference_mode(self):
        bn = M.BatchNorm1d(5)
        bn.forward(self.x, training=True)  # populate running stats
        num = self._numeric_input_grad(bn, training=False)
        bn.forward(self.x, training=False)
        self.assertLess(relative_error(bn.backward(self.g), num), 1e-7)

    def test_mlp_with_batchnorm_gradcheck(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(10, 4)), rng.integers(0, 3, 10)
        model = M.MLP([4, 6, 5, 3], activation='tanh', batchnorm=True, seed=0)
        model.forward(x, training=True)  # non-trivial running statistics
        self.assertTrue(check_model_gradients(model, M.SoftmaxCrossEntropyLoss(), x, y)['passed'])

    def test_running_stats_saved(self):
        model = M.MLP([4, 6, 2], batchnorm=True, seed=0)
        model.forward(np.random.default_rng(2).normal(5, 3, size=(16, 4)), training=True)
        other = M.MLP([4, 6, 2], batchnorm=True, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bn.npz')
            model.save(path)
            other.load(path)
        x = np.ones((3, 4))
        np.testing.assert_allclose(other.forward(x, training=False), model.forward(x, training=False))

    def test_single_sample_batch_rejected(self):
        with self.assertRaises(ValueError):
            M.BatchNorm1d(3).forward(np.zeros((1, 3)), training=True)


class TestDropout(unittest.TestCase):
    def test_identity_at_inference(self):
        x = np.ones((4, 10))
        np.testing.assert_array_equal(M.Dropout(0.5).forward(x, training=False), x)

    def test_expectation_preserved(self):
        out = M.Dropout(0.3, rng=np.random.default_rng(0)).forward(np.ones((2000, 50)), training=True)
        self.assertAlmostEqual(out.mean(), 1.0, delta=0.01)
        self.assertAlmostEqual(np.mean(out == 0), 0.3, delta=0.01)

    def test_backward_uses_same_mask(self):
        layer = M.Dropout(0.5, rng=np.random.default_rng(1))
        out = layer.forward(np.ones((5, 6)), training=True)
        np.testing.assert_array_equal(layer.backward(np.ones((5, 6))), out)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            M.Dropout(1.0)


class TestTrainingExtensions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(40, 3))
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).astype(int)

    def test_augment_fn_applied_to_training_batches_only(self):
        seen = []

        def spy(xb, rng):
            seen.append(xb.shape[0])
            return xb

        M.train(M.MLP([3, 4, 2]), M.SoftmaxCrossEntropyLoss(), M.GradientDescent(0.1), self.x, self.y,
                epochs=2, batch_size=10, x_val=self.x, y_val=self.y, augment_fn=spy)
        self.assertEqual(seen, [10] * 8)

    def test_restore_best_returns_best_epoch_weights(self):
        model = M.MLP([3, 16, 2], seed=0)
        hist = M.train(model, M.SoftmaxCrossEntropyLoss(), M.Momentum(0.5), self.x, self.y, epochs=30,
                       batch_size=8, x_val=self.x[:10], y_val=1 - self.y[:10], restore_best=True)
        val_loss = M.SoftmaxCrossEntropyLoss().forward(model.forward(self.x[:10], training=False), 1 - self.y[:10])
        self.assertAlmostEqual(val_loss, min(hist['val_loss']), places=10)
        self.assertEqual(hist['best_epoch'], int(np.argmin(hist['val_loss'])) + 1)

    def test_restore_best_requires_validation(self):
        with self.assertRaises(ValueError):
            M.train(M.MLP([3, 2]), M.SoftmaxCrossEntropyLoss(), M.GradientDescent(0.1), self.x, self.y,
                    epochs=1, restore_best=True)

    def test_divergence_detected_and_stops(self):
        # MSE with an oversized step grows geometrically until it overflows to inf / NaN.
        with np.errstate(all='ignore'):
            hist = M.train(M.MLP([3, 8, 1], activation='identity', seed=0), M.MSELoss(),
                           M.GradientDescent(50.0), self.x, self.y.astype(float), epochs=200, metric_fn=None)
        self.assertTrue(hist['diverged'])
        self.assertLess(len(hist['train_loss']), 50)

    def test_batchnorm_skips_singleton_batch(self):
        # 41 samples with batch 8 leaves a final batch of 1, which BatchNorm cannot use.
        x, y = np.vstack([self.x, self.x[:1]]), np.append(self.y, self.y[0])
        hist = M.train(M.MLP([3, 4, 2], batchnorm=True), M.SoftmaxCrossEntropyLoss(), M.GradientDescent(0.1),
                       x, y, epochs=1, batch_size=8)
        self.assertEqual(hist['grad_steps'], [5])


class TestLandcoverToy(unittest.TestCase):
    def test_shapes_range_balance(self):
        x, y = make_landcover_patches(40, patch=10, seed=0)
        self.assertEqual(x.shape, (40, 3, 10, 10))
        self.assertTrue(np.all((x >= 0) & (x <= 1)))
        np.testing.assert_array_equal(np.bincount(y, minlength=NUM_CLASSES), [10] * NUM_CLASSES)

    def test_deterministic_and_splits_disjoint(self):
        a, b = make_splits(20, 20, 20, seed=1), make_splits(20, 20, 20, seed=1)
        np.testing.assert_array_equal(a['train'][0], b['train'][0])
        self.assertFalse(np.allclose(a['train'][0], a['val'][0]))

    def test_label_noise_fraction(self):
        _, clean = make_landcover_patches(2000, seed=3)
        _, noisy = make_landcover_patches(2000, seed=3, label_noise=0.2)
        self.assertAlmostEqual(np.mean(clean != noisy), 0.2, delta=0.03)

    def test_standardizer_uses_train_stats(self):
        x, _ = make_landcover_patches(40, seed=0)
        scaler = ChannelStandardizer().fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z.mean(axis=(0, 2, 3)), 0.0, atol=1e-10)
        with self.assertRaises(RuntimeError):
            ChannelStandardizer().transform(x)

    def test_dihedral_augment_is_a_permutation_of_pixels(self):
        x, _ = make_landcover_patches(6, patch=10, seed=0)
        flat = flatten(x)
        out = dihedral_augment(3, 10)(flat, np.random.default_rng(0))
        self.assertEqual(out.shape, flat.shape)
        for a, b in zip(flat.reshape(6, 3, -1), out.reshape(6, 3, -1)):
            np.testing.assert_allclose(np.sort(a, axis=1), np.sort(b, axis=1))


class TestExperimentRunner(unittest.TestCase):
    def test_quick_run_produces_outputs(self):
        from src.fundamentals import experiments as E
        with tempfile.TemporaryDirectory() as tmp:
            res = E.main(['--quick', '--seeds', '1', '--out', tmp, '--studies', 'regularization,augmentation'])
            for name in ['results.json', 'summary.csv', 'fig_regularization.png', 'fig_augmentation.png',
                         'fig_landcover_samples.png']:
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)), msg=name)
        self.assertIn('none', res['regularization'])
        self.assertGreater(res['augmentation']['none']['train_acc_mean'], 1 / NUM_CLASSES)

    def test_quadratic_paths_momentum_beats_gd(self):
        from src.fundamentals.experiments import quadratic_paths
        paths = quadratic_paths()
        self.assertLess(paths['Momentum']['final_loss'], paths['GD']['final_loss'])
        self.assertLess(paths['Nesterov']['final_loss'], paths['GD']['final_loss'])


if __name__ == '__main__':
    unittest.main()
