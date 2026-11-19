"""
Unit tests for the from-scratch tensor sandbox (src/fundamentals/tensor_ops.py)
and numerical gradient checking (src/fundamentals/gradcheck.py).
NumPy's own routines are used only as the reference answer.
"""

import unittest

import numpy as np

from src.fundamentals import tensor_ops as T
from src.fundamentals.gradcheck import gradient_check, numerical_gradient, relative_error


class TestBroadcasting(unittest.TestCase):
    CASES = [((3, 1), (1, 4)), ((5,), (2, 5)), ((2, 1, 3), (4, 1)), ((), (3, 2)), ((1,), (7,))]

    def test_broadcast_shapes_matches_numpy(self):
        for a, b in self.CASES:
            self.assertEqual(T.broadcast_shapes(a, b), np.broadcast_shapes(a, b), msg=f"{a} vs {b}")

    def test_incompatible_shapes_raise(self):
        with self.assertRaises(ValueError):
            T.broadcast_shapes((3, 2), (4, 2))

    def test_broadcast_to_matches_numpy(self):
        x = np.arange(3).reshape(3, 1)
        np.testing.assert_array_equal(T.broadcast_to(x, (2, 3, 4)), np.broadcast_to(x, (2, 3, 4)))

    def test_broadcast_to_rejects_bad_target(self):
        with self.assertRaises(ValueError):
            T.broadcast_to(np.zeros((3, 2)), (3, 3))

    def test_elementwise_matches_numpy(self):
        rng = np.random.default_rng(0)
        for a_shape, b_shape in self.CASES:
            a, b = rng.normal(size=a_shape), rng.normal(size=b_shape)
            np.testing.assert_allclose(T.elementwise(lambda p, q: p * q + p, a, b), a * b + a)


class TestIndexingAndReshape(unittest.TestCase):
    def test_strides(self):
        self.assertEqual(T.row_major_strides((2, 3, 4)), (12, 4, 1))
        arr = np.zeros((2, 3, 4))
        self.assertEqual(tuple(s // arr.itemsize for s in arr.strides), T.row_major_strides(arr.shape))

    def test_ravel_unravel_roundtrip(self):
        shape = (3, 4, 5)
        for flat in range(60):
            idx = T.unravel_index(flat, shape)
            self.assertEqual(idx, tuple(int(i) for i in np.unravel_index(flat, shape)))
            self.assertEqual(T.ravel_index(idx, shape), flat)

    def test_index_bounds(self):
        with self.assertRaises(IndexError):
            T.ravel_index((3, 0), (3, 4))
        with self.assertRaises(IndexError):
            T.unravel_index(12, (3, 4))

    def test_reshape_matches_numpy(self):
        x = np.arange(24).reshape(2, 3, 4)
        for shape in [(4, 6), (24,), (2, -1), (-1, 3, 2), (1, 24)]:
            np.testing.assert_array_equal(T.reshape(x, shape), x.reshape(shape))

    def test_reshape_invalid(self):
        with self.assertRaises(ValueError):
            T.reshape(np.arange(10), (3, -1))
        with self.assertRaises(ValueError):
            T.reshape(np.arange(10), (-1, -1))


class TestLinearAlgebra(unittest.TestCase):
    def test_inner_product(self):
        self.assertAlmostEqual(T.inner_product([1, 2, 3], [4, 5, 6]), 32.0)
        with self.assertRaises(ValueError):
            T.inner_product([1, 2], [1, 2, 3])

    def test_matmul_matches_numpy(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(4, 3)), rng.normal(size=(3, 5))
        np.testing.assert_allclose(T.matmul(a, b), a @ b, atol=1e-12)
        with self.assertRaises(ValueError):
            T.matmul(a, a)

    def test_frobenius_norm(self):
        a = np.random.default_rng(2).normal(size=(5, 6))
        self.assertAlmostEqual(T.frobenius_norm(a), np.linalg.norm(a), places=12)

    def test_jacobi_eigh_matches_numpy(self):
        rng = np.random.default_rng(3)
        m = rng.normal(size=(6, 6))
        sym = m + m.T
        vals, vecs = T.jacobi_eigh(sym)
        np.testing.assert_allclose(vals, np.sort(np.linalg.eigvalsh(sym))[::-1], atol=1e-10)
        np.testing.assert_allclose(vecs @ np.diag(vals) @ vecs.T, sym, atol=1e-10)
        np.testing.assert_allclose(vecs.T @ vecs, np.eye(6), atol=1e-10)

    def test_jacobi_rejects_nonsymmetric(self):
        with self.assertRaises(ValueError):
            T.jacobi_eigh(np.array([[1.0, 2.0], [0.0, 1.0]]))

    def _assert_valid_svd(self, a):
        u, s, vt = T.svd(a)
        k = min(a.shape)
        self.assertEqual(u.shape, (a.shape[0], k))
        self.assertEqual(vt.shape, (k, a.shape[1]))
        np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-8)
        np.testing.assert_allclose((u * s) @ vt, a, atol=1e-8)
        np.testing.assert_allclose(u.T @ u, np.eye(k), atol=1e-8)
        np.testing.assert_allclose(vt @ vt.T, np.eye(k), atol=1e-8)

    def test_svd_tall_wide_square(self):
        rng = np.random.default_rng(4)
        for shape in [(8, 5), (5, 8), (6, 6)]:
            self._assert_valid_svd(rng.normal(size=shape))

    def test_svd_rank_deficient(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(7, 2)) @ rng.normal(size=(2, 5))  # rank 2
        self._assert_valid_svd(a)
        _, s, _ = T.svd(a)
        self.assertTrue(np.all(s[2:] < 1e-8))

    def test_eckart_young_error(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=(10, 8))
        s = np.linalg.svd(a, compute_uv=False)
        for r in [1, 3, 5]:
            err = np.linalg.norm(a - T.low_rank_approximation(a, r))
            self.assertAlmostEqual(err, np.sqrt(np.sum(s[r:] ** 2)), places=7)


class TestGradientCheck(unittest.TestCase):
    @staticmethod
    def f(x):
        return float(np.sum(x ** 2 * np.sin(x)))

    @staticmethod
    def grad_f(x):
        return 2 * x * np.sin(x) + x ** 2 * np.cos(x)

    def test_correct_gradient_passes(self):
        x = np.random.default_rng(7).normal(size=(3, 4))
        result = gradient_check(self.f, self.grad_f, x)
        self.assertTrue(result['passed'], msg=f"rel err {result['relative_error']}")

    def test_wrong_gradient_fails(self):
        x = np.random.default_rng(8).normal(size=(5,))
        result = gradient_check(self.f, lambda v: 2 * v * np.sin(v), x)
        self.assertFalse(result['passed'])

    def test_numerical_gradient_does_not_mutate_input(self):
        x = np.array([1.0, 2.0])
        numerical_gradient(self.f, x)
        np.testing.assert_array_equal(x, [1.0, 2.0])

    def test_relative_error_zero_case(self):
        self.assertEqual(relative_error(np.zeros(3), np.zeros(3)), 0.0)


if __name__ == '__main__':
    unittest.main()
