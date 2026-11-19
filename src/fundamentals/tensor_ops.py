"""
Mathematical tensor sandbox implemented from first principles.

The goal of this module is to make the arithmetic behind deep-learning tensors explicit.
NumPy arrays are used only as storage; the rules themselves (broadcasting, row-major
index arithmetic, reshaping, inner products, matrix products and the SVD) are written
out by hand so that every step can be inspected and tested against NumPy as a reference.

Contents:
- broadcast_shapes / broadcast_to / elementwise : NumPy broadcasting rules.
- row_major_strides / ravel_index / unravel_index / reshape : memory-layout arithmetic.
- inner_product / matmul / frobenius_norm : linear-algebra primitives.
- jacobi_eigh / svd / low_rank_approximation : SVD via cyclic Jacobi rotations.
"""

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np

Shape = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

def broadcast_shapes(shape_a: Sequence[int], shape_b: Sequence[int]) -> Shape:
    """
    Compute the broadcast result shape of two shapes.

    Rule: align shapes on the right, left-pad the shorter one with 1s, then each
    dimension pair must be equal or contain a 1; the result takes the larger size.

    Raises:
        ValueError: If any aligned dimension pair is incompatible.
    """
    a, b = tuple(shape_a), tuple(shape_b)
    ndim = max(len(a), len(b))
    padded_a = (1,) * (ndim - len(a)) + a
    padded_b = (1,) * (ndim - len(b)) + b

    result = []
    for axis, (da, db) in enumerate(zip(padded_a, padded_b)):
        if da == db or db == 1:
            result.append(da)
        elif da == 1:
            result.append(db)
        else:
            raise ValueError(
                f"Shapes {a} and {b} are not broadcastable: aligned axis {axis} has sizes {da} vs {db}"
            )
    return tuple(result)


def _source_index(out_index: Shape, src_shape: Shape) -> Shape:
    """Map an index in the broadcast output back to the index in a (smaller) source array."""
    offset = len(out_index) - len(src_shape)
    return tuple(0 if size == 1 else out_index[offset + i] for i, size in enumerate(src_shape))


def broadcast_to(x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """
    Materialise `x` broadcast to `shape` by explicit index mapping (no np.broadcast_to).

    Every output element at index (i_0, ..., i_n) reads the source element whose index is
    i_k on axes where the source has size > 1 and 0 on axes where the source has size 1.
    """
    x = np.asarray(x)
    target = tuple(shape)
    if broadcast_shapes(x.shape, target) != target:
        raise ValueError(f"Cannot broadcast array of shape {x.shape} to {target}")

    out = np.empty(target, dtype=x.dtype)
    for out_index in np.ndindex(*target):
        out[out_index] = x[_source_index(out_index, x.shape)]
    return out


def elementwise(
    op: Callable[[float, float], float],
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """Apply a scalar binary operation element by element with broadcasting."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out_shape = broadcast_shapes(a.shape, b.shape)
    out = np.empty(out_shape, dtype=np.float64)
    for out_index in np.ndindex(*out_shape):
        out[out_index] = op(a[_source_index(out_index, a.shape)], b[_source_index(out_index, b.shape)])
    return out


# ---------------------------------------------------------------------------
# Index arithmetic and reshaping
# ---------------------------------------------------------------------------

def row_major_strides(shape: Sequence[int]) -> Shape:
    """
    Element strides of a C-contiguous (row-major) array.

    stride[k] = product of shape[k+1:]; e.g. shape (2, 3, 4) -> strides (12, 4, 1).
    """
    strides = []
    running = 1
    for size in reversed(tuple(shape)):
        strides.append(running)
        running *= size
    return tuple(reversed(strides))


def ravel_index(index: Sequence[int], shape: Sequence[int]) -> int:
    """Convert a multi-dimensional index into a flat row-major offset."""
    index, shape = tuple(index), tuple(shape)
    if len(index) != len(shape):
        raise ValueError(f"Index {index} has {len(index)} dims but shape {shape} has {len(shape)}")
    for i, s in zip(index, shape):
        if not 0 <= i < s:
            raise IndexError(f"Index {index} out of bounds for shape {shape}")
    return sum(i * stride for i, stride in zip(index, row_major_strides(shape)))


def unravel_index(flat: int, shape: Sequence[int]) -> Shape:
    """Convert a flat row-major offset into a multi-dimensional index."""
    shape = tuple(shape)
    size = math.prod(shape)
    if not 0 <= flat < size:
        raise IndexError(f"Flat index {flat} out of bounds for size {size}")
    index = []
    for stride in row_major_strides(shape):
        index.append(flat // stride)
        flat %= stride
    return tuple(index)


def resolve_shape(size: int, new_shape: Sequence[int]) -> Shape:
    """Resolve at most one -1 placeholder in `new_shape` and validate the element count."""
    new_shape = list(new_shape)
    unknown = [i for i, s in enumerate(new_shape) if s == -1]
    if len(unknown) > 1:
        raise ValueError("Only one dimension may be -1")
    known = math.prod(s for s in new_shape if s != -1)
    if unknown:
        if known == 0 or size % known != 0:
            raise ValueError(f"Cannot reshape size {size} into {tuple(new_shape)}")
        new_shape[unknown[0]] = size // known
    if math.prod(new_shape) != size:
        raise ValueError(f"Cannot reshape size {size} into {tuple(new_shape)}")
    return tuple(new_shape)


def reshape(x: np.ndarray, new_shape: Sequence[int]) -> np.ndarray:
    """
    Reshape by copying elements in row-major order (supports a single -1).

    Element with flat offset k in the source lands at unravel_index(k, new_shape) in the
    output. This is exactly what np.reshape does for a C-contiguous array.
    """
    x = np.asarray(x)
    target = resolve_shape(x.size, new_shape)
    out = np.empty(target, dtype=x.dtype)
    for flat in range(x.size):
        out[unravel_index(flat, target)] = x[unravel_index(flat, x.shape)]
    return out


# ---------------------------------------------------------------------------
# Linear-algebra primitives
# ---------------------------------------------------------------------------

def inner_product(u: Sequence[float], v: Sequence[float]) -> float:
    """<u, v> = sum_i u_i * v_i for two vectors of equal length."""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.shape != v.shape:
        raise ValueError(f"Vectors must have equal length, got {u.size} and {v.size}")
    total = 0.0
    for ui, vi in zip(u, v):
        total += ui * vi
    return float(total)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[i, j] = <row_i(A), col_j(B)> for 2-D matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul expects 2-D matrices, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Inner dimensions differ: {a.shape} @ {b.shape}")
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            out[i, j] = inner_product(a[i, :], b[:, j])
    return out


def frobenius_norm(a: np.ndarray) -> float:
    """||A||_F = sqrt(sum of squared entries) = sqrt(<vec(A), vec(A)>)."""
    flat = np.asarray(a, dtype=np.float64).ravel()
    return math.sqrt(inner_product(flat, flat))


# ---------------------------------------------------------------------------
# Eigen-decomposition and SVD
# ---------------------------------------------------------------------------

def _jacobi_rotation(app: float, aqq: float, apq: float) -> Tuple[float, float]:
    """Cosine and sine of the rotation that zeroes the (p, q) entry of [[app, apq], [apq, aqq]]."""
    theta = (aqq - app) / (2.0 * apq)
    if abs(theta) > 1e150:
        t = 1.0 / (2.0 * theta)  # asymptotic form, avoids overflow in theta^2
    else:
        t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
    c = 1.0 / math.sqrt(t * t + 1.0)
    return c, t * c


def jacobi_eigh(
    sym: np.ndarray,
    tol: float = 1e-12,
    max_sweeps: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.

    Each rotation P(p, q, theta) is chosen so that (P^T S P)[p, q] = 0:
        theta = (S_qq - S_pp) / (2 S_pq),  t = sign(theta) / (|theta| + sqrt(theta^2 + 1)),
        c = 1 / sqrt(t^2 + 1),  s = t c.
    Repeated sweeps over all (p, q) pairs drive the off-diagonal mass to zero; the
    accumulated product of rotations holds the eigenvectors.

    Returns:
        (eigenvalues sorted descending, eigenvectors as columns in the same order).
    """
    a = np.array(sym, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {a.shape}")
    if not np.allclose(a, a.T, atol=1e-10 * max(1.0, np.abs(a).max())):
        raise ValueError("jacobi_eigh requires a symmetric matrix")

    n = a.shape[0]
    v = np.eye(n)
    scale = max(np.linalg.norm(a), 1e-300)

    for _ in range(max_sweeps):
        off = math.sqrt(max(np.sum(a ** 2) - np.sum(np.diag(a) ** 2), 0.0))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if abs(apq) <= 1e-300:
                    continue
                c, s = _jacobi_rotation(a[p, p], a[q, q], apq)

                # A <- A P (columns p, q), then A <- P^T A (rows p, q)
                col_p, col_q = a[:, p].copy(), a[:, q].copy()
                a[:, p] = c * col_p - s * col_q
                a[:, q] = s * col_p + c * col_q
                row_p, row_q = a[p, :].copy(), a[q, :].copy()
                a[p, :] = c * row_p - s * row_q
                a[q, :] = s * row_p + c * row_q
                a[p, q] = a[q, p] = 0.0

                # V <- V P
                vp, vq = v[:, p].copy(), v[:, q].copy()
                v[:, p] = c * vp - s * vq
                v[:, q] = s * vp + c * vq

    eigenvalues = np.diag(a).copy()
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], v[:, order]


def _complete_orthonormal(columns: List[np.ndarray], dim: int, needed: int) -> List[np.ndarray]:
    """Extend a set of orthonormal vectors with Gram-Schmidt on the standard basis."""
    basis = list(columns)
    for j in range(dim):
        if len(basis) >= needed:
            break
        candidate = np.zeros(dim)
        candidate[j] = 1.0
        for u in basis:
            candidate -= inner_product(u, candidate) * u
        norm = np.linalg.norm(candidate)
        if norm > 1e-8:
            basis.append(candidate / norm)
    return basis


def svd(a: np.ndarray, tol: float = 1e-14, max_sweeps: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin singular value decomposition A = U diag(s) V^T by one-sided (Hestenes) Jacobi.

    Idea: right-multiply A by plane rotations until all its columns are mutually
    orthogonal. For a column pair (p, q) let
        alpha = ||a_p||^2,  beta = ||a_q||^2,  gamma = <a_p, a_q>
    and apply the rotation that zeroes gamma (the same rotation Jacobi would use on the
    Gram matrix A^T A, but without ever forming it, so no precision is lost by squaring).
    At convergence  A V = W  with orthogonal columns, so
        s_i = ||w_i||,  u_i = w_i / s_i,  A = U diag(s) V^T.
    Zero singular values get their U columns from Gram-Schmidt completion.

    Returns:
        U [m, k], s [k] (descending), Vt [k, n] where k = min(m, n).
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"svd expects a 2-D matrix, got shape {a.shape}")
    transposed = a.shape[0] < a.shape[1]
    if transposed:
        # Work on the tall orientation so there are fewer column pairs to rotate.
        a = a.T
    m, n = a.shape
    k = n

    w = a.copy()
    v = np.eye(n)
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = float(w[:, p] @ w[:, p])
                beta = float(w[:, q] @ w[:, q])
                gamma = float(w[:, p] @ w[:, q])
                if abs(gamma) <= tol * math.sqrt(alpha * beta) or abs(gamma) <= 1e-300:
                    continue
                rotated = True
                c, s_ = _jacobi_rotation(alpha, beta, gamma)
                wp, wq = w[:, p].copy(), w[:, q].copy()
                w[:, p] = c * wp - s_ * wq
                w[:, q] = s_ * wp + c * wq
                vp, vq = v[:, p].copy(), v[:, q].copy()
                v[:, p] = c * vp - s_ * vq
                v[:, q] = s_ * vp + c * vq
        if not rotated:
            break

    s = np.linalg.norm(w, axis=0)
    order = np.argsort(s)[::-1]
    s, w, v = s[order], w[:, order], v[:, order]

    cutoff = max(s[0], 1e-300) * 1e-12 if s.size else 0.0
    u_cols: List[np.ndarray] = []
    for i in range(k):
        if s[i] > cutoff:
            u_cols.append(w[:, i] / s[i])
        else:
            s[i] = 0.0
    # Singular values are sorted descending, so zeros are last and the columns appended
    # by Gram-Schmidt completion line up with them.
    u_cols = _complete_orthonormal(u_cols, m, k)
    u = np.stack(u_cols, axis=1)
    vt = v.T

    if transposed:
        return vt.T, s, u.T
    return u, s, vt


def low_rank_approximation(a: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-r approximation (Eckart-Young): A_r = U_r diag(s_r) V_r^T."""
    u, s, vt = svd(a)
    r = max(0, min(rank, s.size))
    return (u[:, :r] * s[:r]) @ vt[:r, :]
