"""
Classical learning rules (Project Step 6), implemented from scratch in NumPy.

Hebbian family (unsupervised, local, correlation-driven)
    hebbian_rule  : dw = lr * y * x                      (y = w.x) -> |w| grows without bound
    oja_rule      : dw = lr * y * (x - y w)              -> converges to the unit 1st principal axis
    sanger_rule   : dW = lr * (y x^T - LT(y y^T) W)      -> first k principal axes, in order (GHA)

Competitive learning (unsupervised, winner-take-all)
    CompetitiveLayer : dw_win = lr * (x - w_win); optional DeSieno "conscience" bias that
                       handicaps frequent winners so no prototype is left dead.
    SOM              : Kohonen self-organising map; the winner AND its grid neighbours move.

Error-correction learning (supervised)
    Perceptron : w <- w + lr * t * x only on mistakes (t in {-1, +1}); converges iff separable.
    Adaline    : LMS / Widrow-Hoff delta rule, w <- w + lr * (t - w.x) x; minimises squared error.

Energy-based models
    Hopfield : symmetric recurrent net, E(s) = -1/2 s^T W s; asynchronous updates never raise E,
               so stored patterns are attractors. Hebbian or pseudo-inverse (projection) storage.
    RBM      : restricted Boltzmann machine, p(v, h) ~ exp(-E(v, h)),
               E = -b.v - c.h - v^T W h. Trained by contrastive divergence (CD-k) or, for tiny
               visible layers, by the exact maximum-likelihood gradient (full enumeration).
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


def _logaddexp0(z: np.ndarray) -> np.ndarray:
    """log(1 + exp(z)), stable for large |z|."""
    return np.logaddexp(0.0, z)


# ---------------------------------------------------------------------------
# Hebbian family
# ---------------------------------------------------------------------------

def hebbian_rule(x: np.ndarray, w0: np.ndarray, lr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Plain Hebb on a stream of samples x [T, d]. Returns final w and the trajectory [T+1, d]."""
    w = np.array(w0, dtype=np.float64)
    path = [w.copy()]
    for xt in x:
        y = w @ xt
        w = w + lr * y * xt
        path.append(w.copy())
    return w, np.array(path)


def oja_rule(x: np.ndarray, w0: np.ndarray, lr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Oja's normalised Hebbian rule. The -y^2 w term is a local weight decay that keeps |w| -> 1."""
    w = np.array(w0, dtype=np.float64)
    path = [w.copy()]
    for xt in x:
        y = w @ xt
        w = w + lr * y * (xt - y * w)
        path.append(w.copy())
    return w, np.array(path)


def sanger_rule(
    x: np.ndarray,
    k: int,
    lr: float,
    epochs: int = 10,
    seed: int = 0,
    decay: float = 0.0
) -> Tuple[np.ndarray, List[float]]:
    """
    Generalised Hebbian Algorithm (Sanger 1989) on zero-mean data x [N, d].

    W [k, d]; for each sample: y = W x,  W += lr_t * (y x^T - lower_triangular(y y^T) W).
    Unit i sees the input with units 0..i-1 already "explained away", so the rows converge
    to the top-k principal axes in decreasing order of variance.
    lr_t = lr / (1 + decay * t) with t the epoch index.

    Returns W and the per-epoch orthonormality error ||W W^T - I||_F.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = rng.normal(0.0, 0.1 / np.sqrt(d), size=(k, d))
    history = []
    for epoch in range(epochs):
        step = lr / (1.0 + decay * epoch)
        for i in rng.permutation(n):
            xt = x[i]
            y = w @ xt
            w += step * (np.outer(y, xt) - np.tril(np.outer(y, y)) @ w)
        history.append(float(np.linalg.norm(w @ w.T - np.eye(k))))
    return w, history


# ---------------------------------------------------------------------------
# Competitive learning
# ---------------------------------------------------------------------------

class CompetitiveLayer:
    """
    Winner-take-all vector quantiser.

    Winner j* = argmin_j ||x - w_j||^2 - bias_j, and only w_j* moves toward x.
    With conscience=True (DeSieno 1988) each unit tracks its win frequency p_j and receives
    bias_j = C * (1/k - p_j): units that win too often are handicapped and starved units are
    helped, which removes "dead" prototypes caused by poor initialisation.
    """

    def __init__(self, k: int, dim: int, seed: int = 0):
        self.k = k
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((k, dim))
        self.win_freq = np.full(k, 1.0 / k)

    def init_from_samples(self, x: np.ndarray) -> 'CompetitiveLayer':
        self.w = x[self.rng.choice(len(x), self.k, replace=False)].astype(np.float64).copy()
        return self

    def init_clustered(self, center: np.ndarray, spread: float = 0.05) -> 'CompetitiveLayer':
        """Deliberately poor init: every prototype starts in one tiny region."""
        self.w = center + self.rng.normal(0.0, spread, size=(self.k, self.dim))
        return self

    def _sq_dist(self, x: np.ndarray) -> np.ndarray:
        return np.sum((x[:, None, :] - self.w[None, :, :]) ** 2, axis=2)

    def fit(
        self,
        x: np.ndarray,
        lr: float = 0.05,
        epochs: int = 20,
        conscience: bool = False,
        bias_strength: float = 30.0,
        freq_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        bias_strength (DeSieno's C) must be comparable to k times the squared distance that separates a
        starved prototype from the data; too small (e.g. 10 on z-scored 4-D features) cannot
        pull far-away dead units back in.
        """
        history: Dict[str, List[float]] = {'quantization_error': [], 'active_units': []}
        for _ in range(epochs):
            for i in self.rng.permutation(len(x)):
                xt = x[i]
                dist = np.sum((self.w - xt) ** 2, axis=1)
                if conscience:
                    dist = dist - bias_strength * (1.0 / self.k - self.win_freq)
                j = int(np.argmin(dist))
                self.w[j] += lr * (xt - self.w[j])
                if conscience:
                    won = np.zeros(self.k)
                    won[j] = 1.0
                    self.win_freq += freq_rate * (won - self.win_freq)
            history['quantization_error'].append(self.quantization_error(x))
            history['active_units'].append(int(np.sum(self.win_counts(x) > 0)))
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmin(self._sq_dist(x), axis=1)

    def win_counts(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.predict(x), minlength=self.k)

    def quantization_error(self, x: np.ndarray) -> float:
        return float(np.mean(np.min(self._sq_dist(x), axis=1)))


def cluster_purity(assignments: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples whose cluster's majority label equals their own label."""
    total = 0
    for c in np.unique(assignments):
        total += np.bincount(labels[assignments == c]).max()
    return float(total / len(labels))


class SOM:
    """
    Kohonen self-organising map on a rows x cols grid.

    For each sample: winner j* = argmin ||x - w_j||; every unit moves by
        dw_j = lr_t * h_t(j, j*) * (x - w_j),  h_t = exp(-||g_j - g_j*||^2 / (2 sigma_t^2))
    where g_j is the grid coordinate. lr_t and sigma_t decay exponentially, so the map first
    orders itself globally and then fine-tunes locally.
    """

    def __init__(self, rows: int, cols: int, dim: int, seed: int = 0):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.rng = np.random.default_rng(seed)
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=np.float64)
        self.w = self.rng.normal(0.0, 0.1, size=(rows * cols, dim))

    def fit(self, x: np.ndarray, epochs: int = 20, lr0: float = 0.5, lr_end: float = 0.01,
            sigma0: Optional[float] = None, sigma_end: float = 0.5) -> Dict[str, List[float]]:
        sigma0 = sigma0 if sigma0 is not None else max(self.rows, self.cols) / 2.0
        total = epochs * len(x)
        history: Dict[str, List[float]] = {'quantization_error': [], 'topographic_error': []}
        t = 0
        for _ in range(epochs):
            for i in self.rng.permutation(len(x)):
                frac = t / max(total - 1, 1)
                lr = lr0 * (lr_end / lr0) ** frac
                sigma = sigma0 * (sigma_end / sigma0) ** frac
                xt = x[i]
                j = int(np.argmin(np.sum((self.w - xt) ** 2, axis=1)))
                h = np.exp(-np.sum((self.grid - self.grid[j]) ** 2, axis=1) / (2 * sigma ** 2))
                self.w += lr * h[:, None] * (xt - self.w)
                t += 1
            history['quantization_error'].append(self.quantization_error(x))
            history['topographic_error'].append(self.topographic_error(x))
        return history

    def _sq_dist(self, x: np.ndarray) -> np.ndarray:
        return np.sum((x[:, None, :] - self.w[None, :, :]) ** 2, axis=2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmin(self._sq_dist(x), axis=1)

    def quantization_error(self, x: np.ndarray) -> float:
        return float(np.mean(np.min(self._sq_dist(x), axis=1)))

    def topographic_error(self, x: np.ndarray) -> float:
        """Fraction of samples whose best and second-best units are not grid neighbours."""
        order = np.argsort(self._sq_dist(x), axis=1)[:, :2]
        gap = np.abs(self.grid[order[:, 0]] - self.grid[order[:, 1]]).max(axis=1)
        return float(np.mean(gap > 1))


# ---------------------------------------------------------------------------
# Error-correction learning
# ---------------------------------------------------------------------------

class Perceptron:
    """Rosenblatt perceptron with targets in {-1, +1}; updates only on misclassified samples."""

    def __init__(self, dim: int, seed: int = 0):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.rng = np.random.default_rng(seed)

    def decision(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.decision(x) >= 0, 1, -1)

    def fit(self, x: np.ndarray, t: np.ndarray, lr: float = 1.0, epochs: int = 100) -> Dict[str, Any]:
        t = np.asarray(t)
        if not set(np.unique(t)) <= {-1, 1}:
            raise ValueError("Perceptron targets must be -1 or +1")
        errors = []
        for _ in range(epochs):
            mistakes = 0
            for i in self.rng.permutation(len(x)):
                if t[i] * (x[i] @ self.w + self.b) <= 0:
                    self.w += lr * t[i] * x[i]
                    self.b += lr * t[i]
                    mistakes += 1
            errors.append(mistakes)
            if mistakes == 0:
                break
        return {'errors_per_epoch': errors, 'converged': errors[-1] == 0, 'epochs_run': len(errors)}


class Adaline:
    """ADALINE trained with the LMS (Widrow-Hoff / delta) rule on a linear output."""

    def __init__(self, dim: int, seed: int = 0):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.rng = np.random.default_rng(seed)

    def decision(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.decision(x) >= 0, 1, -1)

    def fit(self, x: np.ndarray, t: np.ndarray, lr: float = 0.01, epochs: int = 50) -> Dict[str, Any]:
        t = np.asarray(t, dtype=np.float64)
        mse, errors = [], []
        for _ in range(epochs):
            for i in self.rng.permutation(len(x)):
                err = t[i] - (x[i] @ self.w + self.b)
                self.w += lr * err * x[i]
                self.b += lr * err
            mse.append(float(np.mean((t - self.decision(x)) ** 2)))
            errors.append(int(np.sum(self.predict(x) != t)))
        return {'mse_per_epoch': mse, 'errors_per_epoch': errors}


# ---------------------------------------------------------------------------
# Hopfield network
# ---------------------------------------------------------------------------

class Hopfield:
    """Binary (+1/-1) Hopfield network with asynchronous updates."""

    def __init__(self, n_units: int):
        self.n = n_units
        self.W = np.zeros((n_units, n_units))

    def store(self, patterns: np.ndarray, rule: str = 'hebbian') -> 'Hopfield':
        """
        hebbian      : W = (1/N) sum_p x_p x_p^T with zero diagonal. Capacity ~0.138 N for random
                       patterns; correlated patterns interfere and create spurious attractors.
        pseudoinverse: W = X^T (X X^T)^-1 X (projection onto the pattern span). Every linearly
                       independent pattern is an exact fixed point, even if patterns are correlated.
        """
        x = np.asarray(patterns, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(f"Expected patterns [P, {self.n}], got {x.shape}")
        if not np.all(np.isin(x, (-1.0, 1.0))):
            raise ValueError("Hopfield patterns must be +1/-1")
        if rule == 'hebbian':
            self.W = x.T @ x / self.n
            np.fill_diagonal(self.W, 0.0)
        elif rule == 'pseudoinverse':
            self.W = x.T @ np.linalg.pinv(x @ x.T) @ x
        else:
            raise ValueError(f"Unknown storage rule '{rule}'")
        return self

    def energy(self, s: np.ndarray) -> float:
        return float(-0.5 * s @ self.W @ s)

    def recall(self, s0: np.ndarray, max_sweeps: int = 50, seed: int = 0) -> Tuple[np.ndarray, List[float]]:
        """
        Asynchronous dynamics: visit units in random order, s_i <- sign(W_i . s).
        With symmetric W and non-negative diagonal each flip can only lower (or keep) E.
        Returns the final state and the energy after every single-unit update.
        """
        rng = np.random.default_rng(seed)
        s = np.array(s0, dtype=np.float64)
        energies = [self.energy(s)]
        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.n):
                field = self.W[i] @ s
                new = 1.0 if field > 0 else (-1.0 if field < 0 else s[i])
                if new != s[i]:
                    s[i] = new
                    changed = True
                energies.append(self.energy(s))
            if not changed:
                break
        return s, energies


def corrupt(pattern: np.ndarray, flip_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a random fraction of +1/-1 units."""
    s = np.array(pattern, dtype=np.float64)
    idx = rng.choice(s.size, int(round(flip_fraction * s.size)), replace=False)
    s[idx] *= -1
    return s


def hopfield_capacity(
    n_units: int,
    loads: List[int],
    trials: int = 10,
    flip_fraction: float = 0.1,
    seed: int = 0,
    rule: str = 'hebbian'
) -> Dict[int, float]:
    """Fraction of random stored patterns recalled exactly from a 10%-corrupted cue."""
    rng = np.random.default_rng(seed)
    result = {}
    for p in loads:
        hits, total = 0, 0
        for trial in range(trials):
            patterns = rng.choice([-1.0, 1.0], size=(p, n_units))
            net = Hopfield(n_units).store(patterns, rule=rule)
            for j, pat in enumerate(patterns):
                out, _ = net.recall(corrupt(pat, flip_fraction, rng), seed=trial * 1000 + j)
                hits += int(np.array_equal(out, pat))
                total += 1
        result[p] = hits / total
    return result


# ---------------------------------------------------------------------------
# Restricted Boltzmann machine
# ---------------------------------------------------------------------------

class RBM:
    """Bernoulli-Bernoulli RBM with CD-k training and exact likelihood for small visible layers."""

    def __init__(self, n_visible: int, n_hidden: int, seed: int = 0):
        self.nv, self.nh = n_visible, n_hidden
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0.0, 0.01, size=(n_visible, n_hidden))
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

    # --- conditionals -----------------------------------------------------
    def p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return _sigmoid(v @ self.W + self.c)

    def p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return _sigmoid(h @ self.W.T + self.b)

    def _sample(self, p: np.ndarray) -> np.ndarray:
        return (self.rng.random(p.shape) < p).astype(np.float64)

    # --- energies and exact quantities -------------------------------------
    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """F(v) = -b.v - sum_j log(1 + exp(c_j + v.W_j)), so p(v) = exp(-F(v)) / Z."""
        v = np.atleast_2d(v)
        return -(v @ self.b) - np.sum(_logaddexp0(v @ self.W + self.c), axis=1)

    def all_visible_states(self) -> np.ndarray:
        if self.nv > 20:
            raise ValueError("Exact enumeration is only feasible for n_visible <= 20")
        return np.array(list(itertools.product((0.0, 1.0), repeat=self.nv)))

    def log_partition(self) -> float:
        neg_f = -self.free_energy(self.all_visible_states())
        m = neg_f.max()
        return float(m + np.log(np.sum(np.exp(neg_f - m))))

    def log_likelihood(self, v: np.ndarray) -> float:
        """Exact mean log p(v) over the given samples."""
        return float(np.mean(-self.free_energy(v)) - self.log_partition())

    def model_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        states = self.all_visible_states()
        neg_f = -self.free_energy(states)
        p = np.exp(neg_f - neg_f.max())
        return states, p / p.sum()

    # --- training ---------------------------------------------------------
    def _apply(self, dW: np.ndarray, db: np.ndarray, dc: np.ndarray, lr: float) -> None:
        self.W += lr * dW
        self.b += lr * db
        self.c += lr * dc

    def cd_step(self, v0: np.ndarray, k: int = 1, lr: float = 0.1) -> None:
        """Contrastive divergence: positive phase on data, negative phase after k Gibbs steps."""
        ph0 = self.p_h_given_v(v0)
        h = self._sample(ph0)
        vk = v0
        for _ in range(k):
            vk = self._sample(self.p_v_given_h(h))
            phk = self.p_h_given_v(vk)
            h = self._sample(phk)
        n = v0.shape[0]
        self._apply((v0.T @ ph0 - vk.T @ phk) / n, (v0 - vk).mean(axis=0), (ph0 - phk).mean(axis=0), lr)

    def exact_ml_step(self, v0: np.ndarray, lr: float = 0.1) -> None:
        """
        Exact log-likelihood gradient: <v h>_data - <v h>_model, with the model expectation
        computed by enumerating every visible state (h is marginalised analytically).
        """
        states, p = self.model_distribution()
        ph_model = self.p_h_given_v(states)
        ph0 = self.p_h_given_v(v0)
        dW = v0.T @ ph0 / len(v0) - states.T @ (p[:, None] * ph_model)
        db = v0.mean(axis=0) - p @ states
        dc = ph0.mean(axis=0) - p @ ph_model
        self._apply(dW, db, dc, lr)

    def fit(self, v: np.ndarray, epochs: int = 200, method: str = 'cd', k: int = 1,
            lr: float = 0.1, batch_size: int = 16, track_likelihood: bool = True) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'log_likelihood': [], 'reconstruction_error': []}
        for _ in range(epochs):
            order = self.rng.permutation(len(v))
            for start in range(0, len(v), batch_size):
                batch = v[order[start:start + batch_size]]
                if method == 'cd':
                    self.cd_step(batch, k=k, lr=lr)
                elif method == 'exact':
                    self.exact_ml_step(batch, lr=lr)
                else:
                    raise ValueError(f"Unknown RBM training method '{method}'")
            recon = self.p_v_given_h(self.p_h_given_v(v))
            history['reconstruction_error'].append(float(np.mean((v - recon) ** 2)))
            if track_likelihood:
                history['log_likelihood'].append(self.log_likelihood(v))
        return history

    def sample(self, n: int, gibbs_steps: int = 200) -> np.ndarray:
        v = self._sample(np.full((n, self.nv), 0.5))
        for _ in range(gibbs_steps):
            v = self._sample(self.p_v_given_h(self._sample(self.p_h_given_v(v))))
        return v


def bars_and_stripes(size: int = 3) -> np.ndarray:
    """All distinct size x size 'bars and stripes' images (each row, or each column, uniform)."""
    patterns = set()
    for bits in itertools.product((0, 1), repeat=size):
        rows = np.repeat(np.array(bits)[:, None], size, axis=1)
        patterns.add(tuple(rows.ravel()))
        patterns.add(tuple(rows.T.ravel()))
    return np.array(sorted(patterns), dtype=np.float64)
