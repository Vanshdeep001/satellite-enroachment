"""
Single-neuron models.

McCulloch-Pitts (1943) neuron:
    Binary inputs, each either excitatory or inhibitory, and an integer threshold.
    Absolute inhibition: if any inhibitory input is 1 the neuron outputs 0.
    Otherwise it fires (outputs 1) when the number of active excitatory inputs >= threshold.
    There is no learning: the logic is set by hand through the threshold.

Sigmoid neuron:
    y = sigmoid(w . x + b), trained by gradient descent on binary cross-entropy:
        dL/dw = X^T (y_hat - y) / N,   dL/db = mean(y_hat - y)
"""

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.fundamentals.mlp import sigmoid


class McCullochPittsNeuron:
    """Threshold logic unit with excitatory and (absolutely) inhibitory binary inputs."""

    def __init__(self, n_inputs: int, threshold: int, inhibitory: Sequence[int] = ()):
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        bad = [i for i in inhibitory if not 0 <= i < n_inputs]
        if bad:
            raise ValueError(f"Inhibitory indices {bad} out of range for {n_inputs} inputs")
        self.n_inputs = n_inputs
        self.threshold = threshold
        self.inhibitory = frozenset(inhibitory)

    def forward(self, x: Sequence[int]) -> int:
        x = [int(v) for v in x]
        if len(x) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(x)}")
        if any(v not in (0, 1) for v in x):
            raise ValueError("McCulloch-Pitts inputs must be binary (0 or 1)")
        if any(x[i] == 1 for i in self.inhibitory):
            return 0
        excitation = sum(v for i, v in enumerate(x) if i not in self.inhibitory)
        return int(excitation >= self.threshold)

    __call__ = forward

    def truth_table(self) -> List[Tuple[Tuple[int, ...], int]]:
        return [(combo, self.forward(combo)) for combo in itertools.product((0, 1), repeat=self.n_inputs)]


def mp_and(n: int = 2) -> McCullochPittsNeuron:
    """Fires only when all n inputs are on."""
    return McCullochPittsNeuron(n, threshold=n)


def mp_or(n: int = 2) -> McCullochPittsNeuron:
    """Fires when at least one input is on."""
    return McCullochPittsNeuron(n, threshold=1)


def mp_not() -> McCullochPittsNeuron:
    """Single inhibitory input with threshold 0: fires unless the input is on."""
    return McCullochPittsNeuron(1, threshold=0, inhibitory=[0])


def mp_and_not() -> McCullochPittsNeuron:
    """x0 AND NOT x1: x0 excitatory, x1 inhibitory, threshold 1."""
    return McCullochPittsNeuron(2, threshold=1, inhibitory=[1])


def mp_xor(x: Sequence[int]) -> int:
    """
    XOR is not linearly separable, so one MP unit cannot compute it. A two-layer
    network can: XOR(a, b) = OR(a AND NOT b, b AND NOT a).
    """
    a, b = int(x[0]), int(x[1])
    left = mp_and_not()([a, b])
    right = mp_and_not()([b, a])
    return mp_or(2)([left, right])


class SigmoidNeuron:
    """Logistic regression unit trained with full-batch gradient descent."""

    def __init__(self, n_inputs: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.1, size=n_inputs)
        self.b = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return sigmoid(x @ self.w + self.b)

    __call__ = forward

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.forward(x), 1e-12, 1.0 - 1e-12)
        y = np.asarray(y, dtype=np.float64).ravel()
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def gradients(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        err = self.forward(x) - np.asarray(y, dtype=np.float64).ravel()
        return x.T @ err / x.shape[0], float(err.mean())

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 1.0, epochs: int = 500) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'loss': []}
        for _ in range(epochs):
            dw, db = self.gradients(x, y)
            self.w -= lr * dw
            self.b -= lr * db
            history['loss'].append(self.loss(x, y))
        return history

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(x) >= threshold).astype(np.int64)
