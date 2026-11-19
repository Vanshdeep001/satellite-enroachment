"""
Multilayer feed-forward network implemented from scratch in NumPy.

Provides:
- Activations: step, sigmoid, tanh, relu, identity (each with an analytic derivative).
- Dense layer with selectable weight initialisation (xavier, he, normal, large, zeros).
- Losses: MSE, binary cross-entropy on logits, softmax cross-entropy on logits.
- MLP container with manual backpropagation.
- Optimizers: GradientDescent, Momentum, Nesterov (all with optional L2 weight decay).
  Batch GD and SGD use the same optimizer; they differ only in the batch size given to
  train() (None = the full batch, 1 = pure stochastic, k = mini-batch).
- ExperimentTracker separating hyperparameters (chosen before training) from
  parameters (learned weights) and logging both per epoch.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

NamedParam = Tuple[str, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function 1 / (1 + exp(-z))."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    'step': (lambda z: (z >= 0).astype(np.float64), lambda z: np.zeros_like(z, dtype=np.float64)),
    'sigmoid': (sigmoid, _sigmoid_grad),
    'tanh': (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    'relu': (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(np.float64)),
    'identity': (lambda z: np.asarray(z, dtype=np.float64), lambda z: np.ones_like(z, dtype=np.float64)),
}


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Layer:
    """Base interface: forward caches what backward needs; backward returns dL/dinput."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Mapping of parameter name -> (value, gradient). Empty for parameter-free layers."""
        return {}

    def buffers(self) -> Dict[str, np.ndarray]:
        """Non-learned state that must be saved with the model (e.g. BatchNorm running stats)."""
        return {}


class Activation(Layer):
    """Element-wise non-linearity: forward a = f(z), backward dL/dz = dL/da * f'(z)."""

    def __init__(self, name: str):
        if name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(ACTIVATIONS)}")
        self.name = name
        self._fn, self._grad = ACTIVATIONS[name]
        self._z: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._z = x
        return self._fn(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._z is None:
            raise RuntimeError("backward called before forward")
        return grad_out * self._grad(self._z)


def init_weights(scheme: str, fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """
    Weight initialisation schemes.

    xavier : U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))  (tanh / sigmoid)
    he     : N(0, 2/fan_in)                                            (relu)
    normal : N(0, 0.01^2)                                              (small; can vanish)
    large  : N(0, 1)                                                   (large; can saturate)
    zeros  : all zeros                                                 (breaks symmetry badly)
    """
    if scheme == 'xavier':
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))
    if scheme == 'he':
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
    if scheme == 'normal':
        return rng.normal(0.0, 0.01, size=(fan_in, fan_out))
    if scheme == 'large':
        return rng.normal(0.0, 1.0, size=(fan_in, fan_out))
    if scheme == 'zeros':
        return np.zeros((fan_in, fan_out))
    raise ValueError(f"Unknown init scheme '{scheme}'")


class Dense(Layer):
    """
    Fully connected layer y = x W + b.

    Backward with upstream gradient G = dL/dy:
        dL/dW = x^T G,   dL/db = sum_rows(G),   dL/dx = G W^T
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = 'xavier',
        rng: Optional[np.random.Generator] = None
    ):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.in_features = in_features
        self.out_features = out_features
        self.W = init_weights(init, in_features, out_features, rng).astype(np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Dense expects input [B, {self.in_features}], got {x.shape}")
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward called before forward")
        self.dW[...] = self._x.T @ grad_out
        self.db[...] = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {'W': (self.W, self.dW), 'b': (self.b, self.db)}


class BatchNorm1d(Layer):
    """
    Batch normalisation over the batch axis of [B, F] activations.

    Training:  x_hat = (x - mean_B) / sqrt(var_B + eps),  y = gamma * x_hat + beta
               running statistics are updated with exponential averaging.
    Inference: the running mean / variance replace the batch statistics.

    Backward (training mode), with dx_hat = dL/dy * gamma:
        dL/dx = (1 / (B sigma)) * (B dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
    Backward (inference mode) is the affine map dL/dx = dx_hat / sigma_running.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, bool]] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm1d expects [B, {self.num_features}], got {x.shape}")
        if training:
            n = x.shape[0]
            if n < 2:
                raise ValueError("BatchNorm1d needs a batch of at least 2 samples in training mode")
            mean, var = x.mean(axis=0), x.var(axis=0)
            self.running_mean *= (1.0 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1.0 - self.momentum)
            self.running_var += self.momentum * var * n / (n - 1)  # unbiased estimate
        else:
            mean, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        self._cache = (x_hat, inv_std, training)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward called before forward")
        x_hat, inv_std, training = self._cache
        self.dgamma[...] = np.sum(grad_out * x_hat, axis=0)
        self.dbeta[...] = np.sum(grad_out, axis=0)
        dx_hat = grad_out * self.gamma
        if not training:
            return dx_hat * inv_std
        n = grad_out.shape[0]
        return (inv_std / n) * (n * dx_hat - dx_hat.sum(axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0))

    def params(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {'gamma': (self.gamma, self.dgamma), 'beta': (self.beta, self.dbeta)}

    def buffers(self) -> Dict[str, np.ndarray]:
        return {'running_mean': self.running_mean, 'running_var': self.running_var}


class Dropout(Layer):
    """
    Inverted dropout: in training, each unit is kept with probability 1 - p and scaled by
    1 / (1 - p) so the expected activation is unchanged; at inference it is the identity.
    """

    def __init__(self, p: float, rng: Optional[np.random.Generator] = None):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.p > 0.0:
            self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
            return x * self._mask
        self._mask = None
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out if self._mask is None else grad_out * self._mask


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class Loss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class MSELoss(Loss):
    """L = mean((p - t)^2);  dL/dp = 2 (p - t) / N."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        target = np.asarray(target, dtype=np.float64).reshape(pred.shape)
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class BCEWithLogitsLoss(Loss):
    """
    Binary cross-entropy on logits z with targets t in {0, 1}:
        L = mean(max(z, 0) - z t + log(1 + exp(-|z|)))   (stable form)
        dL/dz = (sigmoid(z) - t) / N
    """

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        target = np.asarray(target, dtype=np.float64).reshape(pred.shape)
        self._z, self._t = pred, target
        loss = np.maximum(pred, 0.0) - pred * target + np.log1p(np.exp(-np.abs(pred)))
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        return (sigmoid(self._z) - self._t) / self._z.size


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax cross-entropy on logits [B, K] with integer labels [B]:
        p = softmax(z),  L = -mean_b log p[b, y_b],  dL/dz = (p - onehot(y)) / B
    """

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        labels = np.asarray(target).astype(np.int64).ravel()
        if pred.ndim != 2 or labels.shape[0] != pred.shape[0]:
            raise ValueError(f"Expected logits [B, K] and labels [B], got {pred.shape} and {labels.shape}")
        shifted = pred - pred.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self._probs = np.exp(log_probs)
        self._labels = labels
        return float(-np.mean(log_probs[np.arange(labels.size), labels]))

    def backward(self) -> np.ndarray:
        grad = self._probs.copy()
        grad[np.arange(self._labels.size), self._labels] -= 1.0
        return grad / self._labels.size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP:
    """
    Feed-forward network. Each hidden block is
        Dense -> [BatchNorm1d] -> Activation -> [Dropout]
    and the output layer is a plain Dense returning raw logits.

    Args:
        layer_sizes: e.g. [2, 16, 16, 3] -> two hidden layers of width 16, 3 outputs.
        activation: hidden activation name.
        init: weight initialisation scheme for every Dense layer.
        seed: RNG seed for reproducible initialisation (and dropout masks).
        batchnorm: insert BatchNorm1d after every hidden Dense layer.
        dropout: dropout probability after every hidden activation (0 disables).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = 'tanh',
        init: str = 'xavier',
        seed: int = 0,
        batchnorm: bool = False,
        dropout: float = 0.0
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and an output size")
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.init = init
        self.seed = seed
        self.batchnorm = batchnorm
        self.dropout = dropout

        rng = np.random.default_rng(seed)
        dropout_rng = np.random.default_rng(seed + 1_000_003)
        self.layers: List[Layer] = []
        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(Dense(fan_in, fan_out, init=init, rng=rng))
            if i < len(layer_sizes) - 2:
                if batchnorm:
                    self.layers.append(BatchNorm1d(fan_out))
                self.layers.append(Activation(activation))
                if dropout > 0.0:
                    self.layers.append(Dropout(dropout, rng=dropout_rng))

    def hidden_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Inference-mode outputs of every hidden activation layer (for init diagnostics)."""
        out = np.asarray(x, dtype=np.float64)
        captured = []
        for layer in self.layers:
            out = layer.forward(out, training=False)
            if isinstance(layer, Activation):
                captured.append(out)
        return captured

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    __call__ = forward

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def named_parameters(self) -> List[NamedParam]:
        named = []
        for i, layer in enumerate(self.layers):
            for key, (value, grad) in layer.params().items():
                named.append((f"layer{i}.{key}", value, grad))
        return named

    def num_parameters(self) -> int:
        return int(sum(value.size for _, value, _ in self.named_parameters()))

    def named_buffers(self) -> List[Tuple[str, np.ndarray]]:
        return [(f"layer{i}.{key}", value)
                for i, layer in enumerate(self.layers) for key, value in layer.buffers().items()]

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {name: value.copy() for name, value, _ in self.named_parameters()}
        state.update({name: value.copy() for name, value in self.named_buffers()})
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        targets = [(n, v) for n, v, _ in self.named_parameters()] + self.named_buffers()
        for name, value in targets:
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"State dict missing or mismatched entry for '{name}'")
            value[...] = state[name]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> None:
        with np.load(path) as data:
            self.load_state_dict({k: data[k] for k in data.files})


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Base optimizer. L2 weight decay adds wd * W to the gradient of weight matrices
    (biases are not decayed). All updates are written in place into the parameter arrays.
    """

    name = 'base'

    def __init__(self, lr: float = 0.1, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[str, np.ndarray] = {}

    def step(self, named_params: Sequence[NamedParam]) -> None:
        for name, value, grad in named_params:
            g = grad
            if self.weight_decay and name.endswith('.W'):
                g = grad + self.weight_decay * value
            self._update(name, value, g)

    def _update(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        raise NotImplementedError

    def hyperparameters(self) -> Dict[str, Any]:
        return {'optimizer': self.name, 'lr': self.lr, 'weight_decay': self.weight_decay}


class GradientDescent(Optimizer):
    """theta <- theta - lr * g. Batch GD or SGD depending on the batch size used."""

    name = 'gd'

    def _update(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        value -= self.lr * grad


class Momentum(Optimizer):
    """Polyak heavy ball: v <- beta v - lr g;  theta <- theta + v."""

    name = 'momentum'

    def __init__(self, lr: float = 0.1, beta: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta

    def _update(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        v = self.state.setdefault(name, np.zeros_like(value))
        v *= self.beta
        v -= self.lr * grad
        value += v

    def hyperparameters(self) -> Dict[str, Any]:
        return {**super().hyperparameters(), 'beta': self.beta}


class Nesterov(Momentum):
    """
    Nesterov accelerated gradient in the look-ahead-free form:
        v_prev = v;  v <- beta v - lr g;  theta <- theta - beta v_prev + (1 + beta) v
    which is equivalent to evaluating the gradient at theta + beta v.
    """

    name = 'nesterov'

    def _update(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        v = self.state.setdefault(name, np.zeros_like(value))
        v_prev = v.copy()
        v *= self.beta
        v -= self.lr * grad
        value += -self.beta * v_prev + (1.0 + self.beta) * v


OPTIMIZERS = {'gd': GradientDescent, 'sgd': GradientDescent, 'momentum': Momentum, 'nesterov': Nesterov}


# ---------------------------------------------------------------------------
# Metrics, tracking and training
# ---------------------------------------------------------------------------

def accuracy(logits: np.ndarray, target: np.ndarray) -> float:
    """Accuracy for binary logits [B, 1] (threshold 0) or multi-class logits [B, K]."""
    target = np.asarray(target).ravel()
    if logits.ndim == 2 and logits.shape[1] > 1:
        pred = logits.argmax(axis=1)
    else:
        pred = (logits.ravel() >= 0).astype(target.dtype)
    return float(np.mean(pred == target))


class ExperimentTracker:
    """
    Records the two kinds of quantities in a training run:

    - hyperparameters: fixed before training (architecture, lr, batch size, init, ...).
    - parameters: learned values (weights, biases); logged as per-tensor statistics
      (L2 norm, mean, std, gradient norm) at the end of every epoch.
    """

    def __init__(self, hyperparameters: Dict[str, Any]):
        self.hyperparameters = dict(hyperparameters)
        self.epochs: List[Dict[str, Any]] = []

    @staticmethod
    def parameter_stats(model: MLP) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, value, grad in model.named_parameters():
            stats[name] = {
                'shape': list(value.shape),
                'norm': float(np.linalg.norm(value)),
                'mean': float(value.mean()),
                'std': float(value.std()),
                'grad_norm': float(np.linalg.norm(grad)),
            }
        return stats

    def log_epoch(self, epoch: int, model: MLP, metrics: Dict[str, float]) -> None:
        self.epochs.append({'epoch': epoch, **metrics, 'parameters': self.parameter_stats(model)})

    def history(self, key: str) -> List[float]:
        return [e[key] for e in self.epochs if key in e]

    def summary(self, model: MLP) -> Dict[str, Any]:
        last = {k: v for k, v in self.epochs[-1].items() if k != 'parameters'} if self.epochs else {}
        return {
            'hyperparameters': self.hyperparameters,
            'num_learned_parameters': model.num_parameters(),
            'parameter_shapes': {name: list(v.shape) for name, v, _ in model.named_parameters()},
            'final_epoch': last,
        }

    def save(self, path: str, model: MLP) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({**self.summary(model), 'epochs': self.epochs}, f, indent=2)


def train(
    model: MLP,
    loss_fn: Loss,
    optimizer: Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: Optional[int] = None,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 0,
    tracker: Optional[ExperimentTracker] = None,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = accuracy,
    augment_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    restore_best: bool = False,
    monitor: str = 'val_loss'
) -> Dict[str, Any]:
    """
    Train with (mini-)batch gradient descent.

    batch_size=None uses the full dataset per step (batch GD); 1 gives pure SGD.
    augment_fn(x_batch, rng) is applied to every training batch (never to evaluation data).
    monitor selects the best epoch: 'val_loss' (lower is better) or 'val_metric'
    (higher is better; ties keep the earlier epoch).
    restore_best=True keeps the weights of the best monitored epoch (early stopping by
    checkpoint selection); requires validation data.

    Returns a history dict with per-epoch lists train_loss, grad_steps (+ train_metric,
    val_loss, val_metric), plus scalars best_epoch and diverged.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    n = x.shape[0]
    bs = n if batch_size is None else int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    has_val = x_val is not None and y_val is not None
    if restore_best and not has_val:
        raise ValueError("restore_best requires validation data")
    if monitor not in ('val_loss', 'val_metric'):
        raise ValueError(f"monitor must be 'val_loss' or 'val_metric', got '{monitor}'")
    if monitor == 'val_metric' and metric_fn is None:
        raise ValueError("monitor='val_metric' requires a metric_fn")
    sign = 1.0 if monitor == 'val_loss' else -1.0  # internally always minimise sign * value
    has_batchnorm = any(isinstance(layer, BatchNorm1d) for layer in getattr(model, 'layers', []))
    rng = np.random.default_rng(seed)

    history: Dict[str, Any] = {'train_loss': [], 'grad_steps': [], 'best_epoch': None, 'diverged': False}
    best_val, best_state, steps = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n) if bs < n else np.arange(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            if has_batchnorm and idx.size < 2:
                continue  # batch statistics are undefined for a single sample
            xb = x[idx]
            if augment_fn is not None:
                xb = augment_fn(xb, rng)
            logits = model.forward(xb, training=True)
            loss_fn.forward(logits, y[idx])
            model.backward(loss_fn.backward())
            optimizer.step(model.named_parameters())
            steps += 1

        # End-of-epoch evaluation on the full training set (and validation set if given).
        metrics: Dict[str, float] = {}
        train_logits = model.forward(x, training=False)
        metrics['train_loss'] = loss_fn.forward(train_logits, y)
        if metric_fn is not None:
            metrics['train_metric'] = metric_fn(train_logits, y)
        if has_val:
            val_logits = model.forward(np.asarray(x_val, dtype=np.float64), training=False)
            metrics['val_loss'] = loss_fn.forward(val_logits, y_val)
            if metric_fn is not None:
                metrics['val_metric'] = metric_fn(val_logits, y_val)

        for key, value in metrics.items():
            history.setdefault(key, []).append(value)
        history['grad_steps'].append(steps)
        if tracker is not None:
            tracker.log_epoch(epoch, model, metrics)

        if not np.isfinite(metrics['train_loss']):
            history['diverged'] = True
            break
        if has_val and sign * metrics[monitor] < best_val:
            best_val = sign * metrics[monitor]
            history['best_epoch'] = epoch
            if restore_best:
                best_state = model.state_dict()

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return history
