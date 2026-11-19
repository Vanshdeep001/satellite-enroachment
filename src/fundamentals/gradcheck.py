"""
Numerical gradient checking.

Central difference:  df/dx_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps),  error O(eps^2).

Relative error between analytic gradient g_a and numeric gradient g_n is measured with
the global norm ratio ||g_a - g_n|| / (||g_a|| + ||g_n||), which is robust when
individual entries are close to zero. In float64, values below ~1e-7 indicate a
correct derivation.
"""

from typing import Any, Callable, Dict

import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """Central-difference gradient of a scalar function f at x (x is not modified)."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        original = x[index]
        x[index] = original + eps
        f_plus = f(x)
        x[index] = original - eps
        f_minus = f(x)
        x[index] = original
        grad[index] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """||a - n|| / (||a|| + ||n||), with 0.0 returned when both gradients are zero."""
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denom < 1e-300:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denom)


def gradient_check(
    f: Callable[[np.ndarray], float],
    analytic_grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-7
) -> Dict[str, Any]:
    """Compare an analytic gradient function with central differences at x."""
    numeric = numerical_gradient(f, x, eps=eps)
    analytic = np.asarray(analytic_grad(np.array(x, dtype=np.float64)), dtype=np.float64)
    err = relative_error(analytic, numeric)
    return {
        'relative_error': err,
        'passed': err < tol,
        'analytic': analytic,
        'numeric': numeric,
        'eps': eps,
        'tol': tol,
    }


def check_model_gradients(
    model: Any,
    loss_fn: Any,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Gradient-check every learnable parameter of a fundamentals MLP.

    The model must expose forward(x, training), backward(grad) and named_parameters()
    returning (name, value, grad) triples, and the loss must expose forward/backward.
    Parameters are perturbed in place and restored afterwards.
    """
    logits = model.forward(x, training=False)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())
    analytic = {name: grad.copy() for name, _, grad in model.named_parameters()}

    def loss_at_current_params() -> float:
        return loss_fn.forward(model.forward(x, training=False), y)

    per_param: Dict[str, float] = {}
    for name, value, _ in model.named_parameters():
        numeric = np.zeros_like(value)
        for index in np.ndindex(*value.shape):
            original = value[index]
            value[index] = original + eps
            f_plus = loss_at_current_params()
            value[index] = original - eps
            f_minus = loss_at_current_params()
            value[index] = original
            numeric[index] = (f_plus - f_minus) / (2.0 * eps)
        per_param[name] = relative_error(analytic[name], numeric)

    worst = max(per_param.values()) if per_param else 0.0
    return {'per_parameter': per_param, 'max_relative_error': worst, 'passed': worst < tol}
