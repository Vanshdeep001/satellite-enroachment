"""
Visual explainer for the from-scratch fundamentals package.

Usage:
    python -m src.fundamentals.demo [--out outputs/fundamentals] [--seed 42]

Figures written to --out:
    01_broadcasting.png      shapes (3,1) + (1,4) -> (3,4) with every element annotated
    02_reshape_row_major.png flat offsets of a 2x6 array reshaped to 3x4 and 4x3
    03_svd_low_rank.png      rank-r reconstructions of a satellite tile + singular spectrum
    04_gradient_check.png    central-difference error vs eps, and per-parameter MLP check
    05_neurons.png           McCulloch-Pitts truth tables and sigmoid-neuron learning
    06_activations.png       step / sigmoid / tanh / relu and their derivatives
    07_mlp_training.png      decision regions, loss curves and weight-norm tracking
    mlp_tracker.json         hyperparameters vs learned-parameter statistics per epoch
    summary.json             numeric results of every demo
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.fundamentals import mlp as M  # noqa: E402
from src.fundamentals import tensor_ops as T  # noqa: E402
from src.fundamentals.gradcheck import check_model_gradients, numerical_gradient, relative_error  # noqa: E402
from src.fundamentals.neurons import SigmoidNeuron, mp_and, mp_and_not, mp_not, mp_or, mp_xor  # noqa: E402


def _annotate(ax, arr, fmt='{:.0f}'):
    for (i, j), val in np.ndenumerate(arr):
        ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=9, color='black')
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------

def demo_broadcasting(out_dir: str) -> Dict[str, Any]:
    a = np.array([[1.0], [2.0], [3.0]])           # (3, 1)
    b = np.array([[10.0, 20.0, 30.0, 40.0]])      # (1, 4)
    shape = T.broadcast_shapes(a.shape, b.shape)
    a_b, b_b = T.broadcast_to(a, shape), T.broadcast_to(b, shape)
    total = T.elementwise(lambda x, y: x + y, a, b)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), gridspec_kw={'width_ratios': [1, 4, 4, 4, 4]})
    panels = [(a, 'a  (3,1)'), (b, 'b  (1,4)'), (a_b, 'a stretched to (3,4)'),
              (b_b, 'b stretched to (3,4)'), (total, 'a + b  (3,4)')]
    for ax, (arr, title) in zip(axes, panels):
        ax.imshow(arr, cmap='Blues', vmin=0, vmax=50)
        _annotate(ax, arr)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Broadcasting: align right, size-1 axes are repeated', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_broadcasting.png'), dpi=130)
    plt.close(fig)
    return {'result_shape': list(shape), 'matches_numpy': bool(np.allclose(total, a + b))}


def demo_reshape(out_dir: str) -> Dict[str, Any]:
    x = np.arange(12).reshape(2, 6)
    shapes = [(2, 6), (3, 4), (4, 3)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, shape in zip(axes, shapes):
        arr = T.reshape(x, shape)
        ax.imshow(arr, cmap='viridis')
        _annotate(ax, arr)
        ax.set_title(f'shape {shape}, strides {T.row_major_strides(shape)}', fontsize=10)
    fig.suptitle('Reshape keeps the row-major (flat) order: value = flat offset', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_reshape_row_major.png'), dpi=130)
    plt.close(fig)
    return {'shapes': [list(s) for s in shapes], 'matches_numpy': bool(np.array_equal(T.reshape(x, (4, 3)), x.reshape(4, 3)))}


def load_tile(size: int = 96, seed: int = 0) -> (np.ndarray, str):
    """Grayscale tile from the local dataset if available, else a synthetic scene."""
    candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'data', '*', 'A', '*.png')))
    if candidates:
        img = Image.open(candidates[0]).convert('L').resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float64) / 255.0, os.path.relpath(candidates[0], PROJECT_ROOT)

    # Synthetic "satellite" scene: field gradient, road grid, rectangular rooftops, noise.
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.35 + 0.15 * np.sin(4 * xx) * np.cos(3 * yy)
    img[size // 3 - 2:size // 3 + 2, :] = 0.85
    img[:, size // 2 - 2:size // 2 + 2] = 0.85
    for _ in range(14):
        r, c = rng.integers(0, size - 12, size=2)
        h, w = rng.integers(5, 12, size=2)
        img[r:r + h, c:c + w] = rng.uniform(0.55, 0.95)
    img += rng.normal(0, 0.03, size=img.shape)
    return np.clip(img, 0, 1), 'synthetic (no local LEVIR-CD tiles found)'


def demo_svd(out_dir: str, seed: int) -> Dict[str, Any]:
    tile, source = load_tile(seed=seed)
    start = time.perf_counter()
    u, s, vt = T.svd(tile)
    elapsed = time.perf_counter() - start
    np_s = np.linalg.svd(tile, compute_uv=False)

    ranks = [1, 5, 15, 40]
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    fig, axes = plt.subplots(1, len(ranks) + 2, figsize=(19, 3.6))
    axes[0].imshow(tile, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('original')
    errors = {}
    for ax, r in zip(axes[1:], ranks):
        approx = (u[:, :r] * s[:r]) @ vt[:r, :]
        errors[r] = float(np.linalg.norm(tile - approx) / np.linalg.norm(tile))
        ax.imshow(np.clip(approx, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'rank {r}  (rel err {errors[r]:.3f})', fontsize=10)
    for ax in axes[:-1]:
        ax.axis('off')
    ax = axes[-1]
    ax.semilogy(s, label='singular values')
    ax.set_xlabel('index')
    ax2 = ax.twinx()
    ax2.plot(energy, color='tab:orange', label='cumulative energy')
    ax2.set_ylim(0, 1.02)
    ax.set_title('spectrum', fontsize=10)
    fig.suptitle(f'One-sided Jacobi SVD of a {tile.shape[0]}x{tile.shape[1]} tile: {source}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_svd_low_rank.png'), dpi=130)
    plt.close(fig)
    return {
        'tile_source': source,
        'runtime_seconds': round(elapsed, 3),
        'max_singular_value_error_vs_numpy': float(np.max(np.abs(s - np_s))),
        'relative_reconstruction_error_by_rank': errors,
        'rank_for_95pct_energy': int(np.searchsorted(energy, 0.95) + 1),
    }


def demo_gradient_check(out_dir: str, seed: int) -> Dict[str, Any]:
    def f(x):
        return float(np.sum(np.tanh(x) * x ** 2))

    def grad_f(x):
        return (1 - np.tanh(x) ** 2) * x ** 2 + 2 * x * np.tanh(x)

    x = np.random.default_rng(seed).normal(size=8)
    eps_values = np.logspace(-12, -1, 23)
    errs = [relative_error(grad_f(x), numerical_gradient(f, x, eps=e)) for e in eps_values]

    rng = np.random.default_rng(seed)
    model = M.MLP([3, 6, 5, 3], activation='tanh', seed=seed)
    mlp_check = check_model_gradients(model, M.SoftmaxCrossEntropyLoss(), rng.normal(size=(8, 3)), rng.integers(0, 3, 8))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].loglog(eps_values, errs, 'o-')
    axes[0].set_xlabel('eps')
    axes[0].set_ylabel('relative error')
    axes[0].set_title('Central difference: truncation O(eps^2) vs round-off O(1/eps)')
    axes[0].grid(True, which='both', alpha=0.3)
    names = list(mlp_check['per_parameter'])
    axes[1].bar(names, [max(mlp_check['per_parameter'][n], 1e-16) for n in names])
    axes[1].set_yscale('log')
    axes[1].axhline(1e-6, color='red', ls='--', label='pass threshold 1e-6')
    axes[1].set_title('Backprop vs numeric gradient, per MLP parameter')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_gradient_check.png'), dpi=130)
    plt.close(fig)
    best = int(np.argmin(errs))
    return {
        'best_eps': float(eps_values[best]),
        'best_relative_error': float(errs[best]),
        'mlp_max_relative_error': mlp_check['max_relative_error'],
        'mlp_passed': mlp_check['passed'],
    }


def demo_neurons(out_dir: str, seed: int) -> Dict[str, Any]:
    gates = {'AND': mp_and(), 'OR': mp_or(), 'x0 AND NOT x1': mp_and_not()}
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = [[f'{a}{b}'] + [str(g([a, b])) for g in gates.values()] + [str(mp_xor([a, b]))] for a, b in combos]

    x = np.array(combos, dtype=np.float64)
    y_or = np.array([0, 1, 1, 1])
    neuron = SigmoidNeuron(2, seed=seed)
    hist = neuron.fit(x, y_or, lr=2.0, epochs=300)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].axis('off')
    table = axes[0].table(cellText=rows + [['NOT 0 / NOT 1', f'{mp_not()([0])} / {mp_not()([1])}', '', '', '']],
                          colLabels=['x0 x1'] + list(gates) + ['XOR (2 layers)'], loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.25, 2.0)
    axes[0].set_title('McCulloch-Pitts units (hand-set thresholds)')

    axes[1].plot(hist['loss'])
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('BCE loss')
    axes[1].set_title('Sigmoid neuron learning OR by gradient descent')

    grid = np.linspace(-0.5, 1.5, 200)
    gx, gy = np.meshgrid(grid, grid)
    probs = neuron.forward(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    cs = axes[2].contourf(gx, gy, probs, levels=20, cmap='RdBu_r', alpha=0.8)
    axes[2].contour(gx, gy, probs, levels=[0.5], colors='k')
    axes[2].scatter(x[:, 0], x[:, 1], c=y_or, cmap='RdBu_r', edgecolors='k', s=120)
    axes[2].set_title(f'p(y=1) and 0.5 boundary  (w={np.round(neuron.w, 2)}, b={neuron.b:.2f})', fontsize=10)
    fig.colorbar(cs, ax=axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_neurons.png'), dpi=130)
    plt.close(fig)
    return {
        'mp_truth_tables': {name: [g([a, b]) for a, b in combos] for name, g in gates.items()},
        'mp_xor': [mp_xor(c) for c in combos],
        'sigmoid_or_predictions': neuron.predict(x).tolist(),
        'sigmoid_final_loss': hist['loss'][-1],
    }


def demo_activations(out_dir: str) -> Dict[str, Any]:
    z = np.linspace(-5, 5, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ['step', 'sigmoid', 'tanh', 'relu']:
        fn, grad = M.ACTIVATIONS[name]
        axes[0].plot(z, fn(z), label=name)
        axes[1].plot(z, grad(z), label=name)
    axes[0].set_title('activation f(z)')
    axes[1].set_title("derivative f'(z)  (step: 0 almost everywhere -> no gradient signal)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_activations.png'), dpi=130)
    plt.close(fig)
    return {
        'max_derivative': {n: float(M.ACTIVATIONS[n][1](z).max()) for n in ['step', 'sigmoid', 'tanh', 'relu']},
    }


def make_spiral(n_per_class: int = 100, classes: int = 3, seed: int = 0):
    """Classic interleaved-spiral toy problem: not linearly separable."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + rng.normal(0, 0.2, n_per_class)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n_per_class, c))
    return np.vstack(xs), np.concatenate(ys)


def demo_mlp(out_dir: str, seed: int) -> Dict[str, Any]:
    x, y = make_spiral(seed=seed)
    perm = np.random.default_rng(seed).permutation(len(x))
    split = int(0.8 * len(x))
    tr, va = perm[:split], perm[split:]

    hyper = {
        'layer_sizes': [2, 64, 64, 3], 'activation': 'tanh', 'init': 'xavier',
        'epochs': 400, 'batch_size': 32, 'loss': 'softmax_cross_entropy', 'seed': seed,
    }
    model = M.MLP(hyper['layer_sizes'], activation=hyper['activation'], init=hyper['init'], seed=seed)
    optimizer = M.Momentum(lr=0.1, beta=0.9)
    hyper.update(optimizer.hyperparameters())
    tracker = M.ExperimentTracker(hyper)
    hist = M.train(model, M.SoftmaxCrossEntropyLoss(), optimizer, x[tr], y[tr], epochs=hyper['epochs'],
                   batch_size=hyper['batch_size'], x_val=x[va], y_val=y[va], seed=seed, tracker=tracker)
    tracker.save(os.path.join(out_dir, 'mlp_tracker.json'), model)
    model.save(os.path.join(out_dir, 'mlp_spiral_weights.npz'))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    grid = np.linspace(-1.2, 1.2, 250)
    gx, gy = np.meshgrid(grid, grid)
    pred = model.forward(np.c_[gx.ravel(), gy.ravel()], training=False).argmax(axis=1).reshape(gx.shape)
    axes[0].contourf(gx, gy, pred, alpha=0.35, cmap='brg', levels=[-0.5, 0.5, 1.5, 2.5])
    axes[0].scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=12, edgecolors='k', linewidths=0.3)
    axes[0].set_title(f"decision regions (val acc {hist['val_metric'][-1]:.2%})")

    axes[1].plot(hist['train_loss'], label='train')
    axes[1].plot(hist['val_loss'], label='val')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('epoch')
    axes[1].set_title('softmax cross-entropy')
    axes[1].legend()

    for name in [n for n, _, _ in model.named_parameters() if n.endswith('.W')]:
        axes[2].plot([e['parameters'][name]['norm'] for e in tracker.epochs], label=f'||{name}||')
    axes[2].set_xlabel('epoch')
    axes[2].set_title(f'learned parameters ({model.num_parameters()} total) vs fixed hyperparameters')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '07_mlp_training.png'), dpi=130)
    plt.close(fig)
    return {
        'num_learned_parameters': model.num_parameters(),
        'hyperparameters': hyper,
        'final_train_acc': hist['train_metric'][-1],
        'final_val_acc': hist['val_metric'][-1],
    }


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Render the neural-network fundamentals explainer figures.')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'fundamentals'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    summary: Dict[str, Any] = {}
    steps = [
        ('broadcasting', lambda: demo_broadcasting(args.out)),
        ('reshape', lambda: demo_reshape(args.out)),
        ('svd', lambda: demo_svd(args.out, args.seed)),
        ('gradient_check', lambda: demo_gradient_check(args.out, args.seed)),
        ('neurons', lambda: demo_neurons(args.out, args.seed)),
        ('activations', lambda: demo_activations(args.out)),
        ('mlp', lambda: demo_mlp(args.out, args.seed)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        summary[name] = fn()
        print(f"[{name:>15}] done in {time.perf_counter() - start:5.2f}s")

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Figures and summary written to {args.out}")
    return summary


if __name__ == '__main__':
    main()
