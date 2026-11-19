"""
Project Step 5: optimisation, initialisation, normalisation, regularisation, augmentation
and overfitting / generalisation experiments on the from-scratch NumPy MLP.

Every configuration is trained on the synthetic land-cover patch task
(src/fundamentals/landcover_toy.py) with several seeds. Hyperparameters are selected on
the validation split only; the test split is reported but never used for selection.

Usage:
    python -m src.fundamentals.experiments [--out outputs/fundamentals/experiments]
                                           [--seeds 3] [--quick] [--studies optimizers,init,...]

Studies:
    optimizers      batch GD vs SGD (batch 1) vs mini-batch SGD vs Momentum vs Nesterov,
                    each with its own learning-rate grid; + 2-D ill-conditioned quadratic paths
    init            zeros / normal(0.01) / large N(0,1) / Xavier / He on a 6-hidden-layer net,
                    tanh and ReLU; per-layer activation spread at initialisation
    normalization   raw 0-255 digital numbers vs per-band standardisation vs + BatchNorm
    regularization  none, L2 (3 strengths), dropout (2 rates), L2 + dropout, early stopping
    augmentation    none, dihedral (rot90 + flip), Gaussian jitter, both
    generalization  training-set size sweep (with and without augmentation + L2) and the
                    epoch-wise overfitting curve of the base model
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.fundamentals import mlp as M  # noqa: E402
from src.fundamentals.landcover_toy import (  # noqa: E402
    CLASS_NAMES, NUM_CLASSES, ChannelStandardizer, compose, dihedral_augment, flatten,
    gaussian_jitter, make_splits,
)

PATCH = 10
CHANNELS = 3

BASE_CONFIG: Dict[str, Any] = {
    'hidden': [64, 64],
    'activation': 'relu',
    'init': 'he',
    'batchnorm': False,
    'dropout': 0.0,
    'optimizer': 'momentum',
    'lr': 0.01,
    'beta': 0.9,
    'weight_decay': 0.0,
    'batch_size': 16,
    'epochs': 100,
    'augment': 'none',
    'restore_best': False,
    'monitor': 'val_loss',
    'input': 'standardized',
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class DataBundle:
    """Holds raw splits and lazily builds model-ready inputs for each preprocessing mode."""

    def __init__(self, n_train: int, n_val: int, n_test: int, seed: int = 0):
        self.splits = make_splits(n_train, n_val, n_test, patch=PATCH, seed=seed)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def labels(self, split: str) -> np.ndarray:
        return self.splits[split][1]

    def inputs(self, mode: str) -> Dict[str, np.ndarray]:
        if mode not in self._cache:
            if mode == 'standardized':
                scaler = ChannelStandardizer().fit(self.splits['train'][0])  # train statistics only
                self._cache[mode] = {k: flatten(scaler.transform(v[0])) for k, v in self.splits.items()}
            elif mode == 'raw_dn':
                # 8-bit digital numbers, as delivered by a sensor, with no rescaling at all.
                self._cache[mode] = {k: flatten(np.round(v[0] * 255.0)) for k, v in self.splits.items()}
            else:
                raise ValueError(f"Unknown input mode '{mode}'")
        return self._cache[mode]

    def subset(self, n: int) -> 'DataBundle':
        """Same val/test splits, first n training samples (labels are pre-shuffled)."""
        sub = DataBundle.__new__(DataBundle)
        x, y = self.splits['train']
        sub.splits = {**self.splits, 'train': (x[:n], y[:n])}
        sub._cache = {}
        return sub


def build_augment(name: str) -> Optional[Callable]:
    if name == 'none':
        return None
    if name == 'dihedral':
        return dihedral_augment(CHANNELS, PATCH)
    if name == 'jitter':
        return gaussian_jitter(0.3)
    if name == 'dihedral+jitter':
        return compose(dihedral_augment(CHANNELS, PATCH), gaussian_jitter(0.3))
    raise ValueError(f"Unknown augmentation '{name}'")


def build_optimizer(cfg: Dict[str, Any]) -> M.Optimizer:
    name = cfg['optimizer']
    if name in ('gd', 'sgd'):
        return M.GradientDescent(lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if name == 'momentum':
        return M.Momentum(lr=cfg['lr'], beta=cfg['beta'], weight_decay=cfg['weight_decay'])
    if name == 'nesterov':
        return M.Nesterov(lr=cfg['lr'], beta=cfg['beta'], weight_decay=cfg['weight_decay'])
    raise ValueError(f"Unknown optimizer '{name}'")


def macro_f1(logits: np.ndarray, y: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    pred = logits.argmax(axis=1)
    scores = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Single run and aggregation
# ---------------------------------------------------------------------------

def run_config(cfg: Dict[str, Any], data: DataBundle, seed: int) -> Dict[str, Any]:
    """Train one configuration with one seed; evaluate on train / val / test."""
    x = data.inputs(cfg['input'])
    y_tr, y_va, y_te = data.labels('train'), data.labels('val'), data.labels('test')
    model = M.MLP([x['train'].shape[1], *cfg['hidden'], NUM_CLASSES], activation=cfg['activation'],
                  init=cfg['init'], seed=seed, batchnorm=cfg['batchnorm'], dropout=cfg['dropout'])
    loss_fn = M.SoftmaxCrossEntropyLoss()

    start = time.perf_counter()
    with np.errstate(all='ignore'):
        hist = M.train(model, loss_fn, build_optimizer(cfg), x['train'], y_tr, epochs=cfg['epochs'],
                       batch_size=cfg['batch_size'], x_val=x['val'], y_val=y_va, seed=seed,
                       augment_fn=build_augment(cfg['augment']), restore_best=cfg['restore_best'],
                       monitor=cfg['monitor'])
        runtime = time.perf_counter() - start
        raw_logits = {k: model.forward(x[k], training=False) for k in ('train', 'val', 'test')}
    diverged = bool(hist['diverged']) or not all(np.all(np.isfinite(v)) for v in raw_logits.values())
    # A diverged network still gets scored (its NaN/inf outputs become arbitrary class votes),
    # so its accuracy sits near chance instead of crashing the aggregation.
    logits = {k: np.nan_to_num(v, nan=0.0, posinf=1e300, neginf=-1e300) for k, v in raw_logits.items()}

    def safe(v: float) -> float:
        return float(v) if np.isfinite(v) else float('nan')

    result = {
        'seed': seed,
        'diverged': diverged,
        'train_acc': M.accuracy(logits['train'], y_tr),
        'val_acc': M.accuracy(logits['val'], y_va),
        'test_acc': M.accuracy(logits['test'], y_te),
        'test_macro_f1': macro_f1(logits['test'], y_te),
        'val_loss_final': safe(hist['val_loss'][-1]),
        'val_loss_min': safe(np.nanmin(hist['val_loss'])) if np.any(np.isfinite(hist['val_loss'])) else float('nan'),
        'best_epoch': hist['best_epoch'],
        'grad_steps': hist['grad_steps'][-1] if hist['grad_steps'] else 0,
        'runtime_s': runtime,
        'curves': {k: [safe(v) for v in hist[k]] for k in ('train_loss', 'val_loss', 'train_metric', 'val_metric')},
        'steps_per_epoch': hist['grad_steps'][0] if hist['grad_steps'] else 0,
    }
    return result


SUMMARY_KEYS = ('train_acc', 'val_acc', 'test_acc', 'test_macro_f1', 'val_loss_final', 'val_loss_min', 'runtime_s')


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'n_seeds': len(runs), 'diverged_runs': int(sum(r['diverged'] for r in runs))}
    for key in SUMMARY_KEYS:
        vals = np.array([r[key] for r in runs], dtype=np.float64)
        out[f'{key}_mean'] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')
        out[f'{key}_std'] = float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else float('nan')
    out['generalization_gap_mean'] = out['train_acc_mean'] - out['test_acc_mean']
    epochs = [r['best_epoch'] for r in runs if r['best_epoch'] is not None]
    out['best_epoch_mean'] = float(np.mean(epochs)) if epochs else float('nan')
    out['steps_per_epoch'] = runs[0]['steps_per_epoch']
    # Mean curves over seeds (ragged when a run diverged early -> pad with NaN).
    for curve in ('train_loss', 'val_loss', 'val_metric'):
        length = max(len(r['curves'][curve]) for r in runs)
        stacked = np.full((len(runs), length), np.nan)
        for i, r in enumerate(runs):
            stacked[i, :len(r['curves'][curve])] = r['curves'][curve]
        with np.errstate(all='ignore'), _quiet():
            out[f'{curve}_curve'] = np.nanmean(stacked, axis=0).tolist()
    return out


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore')

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def run_study(
    name: str,
    configs: Dict[str, Dict[str, Any]],
    data: DataBundle,
    seeds: List[int],
    log: Callable[[str], None] = print
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for label, overrides in configs.items():
        cfg = {**BASE_CONFIG, **overrides}
        runs = [run_config(cfg, data, s) for s in seeds]
        agg = aggregate(runs)
        agg['config'] = {k: v for k, v in cfg.items()}
        results[label] = agg
        log(f"  [{name}] {label:<34} train {agg['train_acc_mean']:.3f}  val {agg['val_acc_mean']:.3f}  "
            f"test {agg['test_acc_mean']:.3f}±{agg['test_acc_std']:.3f}  "
            f"{'DIVERGED x' + str(agg['diverged_runs']) if agg['diverged_runs'] else ''}")
    return results


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

OPTIMIZER_FAMILIES = {
    'Batch GD (full batch)': {'optimizer': 'gd', 'batch_size': None},
    'SGD (batch 1)': {'optimizer': 'sgd', 'batch_size': 1},
    'Mini-batch SGD (16)': {'optimizer': 'sgd', 'batch_size': 16},
    'Momentum 0.9 (16)': {'optimizer': 'momentum', 'batch_size': 16},
    'Nesterov 0.9 (16)': {'optimizer': 'nesterov', 'batch_size': 16},
}


def study_optimizers(data: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    lr_grid = [0.01, 0.1] if quick else [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    epochs = 10 if quick else 60
    grid: Dict[str, Dict[str, Any]] = {}
    for family, overrides in OPTIMIZER_FAMILIES.items():
        configs = {f'{family} lr={lr}': {**overrides, 'lr': lr, 'epochs': epochs} for lr in lr_grid}
        grid[family] = run_study('optimizers', configs, data, seeds, log)

    best = {}
    for family, runs in grid.items():
        # Selection criterion: lowest mean minimum validation loss (validation data only).
        label = min(runs, key=lambda k: np.inf if np.isnan(runs[k]['val_loss_min_mean']) else runs[k]['val_loss_min_mean'])
        best[family] = {'label': label, **runs[label]}
    return {'grid': grid, 'best': best, 'lr_grid': lr_grid, 'quadratic_paths': quadratic_paths()}


def quadratic_paths(steps: int = 60) -> Dict[str, Any]:
    """Trajectories on f(w) = 0.5 (w1^2 + 25 w2^2) (condition number 25) from (-9, 2)."""
    curvature = np.array([1.0, 25.0])
    paths = {}
    for name, opt in [('GD', M.GradientDescent(lr=0.035)), ('Momentum', M.Momentum(lr=0.035, beta=0.8)),
                      ('Nesterov', M.Nesterov(lr=0.035, beta=0.8))]:
        w = np.array([-9.0, 2.0])
        path = [w.copy()]
        for _ in range(steps):
            opt.step([('w.W', w, curvature * w)])
            path.append(w.copy())
        path = np.array(path)
        paths[name] = {'path': path.tolist(),
                       'final_loss': float(0.5 * np.sum(curvature * path[-1] ** 2)),
                       'loss': (0.5 * np.sum(curvature * path ** 2, axis=1)).tolist()}
    return paths


def study_init(data: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    depth = [64] * 6
    epochs = 10 if quick else 60
    inits = ['zeros', 'normal', 'large', 'xavier', 'he']
    results: Dict[str, Any] = {}
    x_val = data.inputs('standardized')['val']
    for act in ['tanh', 'relu']:
        configs = {f'{act} / {init}': {'hidden': depth, 'activation': act, 'init': init, 'epochs': epochs}
                   for init in inits}
        res = run_study('init', configs, data, seeds, log)
        for init in inits:
            model = M.MLP([x_val.shape[1], *depth, NUM_CLASSES], activation=act, init=init, seed=seeds[0])
            with np.errstate(all='ignore'):
                acts = model.hidden_activations(x_val)
            res[f'{act} / {init}']['activation_std_per_layer'] = [float(a.std()) for a in acts]
            res[f'{act} / {init}']['dead_or_saturated_fraction'] = [
                float(np.mean(a == 0.0)) if act == 'relu' else float(np.mean(np.abs(a) > 0.99)) for a in acts]
        results[act] = res
    return results


def study_normalization(data: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    epochs = 10 if quick else 100
    configs = {
        'raw DN 0-255, lr 0.01': {'input': 'raw_dn', 'epochs': epochs},
        'raw DN 0-255, lr 1e-5': {'input': 'raw_dn', 'lr': 1e-5, 'epochs': epochs},
        'raw DN + BatchNorm, lr 0.01': {'input': 'raw_dn', 'batchnorm': True, 'epochs': epochs},
        'standardized, lr 0.01': {'epochs': epochs},
        'standardized + BatchNorm, lr 0.01': {'batchnorm': True, 'epochs': epochs},
    }
    return run_study('normalization', configs, data, seeds, log)


def study_regularization(data: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    epochs = 10 if quick else 100
    configs = {
        'none': {},
        'L2 1e-4': {'weight_decay': 1e-4},
        'L2 1e-3': {'weight_decay': 1e-3},
        'L2 1e-2': {'weight_decay': 1e-2},
        'dropout 0.2': {'dropout': 0.2},
        'dropout 0.5': {'dropout': 0.5},
        'L2 1e-3 + dropout 0.2': {'weight_decay': 1e-3, 'dropout': 0.2},
        'early stopping (best val loss)': {'restore_best': True},
        'early stopping (best val acc)': {'restore_best': True, 'monitor': 'val_metric'},
    }
    return run_study('regularization', {k: {**v, 'epochs': epochs} for k, v in configs.items()}, data, seeds, log)


def study_augmentation(data: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    epochs = 10 if quick else 100
    configs = {name: {'augment': name, 'epochs': epochs}
               for name in ['none', 'jitter', 'dihedral', 'dihedral+jitter']}
    return run_study('augmentation', configs, data, seeds, log)


def study_generalization(pool: DataBundle, seeds: List[int], quick: bool, log=print) -> Dict[str, Any]:
    sizes = [40, 160] if quick else [40, 80, 160, 320, 640, 1280]
    epochs = 10 if quick else 100
    results: Dict[str, Any] = {'sizes': sizes, 'plain': {}, 'augment+L2': {}}
    for n in sizes:
        sub = pool.subset(n)
        results['plain'][n] = run_study('generalization', {f'n={n} plain': {'epochs': epochs}}, sub, seeds, log)[f'n={n} plain']
        results['augment+L2'][n] = run_study(
            'generalization', {f'n={n} dihedral + L2 1e-3': {'epochs': epochs, 'augment': 'dihedral', 'weight_decay': 1e-3}},
            sub, seeds, log)[f'n={n} dihedral + L2 1e-3']
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _bars(ax, labels, train, test, test_std, title):
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, train, 0.4, label='train acc', color='#9ecae1')
    ax.bar(idx + 0.2, test, 0.4, yerr=test_std, label='test acc', color='#3182bd', capsize=3)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / NUM_CLASSES, color='grey', ls=':', lw=1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def plot_samples(out_dir: str) -> None:
    from src.fundamentals.landcover_toy import make_patch
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(NUM_CLASSES, 8, figsize=(10, 5.4))
    aug = dihedral_augment(CHANNELS, PATCH)
    for c in range(NUM_CLASSES):
        patches = [make_patch(c, PATCH, rng) for _ in range(4)]
        augmented = [aug(patches[0].reshape(1, -1), rng).reshape(CHANNELS, PATCH, PATCH) for _ in range(4)]
        for j, img in enumerate(patches + augmented):
            axes[c, j].imshow(np.clip(img.transpose(1, 2, 0), 0, 1), interpolation='nearest')
            axes[c, j].set_xticks([])
            axes[c, j].set_yticks([])
        axes[c, 0].set_ylabel(CLASS_NAMES[c], fontsize=9)
    axes[0, 0].set_title('independent samples', loc='left', fontsize=9)
    axes[0, 4].set_title('dihedral augmentations of sample 1', loc='left', fontsize=9)
    fig.suptitle('Synthetic 10x10 land-cover patches (illumination, haze, noise, orientation vary)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_landcover_samples.png'), dpi=130)
    plt.close(fig)


def plot_optimizers(res: Dict[str, Any], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    for family, r in res['best'].items():
        lr = r['config']['lr']
        axes[0].plot(r['val_loss_curve'], label=f'{family}, lr={lr}')
        steps = np.arange(1, len(r['train_loss_curve']) + 1) * max(r['steps_per_epoch'], 1)
        axes[1].plot(steps, r['train_loss_curve'], label=family)
    axes[0].set_title('validation loss vs epoch (best lr per optimizer)', fontsize=10)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylim(0, 2.5)
    axes[0].legend(fontsize=7)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('gradient steps (log)')
    axes[1].set_title('training loss vs number of updates', fontsize=10)

    for family, runs in res['grid'].items():
        lrs = [runs[k]['config']['lr'] for k in runs]
        vals = [runs[k]['val_loss_min_mean'] for k in runs]
        axes[2].plot(lrs, vals, 'o-', label=family)
    axes[2].set_xscale('log')
    axes[2].set_ylim(0, 1.6)
    axes[2].set_xlabel('learning rate')
    axes[2].set_title('lr sensitivity: best validation loss (NaN = diverged)', fontsize=10)
    axes[2].legend(fontsize=7)

    w1, w2 = np.meshgrid(np.linspace(-10, 2, 200), np.linspace(-3, 3, 200))
    axes[3].contour(w1, w2, 0.5 * (w1 ** 2 + 25 * w2 ** 2), levels=np.logspace(-1, 2.2, 14), cmap='Greys', linewidths=0.7)
    for name, p in res['quadratic_paths'].items():
        path = np.array(p['path'])
        axes[3].plot(path[:, 0], path[:, 1], '.-', ms=3, lw=1, label=f"{name} (f={p['final_loss']:.1e})")
    axes[3].set_title('ill-conditioned quadratic, 60 steps, lr=0.035', fontsize=10)
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_optimizers.png'), dpi=130)
    plt.close(fig)


def plot_init(res: Dict[str, Any], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, act in zip(axes[:2], ['tanh', 'relu']):
        for label, r in res[act].items():
            ax.plot(range(1, len(r['activation_std_per_layer']) + 1), r['activation_std_per_layer'], 'o-',
                    label=label.split(' / ')[1])
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.set_xlabel('hidden layer')
        ax.set_ylabel('std of activations at init')
        ax.set_title(f'{act}: signal propagation through 6 layers', fontsize=10)
        ax.legend(fontsize=8)
    labels, test, std = [], [], []
    for act in ['tanh', 'relu']:
        for label, r in res[act].items():
            labels.append(label)
            test.append(r['test_acc_mean'])
            std.append(r['test_acc_std'])
    axes[2].bar(range(len(labels)), test, yerr=std, capsize=3, color=['#fdae6b'] * 5 + ['#74c476'] * 5)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    axes[2].axhline(1 / NUM_CLASSES, color='grey', ls=':')
    axes[2].set_title('test accuracy after training', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_initialization.png'), dpi=130)
    plt.close(fig)


def plot_simple_study(res: Dict[str, Any], out_dir: str, filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    labels = list(res)
    _bars(axes[0], labels, [res[k]['train_acc_mean'] for k in labels], [res[k]['test_acc_mean'] for k in labels],
          [res[k]['test_acc_std'] for k in labels], f'{title}: train vs test accuracy')
    for k in labels:
        axes[1].plot(res[k]['val_loss_curve'], label=k)
    axes[1].set_ylim(0, 3)
    axes[1].set_xlabel('epoch')
    axes[1].set_title('validation loss (mean over seeds)', fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=130)
    plt.close(fig)


def plot_generalization(res: Dict[str, Any], base_curve: Dict[str, Any], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    sizes = res['sizes']
    for key, color in [('plain', 'tab:red'), ('augment+L2', 'tab:blue')]:
        tr = [res[key][n]['train_acc_mean'] for n in sizes]
        te = [res[key][n]['test_acc_mean'] for n in sizes]
        sd = [res[key][n]['test_acc_std'] for n in sizes]
        axes[0].plot(sizes, tr, '--', color=color, label=f'{key} train')
        axes[0].errorbar(sizes, te, yerr=sd, color=color, marker='o', capsize=3, label=f'{key} test')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('training patches')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('more data closes the generalisation gap', fontsize=10)
    axes[0].legend(fontsize=8)

    tl, vl = base_curve['train_loss_curve'], base_curve['val_loss_curve']
    best = int(np.nanargmin(vl)) + 1
    axes[1].plot(range(1, len(tl) + 1), tl, label='train loss')
    axes[1].plot(range(1, len(vl) + 1), vl, label='validation loss')
    axes[1].axvline(best, color='k', ls='--', label=f'min val loss (epoch {best})')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('epoch')
    axes[1].set_title('overfitting: base model, 240 patches, no regularisation', fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_generalization.png'), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _rows(study: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for label, r in results.items():
        rows.append({'study': study, 'config': label,
                     **{k: r[k] for k in r if k.endswith('_mean') or k.endswith('_std')},
                     'diverged_runs': r['diverged_runs'], 'n_seeds': r['n_seeds']})
    return rows


def _strip_curves(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _strip_curves(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_curves(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


ALL_STUDIES = ['optimizers', 'init', 'normalization', 'regularization', 'augmentation', 'generalization']


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Step 5 optimisation and generalisation ablations.')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'fundamentals', 'experiments'))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--quick', action='store_true', help='tiny epochs/grids for smoke testing')
    parser.add_argument('--studies', default=','.join(ALL_STUDIES))
    parser.add_argument('--merge', action='store_true',
                        help='update the studies run here inside an existing results.json instead of replacing it')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    seeds = list(range(args.seeds))
    studies = [s.strip() for s in args.studies.split(',') if s.strip()]
    unknown = set(studies) - set(ALL_STUDIES)
    if unknown:
        parser.error(f"Unknown studies: {sorted(unknown)}")

    n_train, n_val, n_test = (80, 100, 100) if args.quick else (240, 600, 1200)
    pool_size = 160 if args.quick else 1280
    data_pool = DataBundle(pool_size, n_val, n_test, seed=0)
    data = data_pool.subset(n_train)

    plot_samples(args.out)
    results: Dict[str, Any] = {
        'setup': {'n_train': n_train, 'n_val': n_val, 'n_test': n_test, 'patch': PATCH, 'seeds': seeds,
                  'base_config': BASE_CONFIG, 'classes': CLASS_NAMES, 'quick': args.quick},
    }
    results_path = os.path.join(args.out, 'results.json')
    if args.merge and os.path.isfile(results_path):
        with open(results_path) as f:
            previous = json.load(f)
        if previous.get('setup', {}).get('quick') != args.quick:
            parser.error('--merge cannot mix --quick and full runs')
        results = {**previous, 'setup': results['setup']}
    t0 = time.perf_counter()

    if 'optimizers' in studies:
        print('== optimizers')
        res = study_optimizers(data, seeds, args.quick)
        results['optimizers'] = res
        plot_optimizers(res, args.out)
    if 'init' in studies:
        print('== initialization')
        res = study_init(data, seeds, args.quick)
        results['init'] = res
        plot_init(res, args.out)
    for name, fn, fname, title in [
        ('normalization', study_normalization, 'fig_normalization.png', 'normalisation'),
        ('regularization', study_regularization, 'fig_regularization.png', 'regularisation'),
        ('augmentation', study_augmentation, 'fig_augmentation.png', 'augmentation'),
    ]:
        if name in studies:
            print(f'== {name}')
            res = fn(data, seeds, args.quick)
            results[name] = res
            plot_simple_study(res, args.out, fname, title)
    if 'generalization' in studies:
        print('== generalization')
        res = study_generalization(data_pool, seeds, args.quick)
        results['generalization'] = res
        base = results.get('regularization', {}).get('none') or res['plain'][min(res['sizes'], key=lambda n: abs(n - n_train))]
        plot_generalization(res, base, args.out)

    results['last_run_runtime_s'] = time.perf_counter() - t0
    results = _strip_curves(results)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=1)

    rows: List[Dict[str, Any]] = []
    if 'optimizers' in results:
        for family in results['optimizers']['grid'].values():
            rows += _rows('optimizers', family)
    for act in results.get('init', {}).values():
        rows += _rows('init', act)
    for name in ('normalization', 'regularization', 'augmentation'):
        rows += _rows(name, results.get(name, {}))
    if 'generalization' in results:
        for key in ('plain', 'augment+L2'):
            rows += _rows('generalization', {f'n={n} {key}': v for n, v in results['generalization'][key].items()})
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('study', 'config'), k))
        with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Done in {results['last_run_runtime_s']:.1f}s -> {args.out}")
    return results


if __name__ == '__main__':
    main()
