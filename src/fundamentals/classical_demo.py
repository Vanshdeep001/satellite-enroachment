"""
Project Step 6 demonstrations: Hebbian, modified Hebbian, competitive, error-correction
and energy-based (Hopfield + Boltzmann) learning.

Usage:
    python -m src.fundamentals.classical_demo [--out outputs/fundamentals/classical] [--quick]

Figures:
    fig_hebbian.png          Hebb vs Oja on 2-D data; Sanger's GHA recovering the top-4
                             principal axes of land-cover patches (checked against our SVD)
    fig_competitive.png      winner-take-all clustering with good init, poor init (dead units)
                             and poor init + conscience; 6x6 Kohonen map of patch colours
    fig_error_correction.png perceptron convergence (separable / XOR / real patch features)
                             vs LMS delta rule; decision boundaries
    fig_hopfield.png         recall of building-footprint patterns, energy descent, capacity
    fig_boltzmann.png        RBM on 3x3 bars-and-stripes: exact log-likelihood for CD-1, CD-10
                             and exact ML; probability mass on the data; Gibbs samples
    summary.json             every number quoted in docs/classical_learning.md
"""

import argparse
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

from src.fundamentals import tensor_ops as T  # noqa: E402
from src.fundamentals.classical_learning import (  # noqa: E402
    RBM, SOM, Adaline, CompetitiveLayer, Hopfield, Perceptron, bars_and_stripes, cluster_purity,
    corrupt, hebbian_rule, hopfield_capacity, oja_rule, sanger_rule,
)
from src.fundamentals.landcover_toy import (  # noqa: E402
    CLASS_NAMES, ChannelStandardizer, flatten, make_landcover_patches,
)


def patch_features(x: np.ndarray) -> np.ndarray:
    """Hand-crafted per-patch features: mean R, G, B and luminance texture (std)."""
    means = x.mean(axis=(2, 3))
    texture = x.mean(axis=1).std(axis=(1, 2))
    return np.c_[means, texture]


def _zscore(f: np.ndarray):
    mu, sd = f.mean(axis=0), f.std(axis=0) + 1e-12
    return (f - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# Hebbian family
# ---------------------------------------------------------------------------

def demo_hebbian(out_dir: str, quick: bool, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    cov = np.array([[3.0, 1.5], [1.5, 1.0]])
    x2 = rng.multivariate_normal([0, 0], cov, size=400 if quick else 2000)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    w0 = np.array([0.3, -0.8])
    _, hebb_path = hebbian_rule(x2, w0, lr=0.002)
    _, oja_path = oja_rule(x2, w0, lr=0.002)

    def angle_deg(path):
        cos = np.abs(path @ pc1) / np.maximum(np.linalg.norm(path, axis=1), 1e-300)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    # Sanger GHA on 300-D standardised land-cover patches vs our from-scratch SVD.
    x, _ = make_landcover_patches(320 if quick else 1280, seed=seed)
    xs = flatten(ChannelStandardizer().fit(x).transform(x))
    xs = xs - xs.mean(axis=0)
    _, s, vt = T.svd(xs)
    lam_max = s[0] ** 2 / len(xs)
    k = 4
    w_gha, ortho = sanger_rule(xs, k, lr=0.05 / lam_max, epochs=10 if quick else 40, decay=0.1, seed=seed)
    cosines = [float(abs(w_gha[i] @ vt[i]) / np.linalg.norm(w_gha[i])) for i in range(k)]
    explained = (s[:k] ** 2 / np.sum(s ** 2)).tolist()

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(2, 4, 1)
    ax.scatter(x2[:, 0], x2[:, 1], s=3, alpha=0.3, color='grey')
    ax.plot([-5 * pc1[0], 5 * pc1[0]], [-5 * pc1[1], 5 * pc1[1]], 'k--', lw=1, label='true PC1 axis')
    for vec, color, label in [(oja_path[-1] / np.linalg.norm(oja_path[-1]) * 3, 'tab:green', 'Oja (unit norm)'),
                              (hebb_path[-1] / np.linalg.norm(hebb_path[-1]) * 2.5, 'tab:red', 'Hebb (direction only)')]:
        ax.annotate('', xy=vec, xytext=(0, 0), arrowprops=dict(color=color, width=2, headwidth=8))
        ax.plot([], [], color=color, label=label)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('correlated 2-D input, learned weight direction', fontsize=9)

    ax = fig.add_subplot(2, 4, 2)
    ax.semilogy(np.linalg.norm(hebb_path, axis=1), color='tab:red', label='Hebb |w|')
    ax.semilogy(np.linalg.norm(oja_path, axis=1), color='tab:green', label='Oja |w|')
    ax.set_xlabel('sample')
    ax.set_title('weight norm: Hebb explodes, Oja -> 1', fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 4, 3)
    ax.plot(angle_deg(hebb_path), color='tab:red', label='Hebb')
    ax.plot(angle_deg(oja_path), color='tab:green', label='Oja')
    ax.set_xlabel('sample')
    ax.set_ylabel('angle to PC1 (deg)')
    ax.set_title('both align with the principal axis', fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 4, 4)
    ax.bar(range(1, k + 1), cosines, color='tab:purple')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('component')
    ax.set_title('Sanger GHA vs SVD: |cos(w_i, v_i)|', fontsize=9)
    for i, c in enumerate(cosines):
        ax.text(i + 1, c + 0.01, f'{c:.3f}', ha='center', fontsize=8)

    for i in range(k):
        ax = fig.add_subplot(2, 4, 5 + i)
        comp = w_gha[i].reshape(3, 10, 10).transpose(1, 2, 0)
        comp = (comp - comp.min()) / (comp.max() - comp.min() + 1e-12)
        ax.imshow(comp, interpolation='nearest')
        ax.set_title(f'GHA component {i + 1} ({explained[i]:.1%} var)', fontsize=9)
        ax.axis('off')
    fig.suptitle('Hebbian learning (unsupervised, local): plain Hebb, Oja, Sanger GHA', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_hebbian.png'), dpi=120)
    plt.close(fig)
    return {
        'hebb_final_norm': float(np.linalg.norm(hebb_path[-1])),
        'oja_final_norm': float(np.linalg.norm(oja_path[-1])),
        'hebb_final_angle_deg': float(angle_deg(hebb_path)[-1]),
        'oja_final_angle_deg': float(angle_deg(oja_path)[-1]),
        'gha_cosine_vs_svd': cosines,
        'gha_orthonormality_error': ortho[-1],
        'top4_explained_variance': explained,
    }


# ---------------------------------------------------------------------------
# Competitive learning
# ---------------------------------------------------------------------------

def demo_competitive(out_dir: str, quick: bool, seed: int) -> Dict[str, Any]:
    x, y = make_landcover_patches(400 if quick else 1600, seed=seed + 7)
    feats, mu, sd = _zscore(patch_features(x))
    k, epochs = 8, 5 if quick else 15

    variants = {}
    for name, init, conscience in [('random-sample init', 'samples', False),
                                   ('poor init (one corner)', 'clustered', False),
                                   ('poor init + conscience', 'clustered', True)]:
        layer = CompetitiveLayer(k, feats.shape[1], seed=seed)
        if init == 'samples':
            layer.init_from_samples(feats)
        else:
            layer.init_clustered(feats.min(axis=0) - 1.0)
        hist = layer.fit(feats, lr=0.05, epochs=epochs, conscience=conscience)
        assign = layer.predict(feats)
        variants[name] = {
            'layer': layer, 'hist': hist,
            'purity': cluster_purity(assign, y),
            'active_units': int(np.sum(layer.win_counts(feats) > 0)),
            'quantization_error': hist['quantization_error'][-1],
        }

    som = SOM(6, 6, feats.shape[1], seed=seed)
    som_hist = som.fit(feats, epochs=5 if quick else 20)
    som_assign = som.predict(feats)

    # 2-D view: project features on their first two principal axes (our SVD).
    _, _, vt = T.svd(feats - feats.mean(axis=0))
    proj = lambda a: (a - feats.mean(axis=0)) @ vt[:2].T  # noqa: E731
    p = proj(feats)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))
    for ax, (name, v) in zip(axes[:3], variants.items()):
        ax.scatter(p[:, 0], p[:, 1], c=y, cmap='tab10', s=4, alpha=0.4, vmin=0, vmax=9)
        proto = proj(v['layer'].w)
        ax.scatter(proto[:, 0], proto[:, 1], marker='*', s=220, c='k', edgecolors='yellow')
        ax.set_title(f"{name}\nactive {v['active_units']}/{k}, purity {v['purity']:.2f}", fontsize=9)
    handles = [plt.Line2D([], [], marker='o', ls='', color=plt.cm.tab10(i), label=n) for i, n in enumerate(CLASS_NAMES)]
    axes[0].legend(handles=handles, fontsize=7, loc='best')

    for name, v in variants.items():
        axes[3].plot(v['hist']['quantization_error'], label=name)
    axes[3].plot(som_hist['quantization_error'], label='SOM 6x6 (36 units)', ls='--')
    axes[3].set_xlabel('epoch')
    axes[3].set_title('quantisation error', fontsize=9)
    axes[3].legend(fontsize=7)

    # SOM: show each unit's prototype colour (mean RGB, un-standardised) + majority class.
    colours = np.clip(som.w[:, :3] * sd[:3] + mu[:3], 0, 1).reshape(6, 6, 3)
    axes[4].imshow(colours, interpolation='nearest')
    for j in range(36):
        members = y[som_assign == j]
        label = CLASS_NAMES[np.bincount(members).argmax()][:4] if members.size else '-'
        axes[4].text(j % 6, j // 6, label, ha='center', va='center', fontsize=7, color='white',
                     bbox=dict(facecolor='black', alpha=0.4, pad=1, lw=0))
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    axes[4].set_title(f"SOM prototypes (mean RGB) + majority class\npurity {cluster_purity(som_assign, y):.2f}, "
                      f"topographic err {som_hist['topographic_error'][-1]:.2f}", fontsize=9)
    fig.suptitle('Competitive learning on patch features (mean R, G, B, texture): labels never used for training',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_competitive.png'), dpi=120)
    plt.close(fig)
    return {
        **{name: {k2: v[k2] for k2 in ('purity', 'active_units', 'quantization_error')} for name, v in variants.items()},
        'som': {'purity': cluster_purity(som_assign, y), 'quantization_error': som_hist['quantization_error'][-1],
                'topographic_error': som_hist['topographic_error'][-1]},
    }


# ---------------------------------------------------------------------------
# Error-correction learning
# ---------------------------------------------------------------------------

def demo_error_correction(out_dir: str, quick: bool, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    epochs = 30 if quick else 100

    sep_x = np.vstack([rng.normal([-2, -2], 0.7, (50, 2)), rng.normal([2, 2], 0.7, (50, 2))])
    sep_t = np.r_[-np.ones(50), np.ones(50)]
    xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor_t = np.array([-1, 1, 1, -1])

    # Real task: water (0) vs bare soil (3) from two features (blue - red, mean brightness).
    x, y = make_landcover_patches(400 if quick else 1200, seed=seed + 11)
    sel = np.isin(y, [0, 3])
    m = x[sel].mean(axis=(2, 3))
    lc_x, _, _ = _zscore(np.c_[m[:, 2] - m[:, 0], m.mean(axis=1)])
    lc_t = np.where(y[sel] == 0, 1, -1)
    half = len(lc_x) // 2
    tr, te = slice(0, half), slice(half, None)

    runs = {}
    for name, (xx, tt) in {'separable blobs': (sep_x, sep_t), 'XOR': (xor_x, xor_t),
                           'water vs bare soil': (lc_x[tr], lc_t[tr])}.items():
        p = Perceptron(2, seed=seed)
        runs[name] = {'model': p, 'result': p.fit(xx, tt, lr=1.0, epochs=epochs)}
    ada = Adaline(2, seed=seed)
    ada_res = ada.fit(lc_x[tr], lc_t[tr], lr=0.005, epochs=epochs)
    p_lc = runs['water vs bare soil']['model']

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for name, r in runs.items():
        errs = np.array(r['result']['errors_per_epoch'], dtype=float)
        n = {'separable blobs': len(sep_x), 'XOR': 4, 'water vs bare soil': half}[name]
        axes[0].plot(range(1, len(errs) + 1), errs / n, marker='.' if len(errs) < 10 else None,
                     label=f"{name} ({'converged' if r['result']['converged'] else 'never converges'})")
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('fraction of updates (mistakes)')
    axes[0].set_title('Perceptron: stops only when every sample is correct', fontsize=9)
    axes[0].legend(fontsize=8)

    axes[1].plot(np.array(ada_res['errors_per_epoch']) / half, label='LMS training error rate')
    lc_err = [e / half for e in runs['water vs bare soil']['result']['errors_per_epoch']]
    axes[1].plot(lc_err, alpha=0.6, label='perceptron mistakes / epoch')
    ax2 = axes[1].twinx()
    ax2.plot(ada_res['mse_per_epoch'], color='tab:green', label='LMS mean squared error')
    ax2.set_ylabel('MSE', color='tab:green')
    axes[1].set_xlabel('epoch')
    axes[1].set_title('water vs bare soil: LMS settles, perceptron keeps jumping', fontsize=9)
    axes[1].legend(fontsize=8, loc='upper right')

    axes[2].scatter(lc_x[te][:, 0], lc_x[te][:, 1], c=lc_t[te], cmap='coolwarm', s=8, alpha=0.6)
    grid = np.linspace(lc_x[:, 0].min(), lc_x[:, 0].max(), 50)
    for model, color, name in [(p_lc, 'k', 'perceptron'), (ada, 'tab:green', 'LMS / Adaline')]:
        if abs(model.w[1]) > 1e-12:
            axes[2].plot(grid, -(model.w[0] * grid + model.b) / model.w[1], color=color, label=name)
    axes[2].set_ylim(lc_x[:, 1].min(), lc_x[:, 1].max())
    axes[2].set_xlabel('blue - red (z)')
    axes[2].set_ylabel('brightness (z)')
    axes[2].set_title('held-out test patches and learned boundaries', fontsize=9)
    axes[2].legend(fontsize=8)
    fig.suptitle('Error-correction learning: perceptron rule vs Widrow-Hoff delta rule', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_error_correction.png'), dpi=120)
    plt.close(fig)
    return {
        'perceptron_converged': {n: r['result']['converged'] for n, r in runs.items()},
        'perceptron_epochs_to_converge_separable': runs['separable blobs']['result']['epochs_run'],
        'water_vs_soil_test_acc': {
            'perceptron_last_iterate': float(np.mean(p_lc.predict(lc_x[te]) == lc_t[te])),
            'lms': float(np.mean(ada.predict(lc_x[te]) == lc_t[te])),
        },
        'lms_final_mse': ada_res['mse_per_epoch'][-1],
    }


# ---------------------------------------------------------------------------
# Energy-based: Hopfield
# ---------------------------------------------------------------------------

def footprint_patterns() -> Dict[str, np.ndarray]:
    """10x10 building-footprint masks (+1 = roof, -1 = ground)."""
    shapes = {}
    for name in ['L-block', 'courtyard', 'twin slabs', 'junction']:
        g = -np.ones((10, 10))
        if name == 'L-block':
            g[1:9, 1:4] = 1
            g[6:9, 1:9] = 1
        elif name == 'courtyard':
            g[1:9, 1:9] = 1
            g[3:7, 3:7] = -1
        elif name == 'twin slabs':
            g[1:9, 1:4] = 1
            g[1:9, 6:9] = 1
        else:
            g[4:6, :] = 1
            g[:, 4:6] = 1
        shapes[name] = g.ravel()
    return shapes


def demo_hopfield(out_dir: str, quick: bool, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    shapes = footprint_patterns()
    patterns = np.stack(list(shapes.values()))
    nets = {rule: Hopfield(100).store(patterns, rule=rule) for rule in ('hebbian', 'pseudoinverse')}

    cues = [corrupt(p, 0.15, rng) for p in patterns]
    recalls = {rule: [net.recall(c, seed=seed + i) for i, c in enumerate(cues)] for rule, net in nets.items()}
    trials = 3 if quick else 20
    recall_rate = {}
    for rule, net in nets.items():
        hits = sum(np.array_equal(net.recall(corrupt(p, 0.15, rng), seed=t)[0], p)
                   for p in patterns for t in range(trials))
        recall_rate[rule] = hits / (len(patterns) * trials)
    overlap = (patterns @ patterns.T / 100)[np.triu_indices(4, 1)]

    loads = [2, 5, 8, 10, 12, 14, 17, 20, 25, 30]
    capacity = hopfield_capacity(100, loads, trials=2 if quick else 6, seed=seed)

    fig = plt.figure(figsize=(18, 7.5))
    names = list(shapes)
    rows = [('stored', patterns), ('15% corrupted cue', cues),
            ('Hebbian recall', [r[0] for r in recalls['hebbian']]),
            ('pseudo-inverse recall', [r[0] for r in recalls['pseudoinverse']])]
    for r, (row_name, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(4, 7, r * 7 + c + 1)
            ax.imshow(np.asarray(img).reshape(10, 10), cmap='gray_r', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=8)
            if r >= 2:
                ok = np.array_equal(np.asarray(img), patterns[c])
                for spine in ax.spines.values():
                    spine.set_edgecolor('tab:green' if ok else 'tab:red')
                    spine.set_linewidth(2.5)
    ax = fig.add_subplot(1, 7, (5, 5))
    for rule, color in [('hebbian', 'tab:red'), ('pseudoinverse', 'tab:green')]:
        e = recalls[rule][0][1]
        ax.plot(e, color=color, label=rule)
    ax.set_xlabel('single-unit update')
    ax.set_title('energy E = -1/2 s^T W s\n(L-block cue): never increases', fontsize=9)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 7, (6, 7))
    ax.plot(np.array(loads) / 100, [capacity[p] for p in loads], 'o-')
    ax.axvline(0.138, color='k', ls='--', label='theoretical 0.138 N')
    ax.set_xlabel('load P / N (random patterns, N = 100)')
    ax.set_ylabel('exact recall from 10% noise')
    ax.set_title('Hebbian storage capacity', fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle('Hopfield network: footprints as attractors. Correlated patterns break Hebbian storage; '
                 'the projection rule fixes it', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_hopfield.png'), dpi=120)
    plt.close(fig)

    monotone = all(np.all(np.diff(r[1]) <= 1e-9) for rs in recalls.values() for r in rs)
    return {
        'footprint_recall_rate_15pct_noise': recall_rate,
        'mean_pairwise_pattern_overlap': float(overlap.mean()),
        'energy_monotone_non_increasing': bool(monotone),
        'capacity_recall_by_load': {str(k): v for k, v in capacity.items()},
    }


# ---------------------------------------------------------------------------
# Energy-based: restricted Boltzmann machine
# ---------------------------------------------------------------------------

def demo_boltzmann(out_dir: str, quick: bool, seed: int) -> Dict[str, Any]:
    data = bars_and_stripes(3)
    optimum = float(np.log(1.0 / len(data)))
    epochs = 300 if quick else 3000
    runs = {}
    for label, method, k in [('CD-1', 'cd', 1), ('CD-10', 'cd', 10), ('exact ML gradient', 'exact', 1)]:
        rbm = RBM(9, 16, seed=seed)
        start = time.perf_counter()
        hist = rbm.fit(data, epochs=epochs, method=method, k=k, lr=0.5, batch_size=len(data))
        states, probs = rbm.model_distribution()
        data_set = {tuple(d) for d in data}
        mass = float(sum(p for s, p in zip(states, probs) if tuple(s) in data_set))
        runs[label] = {'rbm': rbm, 'hist': hist, 'mass_on_data': mass, 'runtime_s': time.perf_counter() - start}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={'width_ratios': [1.3, 1, 1.2]})
    for label, r in runs.items():
        axes[0].plot(r['hist']['log_likelihood'], label=f"{label} (final {r['hist']['log_likelihood'][-1]:.2f})")
    axes[0].axhline(optimum, color='k', ls='--', label=f'optimum log(1/14) = {optimum:.2f}')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('exact mean log-likelihood')
    axes[0].set_title('RBM (9 visible, 16 hidden): exact likelihood via 512-state enumeration', fontsize=9)
    axes[0].legend(fontsize=8)

    labels = list(runs)
    axes[1].bar(labels, [runs[k]['mass_on_data'] for k in labels], color=['#9ecae1', '#4292c6', '#08519c'])
    axes[1].set_ylim(0, 1)
    axes[1].axhline(14 / 512, color='grey', ls=':', label='uniform model (14/512)')
    axes[1].set_title('model probability mass on the 14 training patterns', fontsize=9)
    axes[1].legend(fontsize=8)

    samples = runs['exact ML gradient']['rbm'].sample(24, gibbs_steps=200)
    canvas = np.ones((4 * 4 - 1, 6 * 4 - 1)) * 0.5
    for i, s in enumerate(samples):
        r, c = divmod(i, 6)
        canvas[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = s.reshape(3, 3)
    axes[2].imshow(canvas, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    valid = sum(tuple(s) in {tuple(d) for d in data} for s in samples)
    axes[2].set_title(f'24 Gibbs samples (exact-ML RBM): {valid}/24 are valid bars/stripes', fontsize=9)
    axes[2].axis('off')
    fig.suptitle('Boltzmann learning: p(v) ~ exp(-F(v)); CD approximates the negative phase, '
                 'exact ML enumerates it', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_boltzmann.png'), dpi=120)
    plt.close(fig)
    return {
        'optimum_log_likelihood': optimum,
        **{label: {'final_log_likelihood': r['hist']['log_likelihood'][-1],
                   'best_log_likelihood': max(r['hist']['log_likelihood']),
                   'mass_on_data': r['mass_on_data'], 'runtime_s': r['runtime_s']} for label, r in runs.items()},
        'valid_samples_of_24': int(valid),
    }


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Step 6 classical learning demonstrations.')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'fundamentals', 'classical'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    summary: Dict[str, Any] = {}
    for name, fn in [('hebbian', demo_hebbian), ('competitive', demo_competitive),
                     ('error_correction', demo_error_correction), ('hopfield', demo_hopfield),
                     ('boltzmann', demo_boltzmann)]:
        start = time.perf_counter()
        summary[name] = fn(args.out, args.quick, args.seed)
        print(f"[{name:>16}] {time.perf_counter() - start:6.1f}s")
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    main()
