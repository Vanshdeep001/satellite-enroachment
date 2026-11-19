"""
Step 9 experiment: do VAE / GAN samples help when labelled data is scarce?

Setup (synthetic 16x16 land-cover patches, 4 classes):
    scarce training set: 20 real patches per class (80 total)
    validation 400 (epoch selection only), test 1600, all generated from disjoint seeds.
For each seed a conditional VAE and a conditional GAN are trained on the 80 real patches
only and asked for 200 new samples per class. A small CNN is then trained under:
    real | real + dihedral | real + CVAE | real + CGAN | real + CVAE + dihedral |
    CVAE only | CGAN only   (the last two = "train on synthetic, test on real", TSTR)
Sample quality: class consistency (accuracy of an oracle CNN trained on 4000 real patches
when it classifies generated samples) and diversity (mean pairwise distance).
Latent organisation: an unconditional VAE on 1000 patches, latent means projected to 2-D.

Usage:
    python -m src.advanced.run_generative [--seeds 3] [--quick] [--out outputs/advanced/generative]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.advanced.generative import (  # noqa: E402
    diversity, evaluate_classifier, train_cgan, train_classifier, train_cvae,
)
from src.fundamentals import tensor_ops as T  # noqa: E402
from src.fundamentals.landcover_toy import CLASS_NAMES, make_landcover_patches  # noqa: E402

SIZE = 16
K = len(CLASS_NAMES)


def tensors(n: int, seed: int):
    x, y = make_landcover_patches(n, patch=SIZE, seed=seed)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def grid_image(samples: torch.Tensor, rows: int, cols: int, pad: int = 1) -> np.ndarray:
    s = samples.detach().cpu().numpy().transpose(0, 2, 3, 1)
    h, w = s.shape[1:3]
    canvas = np.ones((rows * (h + pad) - pad, cols * (w + pad) - pad, 3))
    for i in range(min(len(s), rows * cols)):
        r, c = divmod(i, cols)
        canvas[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = s[i]
    return canvas


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Step 9: VAE / GAN augmentation study')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'advanced', 'generative'))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    q = args.quick
    n_train, n_val, n_test, n_oracle = (40, 80, 200, 400) if q else (80, 400, 1600, 4000)
    vae_epochs, gan_steps, clf_epochs, per_class = (30, 200, 8, 20) if q else (400, 3000, 60, 200)

    x_val, y_val = tensors(n_val, 9001)
    x_te, y_te = tensors(n_test, 9002)
    x_or, y_or = tensors(n_oracle, 9003)
    t0 = time.perf_counter()
    oracle = train_classifier(x_or, y_or, x_val, y_val, epochs=clf_epochs, seed=0)
    oracle_test = evaluate_classifier(oracle, x_te, y_te)
    print(f"oracle CNN (trained on {n_oracle} real): test acc {oracle_test['accuracy']:.3f}")

    conditions = ['real', 'real + dihedral', 'real + CVAE', 'real + CGAN', 'real + CVAE + dihedral',
                  'CVAE only (TSTR)', 'CGAN only (TSTR)']
    results: Dict[str, list] = {c: [] for c in conditions}
    quality = {'CVAE': [], 'CGAN': []}
    keep: Dict[str, Any] = {}
    for seed in range(args.seeds):
        x_tr, y_tr = tensors(n_train, 100 + seed)
        cvae, vae_hist = train_cvae(x_tr, y_tr, epochs=vae_epochs, seed=seed)
        gen, gan_hist = train_cgan(x_tr, y_tr, steps=gan_steps, seed=seed)
        y_syn = torch.arange(K).repeat_interleave(per_class)
        torch.manual_seed(seed)
        synth = {'CVAE': cvae.sample(y_syn), 'CGAN': gen.sample(y_syn)}
        for name, xs in synth.items():
            quality[name].append({'class_consistency': evaluate_classifier(oracle, xs, y_syn)['accuracy'],
                                  'diversity': diversity(xs[:400])})
        if seed == 0:
            keep = {'cvae': cvae, 'gen': gen, 'vae_hist': vae_hist, 'gan_hist': gan_hist, 'x_tr': x_tr, 'y_tr': y_tr}

        train_sets = {
            'real': (x_tr, y_tr, False),
            'real + dihedral': (x_tr, y_tr, True),
            'real + CVAE': (torch.cat([x_tr, synth['CVAE']]), torch.cat([y_tr, y_syn]), False),
            'real + CGAN': (torch.cat([x_tr, synth['CGAN']]), torch.cat([y_tr, y_syn]), False),
            'real + CVAE + dihedral': (torch.cat([x_tr, synth['CVAE']]), torch.cat([y_tr, y_syn]), True),
            'CVAE only (TSTR)': (synth['CVAE'], y_syn, False),
            'CGAN only (TSTR)': (synth['CGAN'], y_syn, False),
        }
        for cond, (xx, yy, aug) in train_sets.items():
            clf = train_classifier(xx, yy, x_val, y_val, epochs=clf_epochs, augment=aug, seed=seed)
            results[cond].append(evaluate_classifier(clf, x_te, y_te))
        print(f"seed {seed}: " + ', '.join(f"{c} {results[c][-1]['accuracy']:.3f}" for c in conditions))

    real_div = diversity(x_te[:400])
    summary = {
        'setup': {'n_train_real': n_train, 'n_val': n_val, 'n_test': n_test, 'patch': SIZE,
                  'synthetic_per_class': per_class, 'seeds': args.seeds, 'quick': q},
        'oracle_test_accuracy': oracle_test['accuracy'],
        'real_test_diversity': real_div,
        'classification': {c: {'accuracy_mean': float(np.mean([r['accuracy'] for r in rs])),
                               'accuracy_std': float(np.std([r['accuracy'] for r in rs])),
                               'macro_f1_mean': float(np.mean([r['macro_f1'] for r in rs]))}
                           for c, rs in results.items()},
        'sample_quality': {m: {'class_consistency_mean': float(np.mean([v['class_consistency'] for v in vs])),
                               'diversity_mean': float(np.mean([v['diversity'] for v in vs]))}
                           for m, vs in quality.items()},
    }

    # ---- latent organisation: unconditional VAE on 1000 real patches -------------------
    x_lat, y_lat = tensors(250 if q else 1000, 9004)
    vae, _ = train_cvae(x_lat, y_lat, num_classes=0, epochs=30 if q else 150, seed=0)
    with torch.no_grad():
        mu, _ = vae.encode(x_te, y_te)
    mu_np = mu.numpy().astype(np.float64)
    centred = mu_np - mu_np.mean(axis=0)
    _, s, vt = T.svd(centred)
    proj = centred @ vt[:2].T
    summary['latent'] = {'explained_by_2_pcs': float(np.sum(s[:2] ** 2) / np.sum(s ** 2))}
    cent = np.stack([proj[y_te.numpy() == c].mean(axis=0) for c in range(K)])
    within = np.mean([np.linalg.norm(proj[y_te.numpy() == c] - cent[c], axis=1).mean() for c in range(K)])
    between = np.mean([np.linalg.norm(cent[i] - cent[j]) for i in range(K) for j in range(i + 1, K)])
    summary['latent']['between_over_within_class_spread'] = float(between / within)
    summary['runtime_s'] = time.perf_counter() - t0

    # ---- figures -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    xs_real = torch.cat([keep['x_tr'][keep['y_tr'] == c][:6] for c in range(K)])
    y_show = torch.arange(K).repeat_interleave(6)
    torch.manual_seed(1)
    for ax, (title, imgs) in zip(axes[:3], [('real (scarce training set)', xs_real),
                                            ('CVAE samples', keep['cvae'].sample(y_show)),
                                            ('CGAN samples', keep['gen'].sample(y_show))]):
        ax.imshow(grid_image(imgs, K, 6), interpolation='nearest')
        ax.set_yticks([(SIZE + 1) * r + SIZE / 2 for r in range(K)])
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xticks([])
        ax.set_title(title)
    for c in range(K):
        sel = y_te.numpy() == c
        axes[3].scatter(proj[sel, 0], proj[sel, 1], s=4, alpha=0.5, label=CLASS_NAMES[c])
    axes[3].legend(markerscale=3, fontsize=8)
    axes[3].set_title(f"unconditional VAE latent (PCA of mu)\nbetween/within spread "
                      f"{summary['latent']['between_over_within_class_spread']:.2f}")
    fig.suptitle('Generative augmenters trained on 20 real patches per class', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_generative_samples_latent.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    vh = keep['vae_hist']
    axes[0].plot([h['reconstruction'] for h in vh], label='reconstruction (BCE)')
    axes[0].plot([h['kl'] for h in vh], label='KL')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('epoch')
    axes[0].set_title('CVAE loss terms (negative ELBO = rec + KL)')
    axes[0].legend()
    gh = keep['gan_hist']
    axes[1].plot([h['step'] for h in gh], [h['d_loss'] for h in gh], label='discriminator')
    axes[1].plot([h['step'] for h in gh], [h['g_loss'] for h in gh], label='generator')
    axes[1].set_xlabel('step')
    axes[1].set_title('CGAN losses')
    axes[1].legend()
    labels = list(summary['classification'])
    means = [summary['classification'][c]['accuracy_mean'] for c in labels]
    stds = [summary['classification'][c]['accuracy_std'] for c in labels]
    axes[2].bar(range(len(labels)), means, yerr=stds, capsize=3,
                color=['grey', '#74c476', '#6baed6', '#fd8d3c', '#3182bd', '#c6dbef', '#fdd0a2'])
    axes[2].axhline(oracle_test['accuracy'], color='k', ls='--', label=f'oracle ({n_oracle} real)')
    axes[2].axhline(1 / K, color='grey', ls=':', label='chance')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('test accuracy of a CNN trained on ...')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_generative_training_and_value.png'), dpi=130)
    plt.close(fig)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ('classification', 'sample_quality', 'latent')}, indent=1))
    print(f"done in {summary['runtime_s']:.0f}s -> {args.out}")
    return summary


if __name__ == '__main__':
    main()
