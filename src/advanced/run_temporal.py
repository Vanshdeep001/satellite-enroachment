"""
Step 9 (+ Step 7 3-D convolution) experiment: temporal change models on synthetic 6-date sequences.

Data: 48x48 sequences, T = 6. New buildings appear at a random date, existing ones persist,
plus illumination changes, a seasonal vegetation cycle, sensor noise and random clouds
(probability 0.3 per date after the first).
    train 240, val 60 (epoch selection), test 100 (natural clouds),
    test "cloudy last date" 100 (the final image is always cloud-covered).

Reports per model: change F1 / IoU / recall, date-of-change accuracy, parameters, runtime.
Figures: example predictions, accuracy comparison, transformer attention (incl. attention
paid to cloudy dates), 3-D convolution kernels and feature volumes, 2-D encoder feature maps,
ConvLSTM gate distributions and the effect of replacing sigmoid / tanh gates with step / sign.

Usage:
    python -m src.advanced.run_temporal [--quick] [--seeds 1] [--out outputs/advanced/temporal]
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict

# Undetected-change runs give NaN date accuracy; nanmean over all-NaN seeds is expected, not an error.
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.advanced.temporal import (  # noqa: E402
    MODEL_BUILDERS, ConvLSTMCD, Conv3DCD, TemporalTransformerCD, count_parameters, make_sequence_dataset,
    predict, temporal_metrics, train_temporal,
)

STEPS = 6


def _norm(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (np.ptp(img) + 1e-12)


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Step 9: temporal change models')
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'advanced', 'temporal'))
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    q = args.quick
    n_tr, n_val, n_te, epochs = (24, 8, 8, 2) if q else (240, 60, 100, 20)
    size = 32 if q else 48
    train = make_sequence_dataset(n_tr, seed=1, size=size, steps=STEPS)
    val = make_sequence_dataset(n_val, seed=2, size=size, steps=STEPS)
    test = make_sequence_dataset(n_te, seed=3, size=size, steps=STEPS)
    test_cloudy = make_sequence_dataset(n_te, seed=4, size=size, steps=STEPS, force_cloudy=[STEPS - 1])

    t0 = time.perf_counter()
    results: Dict[str, Any] = {}
    trained: Dict[str, torch.nn.Module] = {}
    for name, builder in MODEL_BUILDERS.items():
        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = builder(STEPS)
            info = train_temporal(model, train, val, STEPS, epochs=epochs, seed=seed)
            start = time.perf_counter()
            pred_clean = predict(model, test['x'])
            ms = 1000 * (time.perf_counter() - start) / len(test['x'])
            runs.append({'clean': temporal_metrics(pred_clean, test['y']),
                         'cloudy_last': temporal_metrics(predict(model, test_cloudy['x']), test_cloudy['y']),
                         'train_seconds': info['train_seconds'], 'ms_per_sequence': ms,
                         'history': info['history']})
            if seed == 0:
                trained[name] = model
        agg = {'parameters': count_parameters(trained[name])}
        for split in ('clean', 'cloudy_last'):
            for k in runs[0][split]:
                vals = [r[split][k] for r in runs]
                agg[f'{split}_{k}_mean'] = float(np.nanmean(vals))
                agg[f'{split}_{k}_std'] = float(np.nanstd(vals))
        agg['train_seconds'] = float(np.mean([r['train_seconds'] for r in runs]))
        agg['ms_per_sequence'] = float(np.mean([r['ms_per_sequence'] for r in runs]))
        agg['val_f1_curve'] = [h['val_f1'] for h in runs[0]['history']]
        results[name] = agg
        print(f"{name:<28} params {agg['parameters']:>6}  clean F1 {agg['clean_f1_mean']:.3f}  "
              f"cloudy-last F1 {agg['cloudy_last_f1_mean']:.3f}  date acc {agg['clean_date_accuracy_mean']:.3f}  "
              f"({agg['train_seconds']:.0f}s train)")

    # ---- step / sign vs sigmoid / tanh gates in the trained ConvLSTM -----------------------
    lstm: ConvLSTMCD = trained['ConvLSTM']
    lstm.cell.hard_gates = True
    hard = temporal_metrics(predict(lstm, test['x']), test['y'])
    lstm.cell.hard_gates = False
    smooth = temporal_metrics(predict(lstm, test['x']), test['y'])
    gate_study = {'smooth_sigmoid_tanh': smooth, 'hard_step_sign': hard}

    lstm.cell.record_gates, lstm.cell.gates = True, []
    with torch.no_grad():
        lstm(test['x'][:4])
    lstm.cell.record_gates = False
    gate_values = {k: torch.cat([g[k].flatten() for g in lstm.cell.gates]).numpy() for k in lstm.cell.gates[0]}
    gate_study['saturation'] = {
        k: float(np.mean((v < 0.05) | (v > 0.95))) if k != 'candidate' else float(np.mean(np.abs(v) > 0.95))
        for k, v in gate_values.items()}

    # ---- attention analysis on cloudy test sequences ---------------------------------------
    tr_model: TemporalTransformerCD = trained['Temporal transformer']
    n_seq = min(10, n_te)
    tr_model.attn.keep_attention = True
    with torch.no_grad():
        tr_model(test_cloudy['x'][:n_seq])
    tr_model.attn.keep_attention = False
    attn = tr_model.attn.last_attention.mean(dim=1)  # [N_pix, T, T], heads averaged
    attn = attn.view(n_seq, size, size, STEPS, STEPS)
    y10 = test_cloudy['y'][:n_seq]
    cl10 = test_cloudy['clouds'][:n_seq].permute(0, 2, 3, 1)  # [N, H, W, T]
    changed = y10 > 0
    attn_changed = attn[changed].mean(dim=0).numpy()
    attn_static = attn[~changed].mean(dim=0).numpy()
    received = attn.mean(dim=3)  # attention each key date receives, averaged over queries: [N, H, W, T]
    cloudy_keys = cl10 > 0.5
    clear_keys = cl10 < 0.05
    attention_cloud = {'mean_attention_to_cloudy_dates': float(received[cloudy_keys].mean()),
                       'mean_attention_to_clear_dates': float(received[clear_keys].mean()),
                       'uniform_reference': 1.0 / STEPS}

    summary = {'setup': {'size': size, 'steps': STEPS, 'n_train': n_tr, 'n_val': n_val, 'n_test': n_te,
                         'epochs': epochs, 'seeds': args.seeds, 'quick': q},
               'models': results, 'gate_study': gate_study, 'attention_vs_clouds': attention_cloud,
               'runtime_s': time.perf_counter() - t0}

    # ---- figures ---------------------------------------------------------------------------
    idx = 0
    seq = test_cloudy['x'][idx].permute(0, 2, 3, 1).numpy()
    preds = {name: predict(m, test_cloudy['x'][idx:idx + 1])[0].numpy() for name, m in trained.items()}
    fig, axes = plt.subplots(2, max(STEPS, 1 + len(preds)), figsize=(3 * max(STEPS, 5), 6.6))
    for t in range(STEPS):
        axes[0, t].imshow(seq[t])
        axes[0, t].set_title(f'date {t}' + (' (cloudy)' if test_cloudy['clouds'][idx, t].max() > 0.3 else ''), fontsize=9)
    cmap = plt.get_cmap('viridis', STEPS)
    axes[1, 0].imshow(test_cloudy['y'][idx].numpy(), cmap=cmap, vmin=-0.5, vmax=STEPS - 0.5)
    axes[1, 0].set_title('truth: date of change', fontsize=9)
    for j, (name, p) in enumerate(preds.items(), start=1):
        m = temporal_metrics(torch.tensor(p)[None], test_cloudy['y'][idx:idx + 1])
        axes[1, j].imshow(p, cmap=cmap, vmin=-0.5, vmax=STEPS - 0.5)
        axes[1, j].set_title(f'{name}\nF1 {m["f1"]:.2f}, date acc {m["date_accuracy"]:.2f}', fontsize=9)
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle('Multi-date change detection with a cloud-covered final date (0 = no change, k = appeared at date k)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_temporal_example.png'), dpi=120)
    plt.close(fig)

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [results[n]['clean_f1_mean'] for n in names], 0.4, label='natural clouds')
    axes[0].bar(x + 0.2, [results[n]['cloudy_last_f1_mean'] for n in names], 0.4, label='final date cloudy')
    axes[0].set_title('change F1')
    axes[1].bar(x - 0.2, [results[n]['clean_date_accuracy_mean'] for n in names], 0.4, label='natural clouds')
    axes[1].bar(x + 0.2, [results[n]['cloudy_last_date_accuracy_mean'] for n in names], 0.4, label='final date cloudy')
    axes[1].axhline(1 / (STEPS - 1), color='grey', ls=':', label='random date')
    axes[1].set_title('date-of-change accuracy (changed pixels)')
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{results[n]['parameters']:,} params" for n in names], fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    for n in names:
        axes[2].plot(range(1, len(results[n]['val_f1_curve']) + 1), results[n]['val_f1_curve'], label=n)
    axes[2].set_xlabel('epoch')
    axes[2].set_title('validation change F1')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_temporal_results.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, mat, title in [(axes[0], attn_static, 'unchanged pixels'), (axes[1], attn_changed, 'changed pixels')]:
        im = ax.imshow(mat, cmap='Blues', vmin=0)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
        ax.set_xlabel('key date (attended to)')
        ax.set_ylabel('query date')
        ax.set_title(f'transformer attention: {title}', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].bar(['cloudy dates\n(opacity > 0.5)', 'clear dates\n(opacity < 0.05)'],
                [attention_cloud['mean_attention_to_cloudy_dates'], attention_cloud['mean_attention_to_clear_dates']],
                color=['lightgrey', 'tab:blue'])
    axes[2].axhline(1 / STEPS, color='k', ls='--', label='uniform 1/T')
    axes[2].set_title('average attention a date receives', fontsize=10)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_attention.png'), dpi=130)
    plt.close(fig)

    c3d: Conv3DCD = trained['3-D CNN']
    k = c3d.conv1.weight.detach().numpy()  # [16, 3, 3(t), 3, 3]
    with torch.no_grad():
        vol = c3d.features(test_cloudy['x'][idx:idx + 1])[0].numpy()  # [C, T, H, W]
    energy = np.abs(vol).mean(axis=(1, 2, 3))
    top = np.argsort(energy)[::-1][:3]
    fig, axes = plt.subplots(4, 9, figsize=(16, 7.6))
    for f_i in range(8):
        for t in range(3):
            ax = axes[f_i // 2, (f_i % 2) * 3 + t]
            ax.imshow(_norm(k[f_i, :, t].transpose(1, 2, 0)), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(f'filter {f_i}', fontsize=7)
            if f_i < 2:
                ax.set_title(['t-1', 't', 't+1'][t], fontsize=8)
    for r, ch in enumerate(top[:3]):
        for t in range(3):
            ax = axes[r, 6 + t]
            ax.imshow(vol[ch, [0, STEPS // 2, STEPS - 1][t]], cmap='magma')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'ch {ch}, date {[0, STEPS // 2, STEPS - 1][t]}', fontsize=7)
    axes[3, 6].imshow(seq[0])
    axes[3, 7].imshow(seq[STEPS - 1])
    axes[3, 8].imshow(test_cloudy['y'][idx].numpy() > 0, cmap='gray')
    for ax, t in zip(axes[3, 6:], ['input date 0', f'input date {STEPS - 1}', 'true change']):
        ax.set_title(t, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('3-D CNN: first-layer kernels as (t-1, t, t+1) RGB slices (left) and 2nd-layer feature volumes (right)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_conv3d.png'), dpi=120)
    plt.close(fig)

    enc = lstm.encoder.net[0]
    with torch.no_grad():
        fmap = torch.relu(enc(test_cloudy['x'][idx, [0, STEPS - 1]])).numpy()  # [2, 16, H, W]
    fig, axes = plt.subplots(2, 9, figsize=(18, 4.4))
    for r, t in enumerate([0, STEPS - 1]):
        axes[r, 0].imshow(seq[t])
        axes[r, 0].set_title(f'date {t}', fontsize=8)
        for c in range(8):
            axes[r, c + 1].imshow(fmap[r, c], cmap='viridis')
            axes[r, c + 1].set_title(f'channel {c}', fontsize=7)
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle('2-D convolution feature maps (ConvLSTM frame encoder, layer 1, after ReLU)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_feature_maps_2d.png'), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (k_name, v) in zip(axes[:4], gate_values.items()):
        ax.hist(v, bins=60, color='tab:purple' if k_name == 'candidate' else 'tab:blue')
        ax.set_title(f"{k_name} gate ({'tanh' if k_name == 'candidate' else 'sigmoid'})\n"
                     f"saturated: {gate_study['saturation'][k_name]:.0%}", fontsize=9)
        ax.set_yscale('log')
    axes[4].bar(['sigmoid / tanh\n(trained)', 'step / sign\n(hard limits)'],
                [smooth['f1'], hard['f1']], color=['tab:blue', 'tab:red'])
    axes[4].set_ylim(0, 1)
    axes[4].set_title('ConvLSTM change F1 with smooth vs hard gates', fontsize=9)
    fig.suptitle('Gate activations: why LSTMs use sigmoid (0..1 soft switches) and tanh (signed values)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_lstm_gates.png'), dpi=120)
    plt.close(fig)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({'gate_study': {k: (v if k == 'saturation' else {'f1': v['f1']}) for k, v in gate_study.items()},
                      'attention_vs_clouds': attention_cloud}, indent=1))
    print(f"done in {summary['runtime_s']:.0f}s -> {args.out}")
    return summary


if __name__ == '__main__':
    main()
