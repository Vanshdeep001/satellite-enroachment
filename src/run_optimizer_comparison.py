"""
Controlled Optimizer Comparison Experiment (Phase 7).

Compares 4 Optimization Algorithms:
1. Vanilla Stochastic Gradient Descent (SGD)
2. SGD with Classical Polyak Momentum (momentum=0.9)
3. SGD with Nesterov Accelerated Gradient (NAG, momentum=0.9, nesterov=True)
4. AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)

Experimental Rules:
- Identical architecture: SiameseUNet (base_channels=16, 1.96M params)
- Identical data splits: Train (445) and Val (64). Test set untouched.
- Standardized 5 epochs on MPS, batch size 4, seed 42.
- Log learning curves and export plots to outputs/plots/optimizer_comparison.png.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.dataset import get_dataloader
from src.losses import CompositeChangeDetectionLoss
from src.train import set_seed, select_device, train_one_epoch, evaluate_validation


def run_single_optimizer_experiment(
    opt_name: str,
    optimizer_fn,
    epochs: int = 5,
    batch_size: int = 4,
    device: torch.device = torch.device('cpu'),
    save_dir: str = "outputs/experiments"
) -> Dict[str, Any]:
    print(f"\n{'-'*60}")
    print(f"EVALUATING OPTIMIZER: {opt_name}")
    print(f"{'-'*60}")

    set_seed(42)
    data_dir = os.path.join(PROJECT_ROOT, 'data')

    train_loader = get_dataloader(root_dir=data_dir, split='train', batch_size=batch_size, target_size=(256, 256), shuffle=True)
    val_loader = get_dataloader(root_dir=data_dir, split='val', batch_size=batch_size, target_size=(256, 256), shuffle=False)

    model = SiameseUNet(in_channels=3, out_channels=1, base_channels=16, bilinear=True).to(device)
    loss_fn = CompositeChangeDetectionLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = optimizer_fn(model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics = None

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        current_lr = scheduler.get_last_lr()[0]

        train_loss, _, _ = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device
        )
        val_m = evaluate_validation(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, threshold=0.50
        )
        scheduler.step()
        ep_duration = time.perf_counter() - ep_start

        rec = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_m['val_loss'], 4),
            'val_precision': round(val_m['precision'], 4),
            'val_recall': round(val_m['recall'], 4),
            'val_f1': round(val_m['f1_score'], 4),
            'val_iou': round(val_m['iou'], 4),
            'lr': float(f"{current_lr:.6f}"),
            'duration_sec': round(ep_duration, 2)
        }
        history.append(rec)

        print(
            f"  Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | "
            f"Val F1: {val_m['f1_score']:.4f} ({val_m['f1_score']*100:.1f}%) | "
            f"Val IoU: {val_m['iou']:.4f} ({val_m['iou']*100:.1f}%) | "
            f"{ep_duration:.1f}s"
        )

        if val_m['f1_score'] > best_val_f1:
            best_val_f1 = val_m['f1_score']
            best_epoch = epoch
            best_metrics = val_m

    total_time = time.perf_counter() - start_time
    print(f"  Completed in {total_time:.1f}s | Peak Val F1: {best_val_f1:.4f} at Epoch {best_epoch}")

    return {
        'optimizer_name': opt_name,
        'best_epoch': best_epoch,
        'best_val_f1': round(best_val_f1, 4),
        'best_val_iou': round(best_metrics['iou'], 4),
        'best_val_precision': round(best_metrics['precision'], 4),
        'best_val_recall': round(best_metrics['recall'], 4),
        'final_val_loss': round(history[-1]['val_loss'], 4),
        'final_train_loss': round(history[-1]['train_loss'], 4),
        'total_duration_sec': round(total_time, 2),
        'history': history
    }


def plot_optimizer_comparison(opt_results: List[Dict[str, Any]], save_path: str):
    """Plot comparative F1, IoU, and loss curves across optimizers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Bar Chart of Peak Performance
    names = [r['optimizer_name'] for r in opt_results]
    f1s = [r['best_val_f1'] * 100 for r in opt_results]
    ious = [r['best_val_iou'] * 100 for r in opt_results]

    x = np.arange(len(names))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, f1s, width, label='Peak Val F1 (%)', color='#e74c3c')
    rects2 = axes[0].bar(x + width/2, ious, width, label='Peak Val IoU (%)', color='#3498db')

    axes[0].set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Optimizer Peak Performance (Validation Split)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9.5, fontweight='semibold', rotation=15, ha='right')
    axes[0].set_ylim(0, 75)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # 2. Learning Curves of Val F1 over Epochs
    colors = ['#7f8c8d', '#e67e22', '#9b59b6', '#2ecc71']
    for i, res in enumerate(opt_results):
        epochs = [h['epoch'] for h in res['history']]
        f1_vals = [h['val_f1'] for h in res['history']]
        axes[1].plot(epochs, f1_vals, '-o', label=res['optimizer_name'], color=colors[i % len(colors)], linewidth=2)

    axes[1].set_title('Validation F1 Convergence vs. Epochs', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('Validation F1-Score', fontsize=10, fontweight='semibold')
    axes[1].set_ylim(0, 0.70)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    device = select_device()
    print(f"Hardware Compute Backend: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    exp_dir = os.path.join(PROJECT_ROOT, 'outputs', 'experiments')
    plots_dir = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    optimizers_to_test = [
        {
            'name': 'Vanilla SGD',
            'fn': lambda params: SGD(params, lr=1e-2, momentum=0.0)
        },
        {
            'name': 'SGD + Momentum (0.9)',
            'fn': lambda params: SGD(params, lr=1e-2, momentum=0.9)
        },
        {
            'name': 'Nesterov Momentum (0.9)',
            'fn': lambda params: SGD(params, lr=1e-2, momentum=0.9, nesterov=True)
        },
        {
            'name': 'AdamW (lr=1e-3, wd=1e-4)',
            'fn': lambda params: AdamW(params, lr=1e-3, weight_decay=1e-4)
        }
    ]

    opt_results = []
    for opt_cfg in optimizers_to_test:
        res = run_single_optimizer_experiment(
            opt_name=opt_cfg['name'],
            optimizer_fn=opt_cfg['fn'],
            epochs=5,
            batch_size=4,
            device=device,
            save_dir=exp_dir
        )
        opt_results.append(res)

    # Export results JSON
    json_path = os.path.join(exp_dir, 'optimizer_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(opt_results, f, indent=2)
    print(f"\nSaved optimizer comparison JSON to '{json_path}'")

    # Export CSV summary
    csv_path = os.path.join(exp_dir, 'optimizer_summary.csv')
    fieldnames = [
        'optimizer_name', 'best_epoch', 'best_val_f1', 'best_val_iou',
        'best_val_precision', 'best_val_recall', 'final_val_loss', 'final_train_loss', 'total_duration_sec'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in opt_results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"Saved optimizer CSV summary to '{csv_path}'")

    # Generate comparative plot
    plot_path = os.path.join(plots_dir, 'optimizer_comparison.png')
    plot_optimizer_comparison(opt_results, plot_path)
    print(f"Saved optimizer comparison plot to '{plot_path}'")

    print(f"\n{'='*60}")
    print("ALL OPTIMIZER EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

