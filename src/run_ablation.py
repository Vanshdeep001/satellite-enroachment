"""
Controlled Ablation Studies and Training Improvement Experiments (Phase 7).

Compares:
1. Siamese U-Net (Absolute Difference Skips) - Standard Control
2. Siamese U-Net (Concatenation Skips) - Ablation Variant 1
3. Siamese U-Net (No Multi-Scale Skips) - Ablation Variant 2
4. Siamese U-Net with Synchronous Spatial Data Augmentation
5. Siamese U-Net with Dice-Dominant Loss (0.25 BCE + 0.75 Dice)

Experimental Rules:
- Evaluated exclusively on Train (445) and Validation (64) splits.
- Test partition strictly withheld.
- Original best_model.pt untouched.
- Standardized 6 epochs, batch size 4, AdamW, Cosine Annealing, seed 42.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.experiments.models import SiameseUNetConc, SiameseUNetNoSkip
from src.experiments.augmentations import SynchronousBitemporalAugmentation
from src.dataset import get_dataloader, LEVIRCDDataset
from src.losses import CompositeChangeDetectionLoss
from src.metrics import MetricTracker
from src.train import set_seed, select_device, train_one_epoch, evaluate_validation, find_best_threshold


def train_ablation_model(
    model_name: str,
    model: nn.Module,
    epochs: int = 6,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    use_augmentation: bool = False,
    device: torch.device = torch.device('cpu'),
    save_dir: str = "outputs/experiments"
) -> Dict[str, Any]:
    """Train a specific ablation model variant under standardized conditions."""
    print(f"\n{'='*65}")
    print(f"RUNNING EXPERIMENT: {model_name}")
    print(f"{'='*65}")

    set_seed(42)
    data_dir = os.path.join(PROJECT_ROOT, 'data')

    # Data loaders
    aug = SynchronousBitemporalAugmentation() if use_augmentation else None
    train_dataset = LEVIRCDDataset(root_dir=data_dir, split='train', target_size=(256, 256), transform=aug)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = get_dataloader(root_dir=data_dir, split='val', batch_size=batch_size, target_size=(256, 256), shuffle=False)

    model = model.to(device)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb = round((params_count * 4) / (1024 * 1024), 2)
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Parameters: {params_count:,} ({model_mb} MB)")
    print(f"  Loss Weights: BCE={bce_weight}, Dice={dice_weight} | Augmentation={use_augmentation}")

    loss_fn = CompositeChangeDetectionLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    best_weights = None

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        lr_now = scheduler.get_last_lr()[0]

        train_loss, train_bce, train_dice = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device
        )

        val_m = evaluate_validation(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, threshold=0.50
        )

        scheduler.step()
        ep_time = time.perf_counter() - ep_start

        rec = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_m['val_loss'], 4),
            'val_precision': round(val_m['precision'], 4),
            'val_recall': round(val_m['recall'], 4),
            'val_f1': round(val_m['f1_score'], 4),
            'val_iou': round(val_m['iou'], 4),
            'lr': float(f"{lr_now:.6f}"),
            'duration_sec': round(ep_time, 2)
        }
        history.append(rec)

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | "
            f"Val F1: {val_m['f1_score']:.4f} | Val IoU: {val_m['iou']:.4f} | "
            f"{ep_time:.1f}s"
        )

        if val_m['f1_score'] > best_val_f1:
            best_val_f1 = val_m['f1_score']
            best_epoch = epoch
            best_metrics = val_m
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    total_duration = time.perf_counter() - start_time

    # Latency benchmark on 20 pairs
    b_t1, b_t2, _ = next(iter(val_loader))
    t1_s = b_t1[:1].to(device)
    t2_s = b_t2[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(t1_s, t2_s)
        t_lat_start = time.perf_counter()
        for _ in range(20):
            _ = model(t1_s, t2_s)
            if device.type == 'mps':
                torch.mps.synchronize()
        latency_ms = round(((time.perf_counter() - t_lat_start) / 20) * 1000, 2)

    # Save candidate weights separately
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name.lower().replace(' ', '_')}.pt")
    torch.save({
        'model_name': model_name,
        'state_dict': best_weights,
        'parameters': params_count,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'best_val_iou': best_metrics['iou'],
        'history': history
    }, ckpt_path)

    print(f"  Saved candidate weights to '{ckpt_path}'")
    print(f"  Best Val F1: {best_val_f1:.4f} at Epoch {best_epoch} | Latency: {latency_ms} ms/pair")

    return {
        'model_name': model_name,
        'parameters': params_count,
        'model_size_mb': model_mb,
        'total_duration_sec': round(total_duration, 2),
        'latency_ms': latency_ms,
        'best_epoch': best_epoch,
        'best_val_f1': round(best_val_f1, 4),
        'best_val_iou': round(best_metrics['iou'], 4),
        'best_val_precision': round(best_metrics['precision'], 4),
        'best_val_recall': round(best_metrics['recall'], 4),
        'final_val_loss': round(history[-1]['val_loss'], 4),
        'final_train_loss': round(history[-1]['train_loss'], 4),
        'history': history,
        'checkpoint_path': ckpt_path
    }


def plot_ablation_comparison(ablation_results: List[Dict[str, Any]], save_path: str):
    """Plot comparative metrics across all ablation variants."""
    names = [r['model_name'] for r in ablation_results]
    f1s = [r['best_val_f1'] * 100 for r in ablation_results]
    ious = [r['best_val_iou'] * 100 for r in ablation_results]
    params = [r['parameters'] / 1e6 for r in ablation_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    rects1 = ax1.bar(x - width/2, f1s, width, label='Validation F1-Score (%)', color='#2ecc71')
    rects2 = ax1.bar(x + width/2, ious, width, label='Validation IoU (%)', color='#3498db')

    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Ablation Study: Architecture & Improvement Comparison (Validation Set)', fontsize=13, fontweight='bold', pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5, fontweight='semibold', rotation=15, ha='right')
    ax1.set_ylim(0, 85)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)

    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_learning_curves(ablation_results: List[Dict[str, Any]], save_path: str):
    """Plot validation F1-score learning curves across epochs for all variants."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = ['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#1abc9c']

    for i, res in enumerate(ablation_results):
        epochs = [h['epoch'] for h in res['history']]
        f1s = [h['val_f1'] for h in res['history']]
        ax.plot(epochs, f1s, '-o', label=f"{res['model_name']} (Peak: {res['best_val_f1']*100:.1f}%)",
                linewidth=2, markersize=5, color=colors[i % len(colors)])

    ax.set_title("Validation F1-Score Learning Curves Across Ablation Variants", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Validation F1-Score", fontsize=11, fontweight='semibold')
    ax.set_ylim(0.2, 0.75)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9.5, loc='lower right', frameon=True, framealpha=0.9)

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

    experiments = [
        {
            'name': 'Siamese U-Net (Diff Skips - Control)',
            'model_fn': lambda: SiameseUNet(in_channels=3, out_channels=1, base_channels=16),
            'bce_w': 0.5, 'dice_w': 0.5, 'aug': False
        },
        {
            'name': 'Siamese U-Net (Concatenation Skips)',
            'model_fn': lambda: SiameseUNetConc(in_channels=3, out_channels=1, base_channels=16),
            'bce_w': 0.5, 'dice_w': 0.5, 'aug': False
        },
        {
            'name': 'Siamese U-Net (No Multi-Scale Skips)',
            'model_fn': lambda: SiameseUNetNoSkip(in_channels=3, out_channels=1, base_channels=16),
            'bce_w': 0.5, 'dice_w': 0.5, 'aug': False
        },
        {
            'name': 'Siamese U-Net + Data Augmentation',
            'model_fn': lambda: SiameseUNet(in_channels=3, out_channels=1, base_channels=16),
            'bce_w': 0.5, 'dice_w': 0.5, 'aug': True
        },
        {
            'name': 'Siamese U-Net (Dice-Dominant Loss)',
            'model_fn': lambda: SiameseUNet(in_channels=3, out_channels=1, base_channels=16),
            'bce_w': 0.25, 'dice_w': 0.75, 'aug': False
        }
    ]

    all_results = []
    for exp in experiments:
        m = exp['model_fn']()
        res = train_ablation_model(
            model_name=exp['name'],
            model=m,
            epochs=6,
            batch_size=4,
            learning_rate=1e-3,
            bce_weight=exp['bce_w'],
            dice_weight=exp['dice_w'],
            use_augmentation=exp['aug'],
            device=device,
            save_dir=exp_dir
        )
        all_results.append(res)

    # Export results JSON
    json_path = os.path.join(exp_dir, 'ablation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ablation experiment results to '{json_path}'")

    # Export CSV summary
    csv_path = os.path.join(exp_dir, 'ablation_summary.csv')
    fieldnames = [
        'model_name', 'parameters', 'model_size_mb', 'best_epoch',
        'best_val_f1', 'best_val_iou', 'best_val_precision', 'best_val_recall',
        'final_val_loss', 'latency_ms', 'total_duration_sec'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"Saved ablation CSV summary to '{csv_path}'")

    # Generate comparative plots
    ablation_chart_path = os.path.join(plots_dir, 'ablation_comparison.png')
    plot_ablation_comparison(all_results, ablation_chart_path)
    print(f"Saved ablation bar chart to '{ablation_chart_path}'")

    curves_path = os.path.join(plots_dir, 'experiment_learning_curves.png')
    plot_learning_curves(all_results, curves_path)
    print(f"Saved experiment learning curves to '{curves_path}'")

    print(f"\n{'='*65}")
    print("ALL ABLATION EXPERIMENTS COMPLETED!")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

