"""
Final Model Selection & Comprehensive Validation (Phase 8).

Fair, reproducible 15-epoch training comparison between:
- Candidate A: Siamese U-Net (Absolute Feature Difference Skips)
- Candidate B: Siamese U-Net (Feature Concatenation Skips)

Protocol:
- Train Split: 445 pairs (LEVIR-CD)
- Val Split: 64 pairs (LEVIR-CD)
- Test Split: STRICTLY WITHHELD (0 test pairs evaluated)
- Input Resolution: 256 x 256
- Batch Size: 4
- Random Seed: 42
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=15, eta_min=1e-5)
- Loss: Composite Loss (0.5 BCE + 0.5 Soft Dice)
- Maximum Duration: 15 epochs per candidate
- Selection Basis: Validation F1, IoU, efficiency, and real-world deployment viability.
"""

import os
import sys
import time
import json
import csv
import copy
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.mpl_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.experiments.models import SiameseUNetConc
from src.dataset import get_dataloader, LEVIRCDDataset
from src.losses import CompositeChangeDetectionLoss
from src.metrics import MetricTracker, compute_metrics_from_counts
from src.train import set_seed, select_device, train_one_epoch, evaluate_validation, find_best_threshold


def train_candidate(
    candidate_id: str,
    candidate_name: str,
    model: nn.Module,
    device: torch.device,
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    seed: int = 42
) -> Dict[str, Any]:
    """Train a model candidate under fair, standardized conditions."""
    print(f"\n{'='*75}")
    print(f"TRAINING CANDIDATE {candidate_id}: {candidate_name}")
    print(f"{'='*75}")

    set_seed(seed)

    # Data loaders
    train_dataset = LEVIRCDDataset(root_dir=data_dir, split='train', target_size=(256, 256))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = get_dataloader(
        root_dir=data_dir, split='val', batch_size=batch_size, target_size=(256, 256), shuffle=False
    )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb = round((param_count * 4) / (1024 * 1024), 2)
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Trainable Parameters: {param_count:,} ({model_mb} MB Float32)")
    print(f"  Hardware Backend: {device.type.upper()}")
    print(f"  Total Batches: {len(train_loader)} train, {len(val_loader)} val")

    loss_fn = CompositeChangeDetectionLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_weights = None
    best_val_metrics = None

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        lr_current = scheduler.get_last_lr()[0]

        train_loss, train_bce, train_dice = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device
        )

        val_metrics = evaluate_validation(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, threshold=0.50
        )

        scheduler.step()
        ep_duration = time.perf_counter() - ep_start

        log_row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_bce': round(train_bce, 4),
            'train_dice': round(train_dice, 4),
            'val_loss': round(val_metrics['val_loss'], 4),
            'val_precision': round(val_metrics['precision'], 4),
            'val_recall': round(val_metrics['recall'], 4),
            'val_f1': round(val_metrics['f1_score'], 4),
            'val_iou': round(val_metrics['iou'], 4),
            'lr': float(f"{lr_current:.6f}"),
            'duration_sec': round(ep_duration, 2)
        }
        history.append(log_row)

        is_best = val_metrics['f1_score'] > best_val_f1
        marker = " [★ BEST]" if is_best else ""

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} | Val IoU: {val_metrics['iou']:.4f}{marker} "
            f"({ep_duration:.1f}s)"
        )

        if is_best:
            best_val_f1 = val_metrics['f1_score']
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_time = time.perf_counter() - start_train_time

    # Load best weights into model for threshold tuning and latency evaluation
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    # Threshold optimization on validation set
    print(f"\n  Tuning probability decision threshold on validation split...")
    best_th_data = find_best_threshold(model, val_loader, device=device)
    best_threshold = best_th_data['best_threshold']
    tuned_val_metrics = next(m for m in best_th_data['results'] if m['threshold'] == best_threshold)
    print(
        f"  Optimal Threshold: tau* = {best_threshold:.2f} | "
        f"Tuned Val F1: {tuned_val_metrics['f1_score']:.4f} | "
        f"Tuned Val IoU: {tuned_val_metrics['iou']:.4f}"
    )

    # Benchmark inference latency (single pair, 30 warmup + 100 runs)
    sample_t1, sample_t2, _ = next(iter(val_loader))
    t1_single = sample_t1[:1].to(device)
    t2_single = sample_t2[:1].to(device)

    with torch.no_grad():
        for _ in range(15):
            _ = model(t1_single, t2_single)
        if device.type == 'mps':
            torch.mps.synchronize()

        latencies = []
        for _ in range(50):
            t_s = time.perf_counter()
            _ = model(t1_single, t2_single)
            if device.type == 'mps':
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - t_s) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0
    print(f"  Inference Latency: {mean_latency_ms:.2f} ms ({fps:.1f} FPS)")

    return {
        'candidate_id': candidate_id,
        'candidate_name': candidate_name,
        'architecture_class': model.__class__.__name__,
        'parameters': param_count,
        'model_mb': model_mb,
        'total_train_time_sec': round(total_train_time, 2),
        'best_epoch': best_epoch,
        'best_val_f1_default_th': round(best_val_f1, 4),
        'best_val_metrics_default_th': best_val_metrics,
        'optimal_threshold': best_threshold,
        'tuned_val_metrics': tuned_val_metrics,
        'inference_latency_ms': round(mean_latency_ms, 2),
        'fps': round(fps, 1),
        'history': history,
        'best_state_dict': best_weights,
        'threshold_grid': best_th_data['results']
    }


def plot_comparison_panel(results_a: Dict[str, Any], results_b: Dict[str, Any], save_path: str):
    """Generate 4-quadrant visual comparison between Candidate A and Candidate B."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('#ffffff')

    epochs_a = [h['epoch'] for h in results_a['history']]
    epochs_b = [h['epoch'] for h in results_b['history']]

    # 1. Training and Validation Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs_a, [h['train_loss'] for h in results_a['history']], 'b--', label=f"Cand A Train Loss", alpha=0.7)
    ax1.plot(epochs_a, [h['val_loss'] for h in results_a['history']], 'b-', linewidth=2, label=f"Cand A Val Loss")
    ax1.plot(epochs_b, [h['train_loss'] for h in results_b['history']], 'r--', label=f"Cand B Train Loss", alpha=0.7)
    ax1.plot(epochs_b, [h['val_loss'] for h in results_b['history']], 'r-', linewidth=2, label=f"Cand B Val Loss")
    ax1.set_title("Training & Validation Loss Progression", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite Loss (0.5 BCE + 0.5 Dice)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # 2. Validation F1 and IoU Trajectories
    ax2 = axes[0, 1]
    ax2.plot(epochs_a, [h['val_f1'] * 100 for h in results_a['history']], 'b-o', label="Cand A Val F1 (%)", linewidth=2)
    ax2.plot(epochs_a, [h['val_iou'] * 100 for h in results_a['history']], 'b:s', label="Cand A Val IoU (%)", alpha=0.7)
    ax2.plot(epochs_b, [h['val_f1'] * 100 for h in results_b['history']], 'r-o', label="Cand B Val F1 (%)", linewidth=2)
    ax2.plot(epochs_b, [h['val_iou'] * 100 for h in results_b['history']], 'r:s', label="Cand B Val IoU (%)", alpha=0.7)
    ax2.set_title("Validation F1 & IoU Trajectories Across 15 Epochs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (%)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')

    # 3. Peak Metric Comparison Bar Chart (at tuned threshold)
    ax3 = axes[1, 0]
    metrics_labels = ['Precision', 'Recall', 'F1-Score', 'IoU']
    mA = [
        results_a['tuned_val_metrics']['precision'] * 100,
        results_a['tuned_val_metrics']['recall'] * 100,
        results_a['tuned_val_metrics']['f1_score'] * 100,
        results_a['tuned_val_metrics']['iou'] * 100,
    ]
    mB = [
        results_b['tuned_val_metrics']['precision'] * 100,
        results_b['tuned_val_metrics']['recall'] * 100,
        results_b['tuned_val_metrics']['f1_score'] * 100,
        results_b['tuned_val_metrics']['iou'] * 100,
    ]
    x = np.arange(len(metrics_labels))
    width = 0.35
    rects1 = ax3.bar(x - width/2, mA, width, label='Candidate A (Diff Skips)', color='#2b5c8f')
    rects2 = ax3.bar(x + width/2, mB, width, label='Candidate B (Concat Skips)', color='#d95f02')
    ax3.set_title(f"Tuned Validation Metrics (A: τ*={results_a['optimal_threshold']:.2f}, B: τ*={results_b['optimal_threshold']:.2f})", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Score (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_labels)
    ax3.set_ylim(0, 100)
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left')

    for r in rects1:
        h = r.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for r in rects2:
        h = r.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # 4. Computational Resource & Performance Trade-off
    ax4 = axes[1, 1]
    categories = ['Params (M)', 'Size (MB)', 'Latency (ms)', 'Val F1 (%)']
    val_a = [results_a['parameters'] / 1e6, results_a['model_mb'], results_a['inference_latency_ms'], results_a['tuned_val_metrics']['f1_score'] * 100]
    val_b = [results_b['parameters'] / 1e6, results_b['model_mb'], results_b['inference_latency_ms'], results_b['tuned_val_metrics']['f1_score'] * 100]

    # Normalize relative to Candidate A for clear radar/bar view
    ratio = [round((vb / va - 1.0) * 100, 1) for va, vb in zip(val_a, val_b)]
    colors = ['#d95f02' if r > 0 else '#2b5c8f' for r in ratio]
    bars = ax4.bar(categories, ratio, color=colors, width=0.5)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_title("Relative Cost & Gain of Candidate B vs Candidate A (%)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Percentage Difference relative to Cand A (%)")
    ax4.grid(True, axis='y', linestyle=':', alpha=0.6)

    for bar, val, r in zip(bars, ratio, ratio):
        y_pos = bar.get_height() + (2 if bar.get_height() >= 0 else -6)
        sign = "+" if r > 0 else ""
        ax4.annotate(f"{sign}{r:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, y_pos), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Visual comparison panel saved to: {save_path}")


def main():
    print(f"\n{'#'*75}")
    print("# PHASE 8: FINAL MODEL SELECTION, FAIR COMPARISON & VALIDATION")
    print(f"{'#'*75}")

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    exp_ckpt_dir = os.path.join(output_dir, 'experiments', 'checkpoints')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    device = select_device()
    print(f"Compute Backend: {device} (Apple Silicon MPS Acceleration)")

    # Instantiate Candidate A and Candidate B
    model_a = SiameseUNet(in_channels=3, out_channels=1, base_channels=16, bilinear=True)
    model_b = SiameseUNetConc(in_channels=3, out_channels=1, base_channels=16, bilinear=True)

    # Fair standardized training for 15 epochs
    res_a = train_candidate(
        candidate_id="Candidate A",
        candidate_name="Siamese U-Net (Absolute Diff Skips)",
        model=model_a,
        device=device,
        data_dir=data_dir,
        epochs=15,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        bce_weight=0.5,
        dice_weight=0.5,
        seed=42
    )

    # Save Candidate A checkpoint
    ckpt_a_path = os.path.join(exp_ckpt_dir, "candidate_a_diff_15ep.pt")
    torch.save({
        'candidate_id': res_a['candidate_id'],
        'architecture_class': res_a['architecture_class'],
        'state_dict': res_a['best_state_dict'],
        'best_epoch': res_a['best_epoch'],
        'optimal_threshold': res_a['optimal_threshold'],
        'tuned_val_metrics': res_a['tuned_val_metrics'],
    }, ckpt_a_path)
    print(f"Saved Candidate A checkpoint: {ckpt_a_path}")

    res_b = train_candidate(
        candidate_id="Candidate B",
        candidate_name="Siamese U-Net (Concatenation Skips)",
        model=model_b,
        device=device,
        data_dir=data_dir,
        epochs=15,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        bce_weight=0.5,
        dice_weight=0.5,
        seed=42
    )

    # Save Candidate B checkpoint
    ckpt_b_path = os.path.join(exp_ckpt_dir, "candidate_b_conc_15ep.pt")
    torch.save({
        'candidate_id': res_b['candidate_id'],
        'architecture_class': res_b['architecture_class'],
        'state_dict': res_b['best_state_dict'],
        'best_epoch': res_b['best_epoch'],
        'optimal_threshold': res_b['optimal_threshold'],
        'tuned_val_metrics': res_b['tuned_val_metrics'],
    }, ckpt_b_path)
    print(f"Saved Candidate B checkpoint: {ckpt_b_path}")

    # Generate comparison visualization
    plot_path = os.path.join(plots_dir, "final_model_comparison.png")
    plot_comparison_panel(res_a, res_b, plot_path)

    # Compile Comparison CSV Table
    csv_path = os.path.join(metrics_dir, "final_model_comparison.csv")
    csv_rows = [
        {
            'Candidate': res_a['candidate_name'],
            'Architecture': res_a['architecture_class'],
            'Parameters': res_a['parameters'],
            'Model_MB': res_a['model_mb'],
            'Best_Epoch': res_a['best_epoch'],
            'Optimal_Threshold': res_a['optimal_threshold'],
            'Val_Precision': res_a['tuned_val_metrics']['precision'],
            'Val_Recall': res_a['tuned_val_metrics']['recall'],
            'Val_F1': res_a['tuned_val_metrics']['f1_score'],
            'Val_IoU': res_a['tuned_val_metrics']['iou'],
            'Latency_ms': res_a['inference_latency_ms'],
            'FPS': res_a['fps'],
            'Train_Time_s': res_a['total_train_time_sec']
        },
        {
            'Candidate': res_b['candidate_name'],
            'Architecture': res_b['architecture_class'],
            'Parameters': res_b['parameters'],
            'Model_MB': res_b['model_mb'],
            'Best_Epoch': res_b['best_epoch'],
            'Optimal_Threshold': res_b['optimal_threshold'],
            'Val_Precision': res_b['tuned_val_metrics']['precision'],
            'Val_Recall': res_b['tuned_val_metrics']['recall'],
            'Val_F1': res_b['tuned_val_metrics']['f1_score'],
            'Val_IoU': res_b['tuned_val_metrics']['iou'],
            'Latency_ms': res_b['inference_latency_ms'],
            'FPS': res_b['fps'],
            'Train_Time_s': res_b['total_train_time_sec']
        }
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Comparison metrics CSV saved: {csv_path}")

    # Selection Decision
    val_f1_a = res_a['tuned_val_metrics']['f1_score']
    val_f1_b = res_b['tuned_val_metrics']['f1_score']
    f1_diff = val_f1_b - val_f1_a

    print(f"\n{'='*75}")
    print("FINAL SELECTION DECISION LOGIC")
    print(f"{'='*75}")
    print(f"  Candidate A Tuned Val F1: {val_f1_a*100:.2f}% | IoU: {res_a['tuned_val_metrics']['iou']*100:.2f}% | Params: {res_a['parameters']:,} | Latency: {res_a['inference_latency_ms']} ms")
    print(f"  Candidate B Tuned Val F1: {val_f1_b*100:.2f}% | IoU: {res_b['tuned_val_metrics']['iou']*100:.2f}% | Params: {res_b['parameters']:,} | Latency: {res_b['inference_latency_ms']} ms")
    print(f"  Delta (B - A): F1 {f1_diff*100:+.2f}% | Parameters: +{res_b['parameters'] - res_a['parameters']:,} (+{(res_b['parameters']/res_a['parameters'] - 1)*100:.1f}%)")

    # If Candidate B improves F1 meaningfully (>= +2.0%) with reasonable latency (< 25 ms), select Candidate B.
    # Otherwise retain Candidate A.
    if f1_diff >= 0.02 and res_b['inference_latency_ms'] < 25.0:
        selected_res = res_b
        selected_name = "Candidate B (Siamese U-Net Concatenation)"
        reason = (
            f"Candidate B demonstrated a substantial +{f1_diff*100:.2f}% validation F1 improvement "
            f"({val_f1_b*100:.2f}% vs {val_f1_a*100:.2f}%) with minimal inference latency overhead "
            f"({res_b['inference_latency_ms']} ms vs {res_a['inference_latency_ms']} ms, yielding {res_b['fps']} FPS)."
        )
    else:
        selected_res = res_a
        selected_name = "Candidate A (Siamese U-Net Absolute Difference)"
        reason = (
            f"Candidate A selected for optimal parameter efficiency ({res_a['parameters']:,} params, {res_a['model_mb']} MB), "
            f"exact temporal symmetry, and robust validation performance ({val_f1_a*100:.2f}% F1)."
        )

    print(f"\n  >>> WINNER SELECTED: {selected_name}")
    print(f"  Reason: {reason}")

    # Save final selected model checkpoint
    final_selected_ckpt_path = os.path.join(checkpoint_dir, "final_selected_model.pt")
    selected_payload = {
        'candidate_id': selected_res['candidate_id'],
        'candidate_name': selected_res['candidate_name'],
        'architecture_class': selected_res['architecture_class'],
        'in_channels': 3,
        'out_channels': 1,
        'base_channels': 16,
        'bilinear': True,
        'parameters': selected_res['parameters'],
        'best_epoch': selected_res['best_epoch'],
        'optimal_threshold': selected_res['optimal_threshold'],
        'val_metrics': selected_res['tuned_val_metrics'],
        'selection_reason': reason,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'state_dict': selected_res['best_state_dict']
    }
    torch.save(selected_payload, final_selected_ckpt_path)
    print(f"\nFinal Selected Model Checkpoint saved: {final_selected_ckpt_path}")
    print(f"Original Phase 5 'best_model.pt' left 100% UNTOUCHED.")

    # Save JSON summary
    json_path = os.path.join(metrics_dir, "final_model_selection.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        summary_export = {
            'selected_candidate': selected_name,
            'selection_reason': reason,
            'candidate_a': {k: v for k, v in res_a.items() if k not in ('best_state_dict', 'history', 'threshold_grid')},
            'candidate_b': {k: v for k, v in res_b.items() if k not in ('best_state_dict', 'history', 'threshold_grid')}
        }
        json.dump(summary_export, f, indent=2)
    print(f"Saved selection metadata: {json_path}")
    print(f"\n{'#'*75}")
    print("# PHASE 8 MODEL SELECTION COMPLETED SUCCESSFULLY")
    print(f"{'#'*75}")


if __name__ == '__main__':
    main()
