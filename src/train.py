"""
Training Pipeline for Siamese U-Net Change Detection.

Features:
- Full LEVIR-CD train (445 pairs) and validation (64 pairs) splits.
- AdamW optimizer with Cosine Annealing learning rate scheduling.
- Composite BCE + Soft Dice Loss with class-imbalance mitigation.
- MPS (Apple Silicon Metal) hardware acceleration with safe CPU fallback.
- Validation checkpointing (best model by F1-score, and final model).
- Logging to outputs/metrics/training_history.csv.
- Probability threshold tuning on validation split.
- Plotting loss, F1, IoU, and LR curves in outputs/plots/.
- Visualizing representative validation predictions.
"""

import os
import sys
import time
import csv
import json
import random
from typing import Dict, Any, Tuple, List, Optional
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
from src.losses import CompositeChangeDetectionLoss
from src.dataset import get_dataloader, LEVIRCDDataset
from src.metrics import MetricTracker, compute_metrics_from_counts
from src.config import CONFIG, ProjectConfig


def set_seed(seed: int = 42):
    """Ensure full reproducibility across Python, NumPy, and PyTorch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device() -> torch.device:
    """Select compute backend with verification."""
    if torch.backends.mps.is_available():
        try:
            # Perform a quick dummy tensor operation to verify MPS driver stability
            test_t = torch.zeros(1, device='mps')
            del test_t
            return torch.device('mps')
        except Exception as e:
            print(f"[Warning] MPS verification failed ({e}). Falling back to CPU.")
            return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Execute one full training epoch over the training split.

    Returns:
        (avg_total_loss, avg_bce_loss, avg_dice_loss)
    """
    model.train()
    total_loss_accum = 0.0
    bce_loss_accum = 0.0
    dice_loss_accum = 0.0
    num_batches = len(loader)

    for b_idx, (t1, t2, mask) in enumerate(loader):
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward pass: raw unactivated logits [B, 1, H, W]
        logits = model(t1, t2)

        # Composite Loss calculation
        loss, components = loss_fn(logits, mask, return_components=True)

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        total_loss_accum += components['total_loss']
        bce_loss_accum += components['bce_loss']
        dice_loss_accum += components['dice_loss']

    return (
        total_loss_accum / max(num_batches, 1),
        bce_loss_accum / max(num_batches, 1),
        dice_loss_accum / max(num_batches, 1)
    )


def evaluate_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate validation split metrics and compute cumulative confusion matrix.

    Returns:
        Dict containing val_loss, precision, recall, f1_score, iou, accuracy, etc.
    """
    model.eval()
    total_loss_accum = 0.0
    num_batches = len(loader)
    tracker = MetricTracker()

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            logits = model(t1, t2)
            loss = loss_fn(logits, mask)
            total_loss_accum += loss.item()

            # Convert logits to binary prediction via sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).to(torch.float32)

            tracker.update(preds.cpu().numpy(), mask.cpu().numpy(), threshold=0.5)

    metrics = tracker.compute()
    metrics['val_loss'] = total_loss_accum / max(num_batches, 1)
    metrics['threshold'] = threshold
    return metrics


def find_best_threshold(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    candidate_thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
) -> Dict[str, Any]:
    """
    Perform a grid search over candidate probability thresholds on the validation split.

    Returns:
        Dict with 'best_threshold', 'best_f1', and results table.
    """
    model.eval()
    trackers = {th: MetricTracker() for th in candidate_thresholds}

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask_np = mask.squeeze(1).cpu().numpy()

            logits = model(t1, t2)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for th in candidate_thresholds:
                preds = (probs >= th).astype(np.float32)
                trackers[th].update(preds, mask_np, threshold=0.5)

    best_th = 0.5
    best_f1 = -1.0
    results = []

    for th in candidate_thresholds:
        m = trackers[th].compute()
        m['threshold'] = th
        results.append(m)
        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_th = th

    return {
        'best_threshold': best_th,
        'best_f1': best_f1,
        'results': results
    }


def generate_training_plots(history: List[Dict[str, Any]], save_dir: str):
    """Generate training curves and save to outputs/plots/."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    val_f1s = [h['val_f1'] for h in history]
    val_ious = [h['val_iou'] for h in history]
    lrs = [h['lr'] for h in history]

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, train_losses, 'b-o', label='Train Loss (BCE + Dice)', linewidth=2)
    ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax.set_title("Training vs. Validation Loss (Siamese U-Net)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Loss", fontsize=10, fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close(fig)

    # 2. Validation F1 and IoU Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, val_f1s, 'g-^', label='Validation F1-Score', linewidth=2.5, markersize=6)
    ax.plot(epochs, val_ious, 'm-d', label='Validation IoU', linewidth=2, markersize=5)
    ax.set_title("Validation F1-Score & IoU vs. Epochs", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Score", fontsize=10, fontweight='semibold')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'val_metrics_curve.png'), bbox_inches='tight')
    plt.close(fig)

    # 3. Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(epochs, lrs, 'k-o', label='Learning Rate (Cosine Annealing)', linewidth=2)
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Learning Rate", fontsize=10, fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lr_schedule.png'), bbox_inches='tight')
    plt.close(fig)

    # 4. Multi-Panel Comprehensive Summary Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-s', label='Val Loss')
    axes[0, 0].set_title("Composite Loss (BCE + Dice)", fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, val_f1s, 'g-^', label='Val F1-Score', linewidth=2)
    axes[0, 1].plot(epochs, val_ious, 'm-d', label='Val IoU', linewidth=2)
    axes[0, 1].set_title("Validation Segmentation Metrics", fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Metric Score")
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].legend()

    val_precisions = [h['val_precision'] for h in history]
    val_recalls = [h['val_recall'] for h in history]
    axes[1, 0].plot(epochs, val_precisions, 'c-o', label='Precision')
    axes[1, 0].plot(epochs, val_recalls, 'y-s', label='Recall')
    axes[1, 0].set_title("Precision vs. Recall", fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, lrs, 'k-o', label='Learning Rate')
    axes[1, 1].set_title("Cosine Annealing LR", fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_summary_panel.png'), bbox_inches='tight')
    plt.close(fig)


def save_validation_sample_predictions(
    model: nn.Module,
    val_dataset: LEVIRCDDataset,
    device: torch.device,
    threshold: float,
    save_dir: str,
    sample_indices: List[int] = [0, 5, 12, 25]
):
    """Generate 4-panel visual validation samples: T1 | T2 | Ground Truth | Prediction."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for idx in sample_indices:
        if idx >= len(val_dataset):
            continue
        t1, t2, mask = val_dataset[idx]
        meta = val_dataset.get_sample_metadata(idx)
        fname = meta['name']

        # Forward pass
        t1_in = t1.unsqueeze(0).to(device)
        t2_in = t2.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model.predict_probability(t1_in, t2_in).squeeze().cpu().numpy()
            pred_mask = (prob >= threshold).astype(np.float32)

        t1_np = t1.permute(1, 2, 0).numpy()
        t2_np = t2.permute(1, 2, 0).numpy()
        gt_np = mask.squeeze().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
        axes[0].imshow(t1_np)
        axes[0].set_title("Pre-Change (T1)", fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(t2_np)
        axes[1].set_title("Post-Change (T2)", fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Ground Truth Mask", fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f"Siamese U-Net (τ={threshold:.2f})", fontweight='bold')
        axes[3].axis('off')

        fig.suptitle(f"Validation Sample: {fname}", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"val_pred_{fname}")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


def run_training(
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
    seed: int = 42
):
    """Master training execution pipeline."""
    print("=" * 75)
    print("STARTING SIAMESE U-NET TRAINING PIPELINE (PHASE 5)")
    print("=" * 75)

    set_seed(seed)
    device = select_device()
    print(f"Selected Compute Backend: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    # Directory layout
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    checkpoints_dir = os.path.join(outputs_dir, 'checkpoints')
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    plots_dir = os.path.join(outputs_dir, 'plots')
    val_samples_dir = os.path.join(outputs_dir, 'val_predictions')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(val_samples_dir, exist_ok=True)

    # 1. Load DataLoaders
    print(f"\n[1/6] Ingesting LEVIR-CD Partitions (Batch Size = {batch_size}, Image Size = 256x256)...")
    train_loader = get_dataloader(
        root_dir=data_dir,
        split='train',
        batch_size=batch_size,
        target_size=(256, 256),
        shuffle=True,
        num_workers=0
    )
    val_loader = get_dataloader(
        root_dir=data_dir,
        split='val',
        batch_size=batch_size,
        target_size=(256, 256),
        shuffle=False,
        num_workers=0
    )
    val_dataset = LEVIRCDDataset(root_dir=data_dir, split='val', target_size=(256, 256))

    print(f"  Training Split:   {len(train_loader.dataset)} pairs ({len(train_loader)} batches)")
    print(f"  Validation Split: {len(val_loader.dataset)} pairs ({len(val_loader)} batches)")

    # 2. Instantiate Model, Loss & Optimizer
    print(f"\n[2/6] Initializing Siamese U-Net (base_channels={base_channels})...")
    model = SiameseUNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
        bilinear=True
    ).to(device)

    total_params = model.count_parameters()
    model_mb = model.model_size_mb()
    print(f"  Trainable Parameters: {total_params:,} ({model_mb} MB in float32)")

    loss_fn = CompositeChangeDetectionLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-6)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop
    print(f"\n[3/6] Starting {epochs}-Epoch Training Loop on {device}...")
    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pt')
    final_checkpoint_path = os.path.join(checkpoints_dir, 'final_model.pt')

    total_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = scheduler.get_last_lr()[0]

        # Train
        train_loss, train_bce, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )

        # Evaluate on validation split (threshold = 0.5 default)
        val_metrics = evaluate_validation(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=0.50
        )

        scheduler.step()
        epoch_duration = time.perf_counter() - epoch_start

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_bce': round(train_bce, 4),
            'train_dice': round(train_dice, 4),
            'val_loss': round(val_metrics['val_loss'], 4),
            'val_precision': round(val_metrics['precision'], 4),
            'val_recall': round(val_metrics['recall'], 4),
            'val_f1': round(val_metrics['f1_score'], 4),
            'val_iou': round(val_metrics['iou'], 4),
            'val_accuracy': round(val_metrics['accuracy'], 4),
            'lr': float(f"{current_lr:.6f}"),
            'duration_sec': round(epoch_duration, 2)
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} ({val_metrics['f1_score']*100:.1f}%) | "
            f"Val IoU: {val_metrics['iou']:.4f} ({val_metrics['iou']*100:.1f}%) | "
            f"LR: {current_lr:.6f} | "
            f"{epoch_duration:.1f}s"
        )

        # Checkpoint: Save best model by validation F1-score
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1_score'],
                'val_iou': val_metrics['iou'],
                'val_loss': val_metrics['val_loss'],
                'train_loss': train_loss,
                'config': {
                    'base_channels': base_channels,
                    'in_channels': 3,
                    'out_channels': 1,
                    'bilinear': True
                },
                'seed': seed
            }, best_checkpoint_path)
            print(f"  ⭐ New Best Model Checkpoint saved to '{best_checkpoint_path}' (Val F1: {best_val_f1:.4f})")

    total_training_duration = time.perf_counter() - total_start_time
    print(f"\nTraining completed in {total_training_duration:.2f} seconds ({total_training_duration/60:.2f} minutes).")
    print(f"Best Validation Model at Epoch {best_epoch} with F1-Score: {best_val_f1:.4f}")

    # Save final model separately
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'base_channels': base_channels,
            'in_channels': 3,
            'out_channels': 1,
            'bilinear': True
        },
        'seed': seed
    }, final_checkpoint_path)
    print(f"Saved final model checkpoint to '{final_checkpoint_path}'")

    # 4. Save Training History CSV
    csv_path = os.path.join(metrics_dir, 'training_history.csv')
    fieldnames = [
        'epoch', 'train_loss', 'train_bce', 'train_dice', 'val_loss',
        'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_accuracy',
        'lr', 'duration_sec'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"\n[4/6] Exported training metrics log to '{csv_path}'")

    # 5. Generate Metric Curves
    print("\n[5/6] Generating loss, F1, IoU, and LR progression plots...")
    generate_training_plots(history, plots_dir)
    print(f"  Saved all plots in '{plots_dir}/'")

    # 6. Post-Training Validation Threshold Search using Best Model
    print("\n[6/6] Tuning Probability Decision Threshold on Validation Split using Best Checkpoint...")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    candidate_thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    th_search_results = find_best_threshold(
        model=model,
        loader=val_loader,
        device=device,
        candidate_thresholds=candidate_thresholds
    )

    opt_threshold = th_search_results['best_threshold']
    opt_val_f1 = th_search_results['best_f1']
    print(f"  >> Optimal Validation Probability Threshold: tau* = {opt_threshold:.2f}")
    print(f"  >> Validation F1-Score at tau*: {opt_val_f1:.4f} ({opt_val_f1*100:.2f}%)")

    # Save threshold search results
    th_json_path = os.path.join(metrics_dir, 'best_threshold.json')
    with open(th_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_threshold': opt_threshold,
            'best_val_f1': opt_val_f1,
            'best_epoch': best_epoch,
            'candidate_evaluations': th_search_results['results']
        }, f, indent=2)
    print(f"  Saved optimal threshold specification to '{th_json_path}'")

    # Save Validation Sample Predictions at Optimal Threshold
    save_validation_sample_predictions(
        model=model,
        val_dataset=val_dataset,
        device=device,
        threshold=opt_threshold,
        save_dir=val_samples_dir,
        sample_indices=[0, 4, 15, 28]
    )
    print(f"  Saved representative validation visual predictions to '{val_samples_dir}/'")

    print("\n" + "=" * 75)
    print("PHASE 5 TRAINING & EXPERIMENT TRACKING COMPLETE!")
    print("=" * 75)

    return {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'optimal_threshold': opt_threshold,
        'history': history,
        'best_checkpoint_path': best_checkpoint_path,
        'total_duration_sec': total_training_duration
    }


if __name__ == '__main__':
    run_training(
        epochs=10,
        batch_size=4,
        learning_rate=1e-3,
        base_channels=16,
        seed=42
    )

