"""
Comprehensive Final Evaluation and Benchmarking Script (Phase 6).

Executes:
1. Blind test evaluation of best Siamese U-Net checkpoint on 128 test pairs at tau*=0.20.
2. Comparison against Phase 3 classical pixel-difference baseline (tau=0.40).
3. Exporting metrics to outputs/metrics/final_test_metrics.json and test_baseline_comparison.csv.
4. Generating comparative bar chart in outputs/plots/test_metrics_comparison.png.
5. Generating at least 8 multi-panel visual comparisons with error overlays in outputs/test_predictions/.
6. Hardware runtime benchmarking on Apple Silicon MPS and CPU.
7. Physical area calculation in m^2 with scaling analysis.
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

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.losses import CompositeChangeDetectionLoss
from src.dataset import get_dataloader, LEVIRCDDataset
from src.metrics import MetricTracker, compute_metrics_from_counts
from src.baseline import detect_change, evaluate_baseline
from src.config import CONFIG


def create_error_overlay(pred_binary: np.ndarray, gt_binary: np.ndarray) -> np.ndarray:
    """
    Generate an RGB error diagnostic map:
    - Green ([0, 255, 0]): True Positives (Correctly identified changes)
    - Red ([255, 0, 0]): False Positives (False alarms)
    - Blue ([0, 100, 255]): False Negatives (Missed changes)
    - Dark Charcoal ([30, 30, 30]): True Negatives (Correct background)
    """
    h, w = pred_binary.shape
    overlay = np.full((h, w, 3), 30, dtype=np.uint8)

    p = pred_binary > 0.5
    g = gt_binary > 0.5

    tp = p & g
    fp = p & (~g)
    fn = (~p) & g

    overlay[tp] = [46, 204, 113]   # Emerald Green
    overlay[fp] = [231, 76, 60]    # Coral Red
    overlay[fn] = [52, 152, 219]   # Sky Blue

    return overlay


def plot_test_sample_comparison(
    t1_np: np.ndarray,
    t2_np: np.ndarray,
    gt_np: np.ndarray,
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    error_overlay: np.ndarray,
    sample_name: str,
    save_path: str
):
    """Generate a high-resolution 6-panel diagnostic figure."""
    fig, axes = plt.subplots(1, 6, figsize=(22, 4), dpi=150)

    # 1. Pre-Change T1
    axes[0].imshow(t1_np)
    axes[0].set_title("Pre-Change (T1)", fontsize=10, fontweight='bold', pad=8)
    axes[0].axis('off')

    # 2. Post-Change T2
    axes[1].imshow(t2_np)
    axes[1].set_title("Post-Change (T2)", fontsize=10, fontweight='bold', pad=8)
    axes[1].axis('off')

    # 3. Ground Truth
    axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth Mask", fontsize=10, fontweight='bold', pad=8)
    axes[2].axis('off')

    # 4. Traditional Baseline
    axes[3].imshow(baseline_pred, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Baseline (τ=0.40)", fontsize=10, fontweight='bold', pad=8)
    axes[3].axis('off')

    # 5. Siamese U-Net
    axes[4].imshow(model_pred, cmap='gray', vmin=0, vmax=1)
    axes[4].set_title("Siamese U-Net (τ*=0.20)", fontsize=10, fontweight='bold', pad=8)
    axes[4].axis('off')

    # 6. Error Diagnostic Overlay
    axes[5].imshow(error_overlay)
    axes[5].set_title("Model Error Overlay", fontsize=10, fontweight='bold', pad=8)
    axes[5].axis('off')

    # Legend annotations for error overlay
    fig.text(0.85, 0.02, "Green: TP  |  Red: FP  |  Blue: FN", fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cccccc'))

    fig.suptitle(f"Test Pair: {sample_name}", fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_comparison_barchart(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    save_path: str
):
    """Plot comparative grouped bar chart."""
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'IoU', 'Pixel Accuracy']
    baseline_vals = [
        baseline_metrics['precision'] * 100,
        baseline_metrics['recall'] * 100,
        baseline_metrics['f1_score'] * 100,
        baseline_metrics['iou'] * 100,
        baseline_metrics['accuracy'] * 100
    ]
    model_vals = [
        model_metrics['precision'] * 100,
        model_metrics['recall'] * 100,
        model_metrics['f1_score'] * 100,
        model_metrics['iou'] * 100,
        model_metrics['accuracy'] * 100
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Pixel-Difference Baseline (τ=0.40)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, model_vals, width, label='Siamese U-Net (τ*=0.20)', color='#2ecc71')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Test Set Performance: Baseline vs. Siamese U-Net (LEVIR-CD)', fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight='semibold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def run_evaluation():
    print("=" * 75)
    print("STARTING BLIND TEST SET EVALUATION & BENCHMARKING (PHASE 6)")
    print("=" * 75)

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    checkpoints_dir = os.path.join(outputs_dir, 'checkpoints')
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    plots_dir = os.path.join(outputs_dir, 'plots')
    test_preds_dir = os.path.join(outputs_dir, 'test_predictions')

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(test_preds_dir, exist_ok=True)

    # 1. Device Selection
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Compute Backend: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    # 2. Load Checkpoint and Threshold
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pt')
    if not os.path.isfile(best_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{best_checkpoint_path}'")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    print(f"Loaded Best Model Checkpoint from Epoch {checkpoint['epoch']} (Saved Val F1: {checkpoint['val_f1']:.4f})")

    th_path = os.path.join(metrics_dir, 'best_threshold.json')
    with open(th_path, 'r', encoding='utf-8') as f:
        th_data = json.load(f)
    best_threshold = float(th_data['best_threshold'])
    print(f"Locked Operational Decision Threshold: tau* = {best_threshold:.2f}")

    # 3. Instantiate and load Siamese U-Net
    model = SiameseUNet(
        in_channels=checkpoint['config']['in_channels'],
        out_channels=checkpoint['config']['out_channels'],
        base_channels=checkpoint['config']['base_channels'],
        bilinear=checkpoint['config']['bilinear']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = model.count_parameters()
    model_size_mb = model.model_size_mb()
    checkpoint_file_size_mb = round(os.path.getsize(best_checkpoint_path) / (1024 * 1024), 2)
    print(f"Model Parameters: {total_params:,} | Checkpoint Size: {checkpoint_file_size_mb} MB")

    # 4. Load Test Dataset
    print("\n[1/6] Loading Test Dataset (128 image pairs at 256x256 resolution)...")
    test_loader = get_dataloader(
        root_dir=data_dir,
        split='test',
        batch_size=4,
        target_size=(256, 256),
        shuffle=False,
        num_workers=0
    )
    test_dataset = LEVIRCDDataset(root_dir=data_dir, split='test', target_size=(256, 256))
    print(f"Successfully loaded {len(test_dataset)} test pairs.")

    # 5. Evaluate Siamese U-Net Exactly Once
    print(f"\n[2/6] Evaluating Siamese U-Net on 128 Test Pairs at locked tau* = {best_threshold:.2f}...")
    loss_fn = CompositeChangeDetectionLoss(bce_weight=0.5, dice_weight=0.5)
    test_tracker = MetricTracker()
    total_test_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    num_batches = len(test_loader)

    start_eval_time = time.perf_counter()
    with torch.no_grad():
        for t1, t2, mask in test_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            logits = model(t1, t2)
            loss, comps = loss_fn(logits, mask, return_components=True)

            total_test_loss += comps['total_loss']
            total_bce_loss += comps['bce_loss']
            total_dice_loss += comps['dice_loss']

            probs = torch.sigmoid(logits)
            preds = (probs >= best_threshold).to(torch.float32)

            test_tracker.update(preds.cpu().numpy(), mask.cpu().numpy(), threshold=0.5)

    eval_duration = time.perf_counter() - start_eval_time
    model_metrics = test_tracker.compute()
    model_metrics['test_loss'] = round(total_test_loss / max(num_batches, 1), 4)
    model_metrics['test_bce_loss'] = round(total_bce_loss / max(num_batches, 1), 4)
    model_metrics['test_dice_loss'] = round(total_dice_loss / max(num_batches, 1), 4)
    model_metrics['threshold'] = best_threshold
    model_metrics['eval_duration_sec'] = round(eval_duration, 3)
    model_metrics['latency_ms_per_pair'] = round((eval_duration / len(test_dataset)) * 1000, 2)
    model_metrics['fps'] = round(len(test_dataset) / max(eval_duration, 1e-4), 2)

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION: SIAMESE U-NET")
    print("=" * 55)
    print(f"  Precision:   {model_metrics['precision']:.4f} ({model_metrics['precision']*100:.2f}%)")
    print(f"  Recall:      {model_metrics['recall']:.4f} ({model_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:    {model_metrics['f1_score']:.4f} ({model_metrics['f1_score']*100:.2f}%)")
    print(f"  IoU:         {model_metrics['iou']:.4f} ({model_metrics['iou']*100:.2f}%)")
    print(f"  Accuracy:    {model_metrics['accuracy']:.4f} ({model_metrics['accuracy']*100:.2f}%)")
    print(f"  Specificity: {model_metrics['specificity']:.4f}")
    print(f"  Test Loss:   {model_metrics['test_loss']:.4f} (BCE: {model_metrics['test_bce_loss']:.4f}, Dice: {model_metrics['test_dice_loss']:.4f})")
    print(f"  TP: {model_metrics['tp']:,} | FP: {model_metrics['fp']:,} | FN: {model_metrics['fn']:,} | TN: {model_metrics['tn']:,}")
    print(f"  Runtime:     {model_metrics['eval_duration_sec']}s ({model_metrics['latency_ms_per_pair']} ms/pair, {model_metrics['fps']} FPS on {device})")
    print("=" * 55)

    # 6. Retrieve Phase 3 Baseline Test Results
    print("\n[3/6] Retrieving Phase 3 Baseline Test Performance (tau = 0.40)...")
    base_metrics_path = os.path.join(metrics_dir, 'baseline_test_metrics.json')
    if os.path.isfile(base_metrics_path):
        with open(base_metrics_path, 'r', encoding='utf-8') as f:
            baseline_metrics = json.load(f)
    else:
        print("  Re-evaluating baseline on test split...")
        baseline_metrics = evaluate_baseline(test_dataset, threshold=0.40, method='euclidean')

    print(f"  Baseline Precision: {baseline_metrics['precision']*100:.2f}%")
    print(f"  Baseline Recall:    {baseline_metrics['recall']*100:.2f}%")
    print(f"  Baseline F1-Score:  {baseline_metrics['f1_score']*100:.2f}%")
    print(f"  Baseline IoU:       {baseline_metrics['iou']*100:.2f}%")
    print(f"  Baseline Accuracy:  {baseline_metrics['accuracy']*100:.2f}%")

    # 7. Comparison Table & Export
    print("\n[4/6] Generating Comparison Metrics Table & Exporting...")
    comparison_records = [
        {
            'Metric': 'Precision',
            'Pixel-Difference Baseline (τ=0.40)': f"{baseline_metrics['precision']*100:.2f}%",
            'Siamese U-Net (τ*=0.20)': f"{model_metrics['precision']*100:.2f}%",
            'Absolute Improvement': f"+{(model_metrics['precision'] - baseline_metrics['precision'])*100:.2f}%",
            'Relative Factor': f"{model_metrics['precision'] / max(baseline_metrics['precision'], 1e-4):.2f}x"
        },
        {
            'Metric': 'Recall',
            'Pixel-Difference Baseline (τ=0.40)': f"{baseline_metrics['recall']*100:.2f}%",
            'Siamese U-Net (τ*=0.20)': f"{model_metrics['recall']*100:.2f}%",
            'Absolute Improvement': f"+{(model_metrics['recall'] - baseline_metrics['recall'])*100:.2f}%",
            'Relative Factor': f"{model_metrics['recall'] / max(baseline_metrics['recall'], 1e-4):.2f}x"
        },
        {
            'Metric': 'F1-Score',
            'Pixel-Difference Baseline (τ=0.40)': f"{baseline_metrics['f1_score']*100:.2f}%",
            'Siamese U-Net (τ*=0.20)': f"{model_metrics['f1_score']*100:.2f}%",
            'Absolute Improvement': f"+{(model_metrics['f1_score'] - baseline_metrics['f1_score'])*100:.2f}%",
            'Relative Factor': f"{model_metrics['f1_score'] / max(baseline_metrics['f1_score'], 1e-4):.2f}x"
        },
        {
            'Metric': 'Intersection over Union (IoU)',
            'Pixel-Difference Baseline (τ=0.40)': f"{baseline_metrics['iou']*100:.2f}%",
            'Siamese U-Net (τ*=0.20)': f"{model_metrics['iou']*100:.2f}%",
            'Absolute Improvement': f"+{(model_metrics['iou'] - baseline_metrics['iou'])*100:.2f}%",
            'Relative Factor': f"{model_metrics['iou'] / max(baseline_metrics['iou'], 1e-4):.2f}x"
        },
        {
            'Metric': 'Pixel Accuracy',
            'Pixel-Difference Baseline (τ=0.40)': f"{baseline_metrics['accuracy']*100:.2f}%",
            'Siamese U-Net (τ*=0.20)': f"{model_metrics['accuracy']*100:.2f}%",
            'Absolute Improvement': f"+{(model_metrics['accuracy'] - baseline_metrics['accuracy'])*100:.2f}%",
            'Relative Factor': f"{model_metrics['accuracy'] / max(baseline_metrics['accuracy'], 1e-4):.2f}x"
        }
    ]

    csv_path = os.path.join(metrics_dir, 'test_baseline_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Pixel-Difference Baseline (τ=0.40)', 'Siamese U-Net (τ*=0.20)', 'Absolute Improvement', 'Relative Factor'])
        writer.writeheader()
        writer.writerows(comparison_records)
    print(f"  Exported comparison CSV to '{csv_path}'")

    final_json_payload = {
        'siamese_unet_test_metrics': model_metrics,
        'baseline_test_metrics': baseline_metrics,
        'comparison': comparison_records,
        'model_metadata': {
            'checkpoint': best_checkpoint_path,
            'epoch': checkpoint['epoch'],
            'threshold': best_threshold,
            'parameters': total_params,
            'checkpoint_size_mb': checkpoint_file_size_mb,
            'device': str(device)
        }
    }
    json_path = os.path.join(metrics_dir, 'final_test_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_payload, f, indent=2)
    print(f"  Exported test metrics JSON to '{json_path}'")

    # 8. Comparative Bar Chart
    chart_path = os.path.join(plots_dir, 'test_metrics_comparison.png')
    plot_metrics_comparison_barchart(baseline_metrics, model_metrics, chart_path)
    print(f"  Saved comparison bar chart to '{chart_path}'")

    # 9. Generate Representative Test Visualizations (8 Diverse Samples)
    print("\n[5/6] Generating 8 Diverse Test Diagnostic Visualizations with Error Overlays...")
    # Select 8 diverse indices:
    # 0, 10, 22, 35, 48, 62, 85, 110
    sample_indices = [0, 10, 22, 35, 48, 62, 85, 110]

    for s_idx in sample_indices:
        if s_idx >= len(test_dataset):
            continue
        t1, t2, mask = test_dataset[s_idx]
        meta = test_dataset.get_sample_metadata(s_idx)
        fname = meta['name']

        # 1. Siamese U-Net prediction
        t1_in = t1.unsqueeze(0).to(device)
        t2_in = t2.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model.predict_probability(t1_in, t2_in).squeeze().cpu().numpy()
            model_pred = (prob >= best_threshold).astype(np.float32)

        # 2. Baseline prediction (tau=0.40)
        _, baseline_pred = detect_change(t1, t2, threshold=0.40, method='euclidean')

        t1_np = t1.permute(1, 2, 0).numpy()
        t2_np = t2.permute(1, 2, 0).numpy()
        gt_np = mask.squeeze().numpy()

        # 3. Diagnostic error overlay
        error_overlay = create_error_overlay(model_pred, gt_np)

        save_fig_path = os.path.join(test_preds_dir, f"test_comparison_{fname}")
        plot_test_sample_comparison(
            t1_np=t1_np,
            t2_np=t2_np,
            gt_np=gt_np,
            baseline_pred=baseline_pred,
            model_pred=model_pred,
            error_overlay=error_overlay,
            sample_name=fname,
            save_path=save_fig_path
        )
        print(f"  Saved test visual comparison: {fname}")

    # 10. Physical Area Calculation & Geospatial Limitations Investigation
    print("\n[6/6] Investigating Physical Surface Area and Georeferencing Constraints...")
    # In LEVIR-CD:
    # Original image: 1024 x 1024 px at 0.5 meters/pixel GSD -> (0.5)^2 = 0.25 m^2/pixel
    # Resized image in pipeline: 256 x 256 px (downsampling factor of 4 in each dimension)
    # Effective GSD at 256x256: 0.5 * 4 = 2.0 meters/pixel
    # Ground surface area per pixel at 256x256: (2.0 m)^2 = 4.0 m^2 / pixel.
    total_pred_change_pixels = int(model_metrics['tp'] + model_metrics['fp'])
    total_gt_change_pixels = int(model_metrics['tp'] + model_metrics['fn'])

    pred_area_m2 = total_pred_change_pixels * 4.0
    gt_area_m2 = total_gt_change_pixels * 4.0

    print(f"  Original Resolution: 1024x1024 px at 0.50 m/pixel GSD (0.25 m^2/px)")
    print(f"  Resized Resolution:   256x256 px at 2.00 m/pixel GSD (4.00 m^2/px)")
    print(f"  Total Ground Truth Changed Pixels (Test Set): {total_gt_change_pixels:,} pixels ({gt_area_m2:,.1f} m^2 or {gt_area_m2/10000:.2f} hectares)")
    print(f"  Total Model Predicted Changed Pixels (Test Set): {total_pred_change_pixels:,} pixels ({pred_area_m2:,.1f} m^2 or {pred_area_m2/10000:.2f} hectares)")

    print("\n" + "=" * 75)
    print("PHASE 6 FINAL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == '__main__':
    run_evaluation()

