"""
Run Baseline Experiment: Threshold Tuning, Test Evaluation & Visualization.

Executes:
1. Candidate threshold grid search on Validation split (64 pairs).
2. Metric tracking (Precision, Recall, F1, IoU, Accuracy, Confusion Matrix).
3. Selection of best threshold by Validation F1-score.
4. Final locked evaluation on Test split (128 pairs).
5. Saving machine-readable metric JSONs in outputs/metrics/.
6. Generating tuning curve and representative 5-panel sample comparisons in outputs/.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import LEVIRCDDataset
from src.baseline import (
    compute_difference_map,
    tune_threshold,
    evaluate_baseline,
    detect_change
)
from src.visualize import plot_change_comparison, plot_threshold_tuning_curve


def main():
    print("=" * 70)
    print("STARTING TRADITIONAL BASELINE EXPERIMENT (PHASE 3)")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    samples_dir = os.path.join(outputs_dir, 'baseline_samples')

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Load Validation Dataset
    print("\n[1/5] Loading Validation Dataset (64 pairs)...")
    val_dataset = LEVIRCDDataset(root_dir=data_dir, split='val', target_size=(256, 256))
    print(f"Loaded {len(val_dataset)} validation pairs at 256x256 resolution.")

    # 2. Candidate Threshold Grid Search
    candidate_thresholds = [
        0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
        0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50
    ]
    print(f"\n[2/5] Tuning Threshold over {len(candidate_thresholds)} candidates on Validation Split...")
    tuning_output = tune_threshold(
        data_source=val_dataset,
        candidate_thresholds=candidate_thresholds,
        method='euclidean'
    )

    best_thresh = tuning_output['best_threshold']
    best_f1 = tuning_output['best_f1']
    best_val_metrics = tuning_output['best_metrics']

    print(f"\nValidation Tuning Complete in {tuning_output['total_tuning_time_seconds']}s!")
    print(f">> Best Threshold: tau* = {best_thresh:.2f}")
    print(f">> Validation F1-Score: {best_f1:.4f}")
    print(f">> Validation Precision: {best_val_metrics['precision']:.4f}")
    print(f">> Validation Recall: {best_val_metrics['recall']:.4f}")
    print(f">> Validation IoU: {best_val_metrics['iou']:.4f}")

    # Save validation tuning JSON
    val_json_path = os.path.join(metrics_dir, 'baseline_val_tuning.json')
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_output, f, indent=2)
    print(f"Saved validation tuning metrics to '{val_json_path}'")

    # Generate Tuning Curve Plot
    curve_path = os.path.join(outputs_dir, 'baseline_tuning_curve.png')
    plot_threshold_tuning_curve(
        tuning_results=tuning_output['results'],
        selected_threshold=best_thresh,
        save_path=curve_path
    )
    print(f"Saved threshold tuning curve plot to '{curve_path}'")

    # 3. Final Evaluation on Test Split (ONCE)
    print("\n[3/5] Evaluating on Test Split (128 pairs) with locked tau*...")
    test_dataset = LEVIRCDDataset(root_dir=data_dir, split='test', target_size=(256, 256))
    print(f"Loaded {len(test_dataset)} test pairs at 256x256 resolution.")

    test_metrics = evaluate_baseline(
        data_source=test_dataset,
        threshold=best_thresh,
        method='euclidean'
    )

    print("\n" + "=" * 50)
    print(f"FINAL TEST SET EVALUATION (tau* = {best_thresh:.2f})")
    print("=" * 50)
    print(f"  Precision:   {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  Recall:      {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:    {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.2f}%)")
    print(f"  IoU:         {test_metrics['iou']:.4f} ({test_metrics['iou']*100:.2f}%)")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  TP: {test_metrics['tp']:,} | FP: {test_metrics['fp']:,} | FN: {test_metrics['fn']:,} | TN: {test_metrics['tn']:,}")
    print(f"  Total Runtime: {test_metrics['total_runtime_seconds']}s")
    print(f"  Avg Latency:   {test_metrics['avg_latency_ms']} ms/pair ({test_metrics['fps']} FPS)")
    print("=" * 50)

    # Save test metrics JSON
    test_json_path = os.path.join(metrics_dir, 'baseline_test_metrics.json')
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved final test metrics to '{test_json_path}'")

    # 4. Generate Visual Comparisons (4 diverse representative test samples)
    print("\n[4/5] Generating representative 5-panel sample visualizations...")
    # Find samples with diverse characteristics:
    # 1: High change (major construction)
    # 2: Moderate change with false alarms
    # 3: Low change / subtle difference
    # 4: Minimal/zero change background
    sample_indices = [2, 10, 24, 45]
    for idx in sample_indices:
        if idx >= len(test_dataset):
            continue
        t1, t2, mask = test_dataset[idx]
        meta = test_dataset.get_sample_metadata(idx)
        sample_name = meta['name']

        diff_map, pred_mask = detect_change(t1, t2, threshold=best_thresh, method='euclidean')

        sample_save_path = os.path.join(samples_dir, f"baseline_comparison_{sample_name}")
        plot_change_comparison(
            img_t1=t1,
            img_t2=t2,
            ground_truth=mask,
            pred_mask=pred_mask,
            diff_map=diff_map,
            threshold=best_thresh,
            title=f"Sample: {sample_name} | Baseline (tau={best_thresh:.2f})",
            save_path=sample_save_path
        )
        print(f"  Saved comparison for '{sample_name}' to '{sample_save_path}'")

    print("\n[5/5] BASELINE EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()

