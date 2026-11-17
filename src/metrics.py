"""
Evaluation metrics for Change Detection in Remote Sensing.

Implements confusion matrix accumulation (TP, FP, FN, TN) and classification
metrics (Precision, Recall, F1-score, Intersection-over-Union) with robust
handling of edge cases and extreme class imbalance.

Supports both per-sample computation and global dataset-level accumulation.
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
import torch


def compute_confusion_matrix(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute binary confusion matrix elements (TP, FP, FN, TN).

    Args:
        pred: Predicted change mask or probabilities (numpy array or torch Tensor).
              Can be float in [0, 1] or binary in {0, 1}.
        target: Ground-truth binary change mask in {0, 1}.
        threshold: Decision threshold for converting continuous predictions to binary.

    Returns:
        Dict with keys: 'tp', 'fp', 'fn', 'tn'.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten arrays
    pred_bin = (pred > threshold).astype(bool).ravel()
    target_bin = (target > 0.5).astype(bool).ravel()

    if pred_bin.shape != target_bin.shape:
        raise ValueError(
            f"Shape mismatch: pred has {pred_bin.shape} elements, target has {target_bin.shape} elements."
        )

    tp = int(np.sum(pred_bin & target_bin))
    fp = int(np.sum(pred_bin & ~target_bin))
    fn = int(np.sum(~pred_bin & target_bin))
    tn = int(np.sum(~pred_bin & ~target_bin))

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def compute_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    zero_division: float = 0.0
) -> Dict[str, float]:
    """
    Calculate classification metrics from confusion matrix counts.

    Handles zero denominators gracefully (e.g. when there are zero positive
    predictions or zero ground truth changes).

    Args:
        tp: True Positives (pixel count)
        fp: False Positives (pixel count)
        fn: False Negatives (pixel count)
        tn: True Negatives (pixel count)
        zero_division: Value to return when denominator is 0 (default: 0.0).

    Returns:
        Dict containing:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: 2 * (Precision * Recall) / (Precision + Recall)
        - iou: TP / (TP + FP + FN)
        - accuracy: (TP + TN) / (TP + FP + FN + TN)
        - specificity: TN / (TN + FP)
        - total_pixels: TP + FP + FN + TN
    """
    total = tp + fp + fn + tn

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else zero_division
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else zero_division

    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = zero_division

    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else zero_division
    accuracy = float((tp + tn) / total) if total > 0 else zero_division
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else zero_division

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total_pixels': total
    }


class MetricTracker:
    """
    Accumulates confusion matrix counts across batches or samples and computes
    dataset-level global metrics.

    In remote sensing change detection, calculating dataset-level metrics via
    global TP, FP, FN accumulation is standard practice to prevent unweighted
    per-image skew from small or empty change masks.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset internal accumulator counts to zero."""
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0
        self.num_samples: int = 0

    def update(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ):
        """
        Update accumulator with a single sample or batch.

        Args:
            pred: Predicted binary mask or probabilities.
            target: Ground-truth binary mask.
            threshold: Decision threshold.
        """
        counts = compute_confusion_matrix(pred, target, threshold=threshold)
        self.tp += counts['tp']
        self.fp += counts['fp']
        self.fn += counts['fn']
        self.tn += counts['tn']
        self.num_samples += 1

    def compute(self, zero_division: float = 0.0) -> Dict[str, float]:
        """Compute accumulated metrics over all observed samples."""
        metrics = compute_metrics_from_counts(
            self.tp, self.fp, self.fn, self.tn, zero_division=zero_division
        )
        metrics['num_samples'] = self.num_samples
        return metrics

