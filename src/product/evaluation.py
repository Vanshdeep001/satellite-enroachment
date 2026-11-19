"""
Evaluation package for the change detector (Project Step 10).

Binary change detection (pixel level, accumulated globally over the dataset):
    precision, recall (= change recall), F1 (change class), IoU, accuracy, specificity,
    F1 of the no-change class, macro F1 = mean of the two class F1s,
    2x2 confusion matrix (counts and row-normalised).
Area:
    total predicted vs ground-truth changed area (m^2), signed relative area error,
    per-image absolute area error and mean absolute percentage error over changed images.
Object level:
    connected-component instances matched by IoU >= 0.5 -> object precision / recall / F1.
Multi-class (for land-use classification):
    confusion matrix, per-class precision / recall / F1, macro F1, accuracy.
"""

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.metrics import compute_confusion_matrix, compute_metrics_from_counts
from src.product.instances import label_instances, match_instances


def binary_change_report(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    pixel_area_m2: float = 0.25,
    min_instance_px: int = 1,
    instance_iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """Full evaluation of binary change masks against ground truth (lists of [H, W] arrays)."""
    if len(preds) != len(gts) or not preds:
        raise ValueError(f"Need equal, non-empty lists of predictions and ground truths ({len(preds)} vs {len(gts)})")

    tp = fp = fn = tn = 0
    obj = {'tp': 0, 'fp': 0, 'fn': 0}
    pred_areas, gt_areas = [], []
    for pred, gt in zip(preds, gts):
        pred_b, gt_b = np.asarray(pred) > 0, np.asarray(gt) > 0
        c = compute_confusion_matrix(pred_b.astype(np.float32), gt_b.astype(np.float32), threshold=0.5)
        tp, fp, fn, tn = tp + c['tp'], fp + c['fp'], fn + c['fn'], tn + c['tn']
        pred_areas.append(pred_b.sum() * pixel_area_m2)
        gt_areas.append(gt_b.sum() * pixel_area_m2)
        pl, _ = label_instances(pred_b, min_area_px=min_instance_px)
        gl, _ = label_instances(gt_b, min_area_px=1)
        m = match_instances(pl, gl, iou_threshold=instance_iou_threshold)
        for k in obj:
            obj[k] += m[k]

    pixel = compute_metrics_from_counts(tp, fp, fn, tn)
    f1_nochange = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)  # rows: true (no-change, change)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    pred_areas, gt_areas = np.array(pred_areas), np.array(gt_areas)
    total_gt, total_pred = float(gt_areas.sum()), float(pred_areas.sum())
    changed = gt_areas > 0
    obj_p = obj['tp'] / (obj['tp'] + obj['fp']) if obj['tp'] + obj['fp'] else 0.0
    obj_r = obj['tp'] / (obj['tp'] + obj['fn']) if obj['tp'] + obj['fn'] else 0.0

    return {
        'num_images': len(preds),
        'pixel': {
            'precision': pixel['precision'], 'recall': pixel['recall'], 'change_recall': pixel['recall'],
            'f1_change': pixel['f1_score'], 'f1_no_change': float(f1_nochange),
            'macro_f1': float((pixel['f1_score'] + f1_nochange) / 2), 'iou': pixel['iou'],
            'accuracy': pixel['accuracy'], 'specificity': pixel['specificity'],
        },
        'confusion_matrix': {'labels': ['no_change', 'change'], 'counts': cm.tolist(),
                             'row_normalized': cm_norm.tolist()},
        'area': {
            'pixel_area_m2': pixel_area_m2,
            'gt_changed_m2': total_gt, 'pred_changed_m2': total_pred,
            'area_error_m2': total_pred - total_gt,
            'relative_area_error': (total_pred - total_gt) / total_gt if total_gt else None,
            'mean_abs_area_error_m2_per_image': float(np.mean(np.abs(pred_areas - gt_areas))),
            'mape_changed_images': (float(np.mean(np.abs(pred_areas[changed] - gt_areas[changed]) / gt_areas[changed]))
                                    if changed.any() else None),
        },
        'object': {**obj, 'precision': obj_p, 'recall': obj_r,
                   'f1': 2 * obj_p * obj_r / (obj_p + obj_r) if obj_p + obj_r else 0.0,
                   'iou_threshold': instance_iou_threshold, 'min_instance_px': min_instance_px},
    }


def multiclass_report(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict[str, Any]:
    """Confusion matrix, per-class P / R / F1, macro F1 and accuracy for k-class labels."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    per_class = {}
    f1s = []
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        p = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        r = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(cm[c, :].sum())}
        f1s.append(f)
    return {'accuracy': float(np.trace(cm) / max(cm.sum(), 1)), 'macro_f1': float(np.mean(f1s)),
            'per_class': per_class, 'confusion_matrix': cm.tolist(), 'class_names': list(class_names)}


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str, title: str = '') -> str:
    """Row-normalised heatmap with raw counts annotated."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(4.0, 0.9 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(norm):
        ax.text(j, i, f'{v:.2f}\n({int(cm[i, j]):,})', ha='center', va='center', fontsize=8,
                color='white' if v > 0.5 else 'black')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title or 'confusion matrix (row-normalised)')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
