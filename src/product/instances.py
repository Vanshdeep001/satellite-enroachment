"""
Instance-level analysis of a binary change mask.

A pixel mask says *where* change happened. Planning and audit users need *what* changed:
how many separate structures appeared, how large each is, and where. Connected-component
labelling turns the mask into individual change instances, and greedy IoU matching against
ground-truth instances gives object-level precision / recall / F1.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class ChangeInstance:
    id: int
    area_px: int
    area_m2: float
    bbox: Tuple[int, int, int, int]      # (row_min, col_min, row_max_exclusive, col_max_exclusive)
    centroid: Tuple[float, float]        # (row, col)
    mean_probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def label_instances(
    mask: np.ndarray,
    pixel_area_m2: float = 1.0,
    min_area_px: int = 1,
    connectivity: int = 8,
    prob_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[ChangeInstance]]:
    """
    Split a binary mask into connected change instances.

    Args:
        mask: [H, W] array; non-zero = change.
        pixel_area_m2: ground area of one pixel (e.g. 0.25 for 0.5 m GSD).
        min_area_px: components smaller than this are discarded as noise.
        connectivity: 4 or 8 neighbourhood.
        prob_map: optional [H, W] probabilities to report a mean confidence per instance.

    Returns:
        labels [H, W] int32 (0 = background, 1..N = instances) and the instance list.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")
    structure = np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    raw, n = ndimage.label(binary, structure=structure)

    sizes = np.bincount(raw.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes >= max(min_area_px, 1))
    keep = keep[keep != 0]
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    labels = remap[raw]

    instances: List[ChangeInstance] = []
    if keep.size:
        slices = ndimage.find_objects(labels)
        centroids = ndimage.center_of_mass(binary, labels, index=np.arange(1, keep.size + 1))
        means = (ndimage.mean(prob_map, labels, index=np.arange(1, keep.size + 1))
                 if prob_map is not None else [None] * keep.size)
        new_sizes = np.bincount(labels.ravel(), minlength=keep.size + 1)
        for i, sl in enumerate(slices, start=1):
            instances.append(ChangeInstance(
                id=i,
                area_px=int(new_sizes[i]),
                area_m2=float(new_sizes[i] * pixel_area_m2),
                bbox=(sl[0].start, sl[1].start, sl[0].stop, sl[1].stop),
                centroid=(float(centroids[i - 1][0]), float(centroids[i - 1][1])),
                mean_probability=None if means[i - 1] is None else float(means[i - 1]),
            ))
    return labels, instances


def match_instances(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Greedy one-to-one matching of predicted and ground-truth instances by IoU.

    Returns object-level TP / FP / FN, precision, recall, F1 and the matched pairs with
    their IoU and area error (pred - gt, in pixels).
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(f"Shape mismatch {pred_labels.shape} vs {gt_labels.shape}")
    n_pred, n_gt = int(pred_labels.max()), int(gt_labels.max())
    if n_pred == 0 or n_gt == 0:
        tp = 0
        pairs: List[Dict[str, Any]] = []
    else:
        # Joint histogram of (pred id, gt id) gives every pairwise intersection in one pass.
        joint = np.bincount(pred_labels.ravel().astype(np.int64) * (n_gt + 1) + gt_labels.ravel(),
                            minlength=(n_pred + 1) * (n_gt + 1)).reshape(n_pred + 1, n_gt + 1)
        inter = joint[1:, 1:].astype(np.float64)
        pred_area = joint[1:, :].sum(axis=1).astype(np.float64)
        gt_area = joint[:, 1:].sum(axis=0).astype(np.float64)
        iou = inter / (pred_area[:, None] + gt_area[None, :] - inter)

        pairs = []
        used_p, used_g = set(), set()
        for flat in np.argsort(iou, axis=None)[::-1]:
            p, g = divmod(int(flat), n_gt)
            if iou[p, g] < iou_threshold:
                break
            if p in used_p or g in used_g:
                continue
            used_p.add(p)
            used_g.add(g)
            pairs.append({'pred_id': p + 1, 'gt_id': g + 1, 'iou': float(iou[p, g]),
                          'area_error_px': int(pred_area[p] - gt_area[g])})
        tp = len(pairs)

    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'n_pred': n_pred, 'n_gt': n_gt,
            'precision': precision, 'recall': recall, 'f1': f1, 'pairs': pairs,
            'iou_threshold': iou_threshold}
