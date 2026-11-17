"""
Traditional Pixel-Difference Baseline for Satellite Image Change Detection.

Implements an unsupervised change detection baseline using direct pixel-level
comparisons across bitemporal optical satellite imagery (T1 and T2).

Features:
- Multi-channel aggregation methods: Euclidean color distance, Mean Absolute
  Difference (MAD), and Grayscale intensity difference.
- Configurable difference thresholding.
- Batch evaluation and threshold tuning utilities with global metric tracking.
"""

import time
from typing import Union, Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.metrics import MetricTracker


def to_numpy_image(
    image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Standardize various image inputs into a float32 RGB numpy array in [0.0, 1.0].

    Args:
        image_input: File path, PIL Image, numpy array, or PyTorch tensor.
        target_size: Optional (H, W) tuple to resize image.

    Returns:
        np.ndarray of shape [H, W, 3] with values in [0.0, 1.0], dtype float32.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    if isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    if isinstance(image_input, torch.Tensor):
        # Expected tensor shape: [3, H, W] or [1, 3, H, W]
        t = image_input.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            arr = t.permute(1, 2, 0).numpy().astype(np.float32)
        else:
            arr = t.numpy().astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    if isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        if arr.ndim == 2:  # Grayscale to RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[0] == 3 and arr.ndim == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def compute_difference_map(
    img1: Union[str, Image.Image, np.ndarray, torch.Tensor],
    img2: Union[str, Image.Image, np.ndarray, torch.Tensor],
    method: str = 'euclidean'
) -> np.ndarray:
    """
    Compute a normalized single-channel difference map from two registered images.

    Methods:
    - 'euclidean': Normalized Euclidean distance in RGB color space:
                   D = (1 / sqrt(3)) * sqrt((R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2)
                   Values are bounded strictly in [0.0, 1.0].
    - 'mad': Mean Absolute Difference across channels:
             D = (1 / 3) * (|R1-R2| + |G1-G2| + |B1-B2|)
    - 'grayscale': Absolute difference of luminance (ITU-R BT.601 weights):
                   Y = 0.2989*R + 0.5870*G + 0.1140*B
                   D = |Y1 - Y2|

    Args:
        img1: Pre-change image (T1).
        img2: Post-change image (T2).
        method: Aggregation strategy ('euclidean', 'mad', or 'grayscale').

    Returns:
        np.ndarray of shape [H, W] with difference intensities in [0.0, 1.0].
    """
    arr1 = to_numpy_image(img1)
    arr2 = to_numpy_image(img2)

    if arr1.shape != arr2.shape:
        raise ValueError(f"Image shape mismatch: img1 has shape {arr1.shape}, img2 has shape {arr2.shape}")

    method_lower = method.lower()
    if method_lower == 'euclidean':
        diff_sq = np.sum((arr1 - arr2) ** 2, axis=-1)  # shape [H, W]
        diff_map = np.sqrt(diff_sq) / np.sqrt(3.0)     # Normalization factor sqrt(3)
    elif method_lower == 'mad':
        diff_map = np.mean(np.abs(arr1 - arr2), axis=-1)
    elif method_lower == 'grayscale':
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray1 = np.tensordot(arr1, weights, axes=([2], [0]))
        gray2 = np.tensordot(arr2, weights, axes=([2], [0]))
        diff_map = np.abs(gray1 - gray2)
    else:
        raise ValueError(f"Unknown difference method '{method}'. Supported: 'euclidean', 'mad', 'grayscale'.")

    return np.clip(diff_map, 0.0, 1.0).astype(np.float32)


def threshold_difference_map(
    diff_map: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Apply a binary decision threshold to a continuous difference map.

    Args:
        diff_map: 2D array [H, W] of difference magnitudes in [0.0, 1.0].
        threshold: Cutoff value in [0.0, 1.0].

    Returns:
        Binary mask of shape [H, W] with values in {0.0, 1.0}, dtype float32.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")

    binary_mask = (diff_map >= threshold).astype(np.float32)
    return binary_mask


def detect_change(
    img1: Union[str, Image.Image, np.ndarray, torch.Tensor],
    img2: Union[str, Image.Image, np.ndarray, torch.Tensor],
    threshold: float = 0.20,
    method: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end baseline change detection function.

    Args:
        img1: Pre-change image (T1).
        img2: Post-change image (T2).
        threshold: Decision threshold for change classification.
        method: Difference aggregation method.

    Returns:
        Tuple of (diff_map, predicted_binary_mask) both of shape [H, W].
    """
    diff_map = compute_difference_map(img1, img2, method=method)
    pred_mask = threshold_difference_map(diff_map, threshold=threshold)
    return diff_map, pred_mask


def evaluate_baseline(
    data_source: Union[Dataset, DataLoader],
    threshold: float,
    method: str = 'euclidean'
) -> Dict[str, Any]:
    """
    Evaluate the traditional baseline across an entire dataset or dataloader.

    Computes global cumulative metrics (Precision, Recall, F1, IoU) and tracks
    execution runtime and latency per sample.

    Args:
        data_source: PyTorch Dataset or DataLoader yielding (t1, t2, mask).
        threshold: Decision threshold.
        method: Difference aggregation method.

    Returns:
        Dict with metrics and execution performance stats.
    """
    tracker = MetricTracker()
    start_time = time.perf_counter()
    num_samples = 0

    if isinstance(data_source, DataLoader):
        for batch in data_source:
            b_t1, b_t2, b_mask = batch
            # Process sample by sample within batch
            batch_size = b_t1.shape[0]
            for i in range(batch_size):
                t1 = b_t1[i]
                t2 = b_t2[i]
                mask = b_mask[i].squeeze().numpy()  # shape [H, W]
                _, pred_mask = detect_change(t1, t2, threshold=threshold, method=method)
                tracker.update(pred_mask, mask, threshold=0.5)
                num_samples += 1
    elif isinstance(data_source, Dataset):
        for i in range(len(data_source)):
            t1, t2, mask = data_source[i]
            mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            _, pred_mask = detect_change(t1, t2, threshold=threshold, method=method)
            tracker.update(pred_mask, mask_np, threshold=0.5)
            num_samples += 1
    else:
        raise TypeError(f"data_source must be Dataset or DataLoader, got {type(data_source)}")

    total_time = time.perf_counter() - start_time
    metrics = tracker.compute()
    metrics['threshold'] = threshold
    metrics['method'] = method
    metrics['num_samples'] = num_samples
    metrics['total_runtime_seconds'] = round(total_time, 4)
    metrics['avg_latency_ms'] = round((total_time / max(num_samples, 1)) * 1000, 2)
    metrics['fps'] = round(num_samples / max(total_time, 1e-6), 2)

    return metrics


def tune_threshold(
    data_source: Union[Dataset, DataLoader],
    candidate_thresholds: List[float],
    method: str = 'euclidean'
) -> Dict[str, Any]:
    """
    Evaluate multiple candidate thresholds on a validation set and identify
    the optimal threshold by F1-score.

    To optimize runtime, the difference map for each sample is computed once,
    and multiple thresholds are evaluated across the cached difference maps.

    Args:
        data_source: Validation Dataset or DataLoader.
        candidate_thresholds: List of thresholds to compare (e.g. [0.10, 0.15, ...]).
        method: Difference calculation method.

    Returns:
        Dict containing:
        - 'results': List of metric dicts for each threshold.
        - 'best_threshold': Threshold achieving highest validation F1-score.
        - 'best_f1': Best validation F1-score.
        - 'best_metrics': Complete metrics dictionary for the best threshold.
    """
    sorted_thresholds = sorted(candidate_thresholds)
    trackers = {thresh: MetricTracker() for thresh in sorted_thresholds}

    start_time = time.perf_counter()
    num_samples = 0

    # Collect samples
    samples = []
    if isinstance(data_source, DataLoader):
        for batch in data_source:
            b_t1, b_t2, b_mask = batch
            for i in range(b_t1.shape[0]):
                samples.append((b_t1[i], b_t2[i], b_mask[i].squeeze().numpy()))
    else:
        for i in range(len(data_source)):
            t1, t2, mask = data_source[i]
            mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            samples.append((t1, t2, mask_np))

    num_samples = len(samples)

    # Compute difference maps once and update all threshold trackers
    for t1, t2, mask in samples:
        diff_map = compute_difference_map(t1, t2, method=method)
        for thresh in sorted_thresholds:
            pred_mask = (diff_map >= thresh).astype(np.float32)
            trackers[thresh].update(pred_mask, mask, threshold=0.5)

    total_time = time.perf_counter() - start_time

    results = []
    best_thresh = None
    best_f1 = -1.0
    best_metrics = None

    for thresh in sorted_thresholds:
        m = trackers[thresh].compute()
        m['threshold'] = thresh
        m['method'] = method
        results.append(m)
        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_thresh = thresh
            best_metrics = m

    return {
        'results': results,
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'best_metrics': best_metrics,
        'num_samples': num_samples,
        'total_tuning_time_seconds': round(total_time, 4)
    }
