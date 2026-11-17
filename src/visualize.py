"""
Visualization Utilities for Remote Sensing Change Detection.

Generates comparative 5-panel figures:
- Pre-change Image (T1)
- Post-change Image (T2)
- Ground-Truth Mask
- Predicted Binary Change Mask
- Continuous Difference Heatmap
"""

import os
from typing import Union, Optional, Tuple, List
import numpy as np
import matplotlib
# Use a non-interactive backend for headless CLI environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.baseline import to_numpy_image, compute_difference_map, detect_change


def plot_change_comparison(
    img_t1: Union[str, Image.Image, np.ndarray, torch.Tensor],
    img_t2: Union[str, Image.Image, np.ndarray, torch.Tensor],
    ground_truth: Union[str, Image.Image, np.ndarray, torch.Tensor],
    pred_mask: Optional[np.ndarray] = None,
    diff_map: Optional[np.ndarray] = None,
    threshold: float = 0.20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 4)
) -> plt.Figure:
    """
    Generate a 5-panel comparison figure.

    Panels:
    1. Pre-Change Image (T1)
    2. Post-Change Image (T2)
    3. Ground-Truth Mask
    4. Baseline Predicted Mask
    5. Continuous Difference Heatmap
    """
    # Standardize inputs
    arr_t1 = to_numpy_image(img_t1)
    arr_t2 = to_numpy_image(img_t2)

    # Convert ground-truth mask to 2D binary float array in {0, 1}
    if isinstance(ground_truth, str):
        gt_img = Image.open(ground_truth).convert('L')
        gt_arr = (np.array(gt_img) > 127).astype(np.float32)
    elif isinstance(ground_truth, Image.Image):
        gt_arr = (np.array(ground_truth.convert('L')) > 127).astype(np.float32)
    elif isinstance(ground_truth, torch.Tensor):
        gt_arr = ground_truth.detach().cpu().squeeze().numpy()
        gt_arr = (gt_arr > 0.5).astype(np.float32)
    elif isinstance(ground_truth, np.ndarray):
        gt_arr = ground_truth.squeeze()
        gt_arr = (gt_arr > 0.5).astype(np.float32)
    else:
        raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)}")

    # Compute difference and prediction if not provided
    if diff_map is None:
        diff_map = compute_difference_map(arr_t1, arr_t2, method='euclidean')
    if pred_mask is None:
        pred_mask = (diff_map >= threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 5, figsize=figsize, dpi=150)

    # 1. Image T1
    axes[0].imshow(arr_t1)
    axes[0].set_title("Pre-Change (T1)", fontsize=11, fontweight='bold', pad=8)
    axes[0].axis('off')

    # 2. Image T2
    axes[1].imshow(arr_t2)
    axes[1].set_title("Post-Change (T2)", fontsize=11, fontweight='bold', pad=8)
    axes[1].axis('off')

    # 3. Ground Truth
    axes[2].imshow(gt_arr, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth Mask", fontsize=11, fontweight='bold', pad=8)
    axes[2].axis('off')

    # 4. Predicted Binary Mask
    axes[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f"Predicted Mask (τ={threshold:.2f})", fontsize=11, fontweight='bold', pad=8)
    axes[3].axis('off')

    # 5. Difference Heatmap
    im5 = axes[4].imshow(diff_map, cmap='magma', vmin=0.0, vmax=1.0)
    axes[4].set_title("Difference Heatmap", fontsize=11, fontweight='bold', pad=8)
    axes[4].axis('off')
    cbar = fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_threshold_tuning_curve(
    tuning_results: List[dict],
    selected_threshold: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision, Recall, F1-Score, and IoU versus Candidate Thresholds.
    """
    thresholds = [r['threshold'] for r in tuning_results]
    precisions = [r['precision'] for r in tuning_results]
    recalls = [r['recall'] for r in tuning_results]
    f1s = [r['f1_score'] for r in tuning_results]
    ious = [r['iou'] for r in tuning_results]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2, markersize=5)
    ax.plot(thresholds, recalls, 'g-s', label='Recall', linewidth=2, markersize=5)
    ax.plot(thresholds, f1s, 'r-^', label='F1-Score', linewidth=2.5, markersize=6)
    ax.plot(thresholds, ious, 'm-d', label='IoU', linewidth=2, markersize=5)

    # Highlight selected threshold
    ax.axvline(
        x=selected_threshold,
        color='black',
        linestyle='--',
        alpha=0.7,
        label=f'Selected τ* = {selected_threshold:.2f}'
    )

    ax.set_title("Validation Metric Curves vs. Difference Threshold (τ)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Difference Threshold (τ)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Metric Score", fontsize=11, fontweight='semibold')
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center right', fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    return fig

