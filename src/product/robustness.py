"""
Robustness suite: how much does change detection degrade under realistic image defects?

Every perturbation targets a failure mode of operational bitemporal imagery:

    gaussian_noise   sensor noise on both dates                (std, images in [0, 1])
    brightness       T2 illumination / sun-angle difference    (multiplier)
    contrast         T2 radiometric difference                 (contrast factor around the mean)
    blur             T2 defocus / resampling blur              (Gaussian sigma, pixels)
    jpeg             T2 compression artefacts                  (JPEG quality)
    misregistration  T2 shifted relative to T1                 (pixels, diagonal shift)
    haze             T2 atmospheric haze                       (blend weight toward bright grey)

Only T2 is altered for the date-specific defects, because a *difference* between the dates
is what can create false change. Ground truth is never altered.
"""

import io
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from src.product.evaluation import binary_change_report

DEFAULT_SUITE: Dict[str, List[float]] = {
    'gaussian_noise': [0.02, 0.05, 0.10],
    'brightness': [0.7, 1.3],
    'contrast': [0.6, 1.4],
    'blur': [1.0, 2.0],
    'jpeg': [50, 20],
    'misregistration': [1, 2, 4],
    'haze': [0.2, 0.4],
}


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB'), dtype=np.float64) / 255.0


def apply_perturbation(
    t1: np.ndarray,
    t2: np.ndarray,
    name: str,
    level: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Return perturbed copies of (t1, t2), both float [H, W, 3] in [0, 1]."""
    t1, t2 = np.asarray(t1, dtype=np.float64).copy(), np.asarray(t2, dtype=np.float64).copy()
    if name == 'gaussian_noise':
        t1 += rng.normal(0, level, t1.shape)
        t2 += rng.normal(0, level, t2.shape)
    elif name == 'brightness':
        t2 *= level
    elif name == 'contrast':
        mean = t2.mean(axis=(0, 1), keepdims=True)
        t2 = (t2 - mean) * level + mean
    elif name == 'blur':
        t2 = ndimage.gaussian_filter(t2, sigma=(level, level, 0))
    elif name == 'jpeg':
        t2 = _jpeg(t2, int(level))
    elif name == 'misregistration':
        s = int(level)
        t2 = ndimage.shift(t2, shift=(s, s, 0), order=1, mode='nearest')
    elif name == 'haze':
        t2 = (1 - level) * t2 + level * 0.8
    else:
        raise ValueError(f"Unknown perturbation '{name}'")
    return np.clip(t1, 0, 1), np.clip(t2, 0, 1)


def run_robustness(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pixel_area_m2: float = 0.25,
    suite: Dict[str, List[float]] = None,
    seed: int = 0,
    min_instance_px: int = 1
) -> Dict[str, Any]:
    """
    Evaluate predict_fn(t1, t2) -> binary mask on clean and perturbed versions of every
    (t1, t2, gt) sample. Returns a row per condition with pixel / area / object metrics and
    the F1 change relative to clean data.
    """
    suite = DEFAULT_SUITE if suite is None else suite
    rng = np.random.default_rng(seed)
    conditions = [('clean', 0.0)] + [(n, lv) for n, levels in suite.items() for lv in levels]

    rows = []
    for name, level in conditions:
        preds, start = [], time.perf_counter()
        for t1, t2, _ in samples:
            a, b = (t1, t2) if name == 'clean' else apply_perturbation(t1, t2, name, level, rng)
            preds.append(np.asarray(predict_fn(a, b)) > 0)
        elapsed = time.perf_counter() - start
        rep = binary_change_report(preds, [gt for _, _, gt in samples], pixel_area_m2=pixel_area_m2,
                                   min_instance_px=min_instance_px)
        rows.append({
            'perturbation': name, 'level': level,
            'f1': rep['pixel']['f1_change'], 'iou': rep['pixel']['iou'],
            'precision': rep['pixel']['precision'], 'change_recall': rep['pixel']['recall'],
            'macro_f1': rep['pixel']['macro_f1'],
            'relative_area_error': rep['area']['relative_area_error'],
            'object_f1': rep['object']['f1'],
            'ms_per_pair': 1000 * elapsed / len(samples),
        })
    clean = rows[0]
    for r in rows:
        r['delta_f1'] = r['f1'] - clean['f1']
    worst = min(rows[1:], key=lambda r: r['delta_f1']) if len(rows) > 1 else clean
    return {'rows': rows, 'clean_f1': clean['f1'],
            'worst_condition': {'perturbation': worst['perturbation'], 'level': worst['level'],
                                'delta_f1': worst['delta_f1']}}


def plot_robustness(result: Dict[str, Any], path: str, title: str = '') -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = result['rows']
    labels = [f"{r['perturbation']}\n{r['level']:g}" if r['perturbation'] != 'clean' else 'clean' for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 0.75 * len(rows) * 2), 4.6))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, [r['f1'] for r in rows], 0.4, label='pixel F1')
    axes[0].bar(x + 0.2, [r['change_recall'] for r in rows], 0.4, label='change recall')
    axes[0].axhline(result['clean_f1'], color='k', ls='--', lw=1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].set_title('accuracy under perturbation', fontsize=10)
    area = [r['relative_area_error'] if r['relative_area_error'] is not None else 0.0 for r in rows]
    axes[1].bar(x, area, color=['tab:red' if a > 0 else 'tab:blue' for a in area])
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_yscale('symlog', linthresh=0.1)
    axes[1].set_title('relative area error (+ over, - under; symlog: x1 = 100% overestimate)', fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    fig.suptitle(title or 'robustness suite', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
