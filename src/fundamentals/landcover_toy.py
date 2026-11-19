"""
Synthetic land-cover patch dataset for fast, controlled optimisation experiments.

Each sample is a small RGB patch [3, P, P] in [0, 1] drawn from one of four classes that
imitate what a very-high-resolution sensor sees:

    0 water       dark blue, smooth low-frequency ripples
    1 vegetation  green, high-frequency canopy texture
    2 built_up    grey background with bright rectangular roofs and cast shadows
    3 bare_soil   brown with a smooth illumination / moisture gradient

Every patch then gets nuisance variation that real imagery has: random orientation
(rot90 / flip), a global illumination factor, a per-channel atmospheric offset and
per-pixel sensor noise. Because orientation is random by construction, the eight
dihedral symmetries (rot90 x flip) are label-preserving augmentations.

This is a stand-in so that ablations run in seconds on a CPU without the LEVIR-CD
download; it is NOT a substitute for evaluation on real imagery.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

CLASS_NAMES: Tuple[str, ...] = ('water', 'vegetation', 'built_up', 'bare_soil')
NUM_CLASSES = len(CLASS_NAMES)


def _clean_patch(cls: int, p: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:p, 0:p] / max(p - 1, 1)
    if cls == 0:  # water
        base = np.array([0.18, 0.26, 0.32])[:, None, None]
        freq, phase = rng.uniform(0.5, 1.5), rng.uniform(0, 1)
        img = base + 0.03 * np.sin(2 * np.pi * (freq * xx + phase))[None]
    elif cls == 1:  # vegetation
        base = np.array([0.22, 0.32, 0.20])[:, None, None]
        canopy = rng.normal(0.0, 0.08, size=(1, p, p))
        img = base + canopy * np.array([0.6, 1.0, 0.6])[:, None, None]
    elif cls == 2:  # built-up
        img = np.full((3, p, p), 0.42)
        for _ in range(rng.integers(1, 4)):
            h, w = rng.integers(2, max(3, p // 2) + 1, size=2)
            r, c = rng.integers(0, p - h + 1), rng.integers(0, p - w + 1)
            img[:, r:r + h, c:c + w] = rng.uniform(0.70, 0.90)
            if r + h < p:
                img[:, r + h, c:c + w] = 0.20  # cast shadow
    elif cls == 3:  # bare soil
        base = np.array([0.46, 0.41, 0.34])[:, None, None]
        img = base + 0.15 * (xx - 0.5)[None] + rng.normal(0.0, 0.02, size=(1, p, p))
    else:
        raise ValueError(f"Unknown class {cls}")
    return img


def make_patch(cls: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """One patch of class `cls` with orientation, illumination and noise variation."""
    img = _clean_patch(cls, p, rng)
    img = np.rot90(img, k=int(rng.integers(4)), axes=(1, 2))
    if rng.random() < 0.5:
        img = img[:, :, ::-1]
    img = img * rng.uniform(0.5, 1.5)                         # illumination / sun angle
    img = img + rng.normal(0.0, 0.08, size=(3, 1, 1))         # atmospheric offset per band
    img = img + rng.normal(0.0, 0.10, size=img.shape)         # sensor noise
    return np.clip(img, 0.0, 1.0)


def make_landcover_patches(
    n: int,
    patch: int = 10,
    seed: int = 0,
    label_noise: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced dataset of n patches.

    Returns:
        X float64 [n, 3, patch, patch] in [0, 1], y int64 [n].
    label_noise flips that fraction of labels to a different random class.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % NUM_CLASSES
    rng.shuffle(labels)
    x = np.stack([make_patch(int(c), patch, rng) for c in labels])
    y = labels.astype(np.int64)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y[flip] = (y[flip] + rng.integers(1, NUM_CLASSES, size=int(flip.sum()))) % NUM_CLASSES
    return x, y


def make_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    patch: int = 10,
    seed: int = 0,
    label_noise: float = 0.0
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Independent train / val / test splits generated from disjoint RNG streams, so no
    patch can appear in more than one split. Label noise is applied to train only.
    """
    return {
        'train': make_landcover_patches(n_train, patch, seed=seed * 3 + 1, label_noise=label_noise),
        'val': make_landcover_patches(n_val, patch, seed=seed * 3 + 2),
        'test': make_landcover_patches(n_test, patch, seed=seed * 3 + 3),
    }


class ChannelStandardizer:
    """
    Per-band standardisation (z = (x - mu_c) / sigma_c) fitted on the training split only.
    Statistics are per channel, not per pixel, so they commute with rotations and flips.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> 'ChannelStandardizer':
        self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
        self.std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform")
        return (x - self.mean) / self.std


def flatten(x: np.ndarray) -> np.ndarray:
    """[N, C, H, W] -> [N, C*H*W] for an MLP."""
    return x.reshape(x.shape[0], -1)


def dihedral_augment(channels: int, patch: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Random rot90 (k in 0..3) and horizontal flip per sample, on flattened batches."""
    def augment(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        imgs = xb.reshape(-1, channels, patch, patch)
        out = np.empty_like(imgs)
        ks = rng.integers(0, 4, size=imgs.shape[0])
        flips = rng.random(imgs.shape[0]) < 0.5
        for i, (img, k, flip) in enumerate(zip(imgs, ks, flips)):
            img = np.rot90(img, k=int(k), axes=(1, 2))
            out[i] = img[:, :, ::-1] if flip else img
        return out.reshape(xb.shape)
    return augment


def gaussian_jitter(std: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Additive Gaussian noise (in the standardised feature space)."""
    def augment(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return xb + rng.normal(0.0, std, size=xb.shape)
    return augment


def compose(*fns: Callable[[np.ndarray, np.random.Generator], np.ndarray]):
    def augment(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for fn in fns:
            xb = fn(xb, rng)
        return xb
    return augment
