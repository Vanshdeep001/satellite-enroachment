"""
Synthetic bitemporal scenes with exact change masks.

Purpose: exercise and test the product pipeline (instances, encroachment, evaluation,
robustness, app) before the LEVIR-CD imagery is available. Not for reporting model accuracy.

T1: fields (smooth colour texture), a road, existing buildings with shadows.
T2: the same scene plus new buildings (the labelled change), with unlabelled nuisance change
    (a global illumination shift, a seasonal shift in vegetation colour, sensor noise), so that
    naive pixel differencing also produces false alarms, as on real imagery.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage


def _place_building(img: np.ndarray, rng: np.random.Generator, occupied: np.ndarray,
                    min_size: int, max_size: int) -> Tuple[int, int, int, int]:
    h, w = occupied.shape
    for _ in range(200):
        bh, bw = rng.integers(min_size, max_size + 1, size=2)
        r, c = rng.integers(2, h - bh - 3), rng.integers(2, w - bw - 3)
        if not occupied[r - 2:r + bh + 3, c - 2:c + bw + 3].any():
            roof = rng.choice([np.array([0.75, 0.72, 0.68]), np.array([0.62, 0.30, 0.25]),
                               np.array([0.55, 0.58, 0.62])])
            img[r:r + bh, c:c + bw] = roof + rng.normal(0, 0.02, 3)
            img[r + bh:r + bh + 2, c + 1:c + bw + 1] *= 0.45  # cast shadow
            occupied[r:r + bh + 2, c:c + bw + 1] = True
            return int(r), int(c), int(r + bh), int(c + bw)
    raise RuntimeError("Could not place building; scene too crowded")


def make_scene_pair(
    size: int = 256,
    seed: int = 0,
    n_existing: int = 6,
    n_new: int = 4,
    min_building: int = 8,
    max_building: int = 22,
    illumination_shift: float = 0.12,
    seasonal_shift: float = 0.06,
    noise: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
        t1, t2: float [size, size, 3] in [0, 1]
        change: bool [size, size], True where a new building appeared
        info: dict with the rectangles of existing and new buildings (row0, col0, row1, col1)
    """
    rng = np.random.default_rng(seed)
    field_noise = ndimage.gaussian_filter(rng.normal(size=(size, size)), sigma=size / 16)
    field_noise = (field_noise - field_noise.min()) / (np.ptp(field_noise) + 1e-12)
    veg = np.array([0.30, 0.45, 0.22])
    soil = np.array([0.55, 0.47, 0.35])
    base = field_noise[..., None] * veg + (1 - field_noise[..., None]) * soil
    base += ndimage.gaussian_filter(rng.normal(0, 0.03, (size, size, 3)), sigma=(1, 1, 0))
    veg_weight = field_noise.copy()

    occupied = np.zeros((size, size), dtype=bool)
    road_row = int(rng.integers(size // 4, 3 * size // 4))
    base[road_row - 3:road_row + 3, :] = 0.42
    occupied[road_row - 5:road_row + 6, :] = True

    t1 = base.copy()
    existing: List[Tuple[int, int, int, int]] = [
        _place_building(t1, rng, occupied, min_building, max_building) for _ in range(n_existing)]

    t2 = t1.copy()
    change = np.zeros((size, size), dtype=bool)
    new: List[Tuple[int, int, int, int]] = []
    for _ in range(n_new):
        r0, c0, r1, c1 = _place_building(t2, rng, occupied, min_building, max_building)
        change[r0:r1, c0:c1] = True
        new.append((r0, c0, r1, c1))

    # Unlabelled nuisance change on T2: illumination, seasonal vegetation shift, noise.
    t2 = t2 * (1.0 + rng.uniform(-illumination_shift, illumination_shift))
    t2[..., 1] -= seasonal_shift * veg_weight * (~occupied)
    t2[..., 0] += 0.5 * seasonal_shift * veg_weight * (~occupied)
    t1 = t1 + rng.normal(0, noise, t1.shape)
    t2 = t2 + rng.normal(0, noise, t2.shape)
    return np.clip(t1, 0, 1), np.clip(t2, 0, 1), change, {'existing': existing, 'new': new, 'road_row': road_row}


def make_dataset(n: int, size: int = 256, seed: int = 0, **kwargs) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [make_scene_pair(size=size, seed=seed * 10_000 + i, **kwargs)[:3] for i in range(n)]


def demo_zones_geojson(info: Dict[str, Any], size: int) -> Dict[str, Any]:
    """
    Two demo restricted zones in pixel coordinates around the first new building:
    a verified 'reserve buffer' (raises alerts) and an unverified 'proposed easement'
    (reported, never alerts). Purely illustrative boundaries.
    """
    r0, c0, r1, c1 = info['new'][0]
    pad = 6
    verified = [[c0 - pad, r0 - pad], [c1 + pad, r0 - pad], [c1 + pad, (r0 + r1) // 2],
                [c0 - pad, (r0 + r1) // 2], [c0 - pad, r0 - pad]]
    rr = info['road_row']
    unverified = [[0, rr - 12], [size, rr - 12], [size, rr + 12], [0, rr + 12], [0, rr - 12]]
    return {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'name': 'Demo reserve buffer', 'verified': True,
                                           'source': 'SYNTHETIC DEMO BOUNDARY (not a real record)',
                                           'category': 'protected_area'},
         'geometry': {'type': 'Polygon', 'coordinates': [verified]}},
        {'type': 'Feature', 'properties': {'name': 'Proposed road easement', 'verified': False,
                                           'category': 'easement'},
         'geometry': {'type': 'Polygon', 'coordinates': [unverified]}},
    ]}
