"""
Restricted-zone encroachment assessment.

Responsible-use design:
- The detector finds *visual change*. Whether a change is unlawful is a legal question the
  model cannot answer.
- Alerts are raised only for zones whose boundary is marked `"verified": true` and carries a
  `source` (e.g. a cadastral or forest-department record). Unverified zones are reported for
  information but never generate alerts.
- With no boundaries supplied, no encroachment statement is made at all.
- Every alert is advisory and intended to prioritise a field survey.

Zones are read from GeoJSON (Polygon / MultiPolygon, holes supported). Coordinates are either
pixel coordinates (x = column, y = row) or map coordinates converted with a north-up affine
GeoTransform (x0, pixel_width, y0, pixel_height), the same convention as GDAL.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path

from src.product.instances import ChangeInstance

DISCLAIMER = (
    "ADVISORY ONLY. Alerts indicate detected visual change overlapping a supplied restricted-area "
    "boundary. They are not a determination of legal violation. Model errors, boundary inaccuracies, "
    "image misregistration and permitted works can all produce alerts. Verify every alert by field "
    "survey and official records before any action."
)


@dataclass
class GeoTransform:
    """North-up affine transform: col = (x - x0) / pixel_width, row = (y - y0) / pixel_height."""
    x0: float
    pixel_width: float
    y0: float
    pixel_height: float  # negative for north-up rasters

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.c_[(xy[:, 0] - self.x0) / self.pixel_width, (xy[:, 1] - self.y0) / self.pixel_height]


@dataclass
class RestrictedZone:
    name: str
    rings: List[List[np.ndarray]]        # list of polygons; each = [exterior, hole, hole, ...] in pixel (x, y)
    verified: bool = False
    source: str = ''
    category: str = 'restricted'

    def summary(self) -> Dict[str, Any]:
        return {'name': self.name, 'verified': self.verified, 'source': self.source, 'category': self.category}


def rasterize_zone(zone: RestrictedZone, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of pixels whose centre lies inside the zone (even-odd rule, holes removed)."""
    h, w = shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    centres = np.c_[cols.ravel(), rows.ravel()]
    mask = np.zeros(h * w, dtype=bool)
    for polygon in zone.rings:
        if not polygon:
            continue
        inside = Path(polygon[0]).contains_points(centres)
        for hole in polygon[1:]:
            inside &= ~Path(hole).contains_points(centres)
        mask |= inside
    return mask.reshape(h, w)


def zones_from_geojson(data: Dict[str, Any], transform: Optional[GeoTransform] = None) -> List[RestrictedZone]:
    """Parse a GeoJSON FeatureCollection (or single Feature) into RestrictedZone objects."""
    features = data['features'] if data.get('type') == 'FeatureCollection' else [data]
    zones = []
    for i, feat in enumerate(features):
        geom = feat.get('geometry') or {}
        props = feat.get('properties') or {}
        if geom.get('type') == 'Polygon':
            polygons = [geom['coordinates']]
        elif geom.get('type') == 'MultiPolygon':
            polygons = geom['coordinates']
        else:
            raise ValueError(f"Feature {i}: unsupported geometry type {geom.get('type')!r}")
        rings = []
        for poly in polygons:
            converted = []
            for ring in poly:
                pts = np.asarray(ring, dtype=np.float64)[:, :2]
                converted.append(transform.to_pixel(pts) if transform else pts)
            rings.append(converted)
        zones.append(RestrictedZone(
            name=str(props.get('name', f'zone_{i + 1}')),
            rings=rings,
            verified=bool(props.get('verified', False)) and bool(props.get('source')),
            source=str(props.get('source', '')),
            category=str(props.get('category', 'restricted')),
        ))
    return zones


def load_zones_geojson(path: str, transform: Optional[GeoTransform] = None) -> List[RestrictedZone]:
    with open(path, encoding='utf-8') as f:
        return zones_from_geojson(json.load(f), transform)


@dataclass
class EncroachmentAlert:
    zone: str
    zone_category: str
    zone_source: str
    instance_id: int
    overlap_px: int
    overlap_m2: float
    fraction_of_change_inside_zone: float
    instance_area_m2: float
    bbox: Tuple[int, int, int, int]
    status: str = 'advisory - field verification required'


def assess_encroachment(
    labels: np.ndarray,
    instances: Sequence[ChangeInstance],
    zones: Sequence[RestrictedZone],
    pixel_area_m2: float,
    min_overlap_px: int = 1,
    min_overlap_fraction: float = 0.0
) -> Dict[str, Any]:
    """
    Intersect change instances with restricted zones.

    Args:
        labels: instance label image from instances.label_instances.
        instances: the matching ChangeInstance list.
        zones: restricted zones (already in pixel coordinates).
        pixel_area_m2: ground area per pixel.
        min_overlap_px / min_overlap_fraction: suppress alerts for tiny boundary touches
            (boundary data and masks both have edge uncertainty).
    """
    if not zones:
        return {'status': 'no_boundaries_supplied', 'alerts': [], 'zones': [],
                'message': 'No restricted-area boundaries were supplied; no encroachment assessment was made.',
                'disclaimer': DISCLAIMER}

    by_id = {inst.id: inst for inst in instances}
    n = int(labels.max())
    alerts: List[EncroachmentAlert] = []
    zone_reports = []
    for zone in zones:
        zmask = rasterize_zone(zone, labels.shape)
        overlap = np.bincount(labels[zmask].ravel(), minlength=n + 1)
        touched = [i for i in range(1, n + 1) if overlap[i] > 0]
        report = {**zone.summary(), 'zone_area_m2': float(zmask.sum() * pixel_area_m2),
                  'changed_area_inside_m2': float(overlap[1:].sum() * pixel_area_m2),
                  'instances_touching': len(touched)}
        if not zone.verified:
            report['note'] = 'Boundary not verified (needs "verified": true and a "source"); no alerts raised.'
        else:
            for i in touched:
                inst = by_id[i]
                frac = overlap[i] / inst.area_px
                if overlap[i] >= min_overlap_px and frac >= min_overlap_fraction:
                    alerts.append(EncroachmentAlert(
                        zone=zone.name, zone_category=zone.category, zone_source=zone.source,
                        instance_id=i, overlap_px=int(overlap[i]),
                        overlap_m2=float(overlap[i] * pixel_area_m2),
                        fraction_of_change_inside_zone=float(frac),
                        instance_area_m2=inst.area_m2, bbox=inst.bbox))
        zone_reports.append(report)

    verified = [z for z in zones if z.verified]
    status = 'alerts_raised' if alerts else ('no_change_in_verified_zones' if verified else 'no_verified_boundaries')
    return {'status': status, 'alerts': [asdict(a) for a in alerts], 'zones': zone_reports,
            'disclaimer': DISCLAIMER}
