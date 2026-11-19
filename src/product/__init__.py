"""
Product layer of the Satellite Land Use Change & Encroachment Detector.

Modules:
- instances: change mask -> individual change instances (area, bbox, centroid) and
  object-level matching against ground truth.
- encroachment: restricted-zone polygons (GeoJSON) -> raster masks -> advisory alerts.
  Only boundaries marked as verified can raise alerts.
- evaluation: pixel metrics, macro F1, change recall, area error, object-level F1,
  confusion matrices (binary and multi-class).
- robustness: perturbation suite (noise, illumination, blur, JPEG, misregistration, haze).
- synthetic_scenes: bitemporal scenes with known change masks for testing and demos.
- app: command-line application (detect / evaluate / robustness / demo).
"""
