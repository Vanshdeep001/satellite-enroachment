# Product Pipeline: Instances, Encroachment, Evaluation, Robustness, CLI (Steps 7 and 10)

`src/product/` turns a change mask into the outputs a planning or audit user needs, and
provides the evaluation and robustness tooling the brief asks for. Everything runs without the
dataset. When a trained checkpoint exists, the same commands use the model instead of the
pixel-difference baseline.

## Commands

```bash
python -m src.product.app demo                                   # synthetic scene, demo zones, full report
python -m src.product.app detect --t1 T1.png --t2 T2.png --gsd 0.5 \
       --zones zones.geojson [--geotransform X0 PX_W Y0 PX_H]    # one image pair
python -m src.product.app evaluate --data-root data --split test  # LEVIR-CD evaluation package
python -m src.product.app robustness --data-root data --split test   # or: --synthetic 12
```

`--method auto|model|baseline` selects the backend. `auto` uses
`outputs/checkpoints/final_selected_model.pt` if present. Any checkpoint can be passed with
`--checkpoint`, including the new Siamese ResNet (`src/train_resnet_cd.py`). After
`pip install -e .`, the same CLI is available as `satcd`.

## Pipeline

```
T1, T2 ──► detector (Siamese U-Net / Siamese ResNet / pixel-difference baseline)
       ──► change score map ──► threshold ──► binary mask
       ──► connected components (8-neighbour, min area in m²) ──► change instances
             (id, area m², bbox, centroid, mean confidence)
       ──► restricted zones (GeoJSON → pixel polygons → raster, holes supported)
       ──► overlap per instance × zone ──► advisory alerts (verified zones only)
       ──► report.json + report.png + change_mask.png
```

## Responsible-use rules (enforced in code, `src/product/encroachment.py`)

| Situation | Behaviour |
|---|---|
| No boundaries supplied | Status `no_boundaries_supplied`; no encroachment statement at all |
| Zone without `"verified": true` **and** a `"source"` | Overlap is reported for information; **no alerts** |
| Verified zone | Alert per change instance overlapping it, with overlap m² and % of that change inside |
| Small edge touches | Suppressed by `--min-overlap` (default 10% of the change instance) |
| Every report and figure | Carries the advisory disclaimer: a change is not a legal finding; field verification is required |

## Evaluation package (`src/product/evaluation.py`)

| Group | Metrics |
|---|---|
| Pixel (global accumulation) | Precision, recall (= change recall), F1 (change), F1 (no change), **macro F1**, IoU, accuracy, specificity |
| Confusion matrix | 2×2 counts and row-normalised, plus a plotted figure |
| Area | Ground-truth vs predicted changed m², **signed relative area error**, mean absolute area error per image, MAPE over images with change |
| Object level | Instances matched one-to-one at IoU ≥ 0.5 → object precision / recall / F1 |
| Multi-class (land use) | Confusion matrix, per-class P / R / F1, macro F1, accuracy |

## Robustness suite (`src/product/robustness.py`)

| Perturbation | Levels | Applied to | Real-world cause |
|---|---|---|---|
| Gaussian noise | σ 0.02 / 0.05 / 0.10 | T1 and T2 | Sensor noise |
| Brightness | ×0.7 / ×1.3 | T2 | Sun angle, season, exposure |
| Contrast | ×0.6 / ×1.4 | T2 | Radiometric differences between sensors |
| Blur | σ 1 / 2 px | T2 | Defocus, resampling |
| JPEG | quality 50 / 20 | T2 | Compression in delivery pipelines |
| Misregistration | 1 / 2 / 4 px shift | T2 | Imperfect co-registration |
| Haze | 20% / 40% | T2 | Atmosphere |

Ground truth is never perturbed. Each condition reports F1, ΔF1 vs clean, change recall, macro
F1, relative area error, object F1 and runtime.

## Results so far (synthetic scenes; the baseline is the only detector without the dataset)

**Demo scene** (256×256, 0.5 m GSD, 4 new buildings, illumination/seasonal nuisance; the
baseline threshold was tuned on 8 separate calibration scenes, τ = 0.12):

| Metric | Value |
|---|---|
| Pixel F1 vs synthetic truth | 0.930 |
| Object F1 (IoU ≥ 0.5) | 1.000 (4 / 4 instances) |
| Relative area error | +14.9% (cast shadows of new buildings are detected as change) |
| Encroachment | 2 advisory alerts in the verified demo zone; the unverified easement raised none |
| Runtime | ~14 ms for the full pipeline on CPU |

**Robustness of the pixel-difference baseline** (12 synthetic scenes):

| Condition | F1 | ΔF1 | Change recall | Relative area error |
|---|---|---|---|---|
| Clean | 0.922 | — | 0.973 | +11% |
| Noise σ 0.02 / 0.05 / 0.10 | 0.877 / 0.195 / 0.043 | −0.04 / −0.73 / −0.88 | ≥ 0.93 | +20% / +762% / +4188% |
| Brightness ×0.7 / ×1.3 | 0.030 / 0.054 | −0.89 / −0.87 | 0.65 / 1.00 | +4080% / +3475% |
| Contrast ×0.6 / ×1.4 | 0.588 / 0.742 | −0.33 / −0.18 | 0.62 / 1.00 | +10% / +69% |
| Blur σ 1 / 2 | 0.803 / 0.671 | −0.12 / −0.25 | 0.89 / 0.78 | +22% / +34% |
| JPEG q50 / q20 | 0.850 / 0.787 | −0.07 / −0.13 | 0.87 / 0.83 | +5% / +11% |
| Misregistration 1 / 2 / 4 px | 0.636 / 0.461 / 0.268 | −0.29 / −0.46 / −0.65 | 0.85 / 0.73 / 0.53 | +66% / +118% / +193% |
| Haze 20% / 40% | 0.244 / 0.028 | −0.68 / −0.89 | 0.96 / 1.00 | +591% / +7008% |

**Reading:** a fixed threshold on raw colour difference flags *every* pixel once global
illumination, haze or noise shifts the colours. Hence the near-perfect recall, collapsed
precision and huge area over-estimates. This is the quantitative case for a learned detector,
and for running this same suite on the trained Siamese U-Net / ResNet once LEVIR-CD is
available (`robustness --data-root data`). Misregistration is the main geometric weakness, and
it will also affect learned models, so co-registration quality should be a deployment
requirement.

## Tests

`tests/test_product.py`, 23 tests: instance areas, bounding boxes, connectivity and IoU
matching; zone rasterisation with holes, GeoJSON parsing, the geotransform, verified vs
unverified behaviour and overlap suppression; hand-computed confusion matrix, macro F1 and
area error; perturbation correctness; and end-to-end `demo`, `detect` and `evaluate` on a
temporary LEVIR-CD-layout folder.
