# Project 50 — Satellite Land Use Change & Encroachment Detector: Status Report

*Last updated: 19 November 2026*

This report compares the project brief (10 project steps and the expected-product checklist)
with what is in the repository today. For each step it gives what is complete, where it lives,
the key results, and what remains.

> **Language decision.** The brief allows "C, C++ or Java". This project is **Python-only**.
> Every item that asked for a C/C++/Java implementation (the tensor sandbox, the
> neurons/MLP module and the final application) is implemented in Python instead.
> To keep the "built from first principles" intent, the fundamentals use NumPy only, with
> no PyTorch, no autograd and no `np.linalg` inside the algorithms.

---

## 1. Summary

| Step | Topic | Status |
|---|---|---|
| 1 | Use case, stakeholder needs, requirements, acceptance tests, constraints | ✅ Complete (documentation) |
| 2 | Data acquisition, provenance, licensing, de-duplication, labels, leakage-safe splits | 🟡 Mostly complete |
| 3 | Mathematical sandbox: tensors, broadcasting, inner products, reshaping, SVD, gradient checking, visuals | ✅ Complete (Python/NumPy) |
| 4 | McCulloch-Pitts and sigmoid neurons, MLP with backprop, loss, parameter vs hyperparameter tracking | ✅ Complete (Python/NumPy) |
| 5 | Batch GD / SGD / Momentum / Nesterov; init, normalisation, regularisation, augmentation, overfitting ablations | ✅ Complete |
| 6 | Hebbian, modified Hebbian, competitive, error-correction learning, Boltzmann / energy-based experiment | ✅ Complete |
| 7 | Main product: land-use classification, segmentation, encroachment, temporal change, ResNet, 2D/3D conv visuals | 🟡 Partial (change segmentation only; no ResNet) |
| 8 | Benchmark R-CNN, Fast R-CNN, Faster R-CNN, YOLO, ResNet, GoogLeNet, RetinaNet | ❌ Not started |
| 9 | LSTM/transformer change model, VAE/GAN augmenter, latent-space visuals, step/sigmoid/tanh comparison | 🟡 Activation comparison only |
| 10 | Integrated reproducible app; IoU, macro F1, change recall, area error, runtime, robustness, error analysis, responsible use | 🟡 Partial |

**Tests:** `python -m pytest tests` gives **111 passed, 12 skipped**. The 12 skipped tests
belong to the original deep-learning pipeline. They need the LEVIR-CD images and trained
checkpoints, which are not on the current machine.

---

## 2. Completed work in detail

### Step 1 — Use-case definition ✅

All in `docs/`:
- `problem_statement.md`: formal problem, stakeholders, ethical boundary.
- `project_objectives.md`: functional and non-functional requirements, success criteria,
  evaluation metrics.
- `literature_review.md`, `references.md`: survey of classical, CNN, Siamese and transformer
  change detection.
- `proposed_architecture.md`, `model_architecture.md`: 9-stage pipeline and the Siamese U-Net
  specification.
- `risk_and_limitations.md`: radiometric noise, class imbalance, domain gap, legal and ethical
  limits.

The responsible-use boundary used throughout: the model detects *visual change*. It flags
*potential* encroachment only when authenticated restricted-area boundaries are supplied, and
the flags are advisory alerts for field audit, not legal findings.

### Step 2 — Data 🟡

| Item | Status | Where |
|---|---|---|
| Dataset chosen: LEVIR-CD (0.5 m/px, 1024×1024 bitemporal RGB building change) | ✅ | `docs/dataset_card.md` |
| Provenance and licence (academic, non-commercial) documented | ✅ | `docs/dataset_card.md` |
| Integrity audit (637 pairs, missing or corrupt files, mask values, change-pixel %) | ✅ | `src/extract_and_audit.py`, `data/audit_summary.json`, `docs/data_audit_report.md` |
| Manifest of all pairs | ✅ | `data/dataset_manifest.csv` |
| Leakage-safe splits: official 445 / 64 / 128 split kept unchanged | ✅ | `docs/data_preprocessing.md` |
| PyTorch dataset / dataloader with unit tests | ✅ | `src/dataset.py`, `tests/test_dataset.py` |
| Duplicate / near-duplicate detection across splits | ❌ | — |
| Own labels or land-use classes (LEVIR-CD only labels building change) | ❌ | — |

Measured class imbalance: 4.6% / 4.2% / 5.1% changed pixels (train / val / test).

### Step 3 — Mathematical sandbox ✅

`src/fundamentals/tensor_ops.py`, `src/fundamentals/gradcheck.py`. Documented in
`docs/fundamentals.md`.

- Broadcasting rules, row-major stride and index arithmetic, reshape (with `-1`), inner
  product, matmul and Frobenius norm, all written out explicitly.
- Symmetric eigendecomposition by cyclic Jacobi rotations, and **SVD by one-sided (Hestenes)
  Jacobi**, plus low-rank (Eckart-Young) approximation.
- Central-difference numerical gradients, relative-error gradient check, and a per-parameter
  gradient check for whole networks.
- Results: SVD singular values match NumPy to **2.4 × 10⁻¹³**. The best finite-difference
  relative error is **1.4 × 10⁻¹¹**.
- Visuals: `outputs/fundamentals/01_broadcasting.png`, `02_reshape_row_major.png`,
  `03_svd_low_rank.png`, `04_gradient_check.png`.

### Step 4 — Neurons and MLP ✅

`src/fundamentals/neurons.py`, `src/fundamentals/mlp.py`.

- McCulloch-Pitts neuron with excitatory/inhibitory inputs: AND, OR, NOT, AND-NOT, and a
  two-layer XOR. A sigmoid neuron is trained by gradient descent.
- MLP:
  - Dense layers with 5 initialisation schemes.
  - Activations: step, sigmoid, tanh, ReLU, identity.
  - Losses: MSE, BCE-with-logits, softmax cross-entropy.
  - Manual backpropagation.
  - Later additions: BatchNorm and Dropout layers, model save/load.
- Optimisers: gradient descent (batch or SGD depending on batch size), Momentum and Nesterov,
  with L2 weight decay.
- `ExperimentTracker` separates **hyperparameters** (fixed beforehand) from **learned
  parameters** (per-tensor norm, mean, std and gradient norm, logged every epoch) and writes
  both to JSON.
- Results: backprop matches the numeric gradient to **8.7 × 10⁻¹⁰**. A 4,547-parameter MLP
  reaches 100% train and validation accuracy on a 3-class spiral.
- Visuals: `outputs/fundamentals/05_neurons.png`, `06_activations.png`, `07_mlp_training.png`.

### Step 5 — Optimisation and generalisation ablations ✅

Two complementary parts:

1. **Deep model (original pipeline, PyTorch):**
   - `src/run_optimizer_comparison.py` compares SGD, Momentum, Nesterov and AdamW on the
     Siamese U-Net.
   - `src/run_ablation.py` covers fusion type (difference / concatenation / no skips),
     augmentation and loss weighting.
   - Write-up: `docs/phase_7_experiment_report.md`.
2. **From-scratch MLP (new):** `src/fundamentals/experiments.py` runs on a synthetic 4-class
   land-cover patch task (`landcover_toy.py`), with 3 seeds per setting and selection on the
   validation set only. Write-up: `docs/optimization_experiments.md`.

| Study | Headline result |
|---|---|
| Optimisers (each with its own lr grid) | All five land within about 1.5 points once lr is tuned. Momentum/Nesterov diverge at a 10× smaller lr than SGD. Batch-1 SGD is about 13× slower. On an ill-conditioned quadratic, final loss is GD 0.56, Momentum 1.3e-4, Nesterov 4.7e-6. |
| Initialisation (6 layers) | Zeros or N(0, 0.01²) stay at chance (signal vanishes). N(0, 1) with ReLU explodes to NaN. He init keeps the activation scale steady. |
| Normalisation | Raw 0–255 inputs diverge. BatchNorm rescues them (60.8%). Per-band standardisation is best (66.2%). |
| Regularisation | L2 and dropout add ≤ 0.5 pt. Early stopping on validation **accuracy** (0.664) beats early stopping on validation **loss** (0.646). |
| Augmentation | Rotations + flips + noise add **+8.1 pt** (74.3%), the largest gain in the study. |
| Data size | Augmentation + L2 reaches 85.4% at 1280 patches vs 73.6% plain. 320 augmented patches beat 1280 plain ones. |

Figures: `outputs/fundamentals/experiments/fig_*.png`.

### Step 6 — Classical learning rules ✅

`src/fundamentals/classical_learning.py`, `src/fundamentals/classical_demo.py`. Write-up and
conceptual comparison table: `docs/classical_learning.md`.

| Paradigm | Implemented | Key result |
|---|---|---|
| Hebbian / modified Hebbian | Plain Hebb, Oja, Sanger GHA | Hebb's ‖w‖ reaches 3.7 × 10⁵ while Oja's converges to 1.0001, both within 0.3° of PC1. GHA recovers the top-4 principal axes of 300-D patches (cosine 0.98–1.00 vs our SVD). |
| Competitive | Winner-take-all, DeSieno conscience, Kohonen SOM | Poor init leaves 3/8 prototypes active (purity 0.46). Conscience revives all 8 (purity 0.70). A 6×6 SOM reaches purity 0.79 with topographic error 0.03. |
| Error-correction | Perceptron, LMS / delta rule (Adaline) | The perceptron converges only on separable data (not XOR, not real patches). Water vs bare soil: perceptron 84.7%, LMS 86.0%. |
| Energy-based | Hopfield (Hebbian vs projection storage), RBM (CD-1, CD-10, exact ML) | Footprint recall from 15% noise: Hebbian 25%, projection 100%. Energy never increases. Capacity breaks at about 0.14N. RBM log-likelihood: CD-1 −3.30, CD-10 −3.11, exact −2.76 (optimum −2.64). |

Figures: `outputs/fundamentals/classical/fig_*.png`.

### Step 7 — Main product (current state) 🟡

These items come from the original Phases 3–8. The README reports the metrics below; they
could not be re-verified on this machine because the checkpoints and data are not present.

| Item | Status | Where |
|---|---|---|
| Classical pixel-difference baseline (Euclidean / MAD / grey) with threshold tuning | ✅ | `src/baseline.py`, `docs/baseline_experiment.md` |
| Siamese U-Net change segmentation (difference fusion) and concatenation / no-skip variants | ✅ | `src/model.py`, `src/experiments/models.py` |
| Composite BCE + Dice loss | ✅ | `src/losses.py` |
| Training with checkpoints, cosine learning-rate schedule, threshold tuning | ✅ | `src/train.py`, `docs/training_report.md` |
| Blind test evaluation: test F1 72.78%, IoU 57.21% (Phase 6 model) | ✅ (reported) | `src/evaluate.py`, `docs/test_evaluation_report.md` |
| Final model selection: validation F1 82.94% | ✅ (reported) | `src/select_final_model.py`, `docs/final_model_selection_report.md` |
| Production inference command-line tool with a 4-panel visualiser | ✅ | `src/inference.py` |
| Changed-area estimate in m² | ✅ | `src/evaluate.py` |
| **ResNet** backbone | ❌ | — |
| Land-use **classification** (multi-class) | ❌ | — |
| Multi-class **semantic** segmentation | ❌ | — |
| **Instance**-level detection of changed buildings | ❌ | — |
| **Encroachment** overlay against restricted-zone polygons | ❌ (planned as Phase 9) | — |
| 2D feature-map and **3D convolution** visualisations | ❌ | — |

### Step 9 — partial 🟡

- Step, sigmoid, tanh and ReLU (with their derivatives) are compared in
  `outputs/fundamentals/06_activations.png`.
- Everything else in Step 9 remains (see section 3).

### Step 10 — partial 🟡

| Item | Status |
|---|---|
| Precision, recall (= change recall), F1, IoU, confusion counts | ✅ `src/metrics.py` |
| Runtime / latency / FPS benchmarking | ✅ `src/benchmark_model.py`, `src/baseline.py` |
| Macro F1 | 🟡 Implemented for the multi-class fundamentals experiments only |
| Area error against ground truth | 🟡 Aggregate over the test set only (−11.7%, see section 3.1); no per-image or per-instance area error yet |
| Robustness tests (noise, brightness, blur, misregistration) | ❌ |
| Systematic error analysis | 🟡 Error-overlay panels in `docs/test_evaluation_report.md` |
| Responsible-use limits | 🟡 `docs/risk_and_limitations.md`; not yet in the app itself |
| Integrated app with a command-line or lightweight GUI | 🟡 Inference CLI only |
| Industry deployment recommendation | ❌ |

---

## 3. Results and metrics

Two sources of numbers, kept separate:

- **3.1–3.3 (deep change-detection pipeline, LEVIR-CD).** Taken from the Phase 3–8 reports in
  `docs/`, produced on Apple Silicon (MPS). The checkpoints and data are not on the current
  machine, so these were **not re-run here**.
- **3.4–3.7 (from-scratch fundamentals).** Produced and verified on the current machine (CPU).
  Raw values are in `outputs/fundamentals/**/summary.json`, `results.json` and `summary.csv`.

All LEVIR-CD metrics are pixel-level, on the changed class, computed after resizing
1024×1024 → 256×256 (2.0 m per model pixel).

### 3.1 Change detection on the blind test set (128 pairs)

| Metric | Pixel-difference baseline (τ = 0.40) | Siamese U-Net, Phase 6 (τ* = 0.20) | Change |
|---|---|---|---|
| Precision | 14.08% | **77.60%** | +63.52 pt (5.5×) |
| Recall (change recall) | 13.88% | **68.53%** | +54.66 pt (4.9×) |
| F1 | 13.98% | **72.78%** | +58.81 pt (5.2×) |
| IoU | 7.51% | **57.21%** | +49.70 pt (7.6×) |
| Pixel accuracy | 91.30% | **97.39%** | +6.09 pt |
| Specificity | 95.45% | — | — |
| False positives (pixels) | 362,021 | **84,550** | −76.6% |
| False negatives (pixels) | 368,106 | **134,499** | −63.5% |

Notes:
- Pixel accuracy is misleading here: the changed class is only about 5% of pixels, so
  predicting "no change" everywhere already scores about 95%. F1 and IoU are the metrics to
  quote.
- False negatives exceed false positives, so the model under-detects. Most missed pixels come
  from small structures (< 25 m²) that are lost in the 4× downscale.

**Area estimation on the test set**

| Quantity | Value |
|---|---|
| Ground-truth changed area | 1,709,656 m² (170.97 ha) |
| Predicted changed area | 1,509,860 m² (150.99 ha) |
| Area error | −199,796 m² (**−11.7%**, under-estimate) |

**Runtime (Phase 6 model, MPS)**

| Metric | Value |
|---|---|
| Parameters | 1,964,097 (7.49 MB float32; 22.57 MB checkpoint with optimiser state) |
| Total test inference (128 pairs) | 6.294 s |
| Latency per pair | 49.17 ms (20.34 FPS, end-to-end including I/O) |

### 3.2 Validation results and model selection

**Phase 5 training (Siamese U-Net, difference skips, 10 epochs, AdamW + cosine schedule)**

| Threshold | Precision | Recall | F1 | IoU | Pixel accuracy |
|---|---|---|---|---|---|
| τ = 0.50 (default) | 81.09% | 64.49% | 71.85% | 56.06% | 97.88% |
| **τ* = 0.20 (tuned on val)** | 74.63% | 72.29% | **73.44%** | **58.03%** | 97.81% |

Best epoch: 9 of 10. About 23 s per epoch, 445 training pairs.

**Phase 8 head-to-head (15 epochs each, same budget)**

| Metric | Candidate A: difference skips | Candidate B: concatenation skips (**selected**) |
|---|---|---|
| Parameters | 1,964,097 (7.49 MB) | 3,930,433 (14.99 MB) |
| Best epoch / val loss | 14 / 0.3005 | 13 / **0.2249** |
| Val F1 / IoU at τ = 0.50 | 72.69% / 57.09% | **82.26% / 69.86%** |
| Tuned val precision | 79.73% | **83.77%** |
| Tuned val recall | 70.00% | **82.13%** |
| **Tuned val F1 (τ* = 0.20)** | 74.55% | **82.94%** |
| **Tuned val IoU** | 59.43% | **70.85%** |
| Model latency / throughput (MPS, forward only) | 9.29 ms / 107.7 FPS | 18.08 ms / 55.3 FPS |
| Training time | 481 s | 704 s |

> ⚠️ The selected Candidate B has **not yet been evaluated on the blind test set**. Its 82.94%
> is a validation number, and the threshold was also tuned on validation, so it is optimistic.
> The only blind test result so far is the Phase 6 model's 72.78% F1. Running `evaluate.py`
> on Candidate B is the first task once LEVIR-CD is back on disk.

### 3.3 Deep-model ablations (Phase 7, validation set)

**Architecture (6 epochs, AdamW)**

| Variant | Parameters | Precision | Recall | F1 | IoU | Latency |
|---|---|---|---|---|---|---|
| Difference skips (control) | 1.96 M | 69.74% | 61.12% | 65.15% | 48.31% | 9.79 ms |
| Concatenation skips | 3.93 M | 83.28% | 75.30% | **79.09%** | **65.41%** | 10.84 ms |
| No multi-scale skips | 1.77 M | 20.37% | 24.63% | 22.30% | 12.55% | 9.10 ms |

**Training changes (6 epochs, difference-skip model)**

| Change | F1 | IoU | Val loss |
|---|---|---|---|
| Control (0.5 BCE + 0.5 Dice) | 65.15% | 48.31% | 0.3758 |
| + synchronous flips / 90° rotations | 55.26% | 38.17% | 0.4144 |
| Dice-dominant loss (0.25 BCE + 0.75 Dice) | **67.65%** | **51.11%** | 0.4726 |

Augmentation *lowered* F1 at 6 epochs. The report attributes this to the harder training
distribution needing longer schedules. This contrasts with the +8.1 pt that augmentation gave
in the from-scratch study (3.5). It is worth re-testing at 15+ epochs.

**Optimisers (5 epochs)**

| Optimiser | lr | Peak val F1 | Peak val IoU | Final val loss |
|---|---|---|---|---|
| SGD | 0.01 | 17.83% | 9.79% | 0.5701 |
| SGD + Momentum 0.9 | 0.01 | 56.41% | 39.28% | 0.3705 |
| Nesterov 0.9 | 0.01 | 58.17% | 41.02% | 0.3575 |
| AdamW (wd 1e-4) | 0.001 | **66.87%** | **50.23%** | **0.3558** |

These optimisers used one fixed lr each, not a per-optimiser lr search, so the gaps include
lr-mismatch effects. The from-scratch study (3.5) tuned lr per optimiser and found the gaps
mostly close.

### 3.4 Fundamentals verification (Steps 3–4)

| Check | Metric | Result |
|---|---|---|
| Broadcasting, reshape, matmul vs NumPy | Exact match | ✅ identical |
| Jacobi SVD vs `np.linalg.svd` (96×96 tile) | Max abs singular-value error | **2.4 × 10⁻¹³** |
| SVD low-rank reconstruction | Relative error at rank 1 / 5 / 15 / 40 | 0.278 / 0.154 / 0.057 / 0.030 |
| SVD energy | Rank for 95% energy | 3 |
| Central-difference gradient | Best relative error (ε ≈ 3.2 × 10⁻⁶) | **1.4 × 10⁻¹¹** |
| MLP backprop vs numeric gradient | Max relative error, all parameters | **8.7 × 10⁻¹⁰** (threshold 10⁻⁶) |
| McCulloch-Pitts AND / OR / AND-NOT / 2-layer XOR | Truth tables | ✅ all correct |
| Sigmoid neuron learning OR | Final BCE loss | 0.0156, 4/4 correct |
| MLP [2, 64, 64, 3] on 3-class spiral | Train / val accuracy | **100% / 100%** (4,547 learned parameters) |

### 3.5 Optimisation and generalisation ablations (Step 5)

Synthetic 4-class land-cover patches. Train 240, val 600, test 1200. Mean ± std over 3 seeds,
selection on validation only. Chance level is 25% accuracy (macro F1 0.10 for a
constant predictor).

**Optimisers (best lr per optimiser, 60 epochs)**

| Optimiser | Selected lr | Test acc | Test macro F1 | Min val loss | Runtime / run |
|---|---|---|---|---|---|
| Batch GD | 0.3 | **0.666 ± 0.005** | 0.665 | **0.829** | 0.4 s |
| SGD (batch 1) | 0.003 | 0.654 ± 0.010 | 0.653 | 0.858 | 8.1 s |
| Mini-batch SGD (16) | 0.03 | 0.653 ± 0.011 | 0.652 | 0.856 | 0.6 s |
| Momentum 0.9 | 0.003 | 0.652 ± 0.006 | 0.652 | 0.855 | 0.9 s |
| Nesterov 0.9 | 0.003 | 0.653 ± 0.006 | 0.653 | 0.853 | 1.1 s |

Ill-conditioned quadratic, final loss after 60 steps: GD 5.6 × 10⁻¹, Momentum 1.3 × 10⁻⁴,
Nesterov 4.7 × 10⁻⁶.

**Initialisation (6 hidden layers, 60 epochs): test accuracy / macro F1**

| Init | tanh | ReLU |
|---|---|---|
| Zeros | 0.250 / 0.100 | 0.250 / 0.100 |
| N(0, 0.01²) | 0.250 / 0.100 | 0.250 / 0.100 |
| N(0, 1) | 0.539 / 0.528 | diverged (3/3 seeds) |
| Xavier | 0.613 / 0.612 | **0.657 / 0.657** |
| He | **0.623 / 0.624** | 0.652 / 0.652 |

**Normalisation, regularisation and augmentation (100 epochs)**

| Study | Configuration | Train acc | Test acc | Test macro F1 |
|---|---|---|---|---|
| Normalisation | Raw DN 0–255, lr 0.01 | 0.308 | 0.246 ± 0.004 | 0.100 |
| | Raw DN 0–255, lr 1e-5 | 1.000 | 0.487 ± 0.023 | 0.483 |
| | Raw DN + BatchNorm | 1.000 | 0.608 ± 0.023 | 0.605 |
| | **Per-band standardised** | 1.000 | **0.662 ± 0.002** | **0.662** |
| | Standardised + BatchNorm | 0.997 | 0.613 ± 0.017 | 0.610 |
| Regularisation | None | 1.000 | 0.662 ± 0.002 | 0.662 |
| | L2 1e-4 / 1e-3 / 1e-2 | 1.000 | 0.663 / 0.664 / 0.667 | 0.663 / 0.664 / 0.668 |
| | Dropout 0.2 / 0.5 | 1.000 / 0.996 | 0.667 / 0.654 | 0.667 / 0.653 |
| | L2 1e-3 + dropout 0.2 | 1.000 | 0.660 ± 0.004 | 0.660 |
| | Early stopping on val loss | 0.847 | 0.646 ± 0.015 | 0.644 |
| | Early stopping on val accuracy | 0.989 | 0.664 ± 0.006 | 0.665 |
| Augmentation | None | 1.000 | 0.662 ± 0.002 | 0.662 |
| | Gaussian jitter | 1.000 | 0.665 ± 0.005 | 0.665 |
| | Rotations + flips (dihedral) | 0.997 | 0.721 ± 0.019 | 0.722 |
| | **Dihedral + jitter** | 0.983 | **0.743 ± 0.008** | **0.746** |

**Training-set size: test accuracy (macro F1)**

| Patches | 40 | 80 | 160 | 320 | 640 | 1280 |
|---|---|---|---|---|---|---|
| Plain | 0.571 (0.555) | 0.606 (0.592) | 0.654 (0.653) | 0.669 (0.670) | 0.692 (0.694) | 0.736 (0.737) |
| Dihedral + L2 1e-3 | 0.605 (0.599) | 0.672 (0.672) | 0.698 (0.700) | 0.767 (0.768) | 0.803 (0.803) | **0.854 (0.854)** |

### 3.6 Classical learning rules (Step 6)

| Experiment | Metric | Result |
|---|---|---|
| Plain Hebb, 2000 samples | Final ‖w‖ / angle to PC1 | 3.7 × 10⁵ / 0.27° |
| Oja, 2000 samples | Final ‖w‖ / angle to PC1 | 1.0001 / 0.27° |
| Sanger GHA, 300-D patches | \|cos\| with SVD components 1–4 | 0.997, 1.000, 1.000, 0.984 |
| | Variance explained by components 1–4 | 49.9%, 7.6%, 5.3%, 1.8% |
| Winner-take-all, k = 8, good init | Active units / purity / quantisation error | 8 / 0.661 / 0.763 |
| Winner-take-all, poor init | Active units / purity / quantisation error | 3 / 0.458 / 1.430 |
| Winner-take-all, poor init + conscience | Active units / purity / quantisation error | 8 / **0.703** / 0.776 |
| Kohonen SOM 6×6 | Purity / quantisation error / topographic error | **0.789** / 0.395 / 0.034 |
| Perceptron, separable blobs | Epochs to converge | 2 |
| Perceptron, XOR and real patches | Converged? | No (as theory predicts) |
| Water vs bare soil, held-out | Test accuracy, perceptron / LMS | 84.7% / **86.0%** |
| Hopfield, 4 footprints, 15% noise | Exact recall, Hebbian / projection | 25% / **100%** |
| Hopfield | Energy monotone non-increasing | ✅ all runs |
| Hopfield capacity, N = 100, 10% noise | Exact recall at P = 5 / 10 / 14 / 20 / 30 | 100% / 100% / 77% / 29% / 0% |
| RBM 9×16, bars and stripes (optimum −2.64) | Final log-likelihood, CD-1 / CD-10 / exact | −3.30 / −3.11 / **−2.76** |
| | Probability mass on training patterns | 97.9% / 98.0% / 98.4% |
| | Valid Gibbs samples (exact-ML model) | 24 / 24 |

### 3.7 Test suite

| Run | Result |
|---|---|
| `python -m pytest tests` (10 test files) | **111 passed, 12 skipped, 0 failed** |

The 12 skipped tests are in the original pipeline suites and need LEVIR-CD data or trained
checkpoints. Every fundamentals test (Steps 3–6) passes.

---

## 4. Remaining work

### Step 2 — finish data curation
- Perceptual-hash duplicate and near-duplicate check within and across train/val/test, with a
  report.
- Add a land-use classification dataset (proposed: **EuroSAT RGB**, 10 classes, 64×64,
  CC-BY-4.0) with provenance, licence, de-duplication and a fixed leakage-safe split.
- Derive building-instance labels from LEVIR-CD masks (connected components) for Step 8
  detection.

### Step 7 — main product
- **ResNet-18/34 Siamese encoder**, replacing the plain convolutional encoder. Compare it
  against the existing Siamese U-Net under the same training budget.
- **Land-use classifier** (ResNet) on EuroSAT, applied per tile to the T1 and T2 images to
  produce land-use *transitions* (e.g. vegetation → built-up).
- **Instance analysis:** split the change mask into individual building changes, giving a
  count, area in m² and bounding box per change.
- **Encroachment module:** load restricted-zone polygons (GeoJSON), rasterise them, intersect
  them with the change instances, and output advisory flags with area and overlap %.
  Nothing is flagged without verified boundaries.
- **Temporal change modelling** across more than two dates where available.
- **Visualisations:** 2D feature maps per encoder stage, and a 3D-convolution demonstration
  over a stacked (T1, T2) temporal volume.

### Step 8 — benchmark
- Classification: **ResNet vs GoogLeNet** on EuroSAT (accuracy, macro F1, parameters,
  latency).
- Detection of changed buildings: **Faster R-CNN vs RetinaNet** (torchvision) **vs YOLO**,
  reporting mAP@0.5, recall and latency.
- R-CNN and Fast R-CNN: a short justified comparison of their region-proposal design and cost
  against Faster R-CNN, possibly with a small region-proposal demo. Full training is
  impractical and superseded, so the brief's "select relevant candidates" wording applies.
- Licence note: Ultralytics YOLO is AGPL-3.0. This must be considered before any deployment.

### Step 9 — advanced models
- **Temporal sequence change model:** ConvLSTM or a small transformer over encoder features
  of an image sequence, with attention/sequence visualisations.
- **VAE and/or GAN augmenter** for rare change patches. Measure whether synthetic samples
  improve change recall, and visualise the latent space (PCA/t-SNE coloured by class or change
  type).

### Step 10 — integration and validation
- One reproducible Python application: a single CLI (`detect`, `evaluate`, `benchmark`) plus
  a lightweight GUI (Gradio, as planned for Phase 9), with fixed seeds and pinned
  dependencies.
- Evaluation package:
  - IoU, macro F1, change recall, **area error** (predicted vs ground-truth m²), runtime and
    confusion matrices.
  - **Robustness suite:** noise, brightness/contrast, blur, JPEG, small misregistration.
  - Error analysis by change size and scene type.
- Responsible-use limits shown in the app output, and an **industry deployment
  recommendation** (hardware, latency, update cadence, human-in-the-loop review).
- Python equivalent of the CMake/Maven/Gradle requirement: `pyproject.toml` with pinned
  dependencies and a one-command test runner.

---

## 5. Blockers and decisions needed

| # | Item | Why it matters |
|---|---|---|
| 1 | **LEVIR-CD images are not on this machine.** The code expects `data/{train,val,test}/{A,B,label}`. | Required to retrain or evaluate any deep model, and to run the 12 skipped tests. |
| 2 | **Permission to download EuroSAT** (about 90 MB). | Needed for land-use classification (Steps 7–8). |
| 3 | **Compute: this machine has CPU-only PyTorch.** | Detector benchmarks and ResNet training need reduced tiles and short schedules on CPU, or the scripts can be run on a GPU machine or Colab. |

---

## 6. How to reproduce the completed work

```bash
# Unit tests (111 pass; 12 skip without LEVIR-CD data / checkpoints)
python -m pytest tests

# Step 3-4 figures                         -> outputs/fundamentals/
python -m src.fundamentals.demo

# Step 5 ablations (~9 min on CPU)         -> outputs/fundamentals/experiments/
python -m src.fundamentals.experiments

# Step 6 classical learning (~70 s)        -> outputs/fundamentals/classical/
python -m src.fundamentals.classical_demo

# Original deep pipeline (requires LEVIR-CD in data/)
python src/train.py
python src/evaluate.py
python src/inference.py --t1 data/val/A/val_1.png --t2 data/val/B/val_1.png --output outputs/sample.png
```

## 7. Documentation index

| Document | Covers |
|---|---|
| `README.md` | Overview, phase table, setup commands |
| `docs/problem_statement.md`, `project_objectives.md` | Step 1 |
| `docs/dataset_card.md`, `data_audit_report.md`, `data_preprocessing.md` | Step 2 |
| `docs/fundamentals.md` | Steps 3–4 |
| `docs/optimization_experiments.md`, `phase_7_experiment_report.md` | Step 5 |
| `docs/classical_learning.md` | Step 6 |
| `docs/model_architecture.md`, `training_report.md`, `test_evaluation_report.md`, `final_model_selection_report.md` | Step 7 (current Siamese U-Net) |
| `docs/risk_and_limitations.md` | Responsible use (Step 10, partial) |
| `docs/PROJECT_STATUS.md` | This report |
