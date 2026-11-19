# Project 50: Satellite Land Use Change & Encroachment Detector — Status

*Updated: 20 November 2026*

> **Language:** the whole project is in **Python**. The brief allowed C, C++ or Java; every
> part that asked for those languages is written in Python instead.

---

## Part A: Completed work

### Quick view

| Step | What it is | Status |
|---|---|---|
| 1 | Problem, users, requirements | ✅ Done |
| 2 | Dataset documentation and checks | ✅ Mostly done |
| 3 | Math sandbox (tensors, SVD, gradient check) | ✅ Done |
| 4 | Neurons + neural network built from scratch | ✅ Done |
| 5 | Optimiser and overfitting experiments | ✅ Done |
| 6 | Hebbian, competitive, perceptron, Hopfield, Boltzmann | ✅ Done |
| 7 | Main change detector | ✅ U-Net done · 🟡 ResNet code ready, not trained |
| 8 | Model benchmark | 🟡 ResNet vs GoogLeNet code ready, not run · ❌ detectors not started |
| 9 | LSTM / transformer and VAE / GAN | 🟡 Code ready and tested, full runs not done |
| 10 | App, evaluation, robustness | ✅ Done (on synthetic data) |

### Step 1: Problem definition ✅
Written in `docs/`: problem statement, requirements, literature review, architecture, and
risks and limits.

**Key rule:** the system only *flags possible* encroachment, and only inside officially
verified boundaries. It never makes a legal decision.

### Step 2: Dataset ✅ (mostly)
- Dataset chosen: **LEVIR-CD**, 637 image pairs at 0.5 m per pixel. Licence and source are
  documented.
- Data checked: no missing or broken files. Split is 445 train / 64 val / 128 test.
- About 5% of pixels show change (heavy class imbalance).

### Step 3: Math sandbox ✅
Built from scratch in NumPy: broadcasting, reshape, matrix multiply, SVD and gradient checking.

| Check | Result |
|---|---|
| Our SVD vs NumPy | Error 0.0000000000002 (matches) |
| Gradient check | Error 0.00000000001 (correct) |

### Step 4: Neurons and neural network ✅
- McCulloch-Pitts neuron (AND, OR, NOT, XOR) and a sigmoid neuron.
- Full neural network with hand-written backpropagation, 3 losses, 3 optimisers, BatchNorm,
  Dropout, and tracking of parameters vs hyperparameters.

| Check | Result |
|---|---|
| Logic gates | All correct |
| Backprop vs numeric gradient | Error 0.0000000009 (correct) |
| Spiral classification | 100% accuracy |

### Step 5: Training experiments ✅
Run on synthetic land-cover patches, 3 seeds each.

| Experiment | Main finding |
|---|---|
| Optimisers (GD, SGD, Momentum, Nesterov) | All reach about 65–67% once tuned; Momentum and Nesterov converge much faster |
| Weight initialisation | Zero or tiny init stays at chance (25%); He / Xavier work (65%) |
| Input normalisation | Raw pixels fail (25%); standardised inputs best (66%) |
| Regularisation (L2, dropout) | Small effect (+0.5 pt) |
| Augmentation (rotate / flip / noise) | **Biggest gain: 66% → 74%** |
| More data | 1280 images + augmentation: **85%** (vs 74% without) |

### Step 6: Classical learning rules ✅

| Method | Result |
|---|---|
| Hebb vs Oja | Hebb weights explode (370,000); Oja stays at 1.0 |
| Sanger (finds main patterns) | Matches SVD (similarity 0.98–1.00) |
| Competitive learning + "conscience" | Fixes dead units: 3/8 → 8/8 active |
| Self-organising map | Groups land types well (purity 0.79) |
| Perceptron vs LMS | Perceptron fails on XOR (as theory says); LMS 86% vs perceptron 85% |
| Hopfield memory | Recalls building shapes 100% (projection rule) vs 25% (Hebbian) |
| Boltzmann machine (RBM) | Exact training best (−2.76 vs best possible −2.64) |

### Step 7: Change detector ✅ (U-Net) / 🟡 (ResNet)
Results reported from earlier training (Apple Mac GPU). These could not be re-run here because
the data and model files are not on this computer.

| Model | Data split | F1 | IoU |
|---|---|---|---|
| Simple pixel-difference baseline | Test | 13.98% | 7.51% |
| Siamese U-Net (Phase 6) | **Test** | **72.78%** | **57.21%** |
| Siamese U-Net, concatenation (selected) | Validation only | 82.94% | 70.85% |

- Area estimate on the test set: predicted 151 ha vs true 171 ha (**−11.7%**).
- Speed: about 49 ms per image pair.
- **New:** a Siamese **ResNet** change detector is written and tested, but **not trained yet**.

### Step 10: App and evaluation tools ✅
One command-line app, `python -m src.product.app` with commands `demo`, `detect`, `evaluate`
and `robustness`. It can:
- Find each changed building separately (count, size in m², location).
- Check changes against restricted-zone maps (GeoJSON) and give **advisory alerts** for
  verified zones only.
- Report F1, IoU, macro F1, change recall, area error, object-level F1 and confusion matrix.
- Test robustness against noise, brightness, blur, JPEG, misalignment and haze.

**Demo result (synthetic scene):** F1 0.93; all 4 new buildings found; 2 alerts in the verified
zone, none in the unverified zone; 14 ms.

**Robustness finding:** the simple baseline breaks easily. A brightness change drops its F1
from 0.92 to 0.05, and haze or a 4-pixel misalignment ruins it. A trained model is needed.

**Packaging:** `pyproject.toml` (install with `pip install -e .`, then run `satcd`) and a list of
exact tested versions.

### Tests
- **153 automated tests pass:** 134 in the full suite plus 19 new ones run separately.
- **12 tests are skipped** because they need the dataset.

---

## Part B: Not yet completed (needs dataset or training)

### 1. Datasets needed

| Dataset | Why | Status |
|---|---|---|
| **LEVIR-CD** (change detection) | Training and testing the main models | ❌ Not on this computer; must be copied into `data/train`, `data/val`, `data/test` |
| **EuroSAT** (land-use types, ~90 MB) | Land-use classification (forest, residential, etc.) | ❌ Not downloaded; needs approval |

### 2. Training and runs still to do

| # | Task | Needs | Code ready? |
|---|---|---|---|
| 1 | Test the selected model (82.94% val) on the **test set** | LEVIR-CD | ✅ Yes |
| 2 | Full evaluation (macro F1, area error, etc.) on real images | LEVIR-CD | ✅ Yes |
| 3 | Robustness tests on the trained model | LEVIR-CD | ✅ Yes |
| 4 | Train the **ResNet change detector** and compare with U-Net | LEVIR-CD | ✅ Yes |
| 5 | Check for duplicate images across splits | LEVIR-CD | ❌ No |
| 6 | Train the land-use classifier (**ResNet vs GoogLeNet**) | EuroSAT | ✅ Yes |
| 7 | Land-use change maps (e.g. farm → buildings) | EuroSAT model | ✅ Yes |
| 8 | Detector benchmark (**Faster R-CNN, RetinaNet, YOLO**) | LEVIR-CD + GPU | ❌ No |
| 9 | **VAE / GAN** full experiment (3 seeds) | Nothing (synthetic) | ✅ Yes, run was interrupted |
| 10 | **LSTM / transformer / 3-D CNN** full experiment | Nothing (synthetic) | ✅ Yes, not run yet |
| 11 | Simple GUI (Gradio) | Approval to install | ❌ No |
| 12 | Deployment recommendation write-up | Results from above | ❌ No |

Items 9 and 10 do **not** need the dataset; they only need time to run (a few minutes each):
```bash
python -m src.advanced.run_generative --seeds 3
python -m src.advanced.run_temporal
```

### 3. Computer limitation
This computer has **no GPU**. Big training jobs (ResNet, detector benchmark) will be slow here.
Use Google Colab or a GPU machine for those.

---

## How to run what is done

```bash
python -m pytest tests                    # all tests
python -m src.fundamentals.demo           # Steps 3–4 figures
python -m src.fundamentals.experiments    # Step 5 experiments (~9 min)
python -m src.fundamentals.classical_demo # Step 6 (~70 s)
python -m src.product.app demo            # Step 10 app demo
python -m src.product.app robustness --synthetic 12
```

Detailed write-ups: `docs/fundamentals.md`, `docs/optimization_experiments.md`,
`docs/classical_learning.md`, `docs/product_pipeline.md`.
