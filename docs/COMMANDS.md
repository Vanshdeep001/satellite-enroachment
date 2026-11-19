# Commands: How to Run and Test Everything

Run every command from the project folder (`Satellite-Encroachment-Detection`). Figures and
results are saved inside `outputs/`.

**Start here (no dataset needed):** run the tests (section 1), then the app demo (section 5).

---

## 0. Setup (once)

```bash
cd "C:\Users\Vansh Deep Srivastav\Pictures\satellite-platfom\Satellite-Encroachment-Detection"
pip install -e .
```

This installs the project and its libraries (numpy, scipy, matplotlib, Pillow, torch,
torchvision). It also adds the `satcd` command, a short name for the app.

Optional extras:
```bash
pip install -e ".[dev]"     # pytest, for running the tests
pip install -e ".[gui]"     # gradio, for the GUI (not built yet)
```

---

## 1. Automated tests

| Command | What it checks | Time |
|---|---|---|
| `python -m pytest tests` | All unit tests (12 are skipped without the dataset) | ~1–2 min |
| `python -m pytest tests/test_fundamentals_tensor.py` | Step 3: math sandbox (broadcasting, reshape, SVD, gradient check) | seconds |
| `python -m pytest tests/test_fundamentals_mlp.py` | Step 4: neurons, neural network, backprop, optimisers | seconds |
| `python -m pytest tests/test_fundamentals_experiments.py` | Step 5: BatchNorm, Dropout, early stopping, augmentation | seconds |
| `python -m pytest tests/test_fundamentals_classical.py` | Step 6: Hebbian, competitive, perceptron, Hopfield, Boltzmann | ~30 s |
| `python -m pytest tests/test_product.py` | Step 10: building instances, encroachment alerts, evaluation, robustness, app | ~10 s |
| `python -m pytest tests/test_advanced.py` | Steps 7–9: VAE/GAN, transformer/LSTM/3D CNN, ResNet, land-use | ~15 s |
| `python -m pytest tests/test_dataset.py tests/test_model.py tests/test_inference.py` | Original U-Net pipeline (some tests skip without data or checkpoints) | seconds |

---

## 2. Steps 3–4: Math sandbox and neural network from scratch

```bash
python -m src.fundamentals.demo
```

| Used for | Output |
|---|---|
| Broadcasting, reshape, SVD, gradient-check, neuron, activation and network-training figures | `outputs/fundamentals/` (7 figures + `summary.json`) |

Time: about 5 seconds.

---

## 3. Step 5: Training experiments

```bash
python -m src.fundamentals.experiments           # full run, 3 seeds (~9 min)
python -m src.fundamentals.experiments --quick   # fast check (seconds)
```

| Used for | Output |
|---|---|
| Compares optimisers (GD, SGD, Momentum, Nesterov), weight initialisation, normalisation, regularisation, augmentation and dataset size | `outputs/fundamentals/experiments/` (figures, `results.json`, `summary.csv`) |

Useful options:
```bash
python -m src.fundamentals.experiments --studies optimizers,augmentation   # run only some studies
python -m src.fundamentals.experiments --studies regularization --merge    # re-run one study, keep the others
```

---

## 4. Step 6: Classical learning rules

```bash
python -m src.fundamentals.classical_demo
```

| Used for | Output |
|---|---|
| Hebb / Oja / Sanger, competitive learning and self-organising map, perceptron vs LMS, Hopfield memory, Boltzmann machine | `outputs/fundamentals/classical/` (5 figures + `summary.json`) |

Time: about 70 seconds.

---

## 5. Step 10: The app (synthetic data, no dataset needed)

| Command | What it does | Output |
|---|---|---|
| `python -m src.product.app demo` | Full pipeline on a synthetic scene: detect change, find each building, check restricted zones, give advisory alerts | `outputs/product/demo/` |
| `python -m src.product.app robustness --synthetic 12` | Tests detection against noise, brightness, blur, JPEG, misalignment and haze | `outputs/product/robustness/` |
| `python -m src.product.app detect --t1 before.png --t2 after.png --zones zones.geojson --gsd 0.5` | Runs on your own pair of images, optionally with a restricted-zone map | `outputs/product/detect/` |

Useful options:

| Option | Meaning |
|---|---|
| `--method baseline` / `--method model` | Force the simple baseline or the trained model (default: model if a checkpoint exists) |
| `--checkpoint path.pt` | Use a specific trained model |
| `--gsd 0.5` | Metres per pixel (LEVIR-CD = 0.5) |
| `--min-area-m2 10` | Ignore changes smaller than this |
| `--min-overlap 0.1` | Minimum share of a change inside a verified zone to raise an alert |
| `--geotransform X0 PX_W Y0 PX_H` | Use when the zone file is in map coordinates instead of pixels |

After `pip install -e .`, `satcd` can replace `python -m src.product.app`, e.g. `satcd demo`.

**Zone file rule:** a zone only raises alerts if it has `"verified": true` **and** a
`"source"`. All other zones are reported for information only.

---

## 6. Step 9: Advanced models (no dataset needed; full runs not done yet)

| Command | What it does | Time |
|---|---|---|
| `python -u -m src.advanced.run_generative --seeds 3` | A VAE and a GAN make extra training images; checks whether they help a classifier; plots the latent space | a few min |
| `python -u -m src.advanced.run_temporal` | Transformer vs ConvLSTM vs 3D CNN vs a two-date baseline on cloudy 6-date sequences; plots attention, 3D kernels, feature maps, and sigmoid/tanh vs step gates | ~5–15 min (estimate) |

Add `--quick` to either one for a fast check. Output goes to `outputs/advanced/generative/`
and `outputs/advanced/temporal/`.

---

## 7. Steps 7–8: ResNet models

| Command | What it does | Needs |
|---|---|---|
| `python src/train_resnet_cd.py --synthetic --epochs 6` | Trains the ResNet change detector on synthetic scenes (checks it works) | nothing |
| `python src/landuse.py benchmark --synthetic` | ResNet-18 vs GoogLeNet land-use classifier on synthetic patches | nothing |
| `python src/train_resnet_cd.py --epochs 15` | Real ResNet change-detector training | **LEVIR-CD** |
| `python src/landuse.py benchmark --data-dir data/EuroSAT_RGB` | Real ResNet vs GoogLeNet benchmark | **EuroSAT** |
| `python src/landuse.py transitions --t1 before.png --t2 after.png --checkpoint outputs/landuse/<run>/resnet18.pt` | Land-use change map (e.g. farmland → buildings) | trained land-use model |

ResNet change-detector options: `--fusion concat|diff`, `--backbone resnet18|resnet34`,
`--pretrained` (downloads ImageNet weights once). The trained checkpoint can be used directly
by the app: `python -m src.product.app detect --checkpoint outputs/resnet_cd/<run>/best_model.pt ...`

---

## 8. Needs the LEVIR-CD dataset

Put the data in `data/train`, `data/val` and `data/test`, each with `A/` (before), `B/` (after)
and `label/` (change mask) folders.

| Command | What it does |
|---|---|
| `python -m src.product.app evaluate --split test` | Full evaluation: F1, IoU, macro F1, change recall, area error, object F1, confusion matrix |
| `python -m src.product.app robustness --split test` | Robustness tests on real images |
| `python src/train.py` | Original U-Net training |
| `python src/evaluate.py` | Original U-Net test-set evaluation |
| `python src/select_final_model.py` | Original model selection (two U-Net candidates) |
| `python src/run_ablation.py` | Original U-Net ablation studies |
| `python src/run_optimizer_comparison.py` | Original U-Net optimiser comparison |
| `python src/inference.py --t1 data/val/A/val_1.png --t2 data/val/B/val_1.png --output outputs/sample.png` | Original single-pair prediction from the U-Net |

---

## 9. Where results go

| Folder | Contents |
|---|---|
| `outputs/fundamentals/` | Steps 3–4 figures |
| `outputs/fundamentals/experiments/` | Step 5 figures and tables |
| `outputs/fundamentals/classical/` | Step 6 figures |
| `outputs/product/` | App reports, robustness results |
| `outputs/advanced/` | Step 9 results |
| `outputs/resnet_cd/`, `outputs/landuse/` | ResNet and land-use runs |
| `outputs/checkpoints/` | Trained U-Net models (not on this computer) |

`outputs/` is not uploaded to GitHub (too large, and it can be regenerated by the commands
above).
