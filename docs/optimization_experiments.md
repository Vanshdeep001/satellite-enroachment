# Step 5 — Optimisation, Initialisation, Normalisation, Regularisation & Generalisation

All experiments use the from-scratch NumPy MLP (`src/fundamentals/mlp.py`). This keeps every
gradient and update rule visible, and a full sweep runs in about 9 minutes on a CPU.

```bash
python -m src.fundamentals.experiments            # full run, 3 seeds -> outputs/fundamentals/experiments/
python -m src.fundamentals.experiments --quick    # smoke test (seconds)
```

## Setup

| Item | Value |
|---|---|
| Task | 4-class land-cover patch classification: water, vegetation, built-up, bare soil |
| Data | Synthetic 10×10 RGB patches (`landcover_toy.py`) with random orientation, illumination ×[0.5, 1.5], per-band haze and sensor noise |
| Splits | 240 train / 600 val / 1200 test, each generated from an independent random seed so no patch appears in two splits |
| Preprocessing | Per-band standardisation fitted on the training split only |
| Base model | 300 → 64 → 64 → 4, ReLU, He init, softmax cross-entropy |
| Base training | Momentum 0.9, lr 0.01, batch 16, 100 epochs |
| Seeds | 3 per configuration, reported as mean ± std |
| Selection rule | Hyperparameters chosen on validation loss only; test accuracy is reported, never used for selection |

> The synthetic data is a controlled stand-in so the ablations can run without the
> LEVIR-CD download. These numbers show *mechanisms*. They are not claims about
> real-imagery accuracy. Chance level is 25%.

## 1. Optimisers (`fig_optimizers.png`)

Each optimiser gets its own learning-rate grid (1e-3 … 0.3), trained for 60 epochs.

| Optimiser | Selected lr | Updates / epoch | Min val loss | Test acc | Runtime / run | Stable lr range |
|---|---|---|---|---|---|---|
| Batch GD (full batch) | 0.3 * | 1 | 0.829 | 0.666 ± 0.005 | 0.4 s | ≥ 0.03 (slow below) |
| SGD (batch 1) | 0.003 | 240 | 0.858 | 0.654 ± 0.010 | 8.1 s | ≤ 0.01 (diverges ≥ 0.03) |
| Mini-batch SGD (16) | 0.03 | 15 | 0.856 | 0.653 ± 0.011 | 0.6 s | 0.003 – 0.1 |
| Momentum 0.9 (16) | 0.003 | 15 | 0.855 | 0.652 ± 0.006 | 0.9 s | ≤ 0.03 |
| Nesterov 0.9 (16) | 0.003 | 15 | 0.853 | 0.653 ± 0.006 | 1.1 s | ≤ 0.03 |

\* 0.3 is the top of the grid, so batch GD might improve further with a larger step.

**Findings**
- Once each optimiser has a tuned lr, all five land within ~1.5 points of each other. On this
  problem the optimiser changes *how fast and how safely* the model fits, not how well it
  generalises.
- Momentum and Nesterov have an effective step of about lr / (1 − β) = 10 × lr. They therefore
  diverge at lr ≥ 0.1, which plain mini-batch SGD still tolerates. A tenfold-smaller lr gives
  them the same behaviour.
- SGD with batch size 1 makes 240 noisy updates per epoch, is about 13× slower in wall-clock
  time, and has the narrowest stable lr range.
- On an ill-conditioned quadratic (curvatures 1 and 25, 60 steps, lr 0.035), the final loss is
  GD **5.6e-1**, Momentum **1.3e-4** and Nesterov **4.7e-6**. GD crawls along the flat direction.
  Momentum builds speed but overshoots. Nesterov's look-ahead damps the overshoot.

## 2. Initialisation (`fig_initialization.png`)

Six hidden layers of width 64, trained for 60 epochs. The middle columns give the standard
deviation of the activations at layer 1 → layer 6, measured at initialisation.

| Activation / init | Activation std, L1 → L6 | Test acc |
|---|---|---|
| tanh / zeros | 0 → 0 | 0.250 (chance) |
| tanh / N(0, 0.01²) | 0.17 → 3e-7 (vanishes) | 0.250 (chance) |
| tanh / N(0, 1) | 0.97 → 0.95 (saturated) | 0.539 ± 0.010 |
| tanh / Xavier | 0.65 → 0.28 | 0.613 ± 0.011 |
| tanh / He | 0.70 → 0.56 | 0.623 ± 0.008 |
| relu / zeros | 0 → 0 | 0.250 (chance) |
| relu / N(0, 0.01²) | 0.11 → 0 (vanishes) | 0.250 (chance) |
| relu / N(0, 1) | 10.7 → 46 506 (explodes) | diverged, 3/3 seeds |
| relu / Xavier | 0.75 → 0.09 (decays) | 0.657 ± 0.010 |
| relu / He | 0.87 → 0.66 (preserved) | 0.652 ± 0.001 |

**Findings**
- **Zero init:** every unit in a layer computes the same thing, so there is no symmetry
  breaking. With ReLU the gradients are exactly zero and nothing ever learns.
- **Too-small init:** the signal shrinks roughly 10× per layer, so both the forward signal and
  the gradients vanish.
- **Too-large init:** tanh saturates at ±1, so its derivative is close to 0 and learning is
  slow. ReLU explodes roughly 5× per layer until the loss becomes NaN.
- **He init** (variance 2 / fan_in) keeps the ReLU signal roughly constant with depth.
  **Xavier** (variance 2 / (fan_in + fan_out)) slowly decays it, but at depth 6 it still trains
  to the same accuracy.

## 3. Input normalisation (`fig_normalization.png`)

| Configuration | Train acc | Test acc |
|---|---|---|
| Raw digital numbers 0–255, lr 0.01 | 0.308 | 0.246 (diverged to chance) |
| Raw DN 0–255, lr 1e-5 | 1.000 | 0.487 ± 0.023 |
| Raw DN + BatchNorm, lr 0.01 | 1.000 | 0.608 ± 0.023 |
| Per-band standardised, lr 0.01 | 1.000 | **0.662 ± 0.002** |
| Standardised + BatchNorm, lr 0.01 | 0.997 | 0.613 ± 0.017 |

**Findings**
- Unscaled 0–255 inputs make the first-layer gradients about 255× too large. The base lr
  diverges, and the lr that survives (1e-5) is far too small for the deeper layers.
- BatchNorm rescues raw inputs almost completely, because it re-standardises after the first
  linear layer.
- On already-standardised inputs, BatchNorm *hurt* by about 5 points. With batch 16 and a
  shallow net, the noise in its per-batch statistics costs more than its conditioning benefit
  gives back. Standardising the inputs is the essential step here; BatchNorm is not.

## 4. Regularisation (`fig_regularization.png`)

| Configuration | Train acc | Test acc |
|---|---|---|
| None | 1.000 | 0.662 ± 0.002 |
| L2 1e-4 | 1.000 | 0.663 ± 0.001 |
| L2 1e-3 | 1.000 | 0.664 ± 0.003 |
| L2 1e-2 | 1.000 | 0.667 ± 0.004 |
| Dropout 0.2 | 1.000 | 0.667 ± 0.006 |
| Dropout 0.5 | 0.996 | 0.654 ± 0.005 |
| L2 1e-3 + dropout 0.2 | 1.000 | 0.660 ± 0.004 |
| Early stopping (best val loss) | 0.847 | 0.646 ± 0.015 |
| Early stopping (best val accuracy) | 0.989 | 0.664 ± 0.006 |

**Findings**
- With 240 patches and 23 684 parameters the network memorises the training set (100% train
  accuracy). Validation loss reaches its minimum at about epoch 2 and then climbs, as shown in
  `fig_generalization.png`, right panel.
- Weight-space regularisers changed test accuracy by at most +0.5 points. They shrink the
  weights but cannot supply the missing invariances.
- Early stopping on *validation loss* gave lower *accuracy* than training to the end. The loss
  minimum comes before the accuracy peak, because the over-confident late model has high loss
  but still ranks classes correctly. A follow-up run monitoring validation *accuracy*
  (`train(..., monitor='val_metric')`) recovered the full 0.664 and also had lower variance
  across seeds. Early stopping should use the metric that is actually reported. Its validation
  accuracy (0.678) is optimistically biased because that split chose the epoch; the test
  figure is the unbiased estimate.

## 5. Augmentation (`fig_augmentation.png`)

| Configuration | Train acc | Test acc | Δ vs none |
|---|---|---|---|
| None | 1.000 | 0.662 ± 0.002 | — |
| Gaussian jitter (σ = 0.3) | 1.000 | 0.665 ± 0.005 | +0.3 |
| Dihedral (rot90 × flip) | 0.997 | 0.721 ± 0.019 | +5.9 |
| Dihedral + jitter | 0.983 | **0.743 ± 0.008** | **+8.1** |

**Finding:** in this study augmentation is the strongest generalisation lever. Overhead
imagery has no canonical "up", so the eight rotations and flips are exact label-preserving
symmetries, and adding them is like having 8× the data. Noise alone does little. Combined with
rotations, it adds a further +2 points.

## 6. Generalisation vs data size (`fig_generalization.png`)

| Training patches | Plain test acc | Dihedral + L2 test acc | Plain train − test gap |
|---|---|---|---|
| 40 | 0.571 | 0.605 | 0.43 |
| 80 | 0.606 | 0.672 | 0.39 |
| 160 | 0.654 | 0.698 | 0.35 |
| 320 | 0.669 | 0.767 | 0.33 |
| 640 | 0.692 | 0.803 | 0.31 |
| 1280 | 0.736 | **0.854** | 0.26 |

**Finding:** the plain model reaches 100% train accuracy at every size, so its gap shrinks only
because test accuracy rises with more data. Augmentation + L2 is worth roughly a 4–8× larger
dataset; for example, 320 augmented patches beat 1280 plain ones.

## Recommendations carried into the deep models (Steps 7–10)

1. Standardise each band with statistics from the training split only. Keep BatchNorm in the
   CNNs, where it is placed per channel and batch statistics are more reliable.
2. Use He init for ReLU networks.
3. Tune the learning rate per optimiser. For momentum-based methods, scale lr by (1 − β)
   relative to SGD.
4. Make rotation and flip augmentation the default for overhead imagery. The existing Siamese
   U-Net pipeline already uses synchronised bitemporal flips and rotations
   (`src/experiments/augmentations.py`).
5. Do early stopping and checkpoint selection on the reported metric (F1 / IoU), not on the
   loss.

## Output files

`outputs/fundamentals/experiments/`
- `results.json`: every configuration with its aggregated metrics and mean curves.
- `summary.csv`: one row per configuration.
- `fig_landcover_samples.png`, `fig_optimizers.png`, `fig_initialization.png`,
  `fig_normalization.png`, `fig_regularization.png`, `fig_augmentation.png`,
  `fig_generalization.png`: the figures referenced above.
