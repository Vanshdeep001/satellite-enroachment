# Step 6 — Hebbian, Competitive, Error-Correction and Energy-Based Learning

The classical learning rules in this step are implemented from scratch in NumPy
(`src/fundamentals/classical_learning.py`). Each is demonstrated on land-cover data, or on a
benchmark small enough to compute exactly.

```bash
python -m src.fundamentals.classical_demo          # ~70 s -> outputs/fundamentals/classical/
python -m pytest tests/test_fundamentals_classical.py
```

## Conceptual comparison

| Paradigm | Rule implemented | Signal used | Supervised? | What it optimises | What it learns here | Typical failure |
|---|---|---|---|---|---|---|
| Hebbian | `Δw = η y x` | Pre × post correlation | No | Nothing bounded (variance grows without limit) | Direction of the 1st principal axis | Weight norm explodes |
| Modified Hebbian (Oja) | `Δw = η y (x − y w)` | Correlation + local decay | No | Variance subject to ‖w‖ = 1 | Unit-norm 1st principal axis | Slow when eigenvalues are close |
| Generalised Hebbian (Sanger) | `ΔW = η (y xᵀ − LT(y yᵀ) W)` | Correlation with the earlier units deflated | No | Top-k principal subspace, in order | 4 PCs of 300-D patches | Step size capped by λ_max, so low PCs learn slowly |
| Competitive (WTA) | `Δw* = η (x − w*)` | Winner only | No | Quantisation error | Colour/texture prototypes | Dead units, local minima |
| Competitive + conscience | WTA with bias `C (1/k − p_j)` | Winner + win frequency | No | Quantisation error with equal usage | All prototypes active | `C` must match the distance scale |
| Kohonen SOM | `Δw_j = η h(j, j*) (x − w_j)` | Winner + grid neighbours | No | Quantisation error + topology | 2-D ordered land-cover map | Neighbourhood schedule needs tuning |
| Error-correction (perceptron) | `w ← w + η t x` on mistakes | Sign error | Yes | Number of mistakes | Linear separator | Never converges unless separable |
| Error-correction (LMS / delta) | `Δw = η (t − w·x) x` | Real-valued error | Yes | Mean squared error (convex) | Least-squares separator | Squared error, not misclassification |
| Energy-based (Hopfield) | Hebbian / projection storage, `s_i ← sign(W_i s)` | Local field | Stores, no labels | Energy `−½ sᵀWs` (monotone descent) | Footprint shapes as attractors | Capacity ≈ 0.138 N; spurious states |
| Energy-based (Boltzmann / RBM) | CD-k or exact `⟨vh⟩_data − ⟨vh⟩_model` | Data vs model statistics | No (generative) | Log-likelihood of `p(v) ∝ e^{−F(v)}` | Distribution of bars/stripes | Partition function intractable at scale; CD gradient is biased |

Backpropagation (Steps 4–5) differs from every rule above in one way. It uses a *global* error
signal carried backwards through the layers. Every rule in this step uses only quantities
available at the synapse, or one sweep of model sampling in the Boltzmann case.

## Results (seed 0)

### Hebbian family (`fig_hebbian.png`)

| Check | Result |
|---|---|
| Plain Hebb after 2000 samples | ‖w‖ = 3.7 × 10⁵ (grows without bound); angle to PC1 0.27° |
| Oja after 2000 samples | ‖w‖ = 1.0001; angle to PC1 0.27° |
| Sanger GHA, 4 components of 300-D standardised patches, vs our Jacobi SVD | \|cos\| = 0.997, 1.000, 1.000, 0.984 |
| Variance explained by those 4 components | 49.9%, 7.6%, 5.3%, 1.8% |

The first component is overall brightness, which reflects the illumination nuisance factor.
The next two are colour contrasts that separate vegetation and water from soil and roofs. The
step size has to stay below about 1/λ_max (λ_max ≈ 147 here). Components with much smaller
eigenvalues therefore converge slowly, which is why the 4th component is the least aligned.

### Competitive learning (`fig_competitive.png`)

Features: mean R, G, B and luminance texture per patch (z-scored), 1600 patches. Labels are
used only to *score* the clusters.

| Variant (k = 8) | Active units | Purity | Quantisation error |
|---|---|---|---|
| Init from random samples | 8 / 8 | 0.661 | 0.763 |
| Poor init (all prototypes in one corner) | 3 / 8 | 0.458 | 1.430 |
| Poor init + conscience (C = 30) | 8 / 8 | 0.703 | 0.776 |
| Kohonen SOM 6 × 6 | 36 units | 0.789 | 0.395 (topographic error 0.034) |

- Without conscience, the unit nearest the data wins every time and the other prototypes never
  move.
- The conscience bias first had to be scaled up. At C = 10, only 5/8 units revived, because the
  bias (≤ C/k) was far smaller than the squared distance from the corner to the data.
- The SOM orders itself spatially: water in one corner, then vegetation, then bare soil, with
  built-up in the bright corner.

### Error-correction learning (`fig_error_correction.png`)

| Data | Perceptron | LMS / delta rule |
|---|---|---|
| Separable Gaussian blobs | Converged in 2 epochs | — |
| XOR | Never converges (≥ 1 mistake in every one of 100 epochs) | — |
| Water vs bare soil (blue − red, brightness), held-out half | Does not converge; last iterate 84.7% test | 86.0% test, MSE settles at 0.376 |

Real patch features overlap, so the perceptron keeps changing its boundary forever. LMS
minimises a convex squared error, so it settles. This is why the delta rule, extended through
hidden layers, became backpropagation.

### Hopfield network (`fig_hopfield.png`)

| Check | Result |
|---|---|
| 4 footprints (L-block, courtyard, twin slabs, junction), 15% of pixels flipped, Hebbian storage | 25% exact recall: the 3 overlapping shapes collapse into a common spurious "U" attractor |
| Same, pseudo-inverse (projection) storage | 100% exact recall |
| Mean pairwise overlap between the stored footprints | 0.26 (random patterns ≈ 0) |
| Energy during every recall | Never increases (verified across all runs) |
| Random-pattern capacity, N = 100, 10% noise | 100% up to P = 10; 77% at P = 14 (0.14 N); 29% at P = 20; 0% at P = 30 |

The measured capacity breakdown sits at the theoretical 0.138 N. The footprint result shows
why Hebbian storage alone is not enough for structured imagery: real shapes are correlated.

### Boltzmann learning, RBM (`fig_boltzmann.png`)

Data: the 14 distinct 3×3 bars-and-stripes images. RBM with 9 visible and 16 hidden units,
3000 full-batch epochs, lr 0.5. Because there are only 2⁹ = 512 visible states, the partition
function, and so the *exact* log-likelihood, can be computed by enumeration.

| Training | Final log-likelihood | Best | Probability mass on the 14 patterns |
|---|---|---|---|
| CD-1 | −3.30 | −3.01 | 97.9% |
| CD-10 | −3.11 | −2.74 | 98.0% |
| Exact ML gradient | **−2.76** | −2.74 | 98.4% |
| *Optimum (uniform over the 14 patterns)* | *−2.64* | | *100%* |

- CD-k replaces the model expectation with k Gibbs steps started from the data. That makes the
  gradient biased, and noisier as k shrinks. The exact gradient climbs smoothly and ends
  closest to the optimum.
- All 24 Gibbs samples from the exact-ML model are valid bars or stripes.
- This is the energy-based counterpart of the generative models in Step 9 (VAE/GAN). Those
  avoid the intractable partition function by learning a decoder instead.

## Notes and limits

- All of these demos run on the synthetic land-cover patches or on toy benchmarks. They
  illustrate mechanisms, not production accuracy.
- Hopfield footprints are hand-drawn 10×10 masks. Real building masks from LEVIR-CD can be
  substituted once the dataset is on disk.
