# Neural-Network Fundamentals Module (Project Steps 3 & 4)

The brief asks for a C/C++/Java mathematical sandbox and neuron/MLP baseline. This project is
Python-only, so both are implemented **from scratch in NumPy** under `src/fundamentals/`.
NumPy arrays are used purely as storage. No PyTorch, no autograd, and no `np.linalg`
routines are used in the implementations. NumPy's routines are used only as the reference
answer in the tests.

## Step 3 — Mathematical sandbox (`tensor_ops.py`, `gradcheck.py`)

| Topic | Function(s) | How it is computed |
|---|---|---|
| Broadcasting | `broadcast_shapes`, `broadcast_to`, `elementwise` | Right-align shapes, pad with 1s, size-1 axes map to index 0 |
| Index arithmetic | `row_major_strides`, `ravel_index`, `unravel_index` | `stride[k] = prod(shape[k+1:])` |
| Reshaping | `reshape` (supports one `-1`) | Copy element with flat offset *k* to `unravel_index(k, new_shape)` |
| Inner products | `inner_product`, `matmul`, `frobenius_norm` | Explicit sums; `C[i,j] = <row_i(A), col_j(B)>` |
| Eigen-decomposition | `jacobi_eigh` | Cyclic two-sided Jacobi rotations on a symmetric matrix |
| SVD | `svd`, `low_rank_approximation` | One-sided (Hestenes) Jacobi: rotate column pairs of A until orthogonal; `s_i = ‖w_i‖`, `u_i = w_i / s_i` |
| Gradient checking | `numerical_gradient`, `gradient_check`, `check_model_gradients` | Central difference `(f(x+ε)−f(x−ε))/2ε`, error `‖a−n‖/(‖a‖+‖n‖)` |

The SVD uses the one-sided form because building `AᵀA` explicitly squares the condition
number and destroyed the orthogonality of U at the 1e-8 level in testing.

## Step 4 — Neurons and MLP (`neurons.py`, `mlp.py`)

- **McCulloch-Pitts neuron:** binary inputs, excitatory or absolutely inhibitory, integer
  threshold, no learning. The module provides AND, OR, NOT and AND-NOT gates, plus XOR built
  from two layers (one unit cannot compute it because XOR is not linearly separable).
- **Sigmoid neuron:** logistic unit trained by gradient descent on BCE, using
  `dL/dw = Xᵀ(ŷ−y)/N`.
- **MLP:** `Dense` layers (init: xavier, he, normal, large or zeros) and activations (step,
  sigmoid, tanh, relu, identity), with backprop written by hand:
  `dW = xᵀG`, `db = ΣG`, `dx = GWᵀ`.
- **Losses:** MSE, BCE-with-logits (stable form) and softmax cross-entropy.
- **Optimizers:** `GradientDescent`, `Momentum` and `Nesterov`, with L2 weight decay applied
  to weights only. Batch GD and SGD are the same optimizer: `train(batch_size=None)` gives
  batch GD and `batch_size=1` gives SGD.
- **Parameter vs hyperparameter tracking:** `ExperimentTracker` stores the hyperparameters
  chosen before training, and per-epoch statistics (norm, mean, std, gradient norm) for every
  learned tensor. It saves both to JSON.

## Running

```bash
python -m src.fundamentals.demo            # figures + summary.json -> outputs/fundamentals/
python -m pytest tests/test_fundamentals_tensor.py tests/test_fundamentals_mlp.py
```

## Verified results (seed 42)

| Check | Result |
|---|---|
| Broadcasting / reshape vs NumPy | identical |
| SVD singular values vs `np.linalg.svd` (96×96 tile) | max abs error 2.4e-13 |
| Central-difference best relative error | 1.4e-11 at ε ≈ 3e-6 |
| Backprop vs numeric gradient (3-layer tanh MLP, softmax CE) | max rel. error 8.7e-10 |
| MP gates incl. 2-layer XOR | all truth tables correct |
| 3-class spiral, MLP [2, 64, 64, 3], Momentum | 100% train / 100% val accuracy, 4,547 learned parameters |
| Unit tests | 45 tests, all passing |

## Figures (`outputs/fundamentals/`)

| File | Content |
|---|---|
| `01_broadcasting.png` | (3,1) + (1,4) → (3,4), every element annotated |
| `02_reshape_row_major.png` | Flat offsets preserved across 2×6, 3×4 and 4×3 |
| `03_svd_low_rank.png` | Rank 1/5/15/40 reconstructions, singular spectrum and cumulative energy |
| `04_gradient_check.png` | Error vs ε (truncation/round-off trade-off), per-parameter MLP check |
| `05_neurons.png` | MP truth tables, sigmoid-neuron loss curve and decision boundary |
| `06_activations.png` | Step, sigmoid, tanh and relu with their derivatives |
| `07_mlp_training.png` | Decision regions, train/val loss and weight-norm trajectories |

The SVD demo uses the first LEVIR-CD tile in `data/*/A/` when one is present, and otherwise
falls back to a synthetic scene (roads and rooftops).
