# Experiment Plan and Ablation Studies

**Project Title:** Satellite Land Use Change and Encroachment Detector  
**Course:** Deep Learning — Project-Based Learning (PBL) / Project 50  
**Status:** Planned Specifications (No models trained; empirical execution scheduled for Phases 2–4)  

---

## 1. Experimental Methodology Overview

To establish academic rigor, this project follows an empirical evaluation methodology:
1. **Unsupervised Lower Bound:** Establish a traditional baseline using pixel-differencing without deep learning.
2. **Supervised Deep Learning Benchmark:** Train the proposed Siamese U-Net under controlled hyperparameters.
3. **Systematic Ablations:** Isolate specific design choices (fusion mechanics, loss functions, threshold sensitivity) to empirically validate architectural contributions.
4. **Independent Test Partition:** All reported evaluation metrics are computed exclusively on the designated, unseen LEVIR-CD test split ($2,048$ patches).

---

## 2. Planned Phase-by-Phase Execution Schedule

| Phase | Target Deliverables | Hardware / Execution Mode | Dependencies |
|---|---|---|---|
| **Phase 2** | • Automatic LEVIR-CD download & integrity check<br>• Preprocessing: non-overlapping $256 \times 256$ patch tiling<br>• Traditional Baseline: Spectral Difference + Otsu thresholding + Morphology<br>• Baseline evaluation on test set | Local Apple Silicon / CPU / `.venv311` | Network access for dataset download |
| **Phase 3** | • PyTorch implementation of Siamese U-Net<br>• Dataloaders with augmentation (rotations, flips)<br>• Training loop with AdamW, Cosine Scheduler, and Composite BCE+Dice Loss<br>• Validation checkpointing (saving best IoU weights) | Apple Silicon MPS (`torch.device("mps")`) | Phase 2 preprocessed data |
| **Phase 4** | • Full inference on unseen test set ($2,048$ patches)<br>• Quantitative metrics table: Precision, Recall, F1, IoU<br>• Generation of visual change maps (TN, TP, FP, FN overlays)<br>• Execution of planned ablation studies | Apple Silicon MPS / CPU | Phase 3 trained weights |
| **Phase 5** | • Geospatial area estimation module ($\text{m}^2$ calculation)<br>• Verified restricted boundary polygon intersection (GeoJSON)<br>• Interactive demonstration interface (Gradio web app / CLI) | Local workstation | Phase 4 model |
| **Phase 6** | • Final academic report compilation<br>• Viva presentation deck and oral examination drill<br>• Final code polish and repository archiving | Markdown / Documentation | All phases |

---

## 3. Training Configuration & Hyperparameters (Planned for Phase 3)

The deep learning model is configured for stable convergence and memory safety on an Apple Silicon Mac:

| Hyperparameter | Planned Value | Justification |
|---|---|---|
| **Input Patch Size** | $256 \times 256 \times 3$ | Standard remote sensing patch dimensions; balances VRAM and receptive field |
| **Batch Size** | $16$ (or $8$ if memory constrained) | Fits comfortably in Apple unified memory without swap thrashing |
| **Optimizer** | `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 10^{-4}$) | Superior weight regularization over classical Adam |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | Standard stable learning rate for U-Net backbones |
| **LR Scheduler** | `CosineAnnealingLR` ($T_{\max} = 30$, $\eta_{\min} = 10^{-6}$) | Smooth decay preventing abrupt gradient shocks |
| **Maximum Epochs** | $30$ epochs (with Early Stopping, patience $= 6$) | Sufficient for convergence on LEVIR-CD without overfitting |
| **Data Augmentation** | Random horizontal/vertical flip ($p=0.5$), Random $90^\circ$ rotation | Enforces rotational invariance for satellite aerial perspectives |
| **Loss Function** | $0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$ | Directly counters extreme foreground/background class imbalance |
| **Device Target** | `torch.device("mps")` | Utilizes Apple Neural Engine & Metal Performance Shaders |

---

## 4. Planned Experiments & Ablation Studies

### Experiment 1: Classical Baseline vs. Proposed Deep Learning Model
* **Objective:** Quantify the performance leap achieved by replacing handcrafted spectral heuristics with hierarchical deep learning representations.
* **Setup:**
  - *Model A (Baseline):* Pixel-wise Euclidean distance in RGB color space + Otsu adaptive thresholding + Morphological opening/closing.
  - *Model B (Proposed):* Siamese U-Net with differential skip fusion trained with Composite Loss.
* **Expected Outcome:** Deep learning model will drastically reduce false positives caused by grass drying, paving color shifts, and building shadows, increasing IoU by $\ge 35\%$.

---

### Experiment 2: Temporal Feature Fusion Strategy Ablation
* **Objective:** Determine the optimal mechanism for integrating bitemporal image information.
* **Configurations to Compare:**
  1. **Early Fusion (EF-UNet):** Bitemporal images are stacked at input ($C = 6$) and passed into a standard single-stream U-Net.
  2. **Siamese Difference (Siam-Diff):** Dual encoders with weight sharing; skip connections pass absolute feature difference $|F_{T_1}^{(l)} - F_{T_2}^{(l)}|$ to the decoder.
  3. **Siamese Concatenation (Siam-Conc):** Dual encoders with weight sharing; skip connections pass concatenated feature maps $[F_{T_1}^{(l)}, F_{T_2}^{(l)}]$ to the decoder.
* **Hypothesis:** Siamese Difference will outperform Early Fusion because independent encoder branches preserve temporal feature integrity prior to comparison.

---

### Experiment 3: Loss Function Sensitivity Ablation
* **Objective:** Evaluate how objective function selection mitigates severe class imbalance ($< 15\%$ changed pixels).
* **Configurations to Compare:**
  1. **Pure BCE Loss:** Standard binary cross-entropy.
  2. **Pure Soft Dice Loss:** Optimizing solely for spatial overlap.
  3. **Composite BCE + Dice Loss ($\alpha = 0.50$):** Hybrid objective function.
* **Hypothesis:** Pure BCE will suffer from high false negatives (under-predicting changes); Pure Dice will exhibit unstable early gradients; Composite Loss will achieve the highest balanced F1 and IoU scores.

---

### Experiment 4: Decision Threshold Sensitivity Analysis
* **Objective:** Analyze model precision-recall trade-offs across varying probability classification thresholds $\tau \in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]$.
* **Application Context:** In strict environmental protection (e.g., zero tolerance for illegal deforestation), high recall ($\tau = 0.3$) is prioritized. In automated municipal alerting, high precision ($\tau = 0.7$) minimizes investigator fatigue.

---

## 5. Result Logging and Traceability Protocol

To prevent accidental data fabrication and ensure academic reproducibility:
- Every training and evaluation run will log metrics into structured JSON and CSV formats under `outputs/metrics/`.
- Generated visual comparisons will be saved under `outputs/change_maps/` with deterministic file naming:
  `outputs/change_maps/{sample_id}_comparison_grid.png`
  (displaying: Image $T_1$, Image $T_2$, Ground Truth, Baseline Prediction, U-Net Prediction, and Error Heatmap).
- Random seeds for PyTorch, NumPy, and Python standard libraries will be pinned explicitly (`seed = 42`).

