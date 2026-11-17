# Project Objectives and System Requirements

**Project Title:** Satellite Land Use Change and Encroachment Detector  
**Course:** Deep Learning — Project-Based Learning (PBL) / Project 50  
**Phase:** Phase 1 — Project Planning & Specifications  

---

## 1. Primary Project Aim

The primary aim of this project is to research, design, develop, and empirically evaluate an end-to-end deep learning framework that ingests registered pairs of high-resolution bitemporal satellite images ($T_1$ and $T_2$) and automatically identifies **pixel-level structural changes (specifically building construction and expansion)**. The system quantitatively compares a classical baseline against a deep learning model, measures changed surface area, and conditionally flags candidate unauthorized developments against verified restricted-area vector boundaries.

---

## 2. Specific Technical Objectives

1. **Dataset Pipeline Formulation:** Curate and preprocess the LEVIR-CD benchmark dataset into normalized $256 \times 256$ patches with paired binary ground-truth masks.
2. **Traditional Baseline Establishment:** Implement an unsupervised pixel-differencing baseline (spectral difference with Otsu thresholding and morphological filtering) to serve as a performance floor.
3. **Deep Learning Architecture Implementation:** Design and implement a lightweight, end-to-end trainable **Siamese U-Net** in PyTorch, capable of extracting dual feature representations and fusing bitemporal signals across multiple spatial scales.
4. **Empirical Evaluation & Benchmarking:** Conduct rigorous quantitative evaluation on unseen test data using standard segmentation metrics: Precision, Recall, F1-Score, and Intersection-over-Union (IoU).
5. **Geospatial & Encroachment Integration:** Implement a post-processing module that calculates physical change area (in $\text{m}^2$) using sensor Ground Sampling Distance (GSD), and evaluates spatial intersections with hypothetical verified restricted boundaries.
6. **Interactive Demonstration:** Develop an accessible, lightweight user interface (CLI and/or Web UI via Gradio) for interactive testing and academic demonstration.

---

## 3. Functional and Non-Functional Requirements

### 3.1 Functional Requirements (FR)

* **FR-01 (Dual Image Ingestion):** The system must accept two registered optical raster images ($T_1$, $T_2$) in common formats (PNG, GeoTIFF, JPEG) of identical spatial dimensions.
* **FR-02 (Automatic Preprocessing):** The pipeline must normalize dynamic range ($[0, 255] \to [0.0, 1.0]$), handle channel formats (RGB), and perform tiling/cropping to $256 \times 256$ patches for model inference.
* **FR-03 (Baseline Generation):** The system must execute classical image differencing and output a binary baseline change map without requiring model weights.
* **FR-04 (Deep Learning Inference):** The system must load trained PyTorch weights and execute forward-pass inference to output a continuous change probability map and a discrete thresholded binary mask ($1 = \text{Change}, 0 = \text{No Change}$).
* **FR-05 (Evaluation Engine):** Given ground-truth masks, the system must compute Confusion Matrix entries (TP, FP, TN, FN) and derive Precision, Recall, F1, and IoU.
* **FR-06 (Physical Area Estimation):** The system must calculate changed land area:
  $$\text{Area } (\text{m}^2) = N_{\text{changed pixels}} \times (\text{GSD}_x \times \text{GSD}_y)$$
  (where $\text{GSD} = 0.5\text{ m}$ for LEVIR-CD).
* **FR-07 (Restricted Boundary Overlay):** When a verified GeoJSON or vector polygon representing a restricted zone is provided, the system must compute the geometric intersection between detected change masks and restricted polygons.
* **FR-08 (Responsible Output Flagging):** Flagged areas must be explicitly reported as *"Potential Encroachment Alert (Pending Field Audit)"*.

### 3.2 Non-Functional Requirements (NFR)

* **NFR-01 (Hardware Portability):** The entire training and inference pipeline must execute efficiently on a standard student laptop (specifically Apple Silicon Mac with PyTorch MPS backend or CPU fallback), without requiring external cloud clusters or high-end NVIDIA GPUs.
* **NFR-02 (Memory & Disk Footprint):** Peak RAM usage during inference must remain under $4\text{ GB}$; model checkpoint size must remain under $150\text{ MB}$; overall disk consumption for virtual environment and code must stay within the available $11\text{ GB}$ ceiling.
* **NFR-03 (Inference Latency):** Inference for a single $256 \times 256$ bitemporal pair must take $< 150\text{ ms}$ on Apple Silicon MPS.
* **NFR-04 (Code Maintainability & Reproducibility):** Modular Python codebase adhering to PEP 8 standards, deterministic random seeding for reproducibility, and complete separation between data, source code, and outputs.

---

## 4. Measurable Success Criteria

| Evaluation Dimension | Metric | Traditional Baseline Target | Deep Learning (U-Net) Target | State-of-the-Art Benchmark (STANet Paper) |
|---|---|---|---|---|
| **Detection Quality** | **IoU (Intersection-over-Union)** | $\ge 0.30$ | $\ge 0.70$ | $\approx 0.78$–$0.83$ |
| **Detection Balance** | **F1-Score** | $\ge 0.45$ | $\ge 0.80$ | $\approx 0.87$–$0.90$ |
| **False Alarm Control**| **Precision** | $\ge 0.40$ | $\ge 0.80$ | $\approx 0.88$–$0.92$ |
| **Missed Detection Control**| **Recall** | $\ge 0.50$ | $\ge 0.78$ | $\approx 0.85$–$0.90$ |
| **Compute Efficiency** | **Model Size** | $0\text{ MB}$ (Rule-based) | $\le 100\text{ MB}$ ($\le 25\text{M}$ params) | $> 200\text{ MB}$ |
| **Inference Time** | **Latency ($256 \times 256$)** | $< 10\text{ ms}$ | $< 100\text{ ms}$ (MPS) | $\approx 250\text{ ms}$ |

*Note: The targets above are planned benchmarks for academic evaluation.*

---

## 5. Evaluation Metrics Formulation

In remote sensing building change detection, **changed pixels occupy a small percentage of the total land area** (severe class imbalance: typically $> 80\%$ unchanged background). Accuracy is therefore misleading. The following four metrics are standard:

1. **Precision (Positive Predictive Value):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   *Interpretation:* What fraction of pixels predicted as "new building" are genuine buildings? High precision means minimal false alarms from seasonal lighting or shadow shifts.

2. **Recall (Sensitivity / True Positive Rate):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *Interpretation:* What fraction of actual building constructions did the model successfully capture? High recall guarantees that unauthorized structures are not missed.

3. **F1-Score (Harmonic Mean of Precision and Recall):**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
   *Interpretation:* Comprehensive measure of balanced detection quality under severe class imbalance.

4. **Intersection-over-Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{|\text{Prediction} \cap \text{Ground Truth}|}{|\text{Prediction} \cup \text{Ground Truth}|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
   *Interpretation:* Stringent spatial overlap metric. Strongly penalizes both false positives and boundary fuzziness.

---

## 6. Initial Model Selection Justification

We select a **Siamese U-Net** as the primary deep learning architecture for the following reasons:
1. **Separation of Temporal Feature Representations:** Unlike an Early Fusion network ($6$-channel input) that immediately blends $T_1$ and $T_2$ at the first convolution layer, a **Siamese encoder** processes each timestamp through identical, weight-shared convolutional filters. This guarantees that representations for $T_1$ and $T_2$ reside in an identical feature space before interaction.
2. **Multi-Scale Skip Connections:** High-resolution spatial details (such as building corners and linear edges) that are lost during progressive downsampling are directly recovered by concatenating encoder feature maps into the decoder.
3. **Execution on Resource-Constrained Hardware:** Complex attention models (like STANet or ChangeFormer) require massive GPU memory $\mathcal{O}((HW)^2)$ and prolonged training. A lightweight Siamese U-Net converges quickly on Apple Silicon (M-series MPS backend), making it well-suited for a student lab project while retaining high structural fidelity.

---

## 7. Initial Innovation Hypothesis

**Hypothesis:**  
*"Integrating multi-scale Siamese feature differencing ($|F_{T_1}^{(l)} - F_{T_2}^{(l)}|$) with a composite Loss Function (Binary Cross-Entropy + Soft Dice Loss) inside a U-Net backbone will achieve an $\text{IoU} \ge 0.70$ on the LEVIR-CD benchmark, while remaining within a $100\text{ MB}$ parameter footprint and running natively on local Apple Silicon hardware."*

*Rationale:* 
- Pure BCE loss struggles under extreme class imbalance (models tend to predict the majority "unchanged" class).
- Dice Loss optimizes directly for spatial mask overlap (equivalent to soft IoU).
- Their linear combination ($\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$) provides stable pixel-level gradients early in training and fine-grained boundary refinement in later epochs.

