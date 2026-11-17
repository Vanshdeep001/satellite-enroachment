# Risks, Limitations, and Ethical Safeguards

**Project Title:** Satellite Land Use Change and Encroachment Detector  
**Course:** Deep Learning — Project-Based Learning (PBL) / Project 50  
**Phase:** Phase 1 — Comprehensive Risk Analysis  

---

## 1. Technical and Algorithmic Risks

### 1.1 Image Misregistration & Geometric Misalignment
* **Risk:** In bitemporal satellite observation, if images $T_1$ and $T_2$ are not sub-pixel aligned, static objects (e.g., unchanged building rooftops or linear roads) exhibit an apparent spatial displacement.
* **Impact:** The model interprets geometric offsets as artificial "edge changes", generating thin linear false-positive fringes around static structures.
* **Mitigation:**
  - LEVIR-CD comes pre-coregistered by remote sensing specialists.
  - In our architecture, the Siamese encoder downsamples high-frequency spatial noise through pooling, extracting invariant semantic representations that are naturally more resilient to minor sub-pixel jitters than raw pixel subtraction.

### 1.2 Seasonal, Illumination, and Phenological Radiometric Shifts
* **Risk:** Image pairs acquired in different seasons (e.g., lush summer vs. arid winter) exhibit stark radiometric differences in grass, soil moisture, and tree foliage. Furthermore, differing sun elevation angles cast long building shadows in winter and minimal shadows in summer.
* **Impact:** Classical differencing algorithms fail catastrophically under these conditions, flagging seasonal vegetation changes as building developments.
* **Mitigation:**
  - Deep CNN encoders learn high-level structural semantics (rectilinear corners, roof textures, wall boundaries) rather than raw spectral reflectance.
  - Data augmentation includes random photometric color jittering and rotation to enforce invariance against lighting shifts.

### 1.3 Severe Class Imbalance
* **Risk:** In typical remote sensing scenes, changed pixels occupy $< 15\%$ of the image, while unchanged background occupies $> 85\%$.
* **Impact:** A standard neural network trained with naive Binary Cross-Entropy (BCE) can achieve $> 85\%$ accuracy simply by predicting "no change" across all pixels.
* **Mitigation:**
  - Implementation of a composite loss function combining BCE with Soft Dice Loss ($\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$).
  - Evaluation relies strictly on balanced metrics (IoU, F1-Score, Precision, Recall) rather than raw classification accuracy.

---

## 2. Hardware and Resource Constraints

### 2.1 Disk Space Availability on Student Laptop
* **Risk:** The local machine has approximately $\approx 11\text{ GB}$ of free storage. Storing multiple uncompressed datasets, large checkpoints, and environment packages could exhaust available disk space.
* **Impact:** System halts, failed checkpoint saves, or macOS operating system instability.
* **Mitigation:**
  - Strict storage budget maintained: LEVIR-CD compressed archive ($\approx 400\text{ MB}$), cropped training patches ($\approx 1.2\text{ GB}$), lightweight model checkpoints ($\approx 50$–$80\text{ MB}$ each, saving only the top-2 checkpoints).
  - All large binary files (`.pth`, `data/raw/`, `outputs/`) are excluded from git version tracking via `.gitignore`.
  - Temporary download archives will be pruned post-extraction in Phase 2.

### 2.2 Memory Constraints on Apple Silicon (Unified Memory / MPS)
* **Risk:** Large batch sizes or high-resolution inputs ($1024 \times 1024$) can trigger out-of-memory (OOM) errors on Apple Silicon Metal Performance Shaders.
* **Impact:** Script crashes during backpropagation.
* **Mitigation:**
  - Mandatory patch tiling to $256 \times 256$ pixels before feeding images to the dataloader.
  - Conservative batch size set to $16$ (with automated fallback to $8$ or gradient accumulation if memory pressure is detected).

---

## 3. Methodological and Generalization Limitations

### 3.1 Geographic and Architectural Domain Gap
* **Limitation:** LEVIR-CD is sourced exclusively from Texas, United States, characterized by detached suburban houses, wide avenues, and distinct roof geometry.
* **Consequence:** The model cannot be assumed to generalize zero-shot to dense organic settlements, informal slums, rural adobe dwellings, or high-rise urban canyons in India or other global regions without fine-tuning.
* **Academic Note:** This limitation is explicitly acknowledged in the PBL documentation. Domain adaptation and fine-tuning on regional Indian satellite data are designated as planned future extensions.

### 3.2 Optical Spectral Limitation (Cloud Cover & Nocturnal Blindness)
* **Limitation:** Optical satellite sensors cannot penetrate dense cloud cover or acquire imagery at night.
* **Consequence:** Monitoring is temporally restricted to cloud-free daylight passes.
* **Future Work:** Multi-modal sensor fusion incorporating Synthetic Aperture Radar (SAR, such as Sentinel-1), which penetrates clouds and weather disturbances.

---

## 4. Legal, Ethical, and Responsible AI Safeguards

### 4.1 Strict Disclaimers Regarding Legal Encroachment
> [!CAUTION]
> **A machine learning model detects visual morphological alterations; it cannot determine legality, ownership, or regulatory authorization.**

* **The Problem:** Erroneously labeling a detected structural modification as "illegal encroachment" could lead to wrongful administrative harassment, reputational damage, or legal disputes.
* **Mandatory System Safeguards:**
  1. **Neutral Semantic Nomenclature:** The core computer vision pipeline outputs only neutral descriptions: *"Detected Structural Modification"* or *"Building Footprint Addition"*.
  2. **Encroachment Gating:** The term *"Potential Encroachment Alert"* is **only generated** when an authenticated, legally ratified vector boundary file (GeoJSON/Shapefile) demarcating a protected/prohibited zone is explicitly provided to the spatial overlay engine.
  3. **Mandatory Human-in-the-Loop Verification:** All alerts are designated as *"Advisory Decision-Support Indicators for Field Survey"*. Automated punitive action based solely on model output is explicitly prohibited.

### 4.2 Cadastral Data Integrity
* Vector boundary shapefiles used for encroachment cross-referencing must be sourced exclusively from authorized land revenue / municipal surveying authorities. Unofficial or unverified boundaries must not be used for compliance decisions.

### 4.3 Data Privacy & Surveillance Ethics
* While commercial high-resolution satellite imagery ($0.5\text{ m/px}$) resolves building perimeters and vehicular outlines, it does not resolve individual human faces or vehicle license plates, maintaining individual privacy while monitoring macroscopic land governance.

