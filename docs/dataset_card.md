# Dataset Card — LEVIR-CD (Building Change Detection)

**Dataset Name:** LEVIR-CD (Large-scale Remote Sensing Building Change Detection Dataset)  
**Curators:** Hao Chen, Zipeng Qi, and Zhenwei Shi (Image Processing Center, Beihang University)  
**Official Portal:** [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)  
**Primary Reference:** Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing, 12(10), 1662. [DOI: 10.3390/rs12101662](https://doi.org/10.3390/rs12101662)  
**Status in Project:** ✅ Fully Acquired, Extracted, Audited & Verified (Phase 2 Complete)

---

## 1. Dataset Overview and Purpose

LEVIR-CD is a dedicated benchmark dataset for **bitemporal building change detection** derived from very high-resolution (VHR) optical remote sensing imagery. The dataset was created to address the scarcity of large, rigorously annotated benchmarks for remote sensing change detection, providing a standardized evaluation platform for deep learning models.

* **Sensor Source:** Google Earth optical aerial / satellite photography.
* **Geographical Coverage:** 20 distinct geographic zones across multiple growing municipalities in Texas, United States (including Austin, Lakeway, Bee Cave, Buda, Kyle, and surrounding suburban areas).
* **Time Span:** Bitemporal image pairs acquired across intervals ranging from **5 to 14 years** (e.g., 2002 to 2016). This extended temporal gap ensures that the dataset captures significant land use transformations alongside challenging environmental noise (seasonal foliage variations, soil dryness, solar azimuth angle shifts, and sensor viewing discrepancies).

---

## 2. Technical Specifications

| Characteristic | Specification | Notes for Implementation |
|---|---|---|
| **Spatial Resolution (GSD)** | **$0.5\text{ meters per pixel}$** | Very High Resolution (VHR); each pixel represents $0.25\text{ m}^2$ on ground |
| **Spectral Bands** | $3$ Channels (Red, Green, Blue) | Standard $8$-bit optical RGB format |
| **Image Dimensions** | $1024 \times 1024\text{ pixels}$ per patch | Sliced to $256 \times 256$ non-overlapping patches for model training |
| **Total Image Pairs** | $637$ pairs ($1274$ images total) | Slices into $10,192$ patches ($256 \times 256$) |
| **Annotated Instances** | Over $31,333$ individual change buildings | Includes large complexes down to small residential extensions |
| **Annotation Format** | Pixel-level binary raster masks (`.png`) | $255$ or $1 = \text{Change}$; $0 = \text{Unchanged Background}$ |
| **File Structure** | Three paired subdirectories: `A/`, `B/`, `label/` | `A/` = Pre-change ($T_1$), `B/` = Post-change ($T_2$), `label/` = Ground Truth Mask |

---

## 3. Official Dataset Splits

The authors established a strict benchmark partition to ensure fair comparison across peer-reviewed literature:

### 3.1 Original Scale ($1024 \times 1024$ image pairs)
- **Training Set:** $445$ image pairs ($\approx 70\%$)
- **Validation Set:** $64$ image pairs ($\approx 10\%$)
- **Test Set:** $128$ image pairs ($\approx 20\%$)
- *Total:* $637$ image pairs

### 3.2 Tiled Benchmark Scale ($256 \times 256$ patches without overlap)
- **Training Patches:** $7,120$ patches
- **Validation Patches:** $1,024$ patches
- **Test Patches:** $2,048$ patches
- *Total:* $10,192$ patches

---

## 4. Types of Changes Captured

LEVIR-CD focuses on **structural anthropogenic building transformations**:
1. **New Building Construction:** Transition from bare land, farmland, or cleared forest to single-family villas, warehouse complexes, or commercial offices.
2. **Suburban Residential Growth:** Massive horizontal expansion of residential subdivisions, including new road networks accompanied by rooftops.
3. **Building Expansion / Structural Additions:** Extensions to preexisting building footprints.
4. **Demolition / Decline:** Structures present in $T_1$ that have been cleared or demolished by $T_2$.

---

## 5. Dataset Strengths and Limitations

### 5.1 Strengths
* **Sub-meter Resolution ($0.5\text{ m/px}$):** Allows crisp boundary extraction and accurate physical area calculation ($\text{m}^2$).
* **High Annotation Fidelity:** Manually annotated and verified by remote sensing specialists.
* **Realistic Environmental Noise:** Captures real-world seasonal differences (green vs. parched grass), sun angle shifts (elongated shadows), and atmospheric haze.
* **Established Benchmark:** Directly comparable with published results in top remote sensing journals (IEEE TGRS, MDPI Remote Sensing, CVPR EarthVision).

### 5.2 Limitations
* **Geographical Bias:** All imagery is sourced from suburban and peri-urban Texas, USA. Architectural styles predominantly feature detached timber/brick single-family homes and warehouses with pitched/flat roofs. Generalization to dense, organic settlements or high-rise urban clusters in Asia or Europe requires domain adaptation.
* **Binary-Only Semantic Granularity:** Distinguishes only between *"Building Change"* and *"No Change"*. It does not classify secondary land-use categories (e.g., water bodies, agricultural crops, paved roads).
* **Three-Band Optical Constraint:** Lacks multispectral infrared bands (NIR/SWIR for vegetation indices like NDVI) and Synthetic Aperture Radar (SAR) data, making it reliant solely on visible spectrum textures.

---

## 6. Licensing and Academic Usage Conditions

* **License Status:** Freely available for **academic, research, and non-commercial educational purposes**.
* **Commercial Restrictions:** The underlying imagery is sourced via Google Earth; proprietary commercial redistribution without licensing from Google/source providers is prohibited.
* **Attribution Requirement:** Any publication, report, or academic evaluation using LEVIR-CD must formally cite the seminal paper:
  ```bibtex
  @article{chen2020spatial,
    title   = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
    author  = {Chen, Hao and Shi, Zhenwei},
    journal = {Remote Sensing},
    volume  = {12},
    number  = {10},
    pages   = {1662},
    year    = {2020},
    doi     = {10.3390/rs12101662}
  }
  ```

---

## 7. Suitability for This Project

LEVIR-CD is well-suited for our project for three reasons:
1. **Direct Alignment with Encroachment Problem:** Unauthorized encroachment in reserved lands almost universally manifests as structural building erection (huts, warehouses, residential villas, boundary walls). LEVIR-CD's explicit focus on building footprints matches this target scenario.
2. **Standardized Reproducibility:** Its predefined train/val/test splits enable transparent, verifiable academic evaluation during viva and project audits.
3. **Storage Feasibility on Student Workstation:** The compressed dataset archive is $\approx 2.35\text{ GB}$, and uncompressed is $\approx 3.8\text{ GB}$. This fits within the local machine's storage ceiling without exhausting disk space.

---

## 8. Empirical Verification & Audit Results (Phase 2)

As documented in `docs/data_audit_report.md` and `data/audit_summary.json`:
* **Total Verified Pairs:** 637 pairs ($1,911$ files across `train`, `val`, `test`).
* **Pair Correspondence:** 100% matched across $T_1$, $T_2$, and mask. Zero missing files. Zero corrupted files.
* **Image Dimensions:** $1024 \times 1024$ pixels, 3-channel optical RGB.
* **Label Representation:** Anti-aliasing edge artifacts present (`[0, 156, 254, 255]`), binarized in pipeline at $\tau = 127$.
* **Empirical Foreground Ratio:** $4.589\%$ in train, $4.197\%$ in val, $5.094\%$ in test ($\approx 4.65\%$ overall), demonstrating severe class imbalance.


