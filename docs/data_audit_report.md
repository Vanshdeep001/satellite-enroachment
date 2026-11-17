# Data Audit Report — LEVIR-CD Benchmark Dataset

**Project:** Satellite Land Use Change and Encroachment Detector  
**Course:** Deep Learning PBL / Project 50  
**Phase:** Phase 2 — Dataset Acquisition, Inspection and Preparation  
**Date of Audit:** November 2, 2026  
**Status:** ✅ Fully Acquired, Extracted, Audited & Verified  

---

## 1. Executive Summary

In accordance with Phase 2 requirements, the official **LEVIR-CD** benchmark dataset was acquired from its primary source, systematically extracted into training, validation, and test partitions, and programmatically audited for structural integrity, correspondence, pixel statistics, and class imbalance.

Zero corrupted images and zero missing counterparts were detected across all partitions. Ground truth masks were verified to contain anti-aliased edge values (`[0, 156, 254, 255]`), necessitating a definitive thresholding step ($\tau = 127$) in the preprocessing pipeline. Across all sets, change pixels account for **$4.19\% - 5.09\%$** of total pixel volume, confirming severe foreground-background class imbalance and reaffirming the necessity of composite loss functions (BCE + Soft Dice) for subsequent model training.

---

## 2. Dataset Acquisition Details

* **Official Source Webpage:** [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)
* **Primary Citation:** Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing, 12(10), 1662.
* **Download Channel:** Official Google Drive storage IDs referenced by the authors:
  - `val.zip`: ID `1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG`
  - `test.zip`: ID `1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0`
  - `train.zip`: ID `18GuoCuBn48oZKAlEo-LrNwABrFhVALU-`
* **Download Methodology:** Automated terminal streaming via Python (`urllib.request`) with cookie handling and confirmation token resolution to bypass large-file Google Drive virus scan prompts safely.
* **Archive Integrity Check:** Verified using Python's `zipfile.ZipFile.testzip()`. All three archives passed CRC-32 checksum checks without a single corrupted member.

---

## 3. Directory Layout and Partition Statistics

The dataset has been extracted and organized in `satellite-change-detector/data/` preserving the canonical directory structure:

```
satellite-change-detector/data/
├── raw/
│   ├── train.zip                 (1,642.2 MB, 1,338 members)
│   ├── val.zip                   (234.7 MB, 195 members)
│   └── test.zip                  (473.3 MB, 387 members)
├── train/
│   ├── A/                        (445 pre-change images, T1)
│   ├── B/                        (445 post-change images, T2)
│   └── label/                    (445 change ground truth masks)
├── val/
│   ├── A/                        (64 pre-change images, T1)
│   ├── B/                        (64 post-change images, T2)
│   └── label/                    (64 change ground truth masks)
└── test/
    ├── A/                        (128 pre-change images, T1)
    ├── B/                        (128 post-change images, T2)
    └── label/                    (128 change ground truth masks)
```

### Partition Distribution Summary

| Partition | Pairs ($T_1, T_2, \text{Mask}$) | Total Image Files | Original Dimensions | Color Space | Split Proportion |
|---|---|---|---|---|---|
| **Training (`train`)** | 445 | 1,335 | $1024 \times 1024$ px | RGB (A/B), 1-Ch (Mask) | 69.86% |
| **Validation (`val`)** | 64 | 192 | $1024 \times 1024$ px | RGB (A/B), 1-Ch (Mask) | 10.05% |
| **Testing (`test`)** | 128 | 384 | $1024 \times 1024$ px | RGB (A/B), 1-Ch (Mask) | 20.09% |
| **Total** | **637 pairs** | **1,911 images** | **$1024 \times 1024$ px** | — | **100.0%** |

*Note: The verified pair counts (445, 64, 128) exactly match the benchmark splits specified in the original literature.*

---

## 4. Programmatic Data Integrity and Correspondence Audit

The programmatic audit was executed via `src/extract_and_audit.py` and output to `data/audit_summary.json`. The results verify:

1. **One-to-One Filename Correspondence:**
   - Every file `train/A/train_{idx}.png` has a corresponding matching counterpart `train/B/train_{idx}.png` and `train/label/train_{idx}.png`.
   - Missing $T_2$ files across all splits: **0** (`missing_b: []`).
   - Missing label files across all splits: **0** (`missing_label: []`).
2. **Corrupted or Unreadable Files:**
   - Zero corrupted files across all splits: **0** (`corrupted: []`).
3. **Format and Channels:**
   - Pre-change images ($T_1$, folder `A`): 100% 3-channel optical RGB.
   - Post-change images ($T_2$, folder `B`): 100% 3-channel optical RGB.
   - Masks (folder `label`): 100% single-channel grayscale raster.

---

## 5. Mask Value and Class Distribution Analysis

### 5.1 Mask Pixel Values and Anti-Aliasing
Inspection of the unique pixel values present in the raw label masks revealed:
$$\text{Unique Values} = \{0, 156, 254, 255\}$$
- Value $0$: Pure background (no change).
- Value $255$: Pure foreground (building change).
- Values $156, 254$: Anti-aliasing interpolation artifacts generated along building edges during rasterization or JPEG compression in earlier dataset processing.

**Resolution in Pipeline:** To prevent soft multi-class ambiguity during loss calculation, all ground-truth masks are strictly binarized in `src/dataset.py`:
$$\text{Binary Mask}(x, y) = \begin{cases} 1.0 & \text{if } I_{\text{mask}}(x, y) > 127 \\ 0.0 & \text{otherwise} \end{cases}$$

### 5.2 Foreground-Background Class Imbalance

| Split | Total Pixels Analyzed | Changed Pixels ($\text{Mask} > 127$) | Change Percentage | Ratio (Bg : Fg) |
|---|---|---|---|---|
| **Train** | $445 \times 1024 \times 1024 = 466,616,320$ | $21,413,023$ | **$4.589\%$** | $\approx 20.8 : 1$ |
| **Validation** | $64 \times 1024 \times 1024 = 67,108,864$ | $2,816,300$ | **$4.197\%$** | $\approx 22.8 : 1$ |
| **Test** | $128 \times 1024 \times 1024 = 134,217,728$ | $6,837,451$ | **$5.094\%$** | $\approx 18.6 : 1$ |
| **Overall** | **$667,942,912$** | **$31,066,774$** | **$4.651\%$** | **$\approx 20.5 : 1$** |

### Key Architectural Takeaway:
Because foreground building change accounts for only **$\approx 4.65\%$** of the spatial area:
1. Standard Binary Cross-Entropy (BCE) alone would allow the network to achieve $>95\%$ pixel accuracy by trivially predicting all zeroes ("no change").
2. A **Composite Loss** combining BCE with **Soft Dice Loss** or **Focal Loss** is mathematically mandatory for effective gradient propagation on changed buildings.

---

## 6. Storage and Resource Utilization Audit

| Resource | Size / Allocation | Notes |
|---|---|---|
| **Raw Zip Archives (`data/raw/`)** | $2.35\text{ GB}$ | Preserved as immutable source backups |
| **Extracted Dataset (`data/{train,val,test}`)** | $3.80\text{ GB}$ | Extracted single-copy uncompressed PNGs |
| **Manifest & Metadata (`data/`)** | $0.08\text{ GB}$ | `dataset_manifest.csv` and `audit_summary.json` |
| **Total Disk Space Used** | **$6.23\text{ GB}$** | Kept inside project directory |
| **Remaining Free Disk Space** | **$8.0\text{ GiB}$** | Sufficient margin for OS operations and checkpoints |

---

## 7. Artifact Manifest and Visual Previews

1. **Dataset Manifest:** Saved to [dataset_manifest.csv](file:///Users/ananyasingh/Documents/UPES/Semester%205/Deep%20Learning%20Lab/satellite-change-detector/data/dataset_manifest.csv). Contains complete path and per-pair metadata for all 637 entries:
   - `split`, `filename`, `path_t1`, `path_t2`, `path_mask`, `width`, `height`, `change_pixels`, `change_percentage`.
2. **Visual Inspection Grid:** Saved to [outputs/sample_pairs_preview.png](file:///Users/ananyasingh/Documents/UPES/Semester%205/Deep%20Learning%20Lab/satellite-change-detector/outputs/sample_pairs_preview.png). Visualizes 4 diverse representative sample pairs ($T_1$, $T_2$, and overlay Ground Truth) validating alignment and high visual quality.

---

## 8. Conclusion and Certification

The LEVIR-CD dataset has been successfully ingested, structured, audited, and verified. The data is 100% complete, uncorrupted, and ready for PyTorch ingestion in Phase 3.

