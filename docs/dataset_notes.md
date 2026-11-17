# Dataset Notes — LEVIR-CD

> **Status:** Dataset NOT yet downloaded. This file documents what is known from the official source.
> Download instructions and verified statistics will be added in Phase 1.

---

## Dataset Identity

| Property | Detail |
|----------|--------|
| **Name** | LEVIR-CD — Large-scale Remote Sensing Building Change Detection Dataset |
| **Authors** | Hao Chen, Zipeng Qi, Zhenwei Shi |
| **Paper** | "Spatial-Temporal Attention-Based Method for Remote Sensing Image Change Detection" (2020) |
| **Official page** | https://justchenhao.github.io/LEVIR/ |
| **Focus** | Binary change detection of **building footprints** between two time periods |
| **Image type** | Bi-temporal RGB image pairs (before / after) |
| **Resolution** | 0.5 m/pixel (very high resolution) |
| **Patch size** | 1024 × 1024 pixels (commonly sub-cropped to 256 × 256 for training) |

---

## Why LEVIR-CD?

- It provides ground-truth binary change masks, making it suitable for supervised learning.
- It focuses specifically on building changes, which aligns with our goal of detecting new construction.
- It is widely used in academic change-detection research, so benchmark numbers are available for comparison.
- The data is publicly available for academic use.

---

## Licensing / Usage Terms

⚠️ **You must read the terms on the official website before downloading.**  
The dataset is intended for **academic and research use**. Commercial use is not covered by the standard academic license. Cite the original paper in all work derived from this dataset.

**Required citation:**
```
@article{chen2020spatial,
  title     = {Spatial-Temporal Attention-Based Method for Remote Sensing Image Change Detection},
  author    = {Chen, Hao and Qi, Zipeng and Shi, Zhenwei},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2020},
  publisher = {IEEE}
}
```

---

## Download Instructions (Phase 1 — not yet executed)

1. Visit https://justchenhao.github.io/LEVIR/ and read the license terms.
2. Request or download the dataset using the link provided on the page.
3. Place the downloaded archive in `data/raw/`.
4. Do **not** commit the data files to git (already excluded in `.gitignore`).
5. Verify the file checksums if provided.

---

## Expected Dataset Statistics (from the paper — to be verified after download)

| Split | Image Pairs | Change Pixels (approx.) |
|-------|------------|--------------------------|
| Train | 445 pairs | ~18% of pixels |
| Val | 64 pairs | — |
| Test | 128 pairs | — |

> These figures are from the published paper. Actual verified numbers will be recorded here after download in Phase 1.

---

## Responsible Use Note

LEVIR-CD labels changes as building additions or removals. This project uses those labels to train a model that detects likely construction. The model output **does not determine legal status** of any detected change. Encroachment classification requires verified boundary data from an authoritative legal/government source, which is outside the scope of this PBL.

