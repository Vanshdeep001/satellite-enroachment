# Literature Review — Remote Sensing Change Detection Methodologies

**Project Title:** Satellite Land Use Change and Encroachment Detector  
**Course:** Deep Learning — Project-Based Learning (PBL) / Project 50  
**Scope:** Comparative Analysis of Classical, Deep Learning, and Attention-Based Change Detection Paradigms  

---

## 1. Introduction and Taxonomy of Change Detection

In optical remote sensing, **Change Detection (CD)** is defined as the process of identifying differences in the state of an Earth surface phenomenon or geographic entity by observing it at bitemporal or multitemporal epochs $T_1$ and $T_2$. Methodologies developed over four decades can be broadly grouped into four generations:
1. **Algebraic / Radiometric Methods:** Pixel-by-pixel mathematical transformations on spectral bands.
2. **Post-Classification Comparison (PCC):** Independent semantic classification followed by symbolic rule comparison.
3. **Deep Convolutional Architectures:** End-to-end representation learning via CNNs, Siamese networks, and Encoder-Decoder models.
4. **Attention and Transformer Architectures:** Modeling long-range spatio-temporal dependencies using self-attention and cross-attention mechanisms.

Below is an extensive comparative review of these methodologies, evaluated against the requirements of our project.

---

## 2. Exhaustive Analysis of Core Methodologies

### 2.1 Traditional Image-Differencing & Change Vector Analysis (CVA)

* **Key Literature:** 
  - Singh, A. (1989). *Digital change detection techniques using remotely-sensed data.* International Journal of Remote Sensing, 10(6), 989–1003. [DOI: 10.1080/01431168908903939](https://doi.org/10.1080/01431168908903939)
  - Bovolo, F., & Bruzzone, L. (2007). *A theoretical framework for unsupervised change detection based on change vector analysis in the polar domain.* IEEE Transactions on Geoscience and Remote Sensing (TGRS), 45(1), 218–236. [DOI: 10.1109/TGRS.2006.885408](https://doi.org/10.1109/TGRS.2006.885408)
* **Working Principle:** Computes an algebraic pixel difference or spectral magnitude vector between registered co-located spectral bands:
  $$\Delta I(x, y) = |I_{T_2}(x, y) - I_{T_1}(x, y)|$$
  A thresholding algorithm (such as Otsu's thresholding or Gaussian Mixture Model fitting) is then applied to the differential image to partition pixels into changed and unchanged classes.
* **Advantages:** Extremely fast computation; zero training data required; fully unsupervised; simple to implement and explain.
* **Limitations:** Sensitive to illumination variations, cloud cover, sensor viewing angles, and phenological (seasonal vegetation) shifts. Produces high false-positive rates on complex urban scenes. Incapable of distinguishing structural changes (e.g., buildings) from non-structural spectral shifts (e.g., grass drying).
* **Relevance to Project:** Serves as our **mandatory traditional baseline** (Phase 2), providing a lower-bound performance metric against which deep learning models are quantified.

---

### 2.2 Post-Classification Comparison (PCC)

* **Key Literature:**
  - Mas, J. F. (1999). *Monitoring land-cover changes: a comparison of change detection techniques.* International Journal of Remote Sensing, 20(1), 139–152. [DOI: 10.1080/014311699213659](https://doi.org/10.1080/014311699213659)
* **Working Principle:** Employs a supervised classifier (e.g., Random Forest, SVM, or single-image semantic segmentation network) to independently generate discrete land-cover maps for both $T_1$ and $T_2$. The resulting thematic maps are compared pixel-by-pixel using a transition matrix (e.g., from *Bare Soil* to *Impervious Surface*).
* **Advantages:** Provides explicit "from-to" land-cover trajectory information; handles bitemporal images acquired by different sensor types with varying radiometric calibrations.
* **Limitations:** Multiplicative error propagation—the overall change accuracy is approximately the product of the two individual classification accuracies ($A_{CD} \approx A_{T_1} \times A_{T_2}$). A small classification error in either date leads directly to a false change detection.
* **Relevance to Project:** Informs our evaluation strategy and highlights why direct end-to-end bitemporal change modeling is superior to two-stage independent classification for building footprints.

---

### 2.3 Fully Convolutional Siamese Networks (FC-EF, FC-Siam-diff, FC-Siam-conc)

* **Key Literature:**
  - Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully Convolutional Siamese Networks for Change Detection.* 2018 25th IEEE International Conference on Image Processing (ICIP), pp. 4063–4067. [arXiv:1810.08462](https://arxiv.org/abs/1810.08462)
* **Working Principle:** Daudt et al. established three foundational deep-learning architectures for remote sensing change detection:
  1. **FC-EF (Early Fusion):** Concatenates bitemporal images at the input stage ($C = 6$ for two RGB images) and feeds them into a single encoder-decoder network.
  2. **FC-Siam-diff (Siamese Feature Difference):** Uses two weight-sharing encoder streams to process $T_1$ and $T_2$ independently. At each skip level and the bottleneck, the absolute difference of feature tensors ($|F_{T_1}^{(l)} - F_{T_2}^{(l)}|$) is passed to the decoder.
  3. **FC-Siam-conc (Siamese Feature Concatenation):** Similar to Siam-diff, but concatenates feature maps ($[F_{T_1}^{(l)}, F_{T_2}^{(l)}]$) across the temporal dimension before passing them into the decoding layers.
* **Advantages:** Weight sharing halves encoder parameter count; explicit multi-scale feature comparison prevents early mixing of noise; end-to-end differentiable.
* **Limitations:** Fixed fusion mechanisms (subtraction/concatenation) treat all spatial channels equally without contextual weighting; struggles with precise edge localization on very small building alterations.
* **Relevance to Project:** Direct architectural foundation for our proposed model. We adopt the Siamese encoder with differential skip fusion within an enhanced U-Net framework.

---

### 2.4 U-Net for Dense Semantic Segmentation and Change Delineation

* **Key Literature:**
  - Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* Medical Image Computing and Computer-Assisted Intervention (MICCAI), LNCS vol. 9351, pp. 234–241. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
* **Working Principle:** A symmetric encoder-decoder network featuring downsampling convolutional blocks (contracting path), upsampling deconvolutional blocks (expansive path), and direct **skip connections** that transfer high-resolution spatial feature maps across layers.
* **Advantages:** Preserves fine spatial geometry and precise boundary details; highly sample-efficient (trains effectively on modest dataset sizes); lightweight compute footprint suitable for local workstation/student laptop execution (MPS / CPU).
* **Limitations:** Standard U-Net expects a single image input; adapting it for bitemporal change detection requires either channel stacking (early fusion) or dual Siamese encoders with differential skip pathways.
* **Relevance to Project:** Core architecture selected for Phase 3. Provides an ideal balance of training stability, memory efficiency on Apple Silicon MPS, and sharp boundary delineation for building footprints.

---

### 2.5 ResNet-Based Backbones and Spatial-Temporal Attention (STANet)

* **Key Literature:**
  - He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016, pp. 770–778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  - Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing, 12(10), 1662. [DOI: 10.3390/rs12101662](https://doi.org/10.3390/rs12101662)
* **Working Principle:** Utilizes deep residual backbones (ResNet-18 or ResNet-34) inside a Siamese feature extractor to avoid vanishing gradients. In STANet, Chen & Shi integrated a Spatial-Temporal Attention module that computes pixel-to-pixel self-attention across space and time, dynamically filtering out misregistration noise and seasonal lighting shifts.
* **Advantages:** Deep feature extraction; attention mechanisms dramatically suppress false positives caused by shadows and seasonal cycles; published benchmark for the LEVIR-CD dataset.
* **Limitations:** Self-attention across all spatial coordinates incurs quadratic computational complexity $\mathcal{O}((HW)^2)$, requiring significant VRAM during backpropagation; higher implementation and debugging overhead.
* **Relevance to Project:** STANet is the benchmark model published alongside the LEVIR-CD dataset. We study its design to inform our ablation experiments and consider a lightweight channel/spatial attention mechanism (e.g., CBAM or Squeeze-and-Excitation) as a planned extension.

---

### 2.6 Object Detection Paradigms (YOLO, Faster R-CNN) vs. Semantic Segmentation

* **Key Literature:**
  - Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
  - Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016, pp. 779–788. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
* **Working Principle:** Object detectors generate rectangular bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ alongside class probabilities around detected discrete entities.
* **Comparative Assessment:**
  - *Why Object Detection Fails for Encroachment Detection:* A bounding box surrounds the entire perimeter of a structure, inevitably including surrounding gardens, driveways, and adjacent land. It cannot compute precise physical square meters of newly constructed foundation.
  - *Why Semantic Segmentation (U-Net) is Superior:* Semantic segmentation predicts a class label ($1$ or $0$) for **every individual pixel**, allowing exact boundary extraction, accurate calculation of structural area ($\text{Area} = \sum \text{pixels} \times \text{GSD}^2$), and crisp polygon intersection with cadastral boundary shapefiles.
* **Relevance to Project:** Justifies our technical decision to reject object detection in favor of dense pixel-level segmentation.

---

### 2.7 Transformer-Based Change Detection (BIT, ChangeFormer)

* **Key Literature:**
  - Chen, H., Qi, Z., & Shi, Z. (2021). *Remote Sensing Image Change Detection With Transformers.* IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14. [DOI: 10.1109/TGRS.2021.3095166](https://doi.org/10.1109/TGRS.2021.3095166)
  - Bandara, W. G. C., & Patel, V. M. (2022). *A Transformer-Based Siamese Network for Change Detection.* IEEE IGARSS 2022, pp. 207–210. [arXiv:2201.01297](https://arxiv.org/abs/2201.01297)
* **Working Principle:** Employs vision transformers (ViT) or hierarchical swin transformers to represent images as token sequences. Bitemporal Image Transformer (BIT) tokens are passed to a transformer encoder/decoder to model context across time.
* **Advantages:** Unmatched global receptive field; state-of-the-art benchmark scores on large remote sensing datasets.
* **Limitations:** Heavy memory footprint; high sample complexity (requires massive pretraining datasets like ImageNet-22k); slow inference and training on non-CUDA local hardware (Apple Silicon MPS).
* **Relevance to Project:** Serves as the theoretical state-of-the-art reference point in our literature review and viva presentation, contrasting with our choice of a lightweight U-Net optimized for local student execution.

---

## 3. Summary Literature Matrix

| Model / Paper | Paradigm | Primary Strength | Primary Bottleneck | Suitability for This Project |
|---|---|---|---|---|
| **CVA (Singh, 1989)** | Pixel Difference | No training required; immediate baseline | Massive false positives from illumination/seasons | **Selected as Classical Baseline (Phase 2)** |
| **PCC (Mas, 1999)** | Two-stage Classification | Explicit from-to categories | Multiplicative error propagation | Excluded (unnecessary error compound) |
| **FC-EF (Daudt, 2018)** | Early Fusion CNN | Fast single-pass inference | Early channel mixing loses temporal nuance | Reference baseline |
| **FC-Siam-diff (Daudt, 2018)** | Siamese Difference | Distinct temporal features; lightweight | Feature subtraction misses subtle semantic cues | **Primary design inspiration for our U-Net** |
| **U-Net (Ronneberger, 2015)** | Dense Segmentation | Pixel-precise boundaries; sample efficient | Standard version is mono-temporal | **Adapted as core model (Siamese U-Net)** |
| **Faster R-CNN / YOLO** | Object Detection | Rapid discrete object counting | Bounding boxes cannot measure exact square meters | Excluded in favor of dense pixel segmentation |
| **STANet (Chen & Shi, 2020)** | Spatio-Temporal Attention | Suppresses pseudo-changes; LEVIR-CD benchmark | Heavy VRAM consumption $\mathcal{O}((HW)^2)$ | Benchmark comparison / planned extension |
| **BIT / ChangeFormer (2021–22)** | Vision Transformer | Global context modeling | Requires extensive GPU compute and pretraining | Theoretical context in literature review |

