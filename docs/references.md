# References and Academic Disclosure

> **Academic Integrity Statement:** This file records all external scientific publications, datasets, software packages, and AI assistance tools utilized in the design, formulation, and implementation of this project.

---

## 1. Primary Benchmark Dataset

1. **LEVIR-CD Dataset:**  
   Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* **Remote Sensing**, 12(10), 1662.  
   DOI: [10.3390/rs12101662](https://doi.org/10.3390/rs12101662) | Official Portal: [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)  
   *Role in Project:* Benchmark dataset providing 637 bitemporal VHR image pairs and binary building change ground-truth masks.

---

## 2. Foundational Change Detection & Deep Learning Literature

2. **Fully Convolutional Siamese Networks for Change Detection:**  
   Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully Convolutional Siamese Networks for Change Detection.* **2018 25th IEEE International Conference on Image Processing (ICIP)**, Athens, Greece, pp. 4063–4067.  
   arXiv: [1810.08462](https://arxiv.org/abs/1810.08462)  
   *Role in Project:* Conceptual foundation for our Siamese feature difference skip architecture (FC-Siam-diff).

3. **U-Net Architecture:**  
   Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* **Medical Image Computing and Computer-Assisted Intervention (MICCAI)**, LNCS vol. 9351, pp. 234–241.  
   arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)  
   *Role in Project:* Core encoder-decoder segmentation architecture with direct multi-scale skip connections.

4. **Deep Residual Learning (ResNet):**  
   He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* **IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 770–778.  
   arXiv: [1512.03385](https://arxiv.org/abs/1512.03385)  
   *Role in Project:* Backbone feature extractor design principles and residual connections.

5. **Traditional Change Detection Survey:**  
   Singh, A. (1989). *Digital change detection techniques using remotely-sensed data.* **International Journal of Remote Sensing**, 10(6), 989–1003.  
   DOI: [10.1080/01431168908903939](https://doi.org/10.1080/01431168908903939)  
   *Role in Project:* Theoretical basis for the classical pixel-differencing baseline.

6. **Change Vector Analysis (CVA):**  
   Bovolo, F., & Bruzzone, L. (2007). *A theoretical framework for unsupervised change detection based on change vector analysis in the polar domain.* **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, 45(1), 218–236.  
   DOI: [10.1109/TGRS.2006.885408](https://doi.org/10.1109/TGRS.2006.885408)  
   *Role in Project:* Formulation of magnitude and direction vectors in spectral space for baseline differencing.

7. **Post-Classification Comparison (PCC):**  
   Mas, J. F. (1999). *Monitoring land-cover changes: a comparison of change detection techniques.* **International Journal of Remote Sensing**, 20(1), 139–152.  
   DOI: [10.1080/014311699213659](https://doi.org/10.1080/014311699213659)  
   *Role in Project:* Theoretical comparison showing error propagation in two-stage classification vs. end-to-end change modeling.

8. **Object Detection Frameworks (YOLO & Faster R-CNN):**  
   - Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* **NeurIPS 2015**. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)  
   - Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* **CVPR 2016**, pp. 779–788. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)  
   *Role in Project:* Analyzed in literature review to demonstrate why bounding-box object detection cannot measure exact physical surface square meters.

9. **Transformer-Based Change Detection (BIT & ChangeFormer):**  
   - Chen, H., Qi, Z., & Shi, Z. (2021). *Remote Sensing Image Change Detection With Transformers.* **IEEE Transactions on Geoscience and Remote Sensing**, vol. 60, pp. 1–14. [DOI: 10.1109/TGRS.2021.3095166](https://doi.org/10.1109/TGRS.2021.3095166)  
   - Bandara, W. G. C., & Patel, V. M. (2022). *A Transformer-Based Siamese Network for Change Detection.* **IEEE IGARSS 2022**, pp. 207–210. [arXiv:2201.01297](https://arxiv.org/abs/2201.01297)  
   *Role in Project:* Theoretical state-of-the-art context in literature review and viva presentation.

---

## 3. Core Software Libraries and Frameworks

10. **PyTorch:**  
    Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* **NeurIPS 2019**, pp. 8024–8035.  
    Website: [https://pytorch.org](https://pytorch.org) | Version: `2.15.0` (Apple Silicon MPS backend).

11. **Torchvision:**  
    Marcel, S., & Rodriguez, P. (2010). *Torchvision the machine-vision library in Torch.* **ACM Multimedia 2010**.  
    Version: `0.29.2`.

12. **Scikit-Learn:**  
    Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* **Journal of Machine Learning Research**, 12, 2825–2830.  
    Website: [https://scikit-learn.org](https://scikit-learn.org) | Version: `1.9.1`.

13. **OpenCV (Open Source Computer Vision Library):**  
    Bradski, G. (2000). *The OpenCV Library.* **Dr. Dobb's Journal of Software Tools**.  
    Version: `opencv-python-headless 5.0.0.93`.

14. **Matplotlib:**  
    Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* **Computing in Science & Engineering**, 9(3), 90–95.  
    Version: `3.11.2`.

---

## 4. Academic AI Assistance Disclosure

Pursuant to academic integrity and transparency policies:
* **Tool:** Google Antigravity / Gemini coding assistant.
* **Scope of Assistance:**
  - Structuring project directories and scaffold configuration (Phase 0).
  - Assisting in literature synthesis, markdown documentation drafting, and Mermaid diagram generation (Phase 1).
  - Formulating experiment plans and viva question prompts.
* **Student Responsibility:**
  - All project decisions (dataset selection, model architecture choices, loss function formulations, and ethical boundaries) were reviewed and directed by the student.
  - The student understands every equation, architecture block, and design decision documented herein and takes full intellectual ownership for the viva and project deliverables.
