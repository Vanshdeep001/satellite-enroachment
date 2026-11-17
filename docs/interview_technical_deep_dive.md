# Interview Guide 2: Deep Learning Technical Concepts & Viva Q&A

**Project:** Satellite Land Use Change and Encroachment Detector  
**Focus:** Deep Learning Fundamentals, Computer Vision, Architecture, Losses, Optimizers & Metrics  

---

## 1. Problem Formulation: What is Bitemporal Change Detection?

- **Input:** Two co-registered optical RGB satellite images $T_1$ (pre-event) and $T_2$ (post-event) of size $[3, H, W]$.
- **Output:** A dense binary segmentation mask $\hat{Y} \in \{0, 1\}^{[1, H, W]}$, where:
  - `1` (Foreground): A structural building change occurred (new construction or extension).
  - `0` (Background): No structural change (unchanged buildings, trees, grass, roads, water, shadows).
- **Core ML Category:** Dense pixel-level binary classification (Computer Vision / Semantic Segmentation).

---

## 2. Model Architecture: Siamese Networks & U-Net

### 2.1 What is a Siamese Neural Network?
A Siamese network consists of **two identical sub-networks with shared weights**. 
- $T_1$ passes through Encoder $E_\theta$.
- $T_2$ passes through the exact same Encoder $E_\theta$ with identical weights $\theta$.

**Why weight-sharing?**
1. **Temporal Symmetry & Invariance:** A building appearing in $T_1 \to T_2$ should be recognized using the exact same visual feature detectors as $T_2 \to T_1$.
2. **Parameter Efficiency:** Halves the number of encoder parameters compared to using two separate feature extractors.
3. **Consistent Feature Space:** Maps both images into the same embedding manifold, making direct feature comparison meaningful.

```
       T1 Image ───► [ Shared Encoder E_θ ] ───► Feature Maps F_T1
                                                     │
                                                     ▼
                                          [ Temporal Feature Fusion ]
                                                     ▲
                                                     │
       T2 Image ───► [ Shared Encoder E_θ ] ───► Feature Maps F_T2
```

---

### 2.2 Why Not "Early Fusion" (Stacking T1 and T2 as 6 Channels)?
- **Early Fusion:** Concatenates $T_1$ and $T_2$ into a single 6-channel tensor $[6, H, W]$ at the input layer.
- **Why it performs poorly:** Early fusion forces early convolutional filters to immediately compare raw pixel RGB values. This makes the network hypersensitive to trivial photometric differences (lighting, sun glint, season, haze) before understanding what objects are present.
- **Siamese Late/Middle Fusion:** First extracts high-level semantic concepts (walls, roofs, corners, geometry) in each image independently, and *then* compares these robust representations.

---

### 2.3 The U-Net Backbone & The Vital Role of Skip Connections
Our architecture uses a 4-stage U-Net encoder-decoder:
- **Encoder (Contracting Path):** 4 blocks of Double Convolutions (`Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU`) followed by $2 \times 2$ Max Pooling. Reduces spatial dimensions by $16\times$ ($256 \to 128 \to 64 \to 32 \to 16$) while expanding feature channels ($16 \to 32 \to 64 \to 128 \to 256$).
- **Bottleneck:** Deepest layer ($16 \times 16$ spatial resolution, $256$ channels), capturing maximum contextual semantics.
- **Decoder (Expansive Path):** 4 stages of progressive bilinear upsampling paired with double convolutions, restoring spatial resolution from $16 \times 16$ back to $256 \times 256$.

#### Why Skip Connections are Indispensable (Our Empirical Proof):
- In deep convolutional networks, downsampling destroys spatial coordinate information. At the $16 \times 16$ bottleneck, the network knows *what* is in the image, but loses *where* building edges are located down to the exact pixel.
- **Skip connections** pass high-resolution feature maps from early encoder layers directly across to the corresponding decoder stages.
- **Our Ablation Finding:** When we removed skip connections (`SiameseUNetNoSkip`), Validation F1-score **collapsed from 65.15% down to 22.30% (a -42.85% absolute drop)**. Without skip connections, the decoder outputs blurry blobs instead of crisp building footprints.

---

### 2.4 Feature Fusion: Absolute Difference vs. Concatenation

| Strategy | Mathematical Formulation | Decoder Channels | Total Params | Best Val F1 | Key Trade-off |
|---|---|:---:|:---:|:---:|---|
| **Absolute Difference (Candidate A)** | $\Delta F_i = \|F_{T1}^{(i)} - F_{T2}^{(i)}\|$ | $C$ | $1.96\text{M}$ ($7.5\text{ MB}$) | $74.55\%$ | Mathematically symmetric ($T_1 \equiv T_2 \implies 0$); compact; slightly lower capacity. |
| **Channel Concatenation (Candidate B - Selected)** | $S_i = [F_{T1}^{(i)}, F_{T2}^{(i)}]$ | $2C$ | $3.93\text{M}$ ($15.0\text{ MB}$) | **$82.94\%$** | $2\times$ channel capacity; learns rich non-linear cross-temporal interactions; $+8.39\%$ higher F1. |

**Why did Concatenation win?**  
Absolute differencing forcibly subtracts features, which can cancel out subtle contextual cues. Concatenation lets the decoder's subsequent $3 \times 3$ convolution learn the optimal mathematical combination of pre- and post-features, boosting Recall from $70.00\%$ to $82.13\%$.

---

## 3. Loss Functions & The Class Imbalance Problem

### 3.1 The Severe Class Imbalance (95% : 5%)
In the LEVIR-CD benchmark:
- **Unchanged pixels (Negative class):** $\approx 95\%$
- **Changed building pixels (Positive class):** $\approx 5\%$

### 3.2 Why Standard Binary Cross-Entropy (BCE) Alone Fails:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$$
- BCE computes loss independently for every single pixel and averages them.
- Because 95% of pixels are zeros ($y_i = 0$), the second term dominates the gradient.
- The model can achieve a very low loss by simply predicting $\hat{p}_i \approx 0$ everywhere (predicting "no change" for everything).

---

### 3.3 Soft Dice Loss (Regional Overlap Loss)
The Dice coefficient measures set overlap between prediction $P$ and ground truth $Y$:
$$\text{Dice} = \frac{2 |P \cap Y|}{|P| + |Y|}$$

To make this differentiable for gradient descent, we use **Soft Dice Loss**:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i^2 + \sum_{i=1}^N y_i^2 + \epsilon}$$
- $p_i \in [0, 1]$ is the predicted probability, $y_i \in \{0, 1\}$ is the ground truth.
- $\epsilon = 10^{-6}$ prevents division by zero.
- **Why Soft Dice is immune to class imbalance:** It measures the *relative intersection* of the positive class. The 95% true negative background pixels don't appear in the numerator, so they cannot drown out the gradient of tiny 5% building clusters!

---

### 3.4 Composite Loss Function: Best of Both Worlds
$$\mathcal{L}_{\text{Composite}} = 0.50 \cdot \mathcal{L}_{\text{BCE}} + 0.50 \cdot \mathcal{L}_{\text{Dice}}$$
- **BCE** provides smooth, convex pixel-wise gradients that stabilize early training.
- **Soft Dice** directly optimizes regional shape overlap and forces the network to focus on the minority building class.

---

## 4. Evaluation Metrics & The "Accuracy Paradox"

### 4.1 Why Accuracy is Completely Misleading:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- If a satellite scene has 95% background, a trivial model that outputs a blank black image (all zeros) achieves **95% accuracy** while finding **zero** buildings.
- **Never use Accuracy for imbalanced segmentation tasks.**

---

### 4.2 The Real Metrics: Precision, Recall, F1, and IoU

```
                        Ground Truth Positive    Ground Truth Negative
                      ┌────────────────────────┬────────────────────────┐
  Predicted Positive  │  True Positive (TP)    │  False Positive (FP)   │
                      ├────────────────────────┼────────────────────────┤
  Predicted Negative  │  False Negative (FN)   │  True Negative (TN)    │
                      └────────────────────────┴────────────────────────┘
```

1. **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$  
   *"When the model flags a change, how often is it actually a building?"* (Measures false alarm resistance).
2. **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$  
   *"Out of all real new buildings on the ground, how many did the model detect?"* (Crucial for illegal encroachment to avoid missed violations).
3. **F1-Score (Dice Score):** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$  
   The harmonic mean balancing Precision and Recall.
4. **IoU (Intersection-over-Union / Jaccard Index):** $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$  
   $$\text{IoU} = \frac{\text{F}_1}{2 - \text{F}_1}$$
   Stricter than F1 because it penalizes false alarms and missed pixels in the denominator without giving double credit to true positives.

---

### 4.3 Optimal Probability Threshold Tuning ($\tau^* = 0.20$)
- By default, sigmoid outputs are thresholded at $\tau = 0.50$.
- Because changed pixels are rare, the model's raw probabilities are naturally skewed lower.
- By sweeping thresholds on the validation set ($\tau \in [0.10, 0.90]$), we discovered that $\tau^* = 0.20$:
  - Boosted Recall from $77.29\% \to 82.13\%$ ($+4.84\%$).
  - Maintained high Precision ($83.77\%$).
  - Maximized harmonic F1 to **$82.94\%$** and IoU to **$70.85\%$**.

---

## 5. Optimizers: Why AdamW Won (Phase 7 Benchmark)

We conducted a controlled 4-optimizer benchmark on identical Siamese U-Net architectures:

```
  Vanilla SGD (eta=0.01)           ───► Val F1: 17.83%  (Struggles in ravines)
  SGD + Momentum (mu=0.9)          ───► Val F1: 56.41%  (+38.58% speedup via inertia)
  Nesterov Momentum (mu=0.9)       ───► Val F1: 58.17%  (+1.76% via look-ahead gradient)
  AdamW (eta=0.001, wd=1e-4)       ───► Val F1: 66.87%  (Clear winner: adaptive per-parameter rates)
```

### Key Intuitions:
1. **Vanilla SGD fails ($17.83\%$):** In deep convnets, loss landscapes have steep ravines. Vanilla SGD bounces back and forth across ravine walls rather than moving along the shallow downhill slope.
2. **Momentum helps ($56.41\%$):** Accumulates velocity vector $v_{t+1} = \mu v_t + \eta \nabla L$. Bouncing oscillations cancel out, while downhill directions accelerate.
3. **Nesterov Momentum ($58.17\%$):** Computes gradient at the *look-ahead* position $\theta + \mu v$, applying an anticipatory brake before overshooting valleys.
4. **AdamW is superior ($66.87\%$ in 5 epochs, $82.94\%$ in 15 epochs):**  
   - Computes individual adaptive learning rates per parameter using first moments ($m_t$, mean) and second moments ($v_t$, variance):
     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t$$
   - For rare building pixels, gradients are sparse. AdamW automatically assigns **larger effective steps** to weights associated with sparse change features!
   - Uses **decoupled weight decay** ($\lambda \theta_t$), ensuring $L_2$ regularization operates cleanly without being distorted by gradient variance.

---

## 6. Training Dynamics & Regularization

1. **Learning Rate Scheduler (Cosine Annealing):**  
   Smoothly decays learning rate following a half-cosine curve:
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
   Starts with exploratory $\eta = 10^{-3}$ and anneals down to fine-tuning $\eta = 10^{-5}$ at epoch 15. Prevents late-stage divergence and settles cleanly into flat minima.
2. **Synchronous Bitemporal Data Augmentation:**  
   Standard data augmentation cannot be applied independently to $T_1$ and $T_2$ (e.g., flipping $T_1$ horizontally while leaving $T_2$ unflipped would completely destroy pixel alignment!).  
   We implemented `SynchronousBitemporalAugmentation`: identical random horizontal flips, vertical flips, and $90^\circ$ orthogonal rotations applied synchronously across $T_1$, $T_2$, and the ground-truth mask.
3. **Overfitting Detection & Checkpoint Strategy:**  
   We saved the best checkpoint based exclusively on **Validation F1** (not training loss). Candidate B achieved peak validation performance at **Epoch 13** (Train Loss: $0.1869$, Val Loss: $0.2249$, Val F1: $82.26\%$), after which training loss continued decreasing while validation loss plateaued, signaling the onset of mild overfitting.

---

## 7. Classical Baseline vs. Deep Learning (Why We Needed DL)

Before building the Siamese network, we implemented a traditional unsupervised image differencing pipeline (`src/baseline.py`):
1. Compute absolute pixel differences in RGB, Grayscale, and Euclidean distance:
   $$D(x, y) = \sqrt{\sum_{c \in \{R,G,B\}} (I_{T1}(x,y,c) - I_{T2}(x,y,c))^2}$$
2. Apply Gaussian smoothing and morphological Opening (erosion + dilation) to reduce speckle noise.
3. Optimize decision threshold $\tau^*$ via grid search.

### Baseline Results:
- **Test Precision:** $25.75\%$
- **Test Recall:** $30.45\%$
- **Test F1-Score:** **$27.90\%$**
- **Test IoU:** **$16.21\%$**

### Why Traditional Differencing Completely Fails:
- **Shadow shifts:** Changes in sun azimuth cause tall buildings to cast shadows in different directions. Classical differencing flags the shadow as a "new building" (false positive).
- **Seasonal vegetation:** Lawns turning from green in spring to brown in summer trigger massive pixel differences.
- **Roof reflections & glare:** Different satellite viewing angles create specular roof highlights.
- **Deep learning solves this** because convolutional kernels learn **spatial context, edges, structural boundaries, and semantic geometry** that are invariant to surface color shifts.

---

## 8. Top 12 "Must-Know" Viva / Interview Questions & Crisp Answers

### Q1: What is the difference between Semantic Segmentation and Change Detection?
> **Answer:** Semantic segmentation classifies every pixel in a *single* image into predefined object classes (e.g., building, tree, road). Change detection ingests *two* registered images of the same location and classifies whether a structural state transition occurred between time $T_1$ and $T_2$ (binary change vs. no-change).

### Q2: Why use a Siamese architecture instead of subtracting images before feeding them to a network?
> **Answer:** Raw pixel subtraction ($T_1 - T_2$) is destroyed by sensor noise, lighting variations, and seasonal changes. A Siamese network maps both images into a learned high-level semantic feature space first. Comparing feature vectors at deep layers compares *concepts* (presence of a building structure) rather than raw photometric colors.

### Q3: What happens if you remove skip connections from U-Net?
> **Answer:** Performance collapses. In our ablation experiment (`SiameseUNetNoSkip`), Validation F1 plummeted from $65.15\%$ to $22.30\%$. Max pooling reduces spatial resolution by $16\times$; skip connections are mathematically required to pass fine-grained coordinate information from early layers to the decoder to recover sharp object boundaries.

### Q4: Why did you use AdamW instead of standard Adam or SGD?
> **Answer:** In standard Adam, $L_2$ regularization is combined with the gradient before computing moments, which scales weight decay incorrectly for frequently updated parameters. AdamW cleanly decouples weight decay from the gradient update. AdamW also vastly outperformed SGD ($66.87\%$ vs $17.83\%$ in 5 epochs) because its per-parameter adaptive learning rates amplify updates for rare building change pixels.

### Q5: Why is Accuracy a dangerous metric in remote sensing?
> **Answer:** Because of severe class imbalance (typically 95% background, 5% change). A model predicting zero everywhere gets 95% accuracy while failing 100% of its intended task. We use F1-score and IoU, which evaluate true positive overlap and penalize false positives and false negatives.

### Q6: What is the relationship between F1-Score and IoU?
> **Answer:** $\text{IoU} = \frac{F_1}{2 - F_1}$. IoU is always numerically lower than F1-score for values $< 1.0$. IoU penalizes misclassified pixels more heavily because the denominator contains the entire union of predicted and ground truth pixels ($\text{TP} + \text{FP} + \text{FN}$).

### Q7: Why did you choose a decision threshold of 0.20 instead of 0.50?
> **Answer:** Because changed pixels represent only 5% of the data, the network's predicted probabilities are naturally calibrated lower. Tuning the threshold on the validation set showed that $\tau^* = 0.20$ maximized the harmonic mean of Precision ($83.77\%$) and Recall ($82.13\%$), lifting validation F1 to $82.94\%$.

### Q8: What is Soft Dice Loss and why do we combine it with BCE?
> **Answer:** Soft Dice Loss optimizes the intersection-over-union of continuous probability maps, making it naturally robust against class imbalance. We combine it with BCE ($0.5\text{BCE} + 0.5\text{Dice}$) because BCE provides smooth, convex pixel-wise gradients that stabilize early optimization, while Dice directly sharpens mask boundary overlap.

### Q9: What is the difference between your Validation set and Test set?
> **Answer:** The validation set (64 pairs) was used iteratively for model checkpointing, hyperparameter tuning, threshold calibration, and candidate selection. The test set (128 pairs) was strictly locked away and evaluated only once as an unbiased benchmark. This prevents test-set data leakage.

### Q10: How does your inference pipeline handle images of different resolutions?
> **Answer:** The pipeline reads original input dimensions $(W, H)$, resizes them using Bilinear interpolation to the model's receptive field ($256 \times 256$), performs GPU forward pass, and then re-interpolates the output probability map back to $(W, H)$. This preserves source GIS coordinate alignment.

### Q11: What is the computational latency of your model, and can it run on edge devices?
> **Answer:** The selected model (`SiameseUNetConc`) has $3.93\text{M}$ parameters ($14.99\text{ MB}$ weights). On Apple Silicon GPU via PyTorch MPS, single-pair inference takes **$18.08\text{ ms}$**, yielding **$55.3\text{ FPS}$**. It can easily run in real-time on edge laptops, embedded devices, or cloud microservices.

### Q12: How do you translate model pixel predictions into actionable real-world measurements?
> **Answer:** Using the sensor Ground Sample Distance ($\text{GSD} = 0.5\text{ m/pixel}$ for LEVIR-CD), each pixel corresponds to $0.5\text{ m} \times 0.5\text{ m} = 0.25\text{ m}^2$. Multiplying the count of positive change pixels by $0.25$ gives the estimated physical footprint of new construction in square meters ($\text{m}^2$).

