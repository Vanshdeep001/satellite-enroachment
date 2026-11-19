# Satellite Land Use Change and Encroachment Detector

**Course:** Deep Learning — Project-Based Learning (PBL) / Project 50  
**Institution:** UPES, Semester 5  
**Language & Framework:** Python 3.11.14 · PyTorch 2.15.0 (Apple Silicon MPS Acceleration)  
**Dataset:** LEVIR-CD Benchmark (Very High-Resolution Building Change Detection)  
**Status:** ✅ Phase 8 Complete — Final Model Selection, Optimisation & Validation  

---

## 1. Project Overview

This project develops an end-to-end deep learning application that ingests bitemporal satellite optical imagery ($T_1$ and $T_2$) of identical geographic coordinates acquired at different calendar epochs and accurately segments **pixel-level structural changes (primarily building construction and horizontal expansion)**.

The application computes quantitative evaluation metrics (Precision, Recall, F1-Score, and Intersection-over-Union), establishes a classical unsupervised baseline, estimates physical altered surface area ($\text{m}^2$), and conditionally cross-references candidate alterations with verified restricted-area boundary polygons.

> [!IMPORTANT]
> **Responsible AI & Legal Safeguard:** The deep learning model is an image processing system that detects visual morphological changes; it does **not** adjudicate legal compliance. Potential encroachment is flagged **only** when authenticated restricted-area vector boundaries (e.g., forest reserves, public easements, wetland buffer zones) are explicitly provided. All flagged incidents are advisory alerts for field survey audits.

---

## 2. Project Lifecycle & Phase Tracking

| Phase | Description | Status | Key Deliverables |
|---|---|---|---|
| **Phase 0** | Environment setup, `.venv311`, PyTorch MPS verification | ✅ Complete | Python 3.11.14, PyTorch 2.15, zero conflicts |
| **Phase 1** | Problem definition, literature review, architecture & experiment design | ✅ Complete | Complete theoretical specification in `docs/` |
| **Phase 2** | LEVIR-CD acquisition, audit, on-the-fly PyTorch Dataset & DataLoader | ✅ Complete | 637 verified pairs, `src/dataset.py`, unit tests |
| **Phase 3** | Traditional pixel-difference baseline, threshold tuning & metrics | ✅ Complete | `src/baseline.py`, `src/metrics.py`, `docs/baseline_experiment.md` |
| **Phase 4** | Siamese U-Net architecture, multi-scale differencing & composite loss | ✅ Complete | `src/model.py`, `src/losses.py`, `src/config.py`, 24 unit tests |
| **Phase 5** | Siamese U-Net training, validation tuning, checkpoints & learning curves | ✅ Complete | `src/train.py`, `best_model.pt`, Val F1: 73.44% |
| **Phase 6** | Blind test set evaluation, comparative benchmarks & area ($\text{m}^2$) | ✅ Complete | `src/evaluate.py`, Test F1: 72.78%, IoU: 57.21% |
| **Phase 7** | Ablation studies, augmentations, loss re-weighting & 4-optimizer benchmark | ✅ Complete | `docs/phase_7_experiment_report.md`, `src/experiments/` |
| **Phase 8** | Final model selection (15 epochs), optimisation, inference engine & validation | ✅ Complete | `src/select_final_model.py`, `src/inference.py`, Val F1: 82.94% |
| **Fundamentals** | From-scratch NumPy tensor sandbox (broadcasting, reshape, Jacobi SVD, gradient checking), McCulloch-Pitts & sigmoid neurons, MLP backprop, GD/Momentum/Nesterov, parameter tracking | ✅ Complete | `src/fundamentals/`, `docs/fundamentals.md`, 45 unit tests |
| **Step 5 Ablations** | From-scratch GD / SGD / mini-batch / Momentum / Nesterov comparison; init, normalisation (incl. BatchNorm), L2 / dropout / early stopping, augmentation and data-size studies (3 seeds) | ✅ Complete | `src/fundamentals/experiments.py`, `docs/optimization_experiments.md` |
| **Step 6 Classical learning** | Hebbian, Oja, Sanger GHA; winner-take-all (+ conscience) and Kohonen SOM; perceptron vs LMS delta rule; Hopfield (Hebbian vs projection storage, capacity) and RBM (CD-1/CD-10 vs exact ML) | ✅ Complete | `src/fundamentals/classical_learning.py`, `docs/classical_learning.md` |
| **Phase 9** | Cadastral boundary encroachment risk overlay, Gradio interactive UI demo & viva defense | 🔲 Next | Interactive web demo, boundary overlays, final code release |

---

## 3. Directory Structure

```
satellite-change-detector/
├── README.md                     ← Master project documentation & roadmap
├── requirements.txt              ← Pinned dependencies (torch 2.15, numpy 2.4, etc.)
├── activate.sh                   ← One-command virtual environment & env variable activation
├── .gitignore                    ← Strict exclusions for data/, outputs/, checkpoints
│
├── src/                          ← Modular Python source code
│   ├── __init__.py
│   ├── config.py                 ← Centralized configuration dataclasses (Model, Loss, Training)
│   ├── model.py                  ← Siamese U-Net with multi-scale absolute feature difference
│   ├── losses.py                 ← SoftDiceLoss & CompositeChangeDetectionLoss (BCE + Dice)
│   ├── dataset.py                ← LEVIRCDDataset & get_dataloader pipeline
│   ├── baseline.py               ← Traditional pixel-difference baseline (Euclidean, MAD, gray)
│   ├── metrics.py                ← MetricTracker & confusion matrix utilities (Precision, Recall, F1, IoU)
│   ├── visualize.py              ← 5-panel sample comparison & tuning curve plotting
│   ├── run_baseline_experiment.py ← End-to-end threshold tuning & test evaluation runner
│   ├── benchmark_model.py        ← Parameter counts, tensor shape traces & MPS benchmarks
│   ├── train.py                  ← End-to-end model training loop with validation checkpointing
│   ├── evaluate.py               ← Blind test evaluation, baseline comparison & area scaling
│   ├── run_ablation.py           ← Multi-architecture & training improvement ablation runner
│   ├── run_optimizer_comparison.py ← 4-optimizer (SGD, Momentum, NAG, AdamW) benchmark runner
│   ├── select_final_model.py     ← Phase 8 standardized 15-epoch head-to-head candidate selector
│   ├── inference.py              ← Decoupled production inference engine with 4-panel visualizer
│   └── experiments/              ← Specialized ablation models & augmentations
│       ├── __init__.py
│       ├── models.py             ← SiameseUNetConc & SiameseUNetNoSkip architectures
│       └── augmentations.py      ← Synchronous bitemporal spatial augmentations
│
├── notebooks/                    ← Interactive Jupyter exploration notebooks
│
├── data/                         ← Local dataset storage (git-ignored)
│   ├── raw/                      ← Original verified archives (train, val, test .zip)
│   ├── train/                    ← 445 verified triplets (A, B, label)
│   ├── val/                      ← 64 verified triplets (A, B, label)
│   ├── test/                     ← 128 verified triplets (A, B, label)
│   ├── dataset_manifest.csv      ← Full 637-pair metadata manifest
│   └── audit_summary.json        ← Programmatic audit statistics
│
├── outputs/                      ← Reproducible artifacts (git-ignored)
│   ├── sample_pairs_preview.png  ← Multi-pair visual verification grid
│   ├── architecture_diagram.png  ← Visual Siamese U-Net architectural dataflow
│   ├── baseline_tuning_curve.png ← Precision/Recall/F1/IoU vs. threshold plot
│   ├── baseline_samples/         ← 5-panel visual comparisons on test pairs
│   ├── val_predictions/          ← 4-panel visual predictions on validation pairs
│   ├── test_predictions/         ← 6-panel visual predictions with error overlays on test pairs
│   ├── checkpoints/              ← Serialized model weights (git-ignored)
│   │   ├── final_selected_model.pt ← Final production model (SiameseUNetConc, Val F1: 82.94%)
│   │   ├── best_model.pt         ← Historical Phase 5 checkpoint (Diff Skips, Epoch 9)
│   │   └── final_model.pt        ← Final epoch checkpoint
│   ├── experiments/              ← Experiment candidate weights and metrics
│   │   ├── checkpoints/          ← Isolated candidate model weights (ablation, 15ep, optimizers)
│   │   ├── ablation_summary.csv  ← Ablation results tabular summary
│   │   ├── ablation_results.json ← Full epoch trajectories for ablations
│   │   ├── optimizer_summary.csv ← Optimizer comparison tabular summary
│   │   └── optimizer_comparison.json ← Optimizer trajectories across epochs
│   ├── plots/                    ← Training curves and loss progressions
│   │   ├── loss_curve.png        ← Composite training vs. validation loss
│   │   ├── val_metrics_curve.png ← Validation F1-score & IoU across epochs
│   │   ├── lr_schedule.png       ← Cosine Annealing learning rate progression
│   │   ├── training_summary_panel.png ← Comprehensive 4-quadrant training panel
│   │   ├── test_metrics_comparison.png ← Grouped bar chart comparing baseline vs. Siamese U-Net
│   │   ├── ablation_comparison.png     ← Phase 7 ablation & improvement comparison bar chart
│   │   ├── experiment_learning_curves.png ← Validation F1 trajectories across ablations
│   │   ├── optimizer_comparison.png    ← 4-optimizer convergence & metric comparison
│   │   └── final_model_comparison.png  ← Phase 8 Candidate A vs. Candidate B 15-epoch panel
│   └── metrics/                  ← Machine-readable JSON/CSV evaluation metrics
│       ├── final_model_comparison.csv ← Head-to-head metrics table (Candidate A vs. Candidate B)
│       ├── final_model_selection.json  ← Comprehensive selection metadata and timing
│       ├── baseline_val_tuning.json   ← Validation threshold grid search results
│       ├── baseline_test_metrics.json ← Baseline test evaluation (tau* = 0.40)
│       ├── model_benchmark_summary.json ← Parameter counts & latency benchmarks
│       ├── training_history.csv       ← Per-epoch training & validation log
│       ├── best_threshold.json        ← Optimal probability decision threshold (tau* = 0.20)
│       ├── test_baseline_comparison.csv ← Test comparison table
│       └── final_test_metrics.json    ← Final blind test evaluation metrics
│
├── tests/                        ← Unit and regression test suites
│   ├── test_dataset.py           ← Automated unit tests for Dataset & DataLoader
│   ├── test_metrics.py           ← Unit tests for confusion matrix & metric calculations
│   ├── test_baseline.py          ← Unit tests for pixel differencing & thresholding
│   ├── test_model.py             ← Unit tests for Siamese U-Net, losses & MPS execution
│   ├── test_experiments.py       ← Unit tests for ablation models, augmentations & loss weights
│   └── test_inference.py         ← Unit tests for production inference pipeline & error handling
│
└── docs/                         ← Comprehensive project documentation & specifications
    ├── problem_statement.md      ← Formal problem formulation, stakeholders, ethical boundary
    ├── literature_review.md      ← Critical survey of classical, CNN, Siamese, & Transformer CD
    ├── project_objectives.md     ← FRs, NFRs, success criteria table, evaluation metrics
    ├── dataset_card.md           ← Complete technical card & licensing for LEVIR-CD
    ├── data_audit_report.md      ← Verified empirical audit, split counts, class imbalance
    ├── data_preprocessing.md     ← On-the-fly resizing, interpolation & PyTorch pipeline
    ├── baseline_experiment.md    ← Traditional baseline mathematical formulation, tuning & test results
    ├── model_architecture.md     ← Complete Siamese U-Net technical specification & tensors
    ├── training_report.md        ← Full Phase 5 training progression, curves & threshold tuning
    ├── test_evaluation_report.md ← Final blind test benchmarking, comparison & error analysis
    ├── phase_7_experiment_report.md ← Phase 7 ablation studies, training improvements, optimizer comparisons & viva Q&A
    ├── final_model_selection_report.md ← Phase 8 Candidate A vs. Candidate B selection report, trade-offs & viva Q&A
    ├── proposed_architecture.md  ← End-to-end 9-stage pipeline & Siamese U-Net Mermaid design
    ├── experiment_plan.md        ← Phase schedule, training config, 4 planned ablation studies
    ├── risk_and_limitations.md   ← Radiometric noise, class imbalance, domain gap, ethics
    ├── references.md             ← Peer-reviewed bibliography & formal AI disclosure
    └── project_setup.md          ← Local hardware, macOS environment, & package audit
```

---

## 4. Setup and Verification

To activate the isolated virtual environment:
```bash
cd "Documents/UPES/Semester 5/Deep Learning Lab/satellite-change-detector"
source activate.sh
```

To run end-to-end inference on any registered satellite pair using the production model:
```bash
python src/inference.py \
  --t1 data/val/A/val_1.png \
  --t2 data/val/B/val_1.png \
  --output outputs/inference_sample_val_1.png
```

To render the neural-network fundamentals explainer figures (tensor sandbox, SVD, gradient checks, neurons, MLP):
```bash
python -m src.fundamentals.demo
```

To reproduce the Step 5 optimiser / initialisation / normalisation / regularisation / augmentation ablations (~9 min on CPU):
```bash
python -m src.fundamentals.experiments
```

To render the Step 6 Hebbian / competitive / error-correction / Hopfield / Boltzmann demonstrations (~70 s):
```bash
python -m src.fundamentals.classical_demo
```

To reproduce the Phase 8 standardized 15-epoch head-to-head model selection:
```bash
python src/select_final_model.py
```

To reproduce the complete blind test evaluation in a single command:
```bash
python src/evaluate.py
```

To reproduce the Phase 7 ablation studies and optimizer benchmarks:
```bash
# Run architectural ablations and training improvement experiments
python src/run_ablation.py

# Run controlled 4-optimizer comparison (SGD, Momentum, NAG, AdamW)
python src/run_optimizer_comparison.py
```

To verify deep learning environment and Apple Silicon GPU acceleration:
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS (Apple GPU) Available:', torch.backends.mps.is_available())"
python -c "import numpy, matplotlib, sklearn; print('Scientific stack verified ✅')"
```

---

## 5. Dataset Information

* **Benchmark:** LEVIR-CD (Large-scale Remote Sensing Building Change Detection Dataset)
* **Official Portal:** [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)
* **Spatial Resolution:** $0.5\text{ meters per pixel}$ (VHR optical RGB)
* **License:** Academic and non-commercial research use only.
* **Dataset Status:** ✅ Fully acquired, audited, and verified. 637 pairs ($1024 \times 1024$ px) partitioned into 445 train, 64 validation, and 128 test pairs. On-the-fly PyTorch ingestion pipeline verified via unit tests.

---

## 6. Academic Integrity and AI Disclosure

This project is conducted for the **Semester 5 Deep Learning PBL (Project 50)** at UPES. Scaffolding, literature synthesis assistance, and markdown templates were developed with AI assistance (Google Antigravity / Gemini). All model architectures, mathematical formulations, experimental designs, and code implementations are directed, audited, and defended by the student. Refer to [`docs/references.md`](docs/references.md) for complete academic disclosures.
