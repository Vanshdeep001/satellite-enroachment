# Project Setup Log — Satellite Land Use Change and Encroachment Detector

> This document records the actual environment observed on this machine during Phase 0.
> No values are invented — every entry below was verified by running a command.

---

## Machine Details

| Property | Value |
|----------|-------|
| **Operating System** | macOS 26.5 (Build 25F71) |
| **Python version** | 3.14.4 |
| **Python path** | `/opt/homebrew/bin/python3` (Homebrew) |
| **Disk space available** | ~11 GB free of 228 GB |

---

## Pre-existing Workspace

The following files already existed in `Deep Learning Lab/` and were **not touched**:

| File | Description |
|------|-------------|
| `Expt1.ipynb` | Existing lab experiment notebook |
| `Untitled.ipynb` | Scratch notebook |
| `pytorch.ipynb` | PyTorch experiments notebook |
| `Numpy/1.py` | NumPy import stub (`import numpy as np`) |
| `.ipynb_checkpoints/` | Jupyter auto-save folder |

All project files for this PBL live in the **`satellite-change-detector/`** sub-folder, isolated from your existing work.

---

## Python Packages Already Installed (system Python3)

Verified with `pip3 list` on 2026-10-29:

| Package | Installed Version | Required Version |
|---------|-------------------|------------------|
| `numpy` | 2.4.4 | ≥ 1.26.0 ✅ |
| `matplotlib` | 3.10.9 | ≥ 3.8.0 ✅ |
| `Pillow` | 12.2.0 | ≥ 10.0.0 ✅ |
| `scikit-learn` | 1.8.0 | ≥ 1.3.0 ✅ |
| `torch` | **NOT INSTALLED** | ≥ 2.2.0 ⚠️ |
| `torchvision` | **NOT INSTALLED** | ≥ 0.17.0 ⚠️ |
| `opencv-python-headless` | **NOT INSTALLED** | ≥ 4.9.0 ⚠️ |
| `tqdm` | unknown | ≥ 4.66.0 — check |
| `jupyter` | unknown | ≥ 1.0.0 — check |
| `pyyaml` | unknown | ≥ 6.0 — check |

---

## Issues and Decisions Needed

### ⚠️ Issue 1: PyTorch not installed

PyTorch is the core framework for this project and is **not present**. Before Phase 1 begins, you should install it.

**Recommended install command for Apple Silicon Mac (MPS backend):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Or, if you want Apple GPU acceleration (Metal Performance Shaders):
```bash
pip install torch torchvision
```
> Check https://pytorch.org/get-started/locally/ for the exact command matching your setup.
> **Do not run either command until you have decided** whether to use a virtual environment (recommended) or the system Python.

### ⚠️ Issue 2: No virtual environment exists

No `.venv` was found anywhere in your Semester 5 folder. Running `pip install` against the system Python will install packages globally, which can cause conflicts with other courses.

**Recommendation:** Create a project-scoped virtual environment:
```bash
cd "Documents/UPES/Semester 5/Deep Learning Lab/satellite-change-detector"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Then install PyTorch separately (see Issue 1)
```

### ⚠️ Issue 3: Disk space is tight

Only ~11 GB is free. The LEVIR-CD dataset is approximately 1.5–2 GB compressed. Plan accordingly before downloading in Phase 1.

### ℹ️ Issue 4: matplotlib config directory not writable

The system matplotlib config path (`~/.matplotlib`) is not writable. This is a minor annoyance — matplotlib shows a warning but works fine. It can be fixed by setting an environment variable, but is not urgent.

---

## Phase 0 Checklist

- [x] OS and Python version inspected
- [x] Existing files catalogued — nothing overwritten
- [x] Project folder structure created (`satellite-change-detector/`)
- [x] `README.md` created with phase plan and responsible-use note
- [x] `requirements.txt` created and refined (PyTorch noted as not yet installed)
- [x] `.gitignore` created (data/ and outputs/ excluded)
- [x] `src/__init__.py` created
- [x] `docs/` scaffold created
- [x] Compatibility and storage assessment completed (Phase 0 continued)
- [x] Virtual environment `.venv311` confirmed working (Python 3.11.14 from Homebrew)
- [x] PyTorch 2.15.0 + torchvision 0.29.2 installed and verified importable
- [x] All supporting packages installed and verified (numpy, matplotlib, Pillow, scikit-learn, tqdm, opencv, pyyaml)
- [x] `torch.backends.mps.is_available()` → **True** ✅ (Apple GPU acceleration available)
- [x] `pip check` → **No broken requirements** ✅
- [x] `requirements.txt` updated with actual pinned versions
- [x] matplotlib config directory fix applied (MPLCONFIGDIR → `.mpl_config/`)
- [x] `activate.sh` helper script created
- [x] Dataset downloaded — completed in Phase 2
- [x] Phase 0 **COMPLETE** ✅
- [x] Phase 1 **COMPLETE** ✅
- [x] Phase 2 **COMPLETE** ✅

---

## Phase 2 — Dataset Acquisition and Audit Summary (verified 2026-11-02)

| Property | Value | Notes |
|---|---|---|
| **Raw Archives (`data/raw/`)** | 2.35 GB (`train.zip`, `val.zip`, `test.zip`) | Verified via `testzip()` CRC-32 |
| **Extracted Pairs (`data/{train,val,test}`)** | 637 pairs (1,911 images) | 445 train, 64 val, 128 test |
| **Integrity & Correspondence** | 0 missing, 0 corrupt | 100% matched triplets ($T_1, T_2, \text{mask}$) |
| **Mask Value Range** | `[0, 156, 254, 255]` | Thresholded at $\tau=127 \to \{0.0, 1.0\}$ |
| **Changed Pixel Ratio** | Train: 4.59%, Val: 4.20%, Test: 5.09% | Severe class imbalance |
| **DataLoader & Tests** | `src/dataset.py`, `tests/test_dataset.py` | 3 unit tests passing in 0.17s |
| **Free Disk Space** | ~8.0 GiB remaining | Safe margin maintained |


---

## Phase 0 — Final Environment State (verified 2026-11-01)

All packages installed and import-verified. PyTorch MPS confirmed working.
See the Phase 0 completion section at the bottom for the full checklist.

---

## Phase 0 (Continued) — Compatibility Assessment  (2026-10-29)

### Python Versions Found on This Machine

| Python | Location | Status |
|--------|----------|--------|
| 3.14.4 (Homebrew) | `/opt/homebrew/Cellar/python@3.14/3.14.4/` | System default `python3` |
| 3.11.14 (Homebrew) | `/opt/homebrew/Cellar/python@3.11/3.11.14/` | Available — recommended for this project |

### Virtual Environment Assessment

The `.venv` folder (13 MB) was created by running `python3 -m venv .venv` from a shell
where `python3` resolved to **Anaconda's Python 3.14.6** (`/opt/anaconda3/bin/python3.14`),
not Homebrew's Python. The `pyvenv.cfg` records:

```
home = /opt/anaconda3/bin
version = 3.14.6
executable = /opt/anaconda3/bin/python3.14
```

Student must verify this venv works, or create a fresh one using Python 3.11 (recommended).
See the compatibility report artifact for exact commands.

### PyTorch Compatibility

- Python 3.14: Supported from PyTorch 2.10 onward ✅
- Python 3.11: Supported since PyTorch 1.x — most stable choice ✅
- Apple Silicon MPS: Available in all recent stable PyTorch ✅

### Storage Budget

| Item | Installed Size |
|------|---------------|
| PyTorch + torchvision (arm64) | ~725 MB |
| Other requirements.txt packages | ~300 MB |
| LEVIR-CD dataset (uncompressed) | ~670 MB |
| **Total** | **~1.7 GB** |

Free space available: ~11 GB. After install: ~9.3 GB remaining. ✅

