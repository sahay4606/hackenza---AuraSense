# AuraSense — Arabic Nativity Classification | Hackenza 2026

> **Binary classification of Arabic speech recordings into Native vs Non-Native speakers using dual pretrained audio models with Weighted Late Fusion.**

---

## 📋 Problem Statement

Given a dataset of Arabic speech audio recordings, classify each speaker as **Native** or **Non-Native**.

| Dataset | Samples | Native | Non-Native |
|---------|---------|--------|------------|
| Training | 160 | 114 (71%) | 46 (29%) |
| Test | 40 | Unlabeled | Unlabeled |

---

## 🚀 How to Run — Complete Step-by-Step Guide

### Step 0 — Get the Code

```bash
git clone https://github.com/Naitik-Gupta96/AuraSense-Hackenza.git
cd AuraSense-Hackenza
```

After cloning, your folder should look like this:

```
AuraSense-Hackenza/
├── aura_sense.py
├── Nativity Assessmet Audio Dataset(Training Dataset).csv
├── Nativity Assessmet Audio Dataset(Test Dataset).csv
├── requirements.txt
└── README.md
```

> ⚠️ **Both CSV files must be present in the same folder as `aura_sense.py`.** The script reads them to find audio download URLs and ground-truth labels. If they are missing, the code will fail immediately with `FileNotFoundError`.

---

### Step 1 — Set Up Python Environment

Check your Python version first:

```bash
python --version
# Must print: Python 3.10.x or higher
```

Create and activate a virtual environment (strongly recommended — keeps packages isolated):

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

> Your terminal prompt will show `(.venv)` when the environment is active.

---

### Step 2 — Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs everything the pipeline needs:

| Package | Version | What it does |
|---------|---------|-------------|
| `torch` | 2.10.0 | Neural network training and inference |
| `torchaudio` | 2.10.0 | Audio file loading and resampling |
| `transformers` | 4.44.0 | Loads WavLM Base Plus from HuggingFace |
| `speechbrain` | 1.0.3 | Loads ECAPA-TDNN from HuggingFace |
| `huggingface_hub` | 0.36.2 | Downloads pretrained models |
| `scikit-learn` | 1.7.2 | Train/test splits, class weights, F1 score |
| `pandas` | 2.3.3 | Reading the CSV files |
| `numpy` | 2.2.6 | Array and tensor operations |
| `tqdm` | latest | Progress bars during extraction |
| `soundfile` | ≥0.12.1 | Audio I/O backend required by torchaudio on CPU/macOS |

> ⏱️ **First-time install takes 5–10 minutes** — PyTorch alone is ~2 GB.

---

### Step 3 — Run the Full Pipeline

```bash
python aura_sense.py
```

The script runs **4 phases automatically, one after the other**. You do not need to do anything between phases — just let it run. Each phase prints a `COMPLETE` message when it finishes.

> ⏱️ **Total runtime: ~50–90 minutes on first run.**  
> Most of the time is spent downloading 160 training audio files and the pretrained models from HuggingFace (~700 MB total). On subsequent runs this is near-instant since everything is cached.

---

### What Happens During Each Phase

#### ⏩ Phase 1A — WavLM Feature Extraction (~20–30 min)

```
Downloads: 160 audio files from URLs in Training CSV  →  raw_audio/
Creates:   extracted_features/{dp_id}.pt               (160 files × 768 floats)
```

1. Reads every row in `Nativity Assessmet Audio Dataset(Training Dataset).csv`
2. Downloads the audio file for each `dp_id` into `raw_audio/`
3. Converts to mono, resamples to 16 kHz
4. Splits waveform into **10-second chunks** (prevents memory overflow on CPU)
5. Passes each chunk through `microsoft/wavlm-base-plus` *(~600 MB download on first run)*
6. Mean-pools across time → averages all chunks → saves one **768-dimensional tensor** per file

Console will show:
```
[Phase 1A] Executing WavLM extraction on: cpu
Loading WavLM Base Plus...
Processing 160 files with VRAM protection...
100%|████████████████████| 160/160 [28:00<00:00]
--- PHASE 1A COMPLETE: WavLM Extraction Done ---
```

> ✅ **Fully resumable** — if it stops halfway (internet drop, power cut), just re-run. Already-extracted `.pt` files are detected and skipped automatically.

---

#### ⏩ Phase 1B — ECAPA-TDNN Feature Extraction (~10–20 min)

```
Reuses:  raw_audio/  (already downloaded in Phase 1A — no re-download)
Creates: extracted_ecapa/{dp_id}.pt  (160 files × 192 floats)
```

1. Loads each audio file already in `raw_audio/`
2. Passes full waveform through `speechbrain/spkrec-ecapa-voxceleb` *(~90 MB download)*
3. Saves one **192-dimensional speaker embedding** per file

Console will show:
```
[Phase 1B] Extracting ECAPA on: cpu
Extracting 192-D Speaker Embeddings...
100%|████████████████████| 160/160 [14:00<00:00]
--- PHASE 1B COMPLETE: ECAPA-TDNN Extraction Done ---
```

---

#### ⏩ Phase 2 — Model Training (~2–5 min)

```
Reads:   extracted_features/*.pt  +  extracted_ecapa/*.pt  +  Training CSV
Creates: best_fusion_model.pth
```

1. Loads all 160 pairs of (WavLM 768-D, ECAPA 192-D) tensors
2. Applies L2 normalization to both (puts them on the same scale)
3. Splits: **128 train / 16 val / 16 test** (stratified 80/10/10)
4. Trains `WeightedLateFusion` for up to 60 epochs with:
   - Class weighting (compensates for 71% Native / 29% Non-Native imbalance)
   - Label smoothing (0.1) to prevent overconfident predictions
   - AdamW optimizer + cosine annealing learning rate schedule
   - Early stopping with patience=15 (stops when val F1 stops improving)
   - Gradient clipping at norm=1.0
5. Saves the best checkpoint (by val F1) to `best_fusion_model.pth`
6. Evaluates on the held-out test split (never seen during training)

Console will show:
```
[Phase 2] Training Weighted Late Fusion (80/10/10 Split) on: cpu
...
┌─────────────────────────────────────────────────┐
│           FINAL RESULTS (80/10/10 Split)         │
│  Train Accuracy:      0.977                     │
│  Validation Accuracy: 0.812  (F1: 0.768)        │
│  TEST Accuracy:       0.750  (F1: 0.667)        │
│  Learned α: 0.847  → WavLM: 84.7%, ECAPA: 15.3%│
└─────────────────────────────────────────────────┘
--- PHASE 2 COMPLETE: Model saved to best_fusion_model.pth ---
```

---

#### ⏩ Phase 3 — Test Inference & Submission CSV (~10–15 min)

```
Reads:   Nativity Assessmet Audio Dataset(Test Dataset).csv  +  best_fusion_model.pth
Creates: submission.csv
```

1. Downloads all 40 test audio files
2. Extracts WavLM + ECAPA features **on-the-fly** (same pipeline as Phase 1)
3. Runs through the trained fusion model
4. Saves one row per test sample with prediction + confidence

Console will show:
```
[Phase 3] Running Weighted Late Fusion Inference on: cpu
Loaded model | α = 0.847 (WavLM: 84.7%, ECAPA: 15.3%)
Processing 40 test files...
100%|████████████████████| 40/40 [12:00<00:00]
--- PHASE 3 COMPLETE: Inference Done ---
Saved 40 predictions to 'submission.csv'
```

**✅ `submission.csv` is your final deliverable.** It has exactly 3 columns:

| dp_id | nativity_status | confidence_score |
|-------|----------------|-----------------|
| 1234  | Native         | 0.8912          |
| 1235  | Non-Native     | 0.7654          |

---

#### ⏩ Phase 4 — Head-to-Head Comparison (OPTIONAL)

This phase **only runs if these 6 files exist on disk**:
```
fusion_fold0.pth  fusion_fold1.pth  fusion_fold2.pth
fusion_fold3.pth  fusion_fold4.pth  fusion_full.pth
```

These are **not included in the repo** (model size). They are produced by the separate ensemble training notebook. If missing, Phase 4 raises `FileNotFoundError` — **this is expected and does not affect Phases 1–3 or your `submission.csv`**.

If the files are present, this phase:
- Trains v1 Aura (naive concat baseline) from scratch for comparison
- Loads the 6-model AuraSense ensemble
- Runs both on the same test set and prints a full comparison table
- Saves `submission_v1_aura.csv` and `submission_copy_of_aura.csv`

---

### Step 4 — Your Output is Ready

After Phases 1–3 complete, your folder will contain:

```
AuraSense-Hackenza/
├── submission.csv          ✅  This is your hackathon submission
├── best_fusion_model.pth       Trained model weights (27,893 params)
├── extracted_features/         768-D WavLM tensors — safe to delete after submission
├── extracted_ecapa/            192-D ECAPA tensors — safe to delete after submission
└── raw_audio/                  Downloaded audio   — safe to delete after submission
```

---

## 🔁 Stopping & Resuming Mid-Run

The pipeline is **safe to interrupt and resume at any point**.

Each phase checks whether output files already exist before processing — already-done work is always skipped. Look for the `═══ STOP POINT ═══` markers in `aura_sense.py` if you want to run one phase at a time:

```
═══ STOP POINT 1A ═══  → WavLM tensors saved to extracted_features/
═══ STOP POINT 1B ═══  → ECAPA tensors saved to extracted_ecapa/
═══ STOP POINT 2  ═══  → Model saved to best_fusion_model.pth
═══ STOP POINT 3  ═══  → submission.csv saved — pipeline complete
```

---

## 🖥️ Hardware Requirements

| Requirement | Minimum | Notes |
|------------|---------|-------|
| Python | 3.10+ | 3.10.x tested |
| RAM | 8 GB | 16 GB recommended |
| Free Storage | 10 GB | Audio + features + models |
| Internet | Required | First run only — then cached |
| GPU | Not required | CUDA gives ~3–5× speedup if available |

> Built and tested on **MacBook Air (Apple Silicon M-series, CPU-only, 8 GB RAM)** — no GPU needed.

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: Nativity Assessmet...` | CSV file not in same folder as script | Move both CSVs next to `aura_sense.py` |
| `ModuleNotFoundError: No module named 'torch'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `AttributeError: 'torchaudio' has no attribute 'list_audio_backends'` | torchaudio version mismatch | Already fixed by monkey-patch in the code |
| `RuntimeError: Couldn't find appropriate backend to handle uri` | `soundfile` not installed — torchaudio needs it on CPU/macOS | Run `pip install soundfile` (already in `requirements.txt`) |
| `FileNotFoundError: fusion_fold0.pth not found` | Ensemble model files missing | Expected — Phase 4 is optional, Phases 1–3 unaffected |
| Downloads stall or fail | Internet interruption | Re-run — resumes from where it stopped |
| `RuntimeError: CUDA out of memory` | GPU too small | Code auto-falls back to chunked CPU processing |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────┐
                    │         Audio Recording          │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌─────────────────────┐           ┌─────────────────────┐
   │  WavLM Base Plus    │           │  ECAPA-TDNN          │
   │  (Linguistic)       │           │  (Speaker Identity)  │
   │  768-D embedding    │           │  192-D embedding     │
   └──────────┬──────────┘           └──────────┬──────────┘
              │  L2 Normalize                   │  L2 Normalize
              ▼                                 ▼
   ┌─────────────────────┐           ┌─────────────────────┐
   │  WavLM Head         │           │  ECAPA Head          │
   │  768→32→BN→GELU     │           │  192→16→BN→GELU      │
   │  →Drop(0.5)→2       │           │  →Drop(0.5)→2        │
   │  → logits_w         │           │  → logits_e          │
   └──────────┬──────────┘           └──────────┬──────────┘
              └──────────┬──────────────────────┘
                         ▼
              ┌──────────────────────────┐
              │  Learned Fusion          │
              │  α · logits_w +          │
              │  (1−α) · logits_e        │
              │  α = sigmoid(param)≈0.85 │
              └──────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Native / Non-Native│
              └─────────────────────┘
```

**Why two models?**
- **WavLM** captures *how* words are pronounced — phonetics, rhythm, intonation
- **ECAPA** captures *who is speaking* — vocal tract shape, accent fingerprint
- **Weighted Late Fusion** lets the model learn how much to trust each source (converges to ~85% WavLM, 15% ECAPA)

---

## 📊 Results

### Training (80/10/10 Split)
| Split | Accuracy | F1 (macro) |
|-------|----------|------------|
| Train | 97.7% | — |
| Validation | 81.2% | 0.768 |
| **Test** | **75.0%** | **0.667** |

### Final Submission — submission.csv (40 test samples)
| Metric | Value |
|--------|-------|
| Native | 26 (65%) |
| Non-Native | 14 (35%) |
| Mean confidence | 0.764 |
| Min / Max confidence | 0.500 / 0.985 |
| Learned α | 0.847 → WavLM 84.7%, ECAPA 15.3% |

---

## 📁 Repository Structure

```
AuraSense-Hackenza/
├── aura_sense.py                                           # Main pipeline (4 phases, fully commented)
├── Nativity Assessmet Audio Dataset(Training Dataset).csv  # 160 labeled training samples ← REQUIRED
├── Nativity Assessmet Audio Dataset(Test Dataset).csv      # 40 unlabeled test samples    ← REQUIRED
├── submission.csv                                          # ✅ Pre-generated predictions (40 rows)
├── requirements.txt                                        # pip install -r requirements.txt
├── README.md                                               # This file (Technical Report)
├── .gitignore                                              # Excludes large generated files
│
└── (generated at runtime — NOT in the repo)
    ├── extracted_features/   # 160 × 768-D WavLM tensors
    ├── extracted_ecapa/      # 160 × 192-D ECAPA tensors
    ├── raw_audio/            # Downloaded training audio
    ├── test_audio_temp/      # Temporary test audio (auto-deleted)
    ├── best_fusion_model.pth # Trained model weights
    └── submission.csv        # ✅ Final predictions
```

---


## 📄 Research Foundations & Core Business Logic

Our approach is grounded in **State-of-the-Art (SOTA)** speech processing research. Each component has a peer-reviewed paper behind it.

### 1. WavLM — Self-Supervised Linguistic Representations

> **Paper:** *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing* (Microsoft Research, 2022)

**What it learns:** Predicts masked speech segments while ignoring noise — learns universal *linguistic* representations (phonetics, rhythm, prosody).

$$\text{WavLM}(x) = \sum_{i=1}^{L} w_i \cdot \text{TransformerLayer}_i(x)$$

Each of the $L=12$ transformer layers captures progressively higher-level speech features — from raw acoustics (layer 1) to abstract linguistic patterns (layer 12). We use the **last hidden state mean-pooled across time** → 768-D embedding per audio file.

**Why it works for nativity:** Native Arabic speakers have different phoneme transitions, vowel durations, and stress patterns than non-native speakers. WavLM encodes exactly these differences.

---

### 2. ECAPA-TDNN — Speaker Identity Fingerprinting

> **Paper:** *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification* (Ghent University, 2020)

**What it learns:** Identifies *who* is speaking by modeling vocal tract shape, formant structure, and speaking style through Squeeze-and-Excitation (SE) attention blocks.

$$z = \text{AttentiveStatPooling}\big(\text{SE-Res2Block}(\text{TDNN}(x))\big) \in \mathbb{R}^{192}$$

The SE blocks perform channel-wise recalibration — they learn *which frequency bands matter most* for distinguishing speakers.

**Why it complements WavLM:** A non-native speaker may pronounce phonemes correctly (fooling WavLM) but still carry an accent *fingerprint* in their vocal quality (caught by ECAPA). The two models look at fundamentally different aspects of the same audio.

---

### 3. Weighted Late Fusion — Learned Modality Blending

> **Inspired by:** *Lahjati at NADI 2025: An ECAPA-WavLM Fusion with Multi-Stage Optimization* (ACL Anthology, 2025)

The Lahjati paper demonstrated ECAPA-WavLM fusion specifically for **Arabic dialect identification** at the NADI 2025 shared task — the closest published work to our exact problem.

**Our fusion formula:**

$$\text{logits}_{\text{fused}} = \alpha \cdot f_W(\text{WavLM}) + (1 - \alpha) \cdot f_E(\text{ECAPA})$$

where:
- $f_W$ : WavLM classification head (768 → 32 → 2)
- $f_E$ : ECAPA classification head (192 → 16 → 2)
- $\alpha = \sigma(\theta) \approx 0.847$ : learned sigmoid-bounded weight

**Key difference from Lahjati:** They use *early/joint fusion* (concatenate before classifier). We use *late fusion* (separate classifiers, blend decisions). Late fusion prevents the 768-D WavLM from drowning out the 192-D ECAPA in a shared feature space.

---

### 4. Training Loss — Weighted Cross-Entropy with Label Smoothing

$$\mathcal{L} = -\sum_{c=1}^{2} w_c \cdot \tilde{y}_c \cdot \log(\hat{y}_c)$$

where:
- $w_c$ = inverse-frequency class weight (Non-Native gets ~2.5× higher weight)
- $\tilde{y}_c = (1 - \epsilon) \cdot y_c + \frac{\epsilon}{C}$ with $\epsilon = 0.1$ (label smoothing)
- $\hat{y}_c = \text{softmax}(\text{logits}_{\text{fused}})_c$

**Why this combination:**
- **Class weights** prevent the model from always predicting "Native" (71% majority)
- **Label smoothing** prevents overconfident predictions on 160 training samples
- Together they force the model to learn *discriminative features* rather than dataset statistics

---

### 5. Design Rationale Summary

| Decision | Why |
|----------|-----|
| Pretrained extractors (WavLM + ECAPA) | Transfer learning from 60,000+ hours — impossible to learn from 160 samples alone |
| Separate classification heads | Prevents 768-D from drowning 192-D in a shared space |
| Learned α (not fixed 50/50) | Model discovers WavLM is 5.7× more informative → α ≈ 0.85 |
| L2 normalization before heads | Puts both modalities on the unit sphere — fair comparison |
| 27,893 total parameters | Deliberately small classifier to prevent overfitting on 160 samples |
| AdamW + cosine annealing | Proper weight decay + cyclic LR for escaping local minima |
| 80/10/10 split (not 80/20) | Separate val (early stopping) from test (honest evaluation) |

---


## 🔧 Environment

| Component | Version |
|-----------|---------|
| Python | 3.9.6+ |
| PyTorch | 2.8.0+ |
| torchaudio | 2.8.0+ |
| Transformers | 4.44.0+ |
| SpeechBrain | 1.0.3 |
| scikit-learn | 1.6.1+ |
| soundfile | 0.12.1+ |
| Hardware | MacBook Air (Apple Silicon, CPU-only) |

---

## 👥 Team

**AuraSense** — Hackenza 2026
