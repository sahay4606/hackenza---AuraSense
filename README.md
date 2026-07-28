<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=4&section=header"/>
  <h1>AuraSense — Hackenza 2026</h1>
  <p><b>Arabic Speech Nativity Classification · Weighted Late Fusion</b></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white"/>
    <img src="https://img.shields.io/badge/WavLM-FF6F00?logo=transformers&logoColor=white"/>
    <img src="https://img.shields.io/badge/SpeechBrain-1E3A5F?logo=speechbrain&logoColor=white"/>
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white"/>
  </p>
</div>

Arabic speech nativity classification (Native vs Non-Native) using a **Weighted Late Fusion** of **WavLM linguistic** + **ECAPA-TDNN speaker** embeddings. Winning submission for Hackenza 2026.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Audio File (.wav/.mp3)               │
└──────────┬────────────────────────────────┬──────────┘
           │                                │
           ▼                                ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Phase 1A: WavLM     │    │  Phase 1B: ECAPA-TDNN     │
│  Microsoft WavLM     │    │  SpeechBrain ECAPA-TDNN   │
│  Base Plus           │    │  (speaker verification)   │
│  768-D linguistic    │    │  192-D speaker embeddings  │
│  embeddings          │    │                          │
│  (chunked for VRAM)  │    │  (full audio, no chunk)  │
└──────────┬───────────┘    └────────────┬─────────────┘
           │                              │
           ▼                              ▼
    extracted_features/            extracted_ecapa/
     {dp_id}.pt (768-D)            {dp_id}.pt (192-D)
           │                              │
           └──────────┬───────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Weighted Late Fusion Model          │
│                                              │
│  WavLM [768] → Linear(768,32) → BN → GELU   │
│     → Dropout(0.5) → Linear(32,2) = logits_w │
│                                              │
│  ECAPA [192] → Linear(192,16) → BN → GELU   │
│     → Dropout(0.5) → Linear(16,2) = logits_e │
│                                              │
│  fused = α · logits_w + (1-α) · logits_e     │
│  α = sigmoid(learnable_param) ~ 0.85         │
└──────────────────────┬───────────────────────┘
                       ▼
              best_fusion_model.pth
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Inference on 40 test samples        │
│  → submission.csv (dp_id, status, confidence) │
└──────────────────────────────────────────────┘
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | ~0.750 |
| Validation Accuracy | ~0.750 |
| Test Accuracy | ~0.750 |
| Test F1 (macro) | ~0.750 |
| Learned α (WavLM weight) | ~0.85 |
| Model Parameters | ~27,893 |

The fusion model learns that **WavLM linguistic embeddings** contribute ~85% of the signal, while **ECAPA-TDNN speaker embeddings** contribute ~15% — indicating nativity is primarily a linguistic rather than speaker-identity trait.

---

## Quick Start

```bash
pip install -r requirements.txt
python aura_sense.py   # runs full pipeline
```

### Requirements
- Python 3.10+
- PyTorch 2.10.0 (CUDA recommended)
- 4GB+ VRAM (GPU) or 8GB+ RAM (CPU)

---

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| WavLM dimension | 768 | Microsoft WavLM Base Plus output |
| ECAPA dimension | 192 | SpeechBrain ECAPA-TDNN output |
| Chunk length | 10s | VRAM safety for long audio |
| Batch size | 16 | Works with 128 training samples |
| Initial α (alpha) | ~0.85 | Learns fusion weight |
| Patience | 15 epochs | Early stopping to prevent overfitting |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| Dropout | 0.5 | Regularization for small dataset |

---

<div align="center">
  <sub>Built with PyTorch, WavLM, SpeechBrain · Hackenza 2026</sub>
</div>
