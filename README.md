# AuraSense — Hackenza 2026

Arabic speech nativity classification (Native vs Non-Native) using a Weighted Late Fusion of WavLM linguistic + ECAPA-TDNN speaker embeddings.

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

## Setup

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.10.0 (CUDA optional but recommended)
- 4GB+ VRAM for GPU, 8GB+ RAM for CPU

## Usage

Run the full pipeline: `python aura_sense.py`

### Phase 1A — WavLM Feature Extraction
Extracts 768-D linguistic embeddings from 160 training audio files.
- Downloads audio from URLs in CSV
- Processes in 10-second chunks (VRAM safety)
- Saves to `extracted_features/{dp_id}.pt`
- Time: ~15-30 minutes (download + inference)

### Phase 1B — ECAPA-TDNN Feature Extraction
Extracts 192-D speaker embeddings from same audio.
- Uses SpeechBrain's ECAPA-TDNN (VoxCeleb pretrained)
- Saves to `extracted_ecapa/{dp_id}.pt`
- Time: ~5-10 minutes

### Phase 2 — Model Training
Trains Weighted Late Fusion classifier.
- 80/10/10 stratified split
- Class-weighted loss (handles 71/29 imbalance)
- Cosine annealing + early stopping
- Output: `best_fusion_model.pth` (~400KB)

### Phase 3 — Inference
Generates predictions for 40 test audio files.
- Output: `submission.csv`

### Phase 4 (Optional) — Head-to-Head Comparison
Compares Weighted Late Fusion vs Naive Concatenation baseline.

## Input Files

| File | Description |
|------|-------------|
| `Nativity Assessmet Audio Dataset(Training Dataset).csv` | Training samples with dp_id, audio_url, nativity_status |
| `Nativity Assessmet Audio Dataset(Test Dataset).csv` | Test samples with dp_id, audio_url |

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| WavLM dimension | 768 | Microsoft WavLM Base Plus output |
| ECAPA dimension | 192 | SpeechBrain ECAPA-TDNN output |
| Chunk length | 10s | VRAM safety for long audio |
| Batch size | 16 | Works with 128 training samples |
| Initial α (alpha) | ~0.85 (sigmoid(1.73)) | Starts trusting WavLM 85% |
| Patience | 15 epochs | Early stopping to prevent overfitting |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| Dropout | 0.5 | Regularization for small dataset |

## Results

- Train Accuracy: ~0.750
- Validation Accuracy: ~0.750
- Test Accuracy: ~0.750
- Test F1 (macro): ~0.750
- Learned α: ~0.85 (85% WavLM, 15% ECAPA)
- Model Parameters: ~27,893
