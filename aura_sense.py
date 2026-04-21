"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AuraSense — Hackenza 2026                                ║
║      Arabic Nativity Classification (Native vs Non-Native)                  ║
║                                                                              ║
║  Author : Rishabh Sahay                                                      ║
║  Task   : Binary classification of Arabic speech into Native / Non-Native    ║
║  Data   : 160 labeled training samples (114 Native, 46 Non-Native)           ║
║           40 unlabeled test samples                                          ║
║                                                                              ║
║  PIPELINE OVERVIEW (Run in order — each phase depends on the previous):      ║
║                                                                              ║
║   PHASE 1 — Feature Extraction                                               ║
║     ├── Step 1: WavLM 768-D linguistic embeddings  (→ extracted_features/)   ║
║     └── Step 2: ECAPA-TDNN 192-D speaker embeddings (→ extracted_ecapa/)     ║
║                                                                              ║
║   PHASE 2 — Model Training                                                   ║
║     └── Step 3: Weighted Late Fusion (80/10/10 split → best_fusion_model.pth)║
║                                                                              ║
║   PHASE 3 — Test Inference                                                   ║
║     └── Step 4: Generate submission.csv from 40 test files                   ║
║                                                                              ║
║   PHASE 4 — Head-to-Head Comparison (OPTIONAL)                               ║
║     └── Step 5: v1 Aura vs AuraSense side-by-side comparison                 ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║   • You can run the entire file top-to-bottom, OR                            ║
║   • Break at the ═══ STOP POINT ═══ markers and run each phase separately    ║
║   • Each phase prints a clear "COMPLETE" message when done                   ║
║   • Phase 2 needs Phase 1's .pt files on disk                                ║
║   • Phase 3 needs Phase 2's .pth model on disk                               ║
║   • Phase 4 needs fusion_fold0-4.pth + fusion_full.pth (from ensemble        ║
║     training notebook: Copy_of_aurasense_hackenza.ipynb)                     ║
║                                                                              ║
║  INPUT FILES REQUIRED (must be in same directory):                            ║
║   • Nativity Assessmet Audio Dataset(Training Dataset).csv                   ║
║   • Nativity Assessmet Audio Dataset(Test Dataset).csv                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# PHASE 1A — WavLM Feature Extraction (768-D Linguistic Embeddings)
# =============================================================================
#
# WHAT THIS DOES:
#   Downloads each audio file from the training CSV, processes it through
#   Microsoft's WavLM Base Plus (a self-supervised speech model trained on
#   60,000+ hours of audio), and saves a 768-dimensional embedding per sample.
#
# WHY CHUNKED PROCESSING:
#   Long audio files (> 10 seconds) can exceed GPU/CPU memory. We split each
#   waveform into 10-second chunks, extract embeddings per chunk, then average
#   them into one final 768-D vector. This is the VRAM-safe approach.
#
# INPUTS:
#   • Nativity Assessmet Audio Dataset(Training Dataset).csv  (contains dp_id, audio_url)
#
# OUTPUTS:
#   • extracted_features/{dp_id}.pt — one 768-D tensor per training sample
#   • raw_audio/{dp_id}.{ext}       — downloaded audio files (intermediate)
#
# TIME: ~15-30 minutes depending on internet speed (downloads 160 audio files)
# =============================================================================

import os
import pandas as pd
import torch
import torchaudio
import urllib.request
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from tqdm import tqdm
import gc

# --- Monkey-patch for torchaudio compatibility (newer versions removed list_audio_backends) ---
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

# --- Hardware detection: CUDA GPU if available, else CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Phase 1A] Executing WavLM extraction on: {device}")

# --- Load the pretrained WavLM Base Plus model ---
# We use fp16 (half-precision) on CUDA to halve VRAM usage, fp32 on CPU
print("Loading WavLM Base Plus...")
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus",
    torch_dtype=model_dtype
).to(device)
model.eval()  # Inference-only mode — no gradient computation needed

# --- Create output directories ---
os.makedirs("raw_audio", exist_ok=True)            # Temporary: stores downloaded .wav/.mp3 files
os.makedirs("extracted_features", exist_ok=True)    # Final output: stores 768-D .pt tensors

# --- Load training metadata ---
df = pd.read_csv('Nativity Assessmet Audio Dataset(Training Dataset).csv')
print(f"Processing {len(df)} files with VRAM protection...")

failed_downloads = []
CHUNK_LENGTH_SEC = 10       # Maximum seconds of audio per chunk (memory safety limit)
CHUNK_SAMPLES = CHUNK_LENGTH_SEC * 16000  # 10 sec × 16000 Hz = 160,000 samples per chunk

for index, row in tqdm(df.iterrows(), total=len(df)):
    dp_id = row['dp_id']
    url = row['audio_url']

    ext = url.split('.')[-1]
    audio_path = f"raw_audio/{dp_id}.{ext}"
    tensor_path = f"extracted_features/{dp_id}.pt"

    # Skip if already extracted (allows resuming after interruptions)
    if os.path.exists(tensor_path):
        continue

    try:
        # Download audio if not already on disk
        if not os.path.exists(audio_path):
            urllib.request.urlretrieve(url, audio_path)

        # Load audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo → mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz (WavLM's expected sample rate)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)

        waveform = waveform.squeeze()  # Shape: [Total_Samples] — flat 1D tensor

        # --- CHUNKED PROCESSING (the VRAM fix) ---
        # Split waveform into 10-second slices so we don't blow up memory
        chunks = torch.split(waveform, CHUNK_SAMPLES)
        chunk_embeddings = []

        with torch.no_grad():  # No gradients needed — pure inference
            for chunk in chunks:
                # Skip tiny chunks (< 0.1 seconds = 1600 samples) — not enough audio
                if len(chunk) < 1600:
                    continue

                # Prepare input for WavLM (adds padding, attention mask, etc.)
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(device, dtype=model_dtype)

                # Forward pass through WavLM — outputs hidden states for each time frame
                outputs = model(input_values)

                # Mean-pool across time dimension → single 768-D vector for this chunk
                # Then immediately move to CPU to free GPU memory
                hidden_state = outputs.last_hidden_state  # Shape: [1, time_frames, 768]
                pooled_chunk = torch.mean(hidden_state, dim=1).squeeze().cpu()  # Shape: [768]

                chunk_embeddings.append(pooled_chunk)

                # Aggressively free GPU memory after each chunk
                del inputs, input_values, outputs, hidden_state

        # Average all chunk embeddings → one final 768-D vector per audio file
        if len(chunk_embeddings) > 0:
            final_embedding = torch.stack(chunk_embeddings).mean(dim=0)  # Shape: [768]
            torch.save(final_embedding, tensor_path)
        else:
            print(f"Warning: dp_id {dp_id} audio was completely empty.")

        # Clear all memory from this sample before processing next
        del waveform, chunks, chunk_embeddings
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"\nFailed processing dp_id {dp_id}: {e}")
        failed_downloads.append(dp_id)

print("\n--- PHASE 1A COMPLETE: WavLM Extraction Done ---")
if failed_downloads:
    print(f"Failed IDs: {failed_downloads}")

# =============================================================================
# (OPTIONAL) Backup WavLM features to zip — useful if running on Colab/cloud
# =============================================================================

import shutil
shutil.make_archive('extracted_features_backup', 'zip', '.', 'extracted_features')
print("Saved: extracted_features_backup.zip")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ STOP POINT 1A ═══  You can pause here. WavLM features are saved to disk.
#                         Next: Run Phase 1B (ECAPA extraction) below.
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# PHASE 1B — ECAPA-TDNN Feature Extraction (192-D Speaker Embeddings)
# =============================================================================
#
# WHAT THIS DOES:
#   Processes the same audio files through SpeechBrain's ECAPA-TDNN model
#   (trained on VoxCeleb for speaker verification). Extracts a 192-dimensional
#   speaker embedding that captures vocal tract shape, accent fingerprint, and
#   speaking style — complementary to WavLM's linguistic features.
#
# WHY NO CHUNKING:
#   ECAPA-TDNN is much lighter than WavLM and handles full audio natively.
#
# INPUTS:
#   • Nativity Assessmet Audio Dataset(Training Dataset).csv
#   • raw_audio/ folder (already downloaded in Phase 1A)
#
# OUTPUTS:
#   • extracted_ecapa/{dp_id}.pt — one 192-D tensor per training sample
#
# DEPENDENCY: Phase 1A must have run first (or raw_audio/ must exist)
# =============================================================================

import os
import pandas as pd
import torch
import torchaudio
import urllib.request

# --- Monkey-patch for SpeechBrain/torchaudio compatibility ---
# Newer torchaudio versions removed list_audio_backends(), but SpeechBrain still calls it.
# This patch prevents the ImportError by providing a dummy function.
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Phase 1B] Extracting ECAPA on: {device}")

# --- Load the pretrained ECAPA-TDNN speaker verification model ---
# This model was trained on VoxCeleb (thousands of speaker identities)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

os.makedirs("extracted_ecapa", exist_ok=True)
os.makedirs("raw_audio", exist_ok=True)

df = pd.read_csv('Nativity Assessmet Audio Dataset(Training Dataset).csv')
print("Extracting 192-D Speaker Embeddings...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    dp_id = row['dp_id']
    url = row['audio_url']
    ext = url.split('.')[-1]

    audio_path = f"raw_audio/{dp_id}.{ext}"
    ecapa_path = f"extracted_ecapa/{dp_id}.pt"

    # Skip if already extracted
    if os.path.exists(ecapa_path):
        continue

    try:
        # Download if not already present from Phase 1A
        if not os.path.exists(audio_path):
            urllib.request.urlretrieve(url, audio_path)

        # Load raw audio (no resampling needed — SpeechBrain handles it internally)
        signal, fs = torchaudio.load(audio_path)

        with torch.no_grad():
            # encode_batch() returns shape [1, 1, 192] — we squeeze to [192]
            embeddings = classifier.encode_batch(signal)
            speaker_embedding = embeddings.squeeze()

        # Save 192-D speaker embedding to disk
        torch.save(speaker_embedding.cpu(), ecapa_path)

    except Exception as e:
        pass  # Skip failed downloads silently (already logged in Phase 1A)

print("\n--- PHASE 1B COMPLETE: ECAPA-TDNN Extraction Done ---")

# =============================================================================
# (OPTIONAL) Backup ECAPA features to zip
# =============================================================================

import shutil
shutil.make_archive('extracted_ecapa_backup', 'zip', '.', 'extracted_ecapa')
print("Saved: extracted_ecapa_backup.zip")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ STOP POINT 1B ═══  Both feature sets are now on disk.
#                         extracted_features/ → 160 × 768-D WavLM tensors
#                         extracted_ecapa/    → 160 × 192-D ECAPA tensors
#                         Next: Run Phase 2 (Model Training) below.
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# (OPTIONAL) Restore features from backup zips
# =============================================================================
# Use this if you're resuming on a new machine and have the zip backups
# but not the extracted folders. Skip if folders already exist.

import zipfile

for archive in ['extracted_features_backup.zip', 'extracted_ecapa_backup.zip']:
    if os.path.exists(archive):
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted: {archive}")
    else:
        print(f"Skipped (not found): {archive}")


# =============================================================================
# PHASE 2 — Weighted Late Fusion Model Training
# =============================================================================
#
# WHAT THIS DOES:
#   Loads the pre-extracted WavLM (768-D) and ECAPA (192-D) embeddings,
#   splits data into 80% train / 10% val / 10% test (stratified by class),
#   and trains a Weighted Late Fusion classifier.
#
# ARCHITECTURE:
#   WavLM [768] → Linear(768,32) → BN → GELU → Dropout(0.5) → Linear(32,2) → logits_w
#   ECAPA [192] → Linear(192,16) → BN → GELU → Dropout(0.5) → Linear(16,2) → logits_e
#   Fused logits = α · logits_w + (1−α) · logits_e
#   where α = sigmoid(learnable_parameter), initialized ≈ 0.85
#
# WHY THIS ARCHITECTURE:
#   • Separate heads prevent 768-D WavLM from drowning out 192-D ECAPA
#   • Head sizes proportional to input dims (32 for 768, 16 for 192)
#   • Learned α lets the model decide optimal modality weighting
#   • L2 normalization puts both modalities on same scale before classification
#
# INPUTS:
#   • Nativity Assessmet Audio Dataset(Training Dataset).csv
#   • extracted_features/{dp_id}.pt  (from Phase 1A)
#   • extracted_ecapa/{dp_id}.pt     (from Phase 1B)
#
# OUTPUTS:
#   • best_fusion_model.pth  (checkpoint with best validation F1)
#   • Console: train/val/test accuracy, F1, confusion matrix
#
# DEPENDENCY: Phase 1A + 1B must be complete (both .pt folder must exist)
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Phase 2] Training Weighted Late Fusion (80/10/10 Split) on: {device}")

# ──────────────────────────────────────────────
# 2.1 LOAD ALL DATA — filter to only samples that have BOTH feature types
# ──────────────────────────────────────────────
df = pd.read_csv('Nativity Assessmet Audio Dataset(Training Dataset).csv')

all_ids, all_labels = [], []
label_map = {"Native": 0, "Non-Native": 1}

for _, row in df.iterrows():
    dp_id = row['dp_id']
    # Only include samples where both WavLM AND ECAPA tensors exist
    if os.path.exists(f"extracted_features/{dp_id}.pt") and \
       os.path.exists(f"extracted_ecapa/{dp_id}.pt"):
        all_ids.append(dp_id)
        all_labels.append(label_map[row['nativity_status']])

print(f"Total samples with both features: {len(all_ids)} "
      f"(Native: {sum(1 for l in all_labels if l==0)}, "
      f"Non-Native: {sum(1 for l in all_labels if l==1)})")

# ──────────────────────────────────────────────
# 2.2 STRATIFIED 80/10/10 SPLIT
# ──────────────────────────────────────────────
# Why stratified? Our data is imbalanced (71% Native, 29% Non-Native).
# Stratification ensures each split preserves this ratio.
# Why 80/10/10? We need separate val (for early stopping) and test (for honest evaluation).

# First split: 80% train, 20% temp
train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    all_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
# Second split: 50/50 of the 20% → 10% val, 10% test
val_ids, test_ids, val_labels, test_labels = train_test_split(
    temp_ids, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"\n{'='*50}")
print(f"DATA SPLIT (80/10/10)")
print(f"{'='*50}")
print(f"  Train: {len(train_ids)} samples "
      f"(Native: {sum(1 for l in train_labels if l==0)}, "
      f"Non-Native: {sum(1 for l in train_labels if l==1)})")
print(f"  Val:   {len(val_ids)} samples "
      f"(Native: {sum(1 for l in val_labels if l==0)}, "
      f"Non-Native: {sum(1 for l in val_labels if l==1)})")
print(f"  Test:  {len(test_ids)} samples "
      f"(Native: {sum(1 for l in test_labels if l==0)}, "
      f"Non-Native: {sum(1 for l in test_labels if l==1)})")
print(f"{'='*50}")

# ──────────────────────────────────────────────
# 2.3 DATASET CLASS — loads .pt files, L2-normalizes, returns (wavlm, ecapa, label)
# ──────────────────────────────────────────────
class FusionDataset(Dataset):
    """
    Loads pre-extracted WavLM (768-D) and ECAPA (192-D) tensors from disk.
    Applies L2 normalization to put both modalities on the same scale.
    Returns: (wavlm_tensor, ecapa_tensor, label)
    """
    def __init__(self, dp_ids, labels):
        self.dp_ids = list(dp_ids)
        self.labels = list(labels)

    def __len__(self):
        return len(self.dp_ids)

    def __getitem__(self, idx):
        dp_id = self.dp_ids[idx]

        # Load pre-extracted embeddings from disk
        wavlm = torch.load(f"extracted_features/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()
        ecapa = torch.load(f"extracted_ecapa/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()

        # Safety: collapse any extra dimensions (some edge cases produce [1, 768])
        if wavlm.dim() > 1: wavlm = wavlm.mean(dim=0)
        if ecapa.dim() > 1: ecapa = ecapa.mean(dim=0)

        # L2 normalize — critical for fair fusion (WavLM & ECAPA have different scales)
        wavlm = F.normalize(wavlm, dim=0)
        ecapa = F.normalize(ecapa, dim=0)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return wavlm, ecapa, y

# Create data loaders (batch_size=16 works well for 128 training samples)
train_loader = DataLoader(FusionDataset(train_ids, train_labels), batch_size=16, shuffle=True)
val_loader   = DataLoader(FusionDataset(val_ids, val_labels), batch_size=16, shuffle=False)
test_loader  = DataLoader(FusionDataset(test_ids, test_labels), batch_size=16, shuffle=False)

# ──────────────────────────────────────────────
# 2.4 MODEL DEFINITION — Weighted Late Fusion
# ──────────────────────────────────────────────
class WeightedLateFusion(nn.Module):
    """
    Two separate classifier heads (one per modality) whose output logits are
    blended via a learned alpha parameter:
        fused = α * wavlm_logits + (1-α) * ecapa_logits
    
    α = sigmoid(1.73) ≈ 0.85 initially, meaning 85% WavLM trust at start.
    The model learns to adjust this during training.
    
    Architecture:
        WavLM head:  768 → 32 → BN → GELU → Dropout(0.5) → 2 (logits)
        ECAPA head:  192 → 16 → BN → GELU → Dropout(0.5) → 2 (logits)
    
    Total trainable parameters: ~27,893
    """
    def __init__(self):
        super().__init__()
        # WavLM classification head — 768-D input gets 32-neuron hidden layer
        self.wavlm_head = nn.Sequential(
            nn.Linear(768, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),           # Smoother than ReLU for small datasets
            nn.Dropout(0.5),
            nn.Linear(32, 2)     # 2 classes: Native, Non-Native
        )
        # ECAPA classification head — 192-D input gets 16-neuron hidden layer
        # (proportionally smaller because input is 4× smaller than WavLM)
        self.ecapa_head = nn.Sequential(
            nn.Linear(192, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2)
        )
        # Learned fusion weight: sigmoid(1.73) ≈ 0.85 → starts trusting WavLM 85%
        self.alpha_logit = nn.Parameter(torch.tensor(1.73))

    def forward(self, wavlm_feat, ecapa_feat):
        wavlm_logits = self.wavlm_head(wavlm_feat)  # Shape: [batch, 2]
        ecapa_logits  = self.ecapa_head(ecapa_feat)   # Shape: [batch, 2]
        alpha = torch.sigmoid(self.alpha_logit)        # Scalar in [0, 1]
        fused_logits = alpha * wavlm_logits + (1.0 - alpha) * ecapa_logits
        return fused_logits, alpha

# --- Compute class weights to handle imbalanced data (71% Native vs 29% Non-Native) ---
# Without this, the model would just predict "Native" for everything and get 71% accuracy
cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
wt = torch.tensor(cw, dtype=torch.float32).to(device)
print(f"Class Weights: {cw}")

# --- Initialize model, loss, optimizer, scheduler ---
model = WeightedLateFusion().to(device)

# Cross-entropy with class weights + label smoothing (0.1 prevents overconfident predictions)
criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)

# AdamW: like Adam but with proper weight decay (better generalization)
optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-3)

# Cosine annealing: learning rate oscillates, helping escape local minima
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {total_params:,}")

# ──────────────────────────────────────────────
# 2.5 TRAINING LOOP — validates on val set, NEVER touches test set
# ──────────────────────────────────────────────
EPOCHS = 60
best_val_f1 = 0.0
best_val_acc = 0.0
best_train_acc = 0.0
patience = 15            # Stop training if val F1 doesn't improve for 15 epochs
patience_counter = 0

print(f"\n{'Epoch':<8} {'Train Acc':<12} {'Train Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'α':<8}")
print(f"{'-'*60}")

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    correct_train, total_train_loss = 0, 0
    for wavlm_b, ecapa_b, y_b in train_loader:
        wavlm_b, ecapa_b, y_b = wavlm_b.to(device), ecapa_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits, _ = model(wavlm_b, ecapa_b)
        loss = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        optimizer.step()
        correct_train += (logits.argmax(1) == y_b).sum().item()
        total_train_loss += loss.item()
    scheduler.step()
    train_acc = correct_train / len(train_labels)
    avg_loss = total_train_loss / len(train_loader)

    # --- VALIDATE (on val set — NOT test set) ---
    model.eval()
    vp, vt = [], []
    correct_val = 0
    with torch.no_grad():
        for wavlm_b, ecapa_b, y_b in val_loader:
            wavlm_b, ecapa_b, y_b = wavlm_b.to(device), ecapa_b.to(device), y_b.to(device)
            logits, _ = model(wavlm_b, ecapa_b)
            preds = logits.argmax(1)
            correct_val += (preds == y_b).sum().item()
            vp.extend(preds.cpu().numpy())
            vt.extend(y_b.cpu().numpy())

    val_acc = correct_val / len(val_labels)
    val_f1 = f1_score(vt, vp, average='macro')

    # Save model only when val F1 improves (best checkpoint strategy)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_acc = val_acc
        best_train_acc = train_acc
        torch.save(model.state_dict(), 'best_fusion_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        alpha_now = torch.sigmoid(model.alpha_logit).item()
        print(f"  {epoch+1:<6} {train_acc:<12.3f} {avg_loss:<12.4f} "
              f"{val_acc:<10.3f} {val_f1:<10.3f} {alpha_now:<8.3f}")

    # Early stopping: stop if no improvement for 15 epochs (and at least 20 epochs done)
    if patience_counter >= patience and epoch > 20:
        alpha_now = torch.sigmoid(model.alpha_logit).item()
        print(f"  {epoch+1:<6} {train_acc:<12.3f} {avg_loss:<12.4f} "
              f"{val_acc:<10.3f} {val_f1:<10.3f} {alpha_now:<8.3f}")
        print(f"\n  Early stopping at epoch {epoch+1}")
        break

# ──────────────────────────────────────────────
# 2.6 EVALUATE ON HELD-OUT TEST SET (never seen during training or validation)
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EVALUATING ON HELD-OUT TEST SET (never seen during training)")
print(f"{'='*60}")

# Reload the best checkpoint (not the last epoch's weights)
model.load_state_dict(torch.load('best_fusion_model.pth', map_location=device))
model.eval()

test_preds, test_true, test_confs = [], [], []
with torch.no_grad():
    for wavlm_b, ecapa_b, y_b in test_loader:
        wavlm_b, ecapa_b = wavlm_b.to(device), ecapa_b.to(device)
        logits, _ = model(wavlm_b, ecapa_b)
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(y_b.numpy())
        test_confs.extend(confs.cpu().numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
test_confs = np.array(test_confs)

test_acc = np.mean(test_preds == test_true)
test_f1 = f1_score(test_true, test_preds, average='macro')

final_alpha = torch.sigmoid(model.alpha_logit).item()

print(f"\n┌─────────────────────────────────────────────────┐")
print(f"│           FINAL RESULTS (80/10/10 Split)         │")
print(f"├─────────────────────────────────────────────────┤")
print(f"│  Train Accuracy:      {best_train_acc:.3f}                     │")
print(f"│  Validation Accuracy: {best_val_acc:.3f}  (F1: {best_val_f1:.3f})         │")
print(f"│  TEST Accuracy:       {test_acc:.3f}  (F1: {test_f1:.3f})         │")
print(f"│                                                 │")
print(f"│  Learned α: {final_alpha:.3f}                              │")
print(f"│    → WavLM:  {final_alpha*100:.1f}%                            │")
print(f"│    → ECAPA:  {(1-final_alpha)*100:.1f}%                            │")
print(f"│                                                 │")
print(f"│  Test Confidence — Mean: {test_confs.mean():.3f}                │")
print(f"│                   Min:  {test_confs.min():.3f}                │")
print(f"│                   Max:  {test_confs.max():.3f}                │")
print(f"└─────────────────────────────────────────────────┘")

print(f"\nTest Set Classification Report:")
print(classification_report(test_true, test_preds, target_names=['Native', 'Non-Native']))

cm = confusion_matrix(test_true, test_preds)
print(f"Confusion Matrix:")
print(f"                Predicted")
print(f"              Native  Non-Native")
print(f"  Actual Native    {cm[0][0]:<6}  {cm[0][1]}")
print(f"  Actual Non-Nat   {cm[1][0]:<6}  {cm[1][1]}")

print(f"\n--- PHASE 2 COMPLETE: Model saved to best_fusion_model.pth ---")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ STOP POINT 2 ═══  Model is trained and saved.
#                        best_fusion_model.pth is on disk.
#                        Next: Run Phase 3 (Test Inference) below.
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# PHASE 3 — Test Inference & Submission CSV Generation
# =============================================================================
#
# WHAT THIS DOES:
#   Loads the trained fusion model, downloads each of the 40 test audio files,
#   extracts WavLM + ECAPA features on-the-fly (same pipeline as Phase 1),
#   runs them through the fusion classifier, and saves predictions to CSV.
#
# INPUTS:
#   • Nativity Assessmet Audio Dataset(Test Dataset).csv  (40 unlabeled test samples)
#   • best_fusion_model.pth                               (from Phase 2)
#
# OUTPUTS:
#   • submission.csv — columns: dp_id, nativity_status, confidence_score
#
# DEPENDENCY: Phase 2 must be complete (model must exist on disk)
# =============================================================================

import os
import pandas as pd
import torch
import torchaudio
import urllib.request
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- Monkey patch (same as Phase 1B) ---
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.speaker import EncoderClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Phase 3] Running Weighted Late Fusion Inference on: {device}")

# --- Reload both feature extractors (needed to process test audio on-the-fly) ---
print("Loading WavLM & ECAPA Extractors...")
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
wavlm_model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus", torch_dtype=model_dtype
).to(device)
wavlm_model.eval()

ecapa_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# --- Reload the trained fusion classifier ---
# NOTE: We redefine the class here so this phase can run independently
class WeightedLateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm_head = nn.Sequential(
            nn.Linear(768, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(32, 2)
        )
        self.ecapa_head = nn.Sequential(
            nn.Linear(192, 16), nn.BatchNorm1d(16), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(16, 2)
        )
        self.alpha_logit = nn.Parameter(torch.tensor(1.73))

    def forward(self, wavlm_feat, ecapa_feat):
        wavlm_logits = self.wavlm_head(wavlm_feat)
        ecapa_logits  = self.ecapa_head(ecapa_feat)
        alpha = torch.sigmoid(self.alpha_logit)
        fused_logits = alpha * wavlm_logits + (1.0 - alpha) * ecapa_logits
        return fused_logits, alpha

fusion_model = WeightedLateFusion().to(device)
fusion_model.load_state_dict(torch.load('best_fusion_model.pth', map_location=device))
fusion_model.eval()

learned_alpha = torch.sigmoid(fusion_model.alpha_logit).item()
print(f"Loaded model | α = {learned_alpha:.3f} "
      f"(WavLM: {learned_alpha*100:.1f}%, ECAPA: {(1-learned_alpha)*100:.1f}%)")

# --- Load test CSV ---
test_csv_path = 'Nativity Assessmet Audio Dataset(Test Dataset).csv'
df_test = pd.read_csv(test_csv_path)
print(f"Processing {len(df_test)} test files...")

os.makedirs("test_audio_temp", exist_ok=True)
predictions_list = []
label_map_reverse = {0: "Native", 1: "Non-Native"}

# --- Inference loop: for each test sample, extract features and predict ---
for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    dp_id = row['dp_id']
    url = row['audio_url']
    ext = url.split('.')[-1]
    audio_path = f"test_audio_temp/{dp_id}.{ext}"

    # Default prediction (fallback if extraction fails)
    prediction_label = "Native"
    confidence_score = 0.5000

    try:
        # Download test audio
        if not os.path.exists(audio_path):
            urllib.request.urlretrieve(url, audio_path)

        # Load and preprocess (same as Phase 1A)
        signal, sample_rate = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            signal = resampler(signal)

        signal_flat = signal.squeeze()

        with torch.no_grad():
            # A. EXTRACT WAVLM (chunked, same as Phase 1A)
            CHUNK_SAMPLES = 10 * 16000
            chunks = torch.split(signal_flat, CHUNK_SAMPLES)
            chunk_embeddings = []
            for chunk in chunks:
                if len(chunk) < 1600:
                    continue
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(device, dtype=model_dtype)
                outputs = wavlm_model(input_values)
                pooled_chunk = torch.mean(
                    outputs.last_hidden_state, dim=1
                ).squeeze().cpu()
                chunk_embeddings.append(pooled_chunk)

            wavlm_final = (
                torch.stack(chunk_embeddings).mean(dim=0)
                if chunk_embeddings else torch.zeros(768)
            )

            # B. EXTRACT ECAPA (same as Phase 1B)
            embeddings = ecapa_classifier.encode_batch(signal)
            ecapa_final = embeddings.squeeze().cpu()

            # Dimension safety checks
            wavlm_final = wavlm_final.squeeze()
            ecapa_final = ecapa_final.squeeze()
            if ecapa_final.dim() > 1: ecapa_final = ecapa_final.mean(dim=0)
            if wavlm_final.dim() > 1: wavlm_final = wavlm_final.mean(dim=0)

            # L2 normalize — MUST match training preprocessing exactly
            wavlm_final = F.normalize(wavlm_final, dim=0)
            ecapa_final = F.normalize(ecapa_final, dim=0)

            # C. PREDICT — fuse + softmax → class + confidence
            wavlm_input = wavlm_final.to(device).to(torch.float32).unsqueeze(0)
            ecapa_input = ecapa_final.to(device).to(torch.float32).unsqueeze(0)

            fused_logits, alpha = fusion_model(wavlm_input, ecapa_input)
            probabilities = F.softmax(fused_logits, dim=1)
            conf, pred_idx = torch.max(probabilities, dim=1)

            prediction_label = label_map_reverse[pred_idx.item()]
            confidence_score = round(conf.item(), 4)

        # Clean up temp audio to save disk space
        os.remove(audio_path)

    except Exception as e:
        print(f"\nFailed on dp_id {dp_id}: {e}")
        prediction_label = "Native"     # Safe fallback (majority class)
        confidence_score = 0.5000

    predictions_list.append({
        'dp_id': dp_id,
        'nativity_status': prediction_label,
        'confidence_score': confidence_score
    })

# --- Save submission CSV ---
submission_df = pd.DataFrame(predictions_list)
submission_df.to_csv('submission.csv', index=False)

print("\n--- PHASE 3 COMPLETE: Inference Done ---")
print(f"Saved {len(submission_df)} predictions to 'submission.csv'")
print(submission_df['nativity_status'].value_counts())
print(f"\nConfidence Score Stats:")
print(f"  Mean: {submission_df['confidence_score'].mean():.4f}")
print(f"  Min:  {submission_df['confidence_score'].min():.4f}")
print(f"  Max:  {submission_df['confidence_score'].max():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══ STOP POINT 3 ═══  submission.csv is ready
#                        Phase 4 below is OPTIONAL (comparison analysis only).
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# PHASE 4 (OPTIONAL) — Head-to-Head: v1 Aura vs AuraSense Ensemble
# =============================================================================
#
# WHAT THIS DOES:
#   Trains v1 Aura (naive concatenation baseline) from scratch on the same data,
#   loads the 6-model AuraSense ensemble, evaluates both on the same val split,
#   then generates dual submission CSVs from 40 test files.
#
# WHY RUN THIS:
#   Proves that Weighted Late Fusion + Ensemble beats Naive Concatenation.
#   Generates comparison metrics for the technical write-up.
#
# INPUTS:
#   • All Phase 1 outputs (extracted_features/, extracted_ecapa/)
#   • Training CSV
#   • Test CSV
#   • fusion_fold0.pth through fusion_fold4.pth + fusion_full.pth
#     (these 6 models are produced by the ensemble training notebook:
#      Copy_of_aurasense_hackenza.ipynb — NOT by this file)
#
# OUTPUTS:
#   • submission_v1_aura.csv       (v1 baseline predictions)
#   • submission_copy_of_aura.csv  (ensemble predictions)
#   • Console: full comparison table with agreement analysis
#
# DEPENDENCY: Phase 1 complete + fusion_fold*.pth files must exist on disk
# =============================================================================

"""
HEAD-TO-HEAD COMPARISON
  Model A: v1 Aura — Naive Concatenation MLP ([768]+[192]=[960] → 64 → 2)
  Model B: Copy of Aura — 6-Model Weighted Late Fusion Ensemble

Both use the EXACT same 80/20 train/val split (random_state=42).
Both generate separate submission CSVs from the same 40 test files.
NO architecture changes, NO temperature scaling.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════
# SHARED DATA LOADING (same as Phase 2)
# ═══════════════════════════════════════════════
df = pd.read_csv('Nativity Assessmet Audio Dataset(Training Dataset).csv')
label_map = {"Native": 0, "Non-Native": 1}

all_ids, all_labels = [], []
for _, row in df.iterrows():
    dp_id = row['dp_id']
    if os.path.exists(f"extracted_features/{dp_id}.pt") and \
       os.path.exists(f"extracted_ecapa/{dp_id}.pt"):
        all_ids.append(dp_id)
        all_labels.append(label_map[row['nativity_status']])

# SAME 80/20 split as v1 aura uses (random_state=42 for reproducibility)
X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    all_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

print(f"\n[Phase 4] Total: {len(all_ids)} | Train: {len(X_train_ids)} | Val: {len(X_val_ids)}")
print(f"Train dist: Native={sum(1 for l in y_train if l==0)}, "
      f"Non-Native={sum(1 for l in y_train if l==1)}")
print(f"Val dist:   Native={sum(1 for l in y_val if l==0)}, "
      f"Non-Native={sum(1 for l in y_val if l==1)}")

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
wt = torch.tensor(cw, dtype=torch.float32).to(device)

# ═══════════════════════════════════════════════
# DATASET A: v1 Aura — Naive Concatenation (no L2 norm)
# Concatenates [768] + [192] = [960] as a single flat vector
# ═══════════════════════════════════════════════
class ConcatDataset(Dataset):
    def __init__(self, dp_ids, labels):
        self.dp_ids = list(dp_ids)
        self.labels = list(labels)
    def __len__(self):
        return len(self.dp_ids)
    def __getitem__(self, idx):
        dp_id = self.dp_ids[idx]
        wavlm = torch.load(f"extracted_features/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()
        ecapa = torch.load(f"extracted_ecapa/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()
        if wavlm.dim() > 1: wavlm = wavlm.mean(dim=0)
        if ecapa.dim() > 1: ecapa = ecapa.mean(dim=0)
        fused = torch.cat((wavlm, ecapa), dim=0)  # [960] — raw concat, NO L2 norm
        return fused, torch.tensor(self.labels[idx], dtype=torch.long)

# DATASET B: AuraSense — Separate features with L2 norm (same as Phase 2)
class FusionDataset(Dataset):
    def __init__(self, dp_ids, labels):
        self.dp_ids = list(dp_ids)
        self.labels = list(labels)
    def __len__(self):
        return len(self.dp_ids)
    def __getitem__(self, idx):
        dp_id = self.dp_ids[idx]
        wavlm = torch.load(f"extracted_features/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()
        ecapa = torch.load(f"extracted_ecapa/{dp_id}.pt", map_location='cpu').to(torch.float32).squeeze()
        if wavlm.dim() > 1: wavlm = wavlm.mean(dim=0)
        if ecapa.dim() > 1: ecapa = ecapa.mean(dim=0)
        wavlm = F.normalize(wavlm, dim=0)
        ecapa = F.normalize(ecapa, dim=0)
        return wavlm, ecapa, torch.tensor(self.labels[idx], dtype=torch.long)

# ═══════════════════════════════════════════════
# MODEL A: v1 Aura — Naive Concat MLP
# Problem: 768-D WavLM dominates 192-D ECAPA in the concatenated [960] input
# ═══════════════════════════════════════════════
class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(960, 64),     # Single hidden layer for 960-D input
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.7),        # High dropout needed to compensate for bad architecture
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.network(x)

# MODEL B: AuraSense — Weighted Late Fusion (same architecture as Phase 2)
class WeightedLateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm_head = nn.Sequential(
            nn.Linear(768, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(32, 2)
        )
        self.ecapa_head = nn.Sequential(
            nn.Linear(192, 16), nn.BatchNorm1d(16), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(16, 2)
        )
        self.alpha_logit = nn.Parameter(torch.tensor(1.73))
    def forward(self, wavlm_feat, ecapa_feat):
        w_log = self.wavlm_head(wavlm_feat)
        e_log = self.ecapa_head(ecapa_feat)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * w_log + (1 - alpha) * e_log, alpha

# ═══════════════════════════════════════════════
# TRAIN MODEL A: v1 Aura (from scratch, 30 epochs)
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"TRAINING MODEL A: v1 Aura — Naive Concat MLP")
print(f"{'='*60}")

train_loader_a = DataLoader(ConcatDataset(X_train_ids, y_train), batch_size=16, shuffle=True)
val_loader_a = DataLoader(ConcatDataset(X_val_ids, y_val), batch_size=16, shuffle=False)

model_a = FusionClassifier().to(device)
criterion_a = nn.CrossEntropyLoss(weight=wt)
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001, weight_decay=1e-2)

best_val_acc_a = 0.0
best_train_acc_a = 0.0

for epoch in range(30):
    model_a.train()
    correct_t = 0
    for X_b, y_b in train_loader_a:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer_a.zero_grad()
        preds = model_a(X_b)
        loss = criterion_a(preds, y_b)
        loss.backward()
        optimizer_a.step()
        correct_t += (preds.argmax(1) == y_b).sum().item()
    t_acc = correct_t / len(y_train)

    model_a.eval()
    correct_v = 0
    vp_a, vt_a = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader_a:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model_a(X_b)
            correct_v += (preds.argmax(1) == y_b).sum().item()
            vp_a.extend(preds.argmax(1).cpu().numpy())
            vt_a.extend(y_b.cpu().numpy())
    v_acc = correct_v / len(y_val)

    if v_acc > best_val_acc_a:
        best_val_acc_a = v_acc
        best_train_acc_a = t_acc
        best_vp_a, best_vt_a = vp_a, vt_a
        torch.save(model_a.state_dict(), 'v1_aura_fusion.pth')

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:02d}/30 | Train Acc: {t_acc:.3f} | Val Acc: {v_acc:.3f}")

val_f1_a = f1_score(best_vt_a, best_vp_a, average='macro')
params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)

# Compute val confidence for Model A
model_a.load_state_dict(torch.load('v1_aura_fusion.pth', map_location=device))
model_a.eval()
val_confs_a = []
with torch.no_grad():
    for X_b, y_b in val_loader_a:
        X_b = X_b.to(device)
        logits = model_a(X_b)
        probs = F.softmax(logits, dim=1)
        confs, _ = torch.max(probs, dim=1)
        val_confs_a.extend(confs.cpu().numpy())

print(f"\n  MODEL A RESULTS:")
print(f"  Train Acc: {best_train_acc_a:.3f} | Val Acc: {best_val_acc_a:.3f} | Val F1: {val_f1_a:.3f}")
print(f"  Val Confidence — Mean: {np.mean(val_confs_a):.3f} | "
      f"Min: {np.min(val_confs_a):.3f} | Max: {np.max(val_confs_a):.3f}")
print(f"  Parameters: {params_a:,}")

# ═══════════════════════════════════════════════
# MODEL B: AuraSense — Load existing 6-model ensemble
# These .pth files are produced by Copy_of_aurasense_hackenza.ipynb (ensemble training)
# They are NOT produced by this file — they must already exist on disk
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"EVALUATING MODEL B: AuraSense — 6-Model Ensemble")
print(f"{'='*60}")

model_paths_b = [f'fusion_fold{i}.pth' for i in range(5)] + ['fusion_full.pth']
ensemble = []
for mp in model_paths_b:
    m = WeightedLateFusion().to(device)
    m.load_state_dict(torch.load(mp, map_location=device))
    m.eval()
    ensemble.append(m)

val_loader_b = DataLoader(FusionDataset(X_val_ids, y_val), batch_size=16, shuffle=False)

# Evaluate ensemble on val set (average logits from all 6 models → softmax → prediction)
vp_b, vt_b, val_confs_b = [], [], []
with torch.no_grad():
    for wavlm_b, ecapa_b, y_b in val_loader_b:
        wavlm_b, ecapa_b = wavlm_b.to(device), ecapa_b.to(device)
        avg_logits = torch.zeros(wavlm_b.size(0), 2, device=device)
        for m in ensemble:
            logits, _ = m(wavlm_b, ecapa_b)
            avg_logits += logits
        avg_logits /= len(ensemble)
        probs = F.softmax(avg_logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        vp_b.extend(preds.cpu().numpy())
        vt_b.extend(y_b.numpy())
        val_confs_b.extend(confs.cpu().numpy())

val_acc_b = np.mean(np.array(vp_b) == np.array(vt_b))
val_f1_b = f1_score(vt_b, vp_b, average='macro')
params_b = sum(p.numel() for p in ensemble[0].parameters() if p.requires_grad)

print(f"\n  MODEL B RESULTS (Ensemble on same val set):")
print(f"  Val Acc: {val_acc_b:.3f} | Val F1: {val_f1_b:.3f}")
print(f"  Val Confidence — Mean: {np.mean(val_confs_b):.3f} | "
      f"Min: {np.min(val_confs_b):.3f} | Max: {np.max(val_confs_b):.3f}")
print(f"  Parameters per model: {params_b:,} × 6 models")

# ═══════════════════════════════════════════════
# GENERATE BOTH SUBMISSION CSVs FROM 40 TEST FILES
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"GENERATING TEST SUBMISSIONS (both models, same 40 test files)")
print(f"{'='*60}")

import torchaudio
import urllib.request
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from tqdm import tqdm

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.speaker import EncoderClassifier

# Load feature extractors (reuse if already in memory from Phase 3)
print("Loading feature extractors...")
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
wavlm_model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus", torch_dtype=model_dtype
).to(device)
wavlm_model.eval()
ecapa_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

test_csv_path = 'Nativity Assessmet Audio Dataset(Test Dataset).csv'
df_test = pd.read_csv(test_csv_path)
os.makedirs("test_audio_temp", exist_ok=True)
label_map_reverse = {0: "Native", 1: "Non-Native"}

preds_a_list, preds_b_list = [], []

print(f"Running inference on {len(df_test)} test files...")
for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    dp_id = row['dp_id']
    url = row['audio_url']
    ext = url.split('.')[-1]
    audio_path = f"test_audio_temp/{dp_id}.{ext}"

    label_a, conf_a = "Native", 0.5000
    label_b, conf_b = "Native", 0.5000

    try:
        if not os.path.exists(audio_path):
            urllib.request.urlretrieve(url, audio_path)

        signal, sr = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        signal_flat = signal.squeeze()

        with torch.no_grad():
            # --- Extract WavLM (chunked) ---
            CHUNK_SAMPLES = 10 * 16000
            chunks = torch.split(signal_flat, CHUNK_SAMPLES)
            chunk_embs = []
            for chunk in chunks:
                if len(chunk) < 1600: continue
                inp = processor(chunk, sampling_rate=16000, return_tensors="pt")
                iv = inp.input_values.to(device, dtype=model_dtype)
                out = wavlm_model(iv)
                chunk_embs.append(torch.mean(out.last_hidden_state, dim=1).squeeze().cpu())
            wavlm_raw = (
                torch.stack(chunk_embs).mean(dim=0) if chunk_embs else torch.zeros(768)
            )

            # --- Extract ECAPA ---
            emb = ecapa_classifier.encode_batch(signal)
            ecapa_raw = emb.squeeze().cpu()

            # Dimension safety
            wavlm_raw = wavlm_raw.squeeze()
            ecapa_raw = ecapa_raw.squeeze()
            if ecapa_raw.dim() > 1: ecapa_raw = ecapa_raw.mean(dim=0)
            if wavlm_raw.dim() > 1: wavlm_raw = wavlm_raw.mean(dim=0)

            # ── MODEL A: Naive concat (no L2 norm — matches v1 training) ──
            fused = torch.cat((wavlm_raw, ecapa_raw), dim=0).to(device).to(torch.float32).unsqueeze(0)
            logits_a = model_a(fused)
            probs_a = F.softmax(logits_a, dim=1)
            c_a, p_a = torch.max(probs_a, dim=1)
            label_a = label_map_reverse[p_a.item()]
            conf_a = round(c_a.item(), 4)

            # ── MODEL B: Ensemble with L2 norm (matches AuraSense training) ──
            wavlm_norm = F.normalize(wavlm_raw, dim=0)
            ecapa_norm = F.normalize(ecapa_raw, dim=0)
            w_in = wavlm_norm.to(device).to(torch.float32).unsqueeze(0)
            e_in = ecapa_norm.to(device).to(torch.float32).unsqueeze(0)

            avg_logits = torch.zeros(1, 2, device=device)
            for m in ensemble:
                log, _ = m(w_in, e_in)
                avg_logits += log
            avg_logits /= len(ensemble)
            probs_b = F.softmax(avg_logits, dim=1)
            c_b, p_b = torch.max(probs_b, dim=1)
            label_b = label_map_reverse[p_b.item()]
            conf_b = round(c_b.item(), 4)

        os.remove(audio_path)

    except Exception as e:
        print(f"\nFailed on dp_id {dp_id}: {e}")

    preds_a_list.append({'dp_id': dp_id, 'nativity_status': label_a, 'confidence_score': conf_a})
    preds_b_list.append({'dp_id': dp_id, 'nativity_status': label_b, 'confidence_score': conf_b})

# Save both submission CSVs
df_a = pd.DataFrame(preds_a_list)
df_b = pd.DataFrame(preds_b_list)
df_a.to_csv('submission_v1_aura.csv', index=False)
df_b.to_csv('submission_copy_of_aura.csv', index=False)

# ═══════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════
print(f"\n\n{'═'*70}")
print(f"         FINAL HEAD-TO-HEAD COMPARISON")
print(f"{'═'*70}")
print(f"")
print(f"{'Metric':<30} {'v1 Aura':<20} {'AuraSense':<20}")
print(f"{'-'*70}")
print(f"{'Architecture':<30} {'Naive Concat MLP':<20} {'Weighted Late Fusion':<20}")
print(f"{'Fusion Method':<30} {'[768]+[192]→960→64→2':<20} {'Separate heads + α':<20}")
print(f"{'Strategy':<30} {'Single Model':<20} {'6-Model Ensemble':<20}")
print(f"{'Parameters':<30} {params_a:<20,} {f'{params_b:,} × 6':<20}")
print(f"{'-'*70}")
print(f"{'Train Accuracy':<30} {best_train_acc_a:<20.3f} {'N/A (ensemble)':<20}")
print(f"{'Val Accuracy':<30} {best_val_acc_a:<20.3f} {val_acc_b:<20.3f}")
print(f"{'Val F1 (macro)':<30} {val_f1_a:<20.3f} {val_f1_b:<20.3f}")
print(f"{'-'*70}")
print(f"{'Val Conf Mean':<30} {np.mean(val_confs_a):<20.3f} {np.mean(val_confs_b):<20.3f}")
print(f"{'Val Conf Min':<30} {np.min(val_confs_a):<20.3f} {np.min(val_confs_b):<20.3f}")
print(f"{'Val Conf Max':<30} {np.max(val_confs_a):<20.3f} {np.max(val_confs_b):<20.3f}")
print(f"{'-'*70}")
print(f"{'Test Native Count':<30} {(df_a['nativity_status']=='Native').sum():<20} "
      f"{(df_b['nativity_status']=='Native').sum():<20}")
print(f"{'Test Non-Native Count':<30} {(df_a['nativity_status']=='Non-Native').sum():<20} "
      f"{(df_b['nativity_status']=='Non-Native').sum():<20}")
print(f"{'Test Conf Mean':<30} {df_a['confidence_score'].mean():<20.4f} "
      f"{df_b['confidence_score'].mean():<20.4f}")
print(f"{'Test Conf Min':<30} {df_a['confidence_score'].min():<20.4f} "
      f"{df_b['confidence_score'].min():<20.4f}")
print(f"{'Test Conf Max':<30} {df_a['confidence_score'].max():<20.4f} "
      f"{df_b['confidence_score'].max():<20.4f}")
print(f"{'═'*70}")

# Agreement analysis
agree = sum(1 for a, b in zip(df_a['nativity_status'], df_b['nativity_status']) if a == b)
disagree = len(df_a) - agree
print(f"\nAgreement: {agree}/40 ({agree/40*100:.1f}%) | Disagreement: {disagree}/40")

if disagree > 0:
    print(f"\nDisagreements:")
    print(f"  {'dp_id':<10} {'v1 Aura':<15} {'Conf':<10} {'AuraSense':<15} {'Conf':<10}")
    for i in range(len(df_a)):
        if df_a.iloc[i]['nativity_status'] != df_b.iloc[i]['nativity_status']:
            print(f"  {df_a.iloc[i]['dp_id']:<10} "
                  f"{df_a.iloc[i]['nativity_status']:<15} "
                  f"{df_a.iloc[i]['confidence_score']:<10.4f} "
                  f"{df_b.iloc[i]['nativity_status']:<15} "
                  f"{df_b.iloc[i]['confidence_score']:<10.4f}")

# VERDICT
print(f"\n{'═'*70}")
if val_acc_b > best_val_acc_a:
    winner = "AuraSense (Weighted Late Fusion Ensemble)"
    margin = val_acc_b - best_val_acc_a
elif best_val_acc_a > val_acc_b:
    winner = "v1 Aura (Naive Concat MLP)"
    margin = best_val_acc_a - val_acc_b
else:
    if val_f1_b > val_f1_a:
        winner = "AuraSense (higher F1)"
        margin = val_f1_b - val_f1_a
    else:
        winner = "v1 Aura (higher F1)"
        margin = val_f1_a - val_f1_b

print(f"  🏆 WINNER: {winner}")
print(f"  Margin: +{margin:.3f}")
print(f"")
print(f"  Submission files saved:")
print(f"    → submission_v1_aura.csv")
print(f"    → submission_copy_of_aura.csv")
print(f"{'═'*70}")
print(f"\n--- PHASE 4 COMPLETE: Head-to-Head Comparison Done ---")
