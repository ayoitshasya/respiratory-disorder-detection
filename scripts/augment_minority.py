"""
Audio augmentation for minority classes (wheeze=2, both=3, crackle=1).
Runs locally on PC. Generates augmented .npy mel features and updates manifest.

Target counts after augmentation (~20k total train):
  Normal  (0): 1643  → ~6572 (3x)
  Crackle (1):  626  → ~5634 (8x)
  Wheeze  (2):  419  → ~5038 (11x)
  Both    (3):  203  → ~4266 (20x)
"""

import os, random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# ── Config ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MANIFEST    = os.path.join(BASE_DIR, 'data', 'processed', 'manifest.csv')
AUG_DIR     = os.path.join(BASE_DIR, 'data', 'processed', 'train_aug_v2')
os.makedirs(AUG_DIR, exist_ok=True)

SR          = 22050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
TARGET_FRAMES = 63

# How many augmented copies per sample per class
COPIES = {0: 3, 1: 8, 2: 11, 3: 20}  # normal=3x, crackle=8x, wheeze=11x, both=20x

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────
def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def to_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return pad_or_truncate(mel_db)


def augment(audio, sr, copy_index=0):
    """Return one augmented version using a combination of techniques."""
    # Use copy_index to ensure variety across many copies
    techniques = copy_index % 6

    if techniques == 0:
        # Time stretch slow
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.80, 0.95))
    elif techniques == 1:
        # Time stretch fast
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(1.05, 1.20))
    elif techniques == 2:
        # Pitch shift up
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(1.0, 3.0))
    elif techniques == 3:
        # Pitch shift down
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-3.0, -1.0))
    elif techniques == 4:
        # Gaussian noise
        audio = audio + 0.005 * np.random.randn(len(audio))
    else:
        # Time stretch + noise combined
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.85, 1.15))
        audio = audio + 0.003 * np.random.randn(len(audio))

    return audio


# ── Main ──────────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST)
train_df = manifest[manifest['split'] == 'train'].copy()

print("Original train counts:")
print(train_df['sound_label'].value_counts().sort_index())
print()

new_rows = []

for label, n_copies in COPIES.items():
    subset = train_df[train_df['sound_label'] == label]
    print(f"Augmenting label {label}: {len(subset)} samples × {n_copies} copies...")

    for _, row in subset.iterrows():
        audio_path = row['audio_path']
        if not os.path.exists(audio_path):
            print(f"  SKIP (missing): {audio_path}")
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=SR, mono=True)
        except Exception as e:
            print(f"  SKIP (load error): {e}")
            continue

        for i in range(n_copies):
            aug_audio = augment(audio, sr, copy_index=i)
            mel = to_mel(aug_audio, sr)

            base_name = os.path.splitext(row['audio_file'])[0]
            out_name  = f"{base_name}_aug{i}.npy"
            out_path  = os.path.join(AUG_DIR, out_name)
            np.save(out_path, mel)

            new_rows.append({
                'audio_file':     out_name,
                'audio_path':     audio_path,
                'features_path':  out_path,
                'sound_label':    row['sound_label'],
                'diagnosis_label': row['diagnosis_label'],
                'patient_id':     row['patient_id'],
                'split':          'train'
            })

print(f"\nGenerated {len(new_rows)} augmented samples.")

# ── Save updated manifest ──────────────────────────────────────
aug_df       = pd.DataFrame(new_rows)
manifest_aug = pd.concat([manifest, aug_df], ignore_index=True)
manifest_aug.to_csv(MANIFEST.replace('manifest.csv', 'manifest_aug.csv'), index=False)

train_aug = manifest_aug[manifest_aug['split'] == 'train']
print("\nAugmented train counts:")
print(train_aug['sound_label'].value_counts().sort_index())
print(f"\nSaved to: data/processed/manifest_aug.csv")
print(f"Augmented .npy files in: data/processed/train_aug/")
