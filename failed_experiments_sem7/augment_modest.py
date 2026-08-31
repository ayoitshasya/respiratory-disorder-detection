"""
Modest real-data augmentation for minority sound classes only.
2x crackle, 3x wheeze, 5x both — enough to shift class balance without
overwhelming the real Normal distribution or causing augmentation collapse.

Uses hop_length=256 to match the re-extracted features.

Run after reextract_features.py:
    python augment_modest.py
"""
import os, random, sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST  = os.path.join(BASE, 'data', 'processed', 'manifest.csv')
AUG_DIR   = os.path.join(BASE, 'data', 'processed', 'train_aug_modest')
os.makedirs(AUG_DIR, exist_ok=True)

SR           = 16000
N_MELS       = 128
N_FFT        = 2048
HOP_LENGTH   = 256    # matches preprocessing.py
TARGET_FRAMES = 126   # 1 + floor(32000/256)
FMIN         = 50
FMAX         = 2000

# Minority classes only — Normal is already the majority
COPIES = {
    1: 2,   # Crackle: 626 × 2 = 1,252 aug  + 626 orig = 1,878
    2: 3,   # Wheeze:  419 × 3 = 1,257 aug  + 419 orig = 1,676
    3: 5,   # Both:    203 × 5 = 1,015 aug  + 203 orig = 1,218
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SOUND_NAMES = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}


def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def to_mel(audio):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * SR
    b, a = butter(4, [100 / nyq, 2000 / nyq], btype='band')
    audio = filtfilt(b, a, audio)
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return pad_or_truncate(mel_db.astype(np.float32))


def augment(audio, copy_index):
    """6 varied transforms — cycle through them for diversity."""
    t = copy_index % 6
    if t == 0:
        return librosa.effects.time_stretch(audio, rate=random.uniform(0.80, 0.92))
    elif t == 1:
        return librosa.effects.time_stretch(audio, rate=random.uniform(1.08, 1.20))
    elif t == 2:
        return librosa.effects.pitch_shift(audio, sr=SR, n_steps=random.uniform(1.5, 3.0))
    elif t == 3:
        return librosa.effects.pitch_shift(audio, sr=SR, n_steps=random.uniform(-3.0, -1.5))
    elif t == 4:
        snr = random.uniform(18, 28)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        noise_amp = rms / (10 ** (snr / 20))
        return audio + noise_amp * np.random.randn(len(audio))
    else:
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.85, 1.15))
        rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        noise_amp = rms / (10 ** (22 / 20))
        return audio + noise_amp * np.random.randn(len(audio))


# ── Main ──────────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST)
train_df  = manifest[manifest['split'] == 'train'].copy()

print("=" * 65)
print("Modest Augmentation (hop_length=256)")
print("=" * 65)
print("Original train counts:")
for lbl in range(4):
    n = (train_df['sound_label'] == lbl).sum()
    print(f"  {SOUND_NAMES[lbl]:8s}: {n}")

new_rows = []

for label, n_copies in COPIES.items():
    subset = train_df[train_df['sound_label'] == label]
    print(f"\nAugmenting {SOUND_NAMES[label]} ({len(subset)} samples x {n_copies} copies)...")

    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        if not os.path.exists(row['audio_path']):
            continue
        try:
            audio, _ = librosa.load(row['audio_path'], sr=SR, mono=True)
        except Exception:
            continue

        for i in range(n_copies):
            try:
                aug_audio = augment(audio, i)
                mel = to_mel(aug_audio)
            except Exception:
                continue

            base    = os.path.splitext(row['audio_file'])[0]
            fname   = f"{base}_aug{i}.npy"
            fpath   = os.path.join(AUG_DIR, fname)
            np.save(fpath, mel)

            new_rows.append({
                'audio_file':      fname,
                'audio_path':      row['audio_path'],
                'features_path':   fpath,
                'sound_label':     label,
                'diagnosis_label': row['diagnosis_label'],
                'patient_id':      row['patient_id'],
                'split':           'train',
                'source':          'augmented',
            })

print(f"\nGenerated {len(new_rows)} augmented samples.")

aug_df = pd.DataFrame(new_rows)
if 'source' not in manifest.columns:
    manifest['source'] = 'real'

manifest_aug = pd.concat([manifest, aug_df], ignore_index=True)
out_csv = os.path.join(BASE, 'data', 'processed', 'manifest_aug_modest.csv')
manifest_aug.to_csv(out_csv, index=False)

print("\nFinal train distribution:")
train_aug = manifest_aug[manifest_aug['split'] == 'train']
for lbl in range(4):
    orig = (train_df['sound_label'] == lbl).sum()
    aug  = int((aug_df['sound_label'] == lbl).sum())
    print(f"  {SOUND_NAMES[lbl]:8s}: {orig} orig + {aug} aug = {orig + aug}")

print(f"\nManifest saved: {out_csv}")
print("Next step: python train_highres.py")
