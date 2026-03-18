"""
preprocessing.py
----------------
Audio preprocessing: Butterworth filter, mel-spectrogram, MFCC extraction.
Features are cached to disk as .npy files to avoid re-computation.
"""

import os
import numpy as np
import librosa
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

SR           = 16000   # resample everything to 16kHz (ICBHI has mixed rates)
MIN_DURATION = 2.0     # pad cycles shorter than this (seconds)
MAX_DURATION = 8.0     # truncate cycles longer than this (seconds)

# Butterworth bandpass (lung sounds are 100–2000 Hz)
LOW_CUT      = 100
HIGH_CUT     = 2000
FILTER_ORDER = 4

# Mel spectrogram
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 50
FMAX       = 2000

# MFCC
N_MFCC = 13   # 13 coef + 13 delta + 13 delta-delta = 39 total


# ─────────────────────────────────────────────
# Butterworth filter
# ─────────────────────────────────────────────

def butterworth_filter(audio, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=SR, order=FILTER_ORDER):
    """
    Apply a 4th-order Butterworth bandpass filter.

    Args:
        audio:   1D numpy array
        lowcut:  Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs:      Sample rate (Hz)
        order:   Filter order

    Returns:
        Filtered audio array
    """
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)


# ─────────────────────────────────────────────
# Duration normalisation
# ─────────────────────────────────────────────

def normalise_duration(audio, sr=SR, min_dur=MIN_DURATION, max_dur=MAX_DURATION):
    """
    Pad (zero-pad) or truncate audio to fit [min_dur, max_dur].

    Args:
        audio:   1D numpy array
        sr:      Sample rate
        min_dur: Minimum duration in seconds (zero-pad if shorter)
        max_dur: Maximum duration in seconds (truncate if longer)

    Returns:
        Audio array of length between min_dur*sr and max_dur*sr samples
    """
    min_len = int(min_dur * sr)
    max_len = int(max_dur * sr)

    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    elif len(audio) > max_len:
        audio = audio[:max_len]

    return audio


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def extract_mel_spectrogram(audio, sr=SR):
    """
    Compute log-scale Mel spectrogram.

    Args:
        audio: 1D numpy array (filtered, duration-normalised)
        sr:    Sample rate

    Returns:
        2D array of shape (N_MELS, T) — float32
    """
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise to zero mean, unit std
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def extract_mfcc(audio, sr=SR):
    """
    Compute MFCC + delta + delta-delta (39 features).

    Args:
        audio: 1D numpy array
        sr:    Sample rate

    Returns:
        2D array of shape (39, T) — float32
    """
    mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH,
                                       fmin=FMIN, fmax=FMAX)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.concatenate([mfcc, delta, delta2], axis=0)  # (39, T)
    features   = (features - features.mean()) / (features.std() + 1e-8)
    return features.astype(np.float32)


def extract_features(audio_path, feature_type='mel'):
    """
    Full preprocessing pipeline for a single audio file:
      load → resample → filter → normalise duration → extract features

    Args:
        audio_path:   Path to .wav file
        feature_type: 'mel' or 'mfcc'

    Returns:
        numpy array: (N_MELS, T) for mel, (39, T) for mfcc
    """
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    audio    = butterworth_filter(audio)
    audio    = normalise_duration(audio)

    if feature_type == 'mel':
        return extract_mel_spectrogram(audio)
    elif feature_type == 'mfcc':
        return extract_mfcc(audio)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'mel' or 'mfcc'.")


# ─────────────────────────────────────────────
# Batch caching
# ─────────────────────────────────────────────

def cache_features(manifest, feature_type='mel', overwrite=False):
    """
    Extract and cache features for all entries in the manifest.
    Skips already-cached files unless overwrite=True.

    Args:
        manifest:     pd.DataFrame from data_loader.build_manifest()
        feature_type: 'mel' or 'mfcc'
        overwrite:    Re-extract even if .npy already exists

    Returns:
        Updated manifest DataFrame with 'features_path' column filled in
    """
    print(f"\nCaching {feature_type} features for {len(manifest)} cycles...")

    errors  = 0
    skipped = 0

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc='Extracting'):
        feat_path = row['features_path']
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        if os.path.exists(feat_path) and not overwrite:
            skipped += 1
            continue

        try:
            feat = extract_features(row['audio_path'], feature_type=feature_type)
            np.save(feat_path, feat)
        except Exception as e:
            errors += 1
            manifest.at[idx, 'features_path'] = ''
            continue

    print(f"Done — {len(manifest) - errors - skipped} extracted, {skipped} skipped, {errors} errors")
    return manifest


def load_feature(features_path):
    """Load a cached .npy feature file."""
    return np.load(features_path)


def get_feature_shape(manifest, feature_type='mel'):
    """
    Get the shape of one feature array (for model input_shape).
    Loads the first available cached feature.
    """
    for _, row in manifest.iterrows():
        if os.path.exists(row['features_path']):
            feat = np.load(row['features_path'])
            return feat.shape
    raise FileNotFoundError("No cached features found. Run cache_features() first.")


if __name__ == '__main__':
    import sys
    from data_loader import load_manifest

    BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(BASE, 'data', 'processed')

    print("Loading manifest...")
    manifest = load_manifest(processed_dir)

    print(f"\nTest: extracting features for first 5 entries...")
    sample = manifest.head(5).copy()
    sample = cache_features(sample, feature_type='mel', overwrite=True)

    for _, row in sample.iterrows():
        if os.path.exists(row['features_path']):
            feat = np.load(row['features_path'])
            print(f"  {os.path.basename(row['audio_file'])}: shape={feat.shape}, "
                  f"min={feat.min():.2f}, max={feat.max():.2f}")
