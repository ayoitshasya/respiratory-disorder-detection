"""
synthetic_generator.py
-----------------------
Mathematical synthesis of respiratory sounds: wheezes, crackles, both.

Approach:
  1. Synthesise pathological waveform (wheeze / crackle / both)
  2. Mix with a real normal-breathing recording as acoustic background
  3. Apply the same Butterworth + mel-spectrogram pipeline as real data
  4. Save .npy feature files to data/processed/train_synthetic/

Avoids augmentation-collapse: each sample is a genuinely novel waveform,
not a pitch-shifted copy of an existing recording.
"""

import os
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

SR           = 16000
MIN_DURATION = 2.0
MAX_DURATION = 4.0   # cap synthetics at 4s (keeps mel shape manageable)
LOW_CUT      = 100
HIGH_CUT     = 2000
FILTER_ORDER = 4

# Most clinically associated diagnosis for each pathological sound class
DIAG_FOR_SOUND = {
    1: 3,   # Crackle → Bronchiectasis (label 3)
    2: 1,   # Wheeze  → COPD           (label 1)
    3: 1,   # Both    → COPD           (label 1)
}

# Synthetic sample targets: small enough that Normal still dominates,
# large enough to shift class weights meaningfully
DEFAULT_TARGETS = {
    1: 200,   # Crackle: 626  → 826
    2: 150,   # Wheeze:  419  → 569
    3: 350,   # Both:    203  → 553
}

# Mix alpha: keep pathological sounds subtle so the model learns to detect
# them within a realistic breathing background — v1 alpha was too high (0.35-0.70)
# causing the model to over-detect pathology on real Normal recordings
ALPHA_RANGE = {
    1: (0.10, 0.25),   # Crackles: very subtle bursts within breath
    2: (0.10, 0.22),   # Wheeze: sits softly on top of breath
    3: (0.15, 0.28),   # Both: slightly more prominent than individual
}

NORMAL_POOL_SIZE = 200  # max normal recordings to keep in RAM


# ─────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────

def _butterworth_filter(audio, fs=SR):
    nyq  = 0.5 * fs
    low  = LOW_CUT  / nyq
    high = HIGH_CUT / nyq
    b, a = butter(FILTER_ORDER, [low, high], btype='band')
    return filtfilt(b, a, audio)


def _normalise_to_duration(audio, duration, sr=SR):
    n = int(duration * sr)
    if len(audio) < n:
        audio = np.pad(audio, (0, n - len(audio)))
    return audio[:n]


def _amplitude_envelope(n, attack_frac=0.08, decay_frac=0.08):
    env = np.ones(n)
    a   = int(n * attack_frac)
    d   = int(n * decay_frac)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[-d:] = np.linspace(1, 0, d)
    return env


def _bandpass_burst(burst, low, high, fs=SR):
    nyq  = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, burst)


# ─────────────────────────────────────────────
# Wheeze synthesis
# ─────────────────────────────────────────────

def synthesise_wheeze(duration=None, sr=SR):
    """
    Continuous quasi-sinusoidal sound with vibrato and harmonics.

    Real wheezes are polyphonic (multiple simultaneous pitches), so we
    layer two independent oscillators at slightly different base frequencies
    with slow vibrato to mimic this character.
    """
    if duration is None:
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
    n = int(duration * sr)
    t = np.arange(n) / sr

    output = np.zeros(n)
    for _ in range(np.random.randint(1, 3)):  # 1–2 simultaneous wheeze tones
        f0           = np.random.uniform(100, 800)
        vibrato_rate = np.random.uniform(1.0, 3.5)
        vibrato_dep  = np.random.uniform(0.03, 0.10)

        f_t   = f0 * (1.0 + vibrato_dep * np.sin(2 * np.pi * vibrato_rate * t))
        phase = 2 * np.pi * np.cumsum(f_t) / sr

        harmonics = (
            np.sin(phase)
            + 0.35 * np.sin(2 * phase)
            + 0.15 * np.sin(3 * phase)
            + 0.05 * np.random.uniform(0, 1) * np.sin(4 * phase)
        )
        # Slow amplitude modulation (makes it less robotic)
        amp_mod = 1.0 + 0.12 * np.sin(2 * np.pi * np.random.uniform(0.3, 0.8) * t)
        output += _amplitude_envelope(n) * harmonics * amp_mod

    peak = np.max(np.abs(output))
    return (output / peak) if peak > 1e-8 else output


# ─────────────────────────────────────────────
# Crackle synthesis
# ─────────────────────────────────────────────

def synthesise_crackles(duration=None, sr=SR, kind='random'):
    """
    Discontinuous explosive bursts.

    fine crackles:   <20ms, 8–20 per cycle   (pneumonia, IPF)
    coarse crackles: 20–100ms, 3–8 per cycle  (COPD, bronchiectasis)
    """
    if duration is None:
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
    n     = int(duration * sr)
    audio = np.zeros(n)

    if kind == 'random':
        kind = np.random.choice(['fine', 'coarse'])

    if kind == 'fine':
        n_bursts      = np.random.randint(8, 21)
        burst_dur_ms  = (5, 20)
        freq_range    = (300, 900)
    else:
        n_bursts      = np.random.randint(3, 9)
        burst_dur_ms  = (20, 100)
        freq_range    = (200, 600)

    for _ in range(n_bursts):
        burst_dur = np.random.uniform(*burst_dur_ms) / 1000.0
        burst_len = int(burst_dur * sr)
        if burst_len < 8:
            continue

        burst     = np.random.randn(burst_len)
        decay_tc  = burst_len * np.random.uniform(0.2, 0.5)
        burst    *= np.exp(-np.arange(burst_len) / decay_tc)

        if burst_len > 20:
            try:
                burst = _bandpass_burst(burst, *freq_range, fs=sr)
            except Exception:
                pass

        onset = np.random.randint(0, max(1, n - burst_len))
        end   = min(onset + burst_len, n)
        audio[onset:end] += burst[:end - onset]

    peak = np.max(np.abs(audio))
    return (audio / peak) if peak > 1e-8 else audio


# ─────────────────────────────────────────────
# Both synthesis
# ─────────────────────────────────────────────

def synthesise_both(duration=None, sr=SR):
    """Layer a wheeze underneath a crackle pattern."""
    if duration is None:
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)

    wheeze  = synthesise_wheeze(duration, sr)
    crackle = synthesise_crackles(duration, sr)

    w_amp = np.random.uniform(0.5, 0.8)
    c_amp = np.random.uniform(0.35, 0.65)
    out   = w_amp * wheeze + c_amp * crackle
    peak  = np.max(np.abs(out))
    return (out / peak) if peak > 1e-8 else out


# ─────────────────────────────────────────────
# Normal breathing pool
# ─────────────────────────────────────────────

def _load_normal_pool(manifest_path, max_samples=NORMAL_POOL_SIZE):
    """
    Load a random subset of real normal training recordings into RAM.
    These serve as the acoustic background for mixing.
    """
    import librosa
    df     = pd.read_csv(manifest_path)
    normal = df[(df['sound_label'] == 0) & (df['split'] == 'train')]
    normal = normal.sample(min(max_samples, len(normal)), random_state=42)

    pool = []
    for _, row in tqdm(normal.iterrows(), total=len(normal), desc='Loading normal pool'):
        try:
            audio, _ = librosa.load(row['audio_path'], sr=SR, mono=True)
            pool.append(audio)
        except Exception:
            pass

    print(f"  Loaded {len(pool)} normal recordings as mixing background")
    return pool


def _mix_with_normal(synth_audio, normal_pool, alpha):
    """
    Mix synthesised pathological sound with a real normal breathing recording.

    output = alpha * synth + (1 - alpha) * background
    Both are RMS-normalised before mixing so neither drowns the other.
    """
    if not normal_pool:
        return synth_audio

    bg = random.choice(normal_pool).copy()
    n  = len(synth_audio)

    if len(bg) < n:
        bg = np.pad(bg, (0, n - len(bg)))
    else:
        start = np.random.randint(0, len(bg) - n + 1)
        bg    = bg[start:start + n]

    rms_s  = np.sqrt(np.mean(synth_audio ** 2) + 1e-8)
    rms_b  = np.sqrt(np.mean(bg ** 2) + 1e-8)
    bg     = bg / rms_b * rms_s      # match RMS to synthetic

    mixed = alpha * synth_audio + (1.0 - alpha) * bg
    peak  = np.max(np.abs(mixed))
    return (mixed / peak) if peak > 1e-8 else mixed


# ─────────────────────────────────────────────
# Feature extraction (mirrors preprocessing.py exactly)
# ─────────────────────────────────────────────

def _extract_mel(audio, sr=SR):
    import librosa
    audio  = _butterworth_filter(audio)
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512,
        n_mels=128, fmin=50, fmax=2000, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


# ─────────────────────────────────────────────
# Main generation pipeline
# ─────────────────────────────────────────────

_SYNTH_FN = {
    1: synthesise_crackles,
    2: synthesise_wheeze,
    3: synthesise_both,
}

_LABEL_NAME = {1: 'crackle', 2: 'wheeze', 3: 'both'}


def generate_synthetic_features(manifest_path, output_dir, targets=None, seed=42):
    """
    Generate synthetic mel-spectrogram features for minority sound classes.

    Args:
        manifest_path: Path to manifest.csv (original data)
        output_dir:    Where to save .npy files
        targets:       {sound_label: n_samples} — defaults to DEFAULT_TARGETS
        seed:          Random seed for reproducibility

    Returns:
        pd.DataFrame: manifest of all generated synthetic samples,
                      also saved as manifest_synthetic.csv next to manifest.csv
    """
    np.random.seed(seed)
    random.seed(seed)

    if targets is None:
        targets = DEFAULT_TARGETS

    os.makedirs(output_dir, exist_ok=True)

    print("Loading normal breathing pool for mixing...")
    normal_pool = _load_normal_pool(manifest_path)

    records  = []
    synth_id = 0

    for sound_label, n_samples in sorted(targets.items()):
        diag_label = DIAG_FOR_SOUND[sound_label]
        alpha_lo, alpha_hi = ALPHA_RANGE[sound_label]
        synth_fn   = _SYNTH_FN[sound_label]
        label_name = _LABEL_NAME[sound_label]

        print(f"\nGenerating {n_samples}x synthetic '{label_name}' samples...")

        for _ in tqdm(range(n_samples)):
            alpha    = np.random.uniform(alpha_lo, alpha_hi)
            duration = np.random.uniform(MIN_DURATION, MAX_DURATION)

            synth = synth_fn(duration=duration)
            synth = _normalise_to_duration(synth, duration)
            audio = _mix_with_normal(synth, normal_pool, alpha)

            # Pad to at least MIN_DURATION after mixing
            min_len = int(MIN_DURATION * SR)
            if len(audio) < min_len:
                audio = np.pad(audio, (0, min_len - len(audio)))

            try:
                mel = _extract_mel(audio)
            except Exception:
                continue

            fname     = f"synth_{label_name}_{synth_id:05d}.npy"
            feat_path = os.path.join(output_dir, fname)
            np.save(feat_path, mel)

            records.append({
                'audio_file':      fname.replace('.npy', '.wav'),
                'audio_path':      '',        # no raw WAV (synthetic)
                'features_path':   feat_path,
                'sound_label':     sound_label,
                'diagnosis_label': diag_label,
                'patient_id':      -1,        # sentinel: not a real patient
                'split':           'train',
                'source':          'synthetic',
            })
            synth_id += 1

    df          = pd.DataFrame(records)
    synth_path  = os.path.join(os.path.dirname(manifest_path), 'manifest_synthetic.csv')
    df.to_csv(synth_path, index=False)

    print(f"\nDone: {len(df)} synthetic features saved")
    print(f"Manifest: {synth_path}")
    return df
