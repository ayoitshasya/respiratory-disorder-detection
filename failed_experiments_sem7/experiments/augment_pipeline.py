import os
import numpy as np
import librosa
import warnings
from scipy.signal import butter, filtfilt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\dataset augmentation\ICBHI_final_database"
OUTPUT_PATH  = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\processed_data"

SR           = 22050   # sample rate
DURATION     = 5.0     # fixed clip length in seconds
N_FFT        = 2048
HOP_LENGTH   = 512
N_MELS       = 128     # mel spectrogram bins
N_MFCC       = 40      # mfcc coefficients (zero-padded to 128 to match mel)
LOW_CUT      = 80      # butterworth low cutoff  (Hz)
HIGH_CUT     = 1500    # butterworth high cutoff (Hz)
FILTER_ORDER = 4

TARGET_PER_CLASS = 3600   # balance all classes to this count

CLASS_NAMES = ['normal', 'crackle', 'wheeze', 'both']

# ============================================================
# BUTTERWORTH BANDPASS FILTER
# ============================================================

def butterworth_filter(audio, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=SR, order=FILTER_ORDER):
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)

# ============================================================
# FEATURE EXTRACTION  —  Mel Spectrogram + MFCC  (2-channel)
# ============================================================

def extract_features(audio, sr=SR):
    """
    Returns array of shape (128, T, 2)
      channel 0 : Mel Spectrogram  (128 bins)
      channel 1 : MFCC             (40 coefs, zero-padded to 128)
    """
    # pad or trim to fixed length
    target_len = int(DURATION * sr)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # --- Mel Spectrogram ---
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)          # (128, T)

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )                                                        # (40,  T)
    mfcc_padded = np.pad(mfcc, ((0, N_MELS - N_MFCC), (0, 0)))  # (128, T)

    # --- Normalise each channel independently ---
    mel_db    = (mel_db    - mel_db.mean())    / (mel_db.std()    + 1e-8)
    mfcc_padded = (mfcc_padded - mfcc_padded.mean()) / (mfcc_padded.std() + 1e-8)

    # --- Stack → (128, T, 2) ---
    return np.stack([mel_db, mfcc_padded], axis=-1).astype(np.float32)

# ============================================================
# AUGMENTATION
# ============================================================

AUGMENTATIONS = [
    'time_stretch_slow',   # rate 0.8  — slower breathing
    'time_stretch_fast',   # rate 1.2  — faster breathing
    'pitch_up',            # +2 semitones
    'pitch_down',          # -2 semitones
    'add_noise',           # Gaussian noise (simulates MEMS ambient noise)
    'gain_up',             # louder
    'gain_down',           # quieter
]

def augment(audio, sr, aug_type):
    if   aug_type == 'time_stretch_slow': return librosa.effects.time_stretch(audio, rate=0.8)
    elif aug_type == 'time_stretch_fast': return librosa.effects.time_stretch(audio, rate=1.2)
    elif aug_type == 'pitch_up':          return librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    elif aug_type == 'pitch_down':        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    elif aug_type == 'add_noise':
        noise = np.random.normal(0, 0.005, len(audio))
        return audio + noise
    elif aug_type == 'gain_up':           return audio * 1.5
    elif aug_type == 'gain_down':         return audio * 0.7
    return audio

# ============================================================
# PARSE ANNOTATION FILE
# ============================================================

def parse_annotation(txt_path):
    """
    Returns list of (start_sec, end_sec, label)
    label: 0=normal, 1=crackle, 2=wheeze, 3=both
    """
    cycles = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            start, end = float(parts[0]), float(parts[1])
            crackle, wheeze = int(parts[2]), int(parts[3])

            if   crackle == 0 and wheeze == 0: label = 0
            elif crackle == 1 and wheeze == 0: label = 1
            elif crackle == 0 and wheeze == 1: label = 2
            else:                              label = 3

            cycles.append((start, end, label))
    return cycles

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    # create output folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

    # ---- STEP 1 : segment all recordings into breath cycles ----
    print("=" * 60)
    print("STEP 1 — Segmenting breath cycles")
    print("=" * 60)

    class_segments = {0: [], 1: [], 2: [], 3: []}   # label → [audio arrays]

    wav_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.wav')]

    for wav_file in tqdm(wav_files, desc="Segmenting"):
        base      = wav_file.replace('.wav', '')
        wav_path  = os.path.join(DATASET_PATH, wav_file)
        txt_path  = os.path.join(DATASET_PATH, base + '.txt')

        if not os.path.exists(txt_path):
            continue

        try:
            audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        except Exception as e:
            print(f"  [skip] {wav_file}: {e}")
            continue

        # apply butterworth filter
        audio = butterworth_filter(audio)

        for start, end, label in parse_annotation(txt_path):
            s = int(start * SR)
            e = int(min(end * SR, len(audio)))
            clip = audio[s:e]

            if len(clip) < int(0.5 * SR):   # skip clips < 0.5 s
                continue

            class_segments[label].append(clip)

    print("\nClass distribution after segmentation:")
    for lbl, segs in class_segments.items():
        print(f"  {CLASS_NAMES[lbl]:10s}: {len(segs):>5} segments")

    # ---- STEP 2 : extract features + augment minority classes ----
    print("\n" + "=" * 60)
    print("STEP 2 — Feature extraction + Augmentation")
    print("=" * 60)

    file_counts = {cls: 0 for cls in CLASS_NAMES}

    for label, segments in class_segments.items():
        cls_name   = CLASS_NAMES[label]
        cls_out    = os.path.join(OUTPUT_PATH, cls_name)
        n_original = len(segments)

        print(f"\n[{cls_name}]  {n_original} original segments")

        # save original segments
        for i, clip in enumerate(tqdm(segments, desc=f"  Originals")):
            feat = extract_features(clip)
            np.save(os.path.join(cls_out, f"{cls_name}_orig_{i:05d}.npy"), feat)
            file_counts[cls_name] += 1

        # augment if below target
        needed = TARGET_PER_CLASS - n_original
        if needed > 0:
            print(f"  Generating {needed} augmented samples...")
            for i in tqdm(range(needed), desc=f"  Augmenting"):
                clip     = segments[i % n_original]
                aug_type = AUGMENTATIONS[i % len(AUGMENTATIONS)]

                aug_clip = augment(clip, SR, aug_type)
                aug_clip = butterworth_filter(aug_clip)   # re-filter after aug

                feat = extract_features(aug_clip)
                np.save(os.path.join(cls_out, f"{cls_name}_aug_{i:05d}.npy"), feat)
                file_counts[cls_name] += 1

    # ---- SUMMARY ----
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output  : {OUTPUT_PATH}")
    print(f"\nFinal file counts:")
    total = 0
    for cls, cnt in file_counts.items():
        print(f"  {cls:10s}: {cnt:>5} .npy files")
        total += cnt
    print(f"  {'TOTAL':10s}: {total:>5} .npy files")
    print(f"\nFeature shape : (128, 216, 2)")
    print(f"  axis 0 — frequency bins  (128)")
    print(f"  axis 1 — time frames     (216 @ hop=512, 5s clip)")
    print(f"  axis 2 — channels        (0=Mel Spectrogram, 1=MFCC)")

if __name__ == "__main__":
    run_pipeline()
