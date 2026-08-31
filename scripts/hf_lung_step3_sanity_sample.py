"""
Step 3 driver — extract a small (~20 event) sample of HF_Lung windows as
actual .wav clips for manual listening review, plus a CSV summary.

Audio is output UNFILTERED (resampled to 16kHz only, no Butterworth bandpass)
so a human reviewer hears the clip as close to the original recording as
possible when checking "does this window really contain what its label
claims, at roughly this timestamp" — the Butterworth filter is a model-input
preprocessing step (applied later in the real feature pipeline), not needed
to audibly verify label/timestamp alignment.

Output: data/raw/hf_lung_v1/sanity_check_samples/
    <label>_<source_file_stem>_<window_start>s.wav
    summary.csv
"""
import os
import sys
import csv
import random

import librosa
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hf_lung_step2_label_counts import build_all_windows, ROOTS
from hf_lung_labels import extract_date_key

SR = 16000  # matches src/preprocessing.py SR_MODEL — NOT the native 4000Hz file rate
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'data', 'raw', 'hf_lung_v1', 'sanity_check_samples')

random.seed(42)

# Target sample composition — mix of classes and both filename prefixes.
TARGET_COUNTS = {'Crackle': 5, 'Wheeze': 5, 'Both': 3, 'Normal': 7}  # sums to 20


def pick_samples(per_file_windows):
    """Pick a mixed sample: target class counts, mixed steth_/trunc_ prefixes."""
    candidates = {'Crackle': [], 'Wheeze': [], 'Both': [], 'Normal': []}
    for wav_name, info in per_file_windows.items():
        prefix = 'steth' if wav_name.startswith('steth_') else 'trunc'
        for w in info['adventitious']:
            candidates[w['label']].append({**w, 'source_file': wav_name,
                                            'split': info['split'], 'prefix': prefix})
        for w in info['normal']:
            candidates['Normal'].append({**w, 'source_file': wav_name,
                                          'split': info['split'], 'prefix': prefix})

    picked = []
    for label, n in TARGET_COUNTS.items():
        pool = candidates[label]
        random.shuffle(pool)
        # Try to get a mix of steth_/trunc_ within this label's picks.
        steth_pool = [c for c in pool if c['prefix'] == 'steth']
        trunc_pool = [c for c in pool if c['prefix'] == 'trunc']
        half = n // 2
        chosen = steth_pool[:half] + trunc_pool[:n - half]
        if len(chosen) < n:
            # backfill from whichever pool has more, if one prefix was short
            remaining = [c for c in pool if c not in chosen]
            chosen += remaining[:n - len(chosen)]
        picked.extend(chosen[:n])
    return picked


def fmt_ts(seconds):
    """Filename-safe timestamp, e.g. 7.500 -> '07.500'."""
    return f"{seconds:06.3f}".replace('.', '-')


def sanitize_stem(wav_name):
    return os.path.splitext(wav_name)[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building full-dataset window candidates (Step 2 logic, reused)...")
    per_file_windows, _, _, _, _ = build_all_windows()

    picked = pick_samples(per_file_windows)
    print(f"Picked {len(picked)} samples: "
          f"{ {l: sum(1 for p in picked if p['label']==l) for l in TARGET_COUNTS} }")

    rows = []
    for item in picked:
        wav_path = os.path.join(ROOTS[item['split']], item['source_file'])
        y, _ = librosa.load(wav_path, sr=SR, mono=True)

        start_sample = int(round(item['window_start'] * SR))
        end_sample = int(round(item['window_end'] * SR))
        clip = y[start_sample:end_sample]
        # Guard against any off-by-one at clip boundary (should not happen, but audible
        # clicks from a short clip are worse than a silent 1-sample pad).
        if len(clip) < SR:
            clip = librosa_pad_or_trim(clip, SR)

        stem = sanitize_stem(item['source_file'])
        out_name = f"{item['label']}_{stem}_{fmt_ts(item['window_start'])}s.wav"
        out_path = os.path.join(OUT_DIR, out_name)
        sf.write(out_path, clip, SR)

        rows.append({
            'output_wav': out_name,
            'label': item['label'],
            'source_file': item['source_file'],
            'split': item['split'],
            'prefix': item['prefix'],
            'date_key': extract_date_key(item['source_file']),
            'window_start_s': round(item['window_start'], 3),
            'window_end_s': round(item['window_end'], 3),
            'source_event_type': item.get('source_type', ''),
            'source_event_start_s': round(item['source_start'], 3) if 'source_start' in item else '',
            'source_event_end_s': round(item['source_end'], 3) if 'source_end' in item else '',
        })

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} wav clips + summary.csv to:\n  {OUT_DIR}")


def librosa_pad_or_trim(clip, target_len):
    import numpy as np
    if len(clip) < target_len:
        return np.pad(clip, (0, target_len - len(clip)))
    return clip[:target_len]


if __name__ == '__main__':
    main()
