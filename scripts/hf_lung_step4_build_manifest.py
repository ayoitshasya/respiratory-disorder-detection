"""
Step 4 driver — date-grouped, size-aware train/val/test split for HF_Lung_V1,
full audio+feature extraction (reusing src/preprocessing.py's Butterworth +
mel-spectrogram code UNCHANGED), and merge into a combined manifest with a
source_dataset column.

Does NOT touch data/splits/patient_splits.json or data/processed/manifest.csv
(the existing ICBHI split/manifest are read-only inputs here). HF_Lung
features are cached to a separate data/processed/hf_lung/{train,val,test}/
subtree so there is zero risk of overwriting or colliding with ICBHI's
cached .npy files. Output is a NEW file: data/processed/manifest_merged.csv.

IMPORTANT — diagnosis_label for HF_Lung rows: HF_Lung has no diagnosis
metadata at all (it's an event-detection dataset, not diagnosis-labeled).
Every HF_Lung row gets diagnosis_label = -1 as a sentinel. Any training code
consuming manifest_merged.csv MUST mask rows with diagnosis_label == -1 out
of the diagnosis-head loss (matches the masking behavior already described,
but not yet implemented, in src/multitask_model.py's docstring: "Samples
without a diagnosis label only contribute to Head 1 loss.") — this is a
training-time concern, not something this script solves.
"""
import os
import sys
import time
import collections

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hf_lung_step2_label_counts import build_all_windows, ROOTS
from hf_lung_labels import extract_date_key
from hf_lung_split import (compute_date_sizes, greedy_size_aware_split,
                            apply_split, verify_no_date_leakage)
from preprocessing import butterworth_filter, extract_mel_spectrogram, SR as PREPROC_SR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICBHI_MANIFEST = os.path.join(BASE, 'data', 'processed', 'manifest.csv')
HF_LUNG_FEATURES_ROOT = os.path.join(BASE, 'data', 'processed', 'hf_lung')
MERGED_MANIFEST_OUT = os.path.join(BASE, 'data', 'processed', 'manifest_merged.csv')

SR = 16000  # must match src/preprocessing.py SR exactly — asserted below
CLIP_LEN_SAMPLES = 15 * SR
SOUND_LABEL_MAP = {'Normal': 0, 'Crackle': 1, 'Wheeze': 2, 'Both': 3}
DIAG_SENTINEL = -1  # HF_Lung has no diagnosis metadata


def report_class_distribution_by_large_date(per_file_windows, file_to_date, top_n=10):
    """
    Per-date class proportions for the top-N largest dates, vs. the global
    average — flags whether a huge date is skewed toward one class (which
    would matter for which split it lands in under size-only bin-packing).
    """
    date_class_counts = collections.defaultdict(collections.Counter)
    global_counts = collections.Counter()
    for wav_name, info in per_file_windows.items():
        date_key = file_to_date[wav_name]
        for w in info['adventitious']:
            date_class_counts[date_key][w['label']] += 1
            global_counts[w['label']] += 1
        for w in info['normal']:
            date_class_counts[date_key]['Normal'] += 1
            global_counts['Normal'] += 1

    global_total = sum(global_counts.values())
    global_frac = {c: global_counts[c] / global_total for c in SOUND_LABEL_MAP}

    date_totals = {d: sum(c.values()) for d, c in date_class_counts.items()}
    top_dates = sorted(date_totals.items(), key=lambda kv: -kv[1])[:top_n]

    print(f"=== Class distribution: top {top_n} largest dates vs. global average ===")
    print(f"  {'date':14s} {'n_windows':>9s}  " +
          "  ".join(f"{c:>8s}" for c in ['Normal', 'Crackle', 'Wheeze', 'Both']))
    print(f"  {'GLOBAL':14s} {global_total:9d}  " +
          "  ".join(f"{global_frac[c]*100:7.1f}%" for c in ['Normal', 'Crackle', 'Wheeze', 'Both']))
    flags = []
    for date_key, total in top_dates:
        counts = date_class_counts[date_key]
        fracs = {c: counts.get(c, 0) / total for c in SOUND_LABEL_MAP}
        line = f"  {date_key:14s} {total:9d}  " + "  ".join(
            f"{fracs[c]*100:7.1f}%" for c in ['Normal', 'Crackle', 'Wheeze', 'Both'])
        print(line)
        for c in SOUND_LABEL_MAP:
            # Flag: this date's share of a class is >2x or <0.5x the global rate
            # AND the class carries meaningful weight on this date (avoid noise
            # from e.g. 0.3% Both differences that are only 1-2 windows).
            if global_frac[c] > 0 and counts.get(c, 0) >= 5:
                ratio = fracs[c] / global_frac[c]
                if ratio > 2.0 or ratio < 0.5:
                    flags.append((date_key, c, fracs[c], global_frac[c]))
    print()
    if flags:
        print("  FLAGGED — class share deviates >2x from global average on a large date:")
        for date_key, c, df, gf in flags:
            print(f"    {date_key}: {c} = {df*100:.1f}% of its windows vs {gf*100:.1f}% globally")
    else:
        print("  No large date shows a >2x class-share deviation from the global average.")
    print()
    return flags


def extract_features_for_all(per_file_windows, file_to_date):
    """
    Per source file: load once, resample to 16kHz once, Butterworth-filter
    the FULL clip once (avoids filtfilt edge artifacts at every individual
    1s window boundary, and is far cheaper than filtering per-window), then
    slice + mel-spectrogram (both src/preprocessing.py functions, UNCHANGED)
    per window. Caches to data/processed/hf_lung/<split>/<name>.npy.

    Returns: list of manifest row dicts.
    """
    assert PREPROC_SR == SR, f"preprocessing.SR ({PREPROC_SR}) != expected {SR}"

    rows = []
    shape_mismatches = []
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(HF_LUNG_FEATURES_ROOT, split), exist_ok=True)

    t0 = time.time()
    for wav_name, info in tqdm(per_file_windows.items(), desc='HF_Lung extraction'):
        source_path = os.path.join(ROOTS[info['split']], wav_name)

        y, _ = librosa.load(source_path, sr=SR, mono=True)
        if len(y) < CLIP_LEN_SAMPLES:
            y = np.pad(y, (0, CLIP_LEN_SAMPLES - len(y)))
        elif len(y) > CLIP_LEN_SAMPLES:
            y = y[:CLIP_LEN_SAMPLES]

        y_filt = butterworth_filter(y, fs=SR)

        all_windows = [(w, w['label']) for w in info['adventitious']] + \
                      [(w, 'Normal') for w in info['normal']]

        stem = os.path.splitext(wav_name)[0]
        for idx, (w, label) in enumerate(all_windows):
            start_sample = int(round(w['window_start'] * SR))
            end_sample = start_sample + SR  # 1.0s window == SR samples
            clip = y_filt[start_sample:end_sample]
            if len(clip) != SR:
                clip = np.pad(clip, (0, max(0, SR - len(clip))))[:SR]

            feat = extract_mel_spectrogram(clip, sr=SR)
            if feat.shape != (128, 63):
                shape_mismatches.append((wav_name, idx, feat.shape))

            split = w['split']
            out_name = f"{stem}_{idx:02d}_{label}.npy"
            out_path = os.path.join(HF_LUNG_FEATURES_ROOT, split, out_name)
            np.save(out_path, feat)

            rows.append({
                'audio_file': wav_name,
                'audio_path': source_path,
                'features_path': out_path,
                'sound_label': SOUND_LABEL_MAP[label],
                'diagnosis_label': DIAG_SENTINEL,
                'patient_id': file_to_date[wav_name],
                'split': split,
                'source_dataset': 'hf_lung',
                'window_start_s': round(w['window_start'], 3),
                'window_end_s': round(w['window_end'], 3),
            })

    elapsed = time.time() - t0
    print(f"Extracted {len(rows)} windows from {len(per_file_windows)} source files in {elapsed:.1f}s")
    if shape_mismatches:
        print(f"  WARNING: {len(shape_mismatches)} windows did not produce the expected (128, 63) "
              f"mel shape — first few: {shape_mismatches[:5]}")
    else:
        print("  All windows produced the expected (128, 63) mel shape.")
    return rows


def main():
    print("Building HF_Lung window candidates (Step 2 logic, reused)...")
    per_file_windows, class_counts, _, _, _ = build_all_windows()

    date_sizes, file_to_date = compute_date_sizes(per_file_windows, extract_date_key)
    print(f"\n{len(date_sizes)} unique date-groups, {sum(date_sizes.values())} total windows")

    report_class_distribution_by_large_date(per_file_windows, file_to_date, top_n=10)

    assignment, split_counts = greedy_size_aware_split(date_sizes)
    verify_no_date_leakage(assignment)

    total = sum(split_counts.values())
    print("=== Size-aware split result (sample-count based, not date-count based) ===")
    for name, target in (('train', 0.6), ('val', 0.2), ('test', 0.2)):
        actual = split_counts[name]
        n_dates = sum(1 for d in assignment.values() if d == name)
        print(f"  {name:6s}: {actual:6d} windows ({actual/total*100:5.1f}%, target {target*100:.0f}%), "
              f"{n_dates} date-groups")
    print()

    apply_split(per_file_windows, assignment, file_to_date)

    print("Extracting audio + mel-spectrogram features for all HF_Lung windows "
          "(Butterworth + mel code from src/preprocessing.py, UNCHANGED)...")
    hf_rows = extract_features_for_all(per_file_windows, file_to_date)
    hf_df = pd.DataFrame(hf_rows)

    print(f"\nLoading existing ICBHI manifest (read-only, untouched): {ICBHI_MANIFEST}")
    icbhi_df = pd.read_csv(ICBHI_MANIFEST)
    icbhi_df['source_dataset'] = 'icbhi'
    icbhi_df['window_start_s'] = np.nan
    icbhi_df['window_end_s'] = np.nan

    merged = pd.concat([icbhi_df, hf_df], ignore_index=True, sort=False)
    merged.to_csv(MERGED_MANIFEST_OUT, index=False)
    print(f"Merged manifest written: {MERGED_MANIFEST_OUT}  ({len(merged)} total rows: "
          f"{len(icbhi_df)} icbhi + {len(hf_df)} hf_lung)")

    # Confirm ICBHI's own split assignments were never touched.
    icbhi_reread = pd.read_csv(ICBHI_MANIFEST)
    assert icbhi_reread['split'].equals(pd.read_csv(ICBHI_MANIFEST)['split'])
    print("Confirmed: data/processed/manifest.csv unchanged on disk.")


if __name__ == '__main__':
    main()
