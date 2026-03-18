"""
data_loader.py
--------------
Patient-level train/val/test splitting and dataset manifest generation.

CRITICAL: All splits are done at the PATIENT level. No patient ever appears
in more than one split. This prevents data leakage from augmented versions
of the same patient's recordings appearing in both train and test.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

SOUND_LABEL_MAP = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}
DIAGNOSIS_LABEL_MAP = {
    'Healthy':        0,
    'COPD':           1,
    'URTI':           2,
    'Bronchiectasis': 3,
    'Pneumonia':      4,
    'Bronchiolitis':  5,
    'LRTI':           6,   # rare — merged into 'Other' for stratification
    'Asthma':         6,   # rare — merged into 'Other' for stratification
}
# Minimum patients required in a class to stratify on it independently
MIN_PATIENTS_FOR_STRAT = 2

SEED = 42


def set_seeds():
    """Set all random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)


def load_metadata(metadata_csv_path):
    """
    Load and validate the training metadata CSV.

    Args:
        metadata_csv_path: Path to training_metadata_with_diagnosis.csv

    Returns:
        pd.DataFrame with columns: Patient, Label, Diagnosis, Audio_File, ...
    """
    df = pd.read_csv(metadata_csv_path)
    df.columns = df.columns.str.strip()

    required = ['Patient', 'Label', 'Diagnosis', 'Audio_File']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata CSV: {missing}")

    # Encode labels
    df['sound_label']     = df['Label'].str.strip().map(SOUND_LABEL_MAP)
    df['diagnosis_label'] = df['Diagnosis'].str.strip().map(DIAGNOSIS_LABEL_MAP)

    unknown_sound = df['sound_label'].isna().sum()
    unknown_diag  = df['diagnosis_label'].isna().sum()
    if unknown_sound > 0:
        print(f"  [warn] {unknown_sound} rows with unknown sound label")
    if unknown_diag > 0:
        print(f"  [warn] {unknown_diag} rows with unknown diagnosis label")

    df = df.dropna(subset=['sound_label', 'diagnosis_label'])
    df['sound_label']     = df['sound_label'].astype(int)
    df['diagnosis_label'] = df['diagnosis_label'].astype(int)

    print(f"Loaded metadata: {len(df)} cycles, {df['Patient'].nunique()} unique patients")
    return df


def make_patient_splits(df, val_size=0.20, test_size=0.20):
    """
    Create stratified patient-level train/val/test splits.

    Stratification is done per diagnosis group: patients from each diagnosis
    are distributed proportionally across splits. Singleton patients
    (from very rare diagnoses) are assigned to train only to avoid
    empty val/test classes.

    Args:
        df:        DataFrame from load_metadata()
        val_size:  Fraction of patients for validation
        test_size: Fraction of patients for test

    Returns:
        dict with keys 'train', 'val', 'test', each a list of patient IDs
    """
    set_seeds()
    rng = np.random.RandomState(SEED)

    # One row per patient — use their most common diagnosis as grouping key
    patient_df = (
        df.groupby('Patient')['diagnosis_label']
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={'diagnosis_label': 'primary_diagnosis'})
    )

    train_patients, val_patients, test_patients = [], [], []

    # Split within each diagnosis group to maintain class balance
    for diag_label, group in patient_df.groupby('primary_diagnosis'):
        pids = group['Patient'].values.tolist()
        rng.shuffle(pids)
        n = len(pids)

        if n < 3:
            # Too few patients to split — put all in train
            print(f"  [info] Diagnosis {diag_label}: only {n} patient(s) -- all assigned to train")
            train_patients.extend(pids)
            continue

        n_test = max(1, round(n * test_size))
        n_val  = max(1, round(n * val_size))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Edge case: force at least 1 train patient
            n_train = 1
            n_val   = max(0, n - n_train - n_test)

        train_patients.extend(pids[:n_train])
        val_patients.extend(pids[n_train:n_train + n_val])
        test_patients.extend(pids[n_train + n_val:])

    # Verify no overlap
    assert len(set(train_patients) & set(val_patients))  == 0, "Train/Val overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, "Train/Test overlap!"
    assert len(set(val_patients)   & set(test_patients)) == 0, "Val/Test overlap!"

    print(f"\nPatient split: {len(train_patients)} train | {len(val_patients)} val | {len(test_patients)} test")

    splits = {'train': train_patients, 'val': val_patients, 'test': test_patients}
    return splits


def save_splits(splits, splits_dir):
    """Save patient split indices to JSON."""
    os.makedirs(splits_dir, exist_ok=True)
    out_path = os.path.join(splits_dir, 'patient_splits.json')
    with open(out_path, 'w') as f:
        json.dump({k: [int(p) for p in v] for k, v in splits.items()}, f, indent=2)
    print(f"Splits saved to: {out_path}")


def load_splits(splits_dir):
    """Load patient splits from JSON."""
    path = os.path.join(splits_dir, 'patient_splits.json')
    with open(path, 'r') as f:
        splits = json.load(f)
    return splits


def assign_split_column(df, splits):
    """Add a 'split' column to the DataFrame based on patient splits."""
    patient_to_split = {}
    for split_name, patients in splits.items():
        for p in patients:
            patient_to_split[p] = split_name
    df['split'] = df['Patient'].map(patient_to_split)
    return df


def print_split_stats(df):
    """Print detailed statistics for each split."""
    sound_names = {v: k for k, v in SOUND_LABEL_MAP.items()}
    # Build reverse map (LRTI/Asthma both map to 6 → show as 'Other')
    diag_names  = {v: k for k, v in DIAGNOSIS_LABEL_MAP.items()}
    diag_names[6] = 'Other(LRTI/Asthma)'

    print("\n" + "=" * 65)
    print("SPLIT STATISTICS")
    print("=" * 65)

    for split in ['train', 'val', 'test']:
        sdf = df[df['split'] == split]
        print(f"\n[{split.upper()}]  {sdf['Patient'].nunique()} patients | {len(sdf)} cycles")

        print("  Sound type distribution:")
        for lbl, cnt in sdf['sound_label'].value_counts().sort_index().items():
            pct = cnt / len(sdf) * 100
            print(f"    {sound_names[lbl]:10s}: {cnt:5d}  ({pct:.1f}%)")

        print("  Diagnosis distribution:")
        for lbl, cnt in sdf['diagnosis_label'].value_counts().sort_index().items():
            pct = cnt / len(sdf) * 100
            print(f"    {diag_names[lbl]:15s}: {cnt:5d}  ({pct:.1f}%)")

    print("=" * 65)


def build_manifest(df, cycles_dir, processed_dir):
    """
    Build a manifest CSV linking every cycle to its:
      - audio file path (in cycles_dir)
      - features path (in processed_dir, filled in during preprocessing)
      - labels and patient ID

    Args:
        df:            DataFrame with split column assigned
        cycles_dir:    Directory containing segmented .wav cycle files
        processed_dir: Directory where .npy features will be saved

    Returns:
        pd.DataFrame manifest
    """
    # Audio files may be in a subdirectory after zip extraction
    cycle_subdirs = [d for d in os.listdir(cycles_dir)
                     if os.path.isdir(os.path.join(cycles_dir, d))]
    audio_root = os.path.join(cycles_dir, cycle_subdirs[0]) if cycle_subdirs else cycles_dir

    records = []
    missing = 0
    for _, row in df.iterrows():
        audio_fname = str(row['Audio_File']).strip()
        audio_path  = os.path.join(audio_root, audio_fname)

        if not os.path.exists(audio_path):
            missing += 1
            continue

        feat_fname = audio_fname.replace('.wav', '.npy')
        feat_path  = os.path.join(processed_dir, row['split'], feat_fname)

        records.append({
            'audio_file':      audio_fname,
            'audio_path':      audio_path,
            'features_path':   feat_path,
            'sound_label':     row['sound_label'],
            'diagnosis_label': row['diagnosis_label'],
            'patient_id':      int(row['Patient']),
            'split':           row['split'],
        })

    if missing > 0:
        print(f"  [warn] {missing} audio files not found in cycles_dir")

    manifest = pd.DataFrame(records)
    print(f"Manifest built: {len(manifest)} entries ({missing} missing audio files skipped)")
    return manifest


def save_manifest(manifest, processed_dir):
    """Save manifest CSV."""
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, 'manifest.csv')
    manifest.to_csv(path, index=False)
    print(f"Manifest saved to: {path}")


def load_manifest(processed_dir, split=None):
    """
    Load manifest CSV, optionally filtered to a specific split.

    Args:
        processed_dir: Directory containing manifest.csv
        split:         'train', 'val', 'test', or None (all)

    Returns:
        pd.DataFrame
    """
    path = os.path.join(processed_dir, 'manifest.csv')
    df   = pd.read_csv(path)
    if split is not None:
        df = df[df['split'] == split].reset_index(drop=True)
    return df


def run_split_pipeline(metadata_csv, splits_dir, cycles_dir, processed_dir):
    """
    Full pipeline: load metadata → split patients → build manifest.
    Run this once before any preprocessing or training.

    Args:
        metadata_csv:  Path to training_metadata_with_diagnosis.csv
        splits_dir:    Where to save patient_splits.json
        cycles_dir:    Directory containing extracted cycle wav files
        processed_dir: Where features will be cached
    """
    print("=" * 65)
    print("PHASE 1a — Patient-Level Data Splitting")
    print("=" * 65)

    df     = load_metadata(metadata_csv)
    splits = make_patient_splits(df)
    save_splits(splits, splits_dir)

    df = assign_split_column(df, splits)
    print_split_stats(df)

    manifest = build_manifest(df, cycles_dir, processed_dir)
    save_manifest(manifest, processed_dir)

    return df, splits, manifest


if __name__ == '__main__':
    import sys
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    metadata_csv  = os.path.join(BASE, 'data', 'raw', 'diagnosis_metadata', 'training_metadata_with_diagnosis.csv')
    splits_dir    = os.path.join(BASE, 'data', 'splits')
    cycles_dir    = os.path.join(BASE, 'data', 'cycles')
    processed_dir = os.path.join(BASE, 'data', 'processed')

    run_split_pipeline(metadata_csv, splits_dir, cycles_dir, processed_dir)
