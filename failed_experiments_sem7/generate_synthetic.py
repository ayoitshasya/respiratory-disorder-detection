"""
Generate synthetic respiratory sound features to reduce class imbalance.

Run once before training with synthetic data:
    python generate_synthetic.py

Output:
    data/processed/train_synthetic/   -- .npy mel-spectrogram files
    data/processed/manifest_synthetic.csv
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.synthetic_generator import generate_synthetic_features, DEFAULT_TARGETS

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE, 'data', 'processed', 'manifest.csv')
OUTPUT_DIR    = os.path.join(BASE, 'data', 'processed', 'train_synthetic')

SOUND_NAMES = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}

print("=" * 65)
print("Synthetic Data Generation — PulmoScan")
print("=" * 65)

# Show what the distribution will look like after generation
orig    = pd.read_csv(MANIFEST_PATH)
train   = orig[orig['split'] == 'train']['sound_label'].value_counts().sort_index()
targets = DEFAULT_TARGETS

print("\nTarget distribution after synthesis:")
for lbl in range(4):
    o = int(train.get(lbl, 0))
    s = targets.get(lbl, 0)
    print(f"  {SOUND_NAMES[lbl]:8s}: {o:4d} orig + {s:4d} synth = {o+s:4d}")

print()
synth_df = generate_synthetic_features(MANIFEST_PATH, OUTPUT_DIR)

print("\nFinal counts:")
for lbl in range(4):
    o = int(train.get(lbl, 0))
    s = int((synth_df['sound_label'] == lbl).sum())
    print(f"  {SOUND_NAMES[lbl]:8s}: {o+s:4d}")

print("\nNext step: python train_with_synthetic.py")



