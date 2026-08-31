"""
Re-extract all mel-spectrogram features with hop_length=256 (was 512).
Overwrites existing .npy files in data/processed/train|val|test/.

Run once after changing HOP_LENGTH in preprocessing.py:
    python reextract_features.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocessing import cache_features

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_CSV = os.path.join(BASE, 'data', 'processed', 'manifest.csv')

print("=" * 65)
print("Feature Re-extraction  (hop_length=256, TARGET_FRAMES=126)")
print("=" * 65)

manifest = pd.read_csv(MANIFEST_CSV)
print(f"Total entries: {len(manifest)}")
for split in ['train', 'val', 'test']:
    n = (manifest['split'] == split).sum()
    print(f"  {split}: {n}")

print()
manifest = cache_features(manifest, feature_type='mel', overwrite=True)
print("Done — all features re-extracted with new hop_length=256.")
