"""
Generate threshold sweep plot — shows ICBHI score vs class weight for each class.
This visualises the threshold tuning process for the report.

Run: python threshold_sweep.py
Output: data/results/threshold_sweep.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE, 'data', 'processed', 'manifest_aug_modest.csv')
MODEL_PATH    = os.path.join(BASE, 'data', 'checkpoints', 'highres_best.keras')
OUT_PATH      = os.path.join(BASE, 'data', 'results', 'threshold_sweep.png')
TARGET_FRAMES = 126
N_MELS        = 128
BEST_WEIGHTS  = [0.8, 0.5, 2.0, 10.0]
CLASS_NAMES   = ['Normal', 'Crackle', 'Wheeze', 'Both']

def icbhi(y_true, y_pred, n=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp = cm[i, i]; fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        se.append(tp / (tp + fn + 1e-9))
        sp.append(tn / (tn + fp + 1e-9))
    return (np.mean(se) + np.mean(sp)) / 2

def load_val():
    df  = pd.read_csv(MANIFEST_PATH)
    val = df[df['split'] == 'val'].reset_index(drop=True)
    X, y = [], []
    for _, row in val.iterrows():
        path = row['features_path']
        if not os.path.exists(path):
            continue
        feat = np.load(path)
        t = feat.shape[-1]
        if t < TARGET_FRAMES:
            feat = np.pad(feat, [(0, 0), (0, TARGET_FRAMES - t)])
        else:
            feat = feat[..., :TARGET_FRAMES]
        X.append(feat[..., np.newaxis])
        y.append(int(row['sound_label']))
    return np.array(X, dtype=np.float32), np.array(y)

def build_model():
    inp = tf.keras.Input(shape=(N_MELS, TARGET_FRAMES, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    shared = tf.keras.layers.Dense(256, activation='relu')(x)
    shared = tf.keras.layers.Dropout(0.5)(shared)
    s = tf.keras.layers.Dense(128, activation='relu')(shared)
    s = tf.keras.layers.Dropout(0.3)(s)
    sound_out = tf.keras.layers.Dense(4, activation='softmax', name='sound')(s)
    d = tf.keras.layers.Dense(128, activation='relu')(shared)
    d = tf.keras.layers.Dropout(0.3)(d)
    diag_out = tf.keras.layers.Dense(7, activation='softmax', name='diagnosis')(d)
    return tf.keras.Model(inp, [sound_out, diag_out])

print("Loading val data...")
X_val, y_val = load_val()
print(f"Val shape: {X_val.shape}")

print("Loading model...")
model = build_model()
model.load_weights(MODEL_PATH)

print("Getting predictions...")
sound_probs, _ = model.predict(X_val, batch_size=32, verbose=0)

# Baseline (no tuning)
baseline = icbhi(y_val, np.argmax(sound_probs, axis=1))
best     = icbhi(y_val, np.argmax(sound_probs * np.array(BEST_WEIGHTS), axis=1))
print(f"Baseline ICBHI: {baseline:.4f}")
print(f"Tuned ICBHI:    {best:.4f}")

# Sweep each weight independently, others fixed at optimal
sweep = np.concatenate([np.arange(0.1, 3.0, 0.1), np.arange(3.0, 15.0, 0.5)])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('ICBHI Score vs Threshold Weight per Class', fontsize=14, y=1.01)

colors = ['#1976d2', '#388e3c', '#f57c00', '#c62828']

for idx, ax in enumerate(axes.flat):
    scores = []
    for w in sweep:
        weights = BEST_WEIGHTS.copy()
        weights[idx] = w
        y_pred = np.argmax(sound_probs * np.array(weights), axis=1)
        scores.append(icbhi(y_val, y_pred))

    ax.plot(sweep, scores, color=colors[idx], linewidth=2)
    ax.axvline(BEST_WEIGHTS[idx], color='red', linestyle='--', linewidth=1.5,
               label=f'Optimal = {BEST_WEIGHTS[idx]}')
    ax.axhline(best, color='gray', linestyle=':', alpha=0.8, label=f'Best = {best:.2%}')
    ax.axhline(baseline, color='black', linestyle=':', alpha=0.4, label=f'No tuning = {baseline:.2%}')
    ax.set_xlabel(f'{CLASS_NAMES[idx]} weight')
    ax.set_ylabel('ICBHI Score')
    ax.set_title(f'{CLASS_NAMES[idx]}')
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, 0.65)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved to {OUT_PATH}")
