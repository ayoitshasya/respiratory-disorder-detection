"""
train_baseline.py
-----------------
Baseline single-task CNN for respiratory sound classification.

Trains a 4-block CNN (32→64→128→256 filters) on mel-spectrogram features
from the ICBHI 2017 dataset to classify lung sounds into 4 categories:
Normal, Crackle, Wheeze, Both.

This is the starting point before multitask learning was introduced.
Achieved 59.80% ICBHI score on the test set.

Run:
    python train_baseline.py

Output:
    data/checkpoints/baseline_best.keras
    data/results/baseline_training_curves.png
    data/results/baseline_confusion_matrix.png
"""

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEED=42; NUM_SOUND=4; BATCH_SIZE=32
EPOCHS=50; LR=1e-3; LR_MIN=1e-6; TARGET_FRAMES=63; N_MELS=128
CKPT_DIR='data/checkpoints'; RESULTS_DIR='data/results'
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
SOUND_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']


def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0,0), (0, t-c)])
    else:
        feat = feat[..., :t]
    return feat


def icbhi_score(y_true, y_pred, n=NUM_SOUND):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=cm.sum()-tp-fn-fp
        se.append(tp/(tp+fn) if tp+fn>0 else 0.0)
        sp.append(tn/(tn+fp) if tn+fp>0 else 0.0)
    return (np.mean(se)+np.mean(sp))/2.0


manifest = pd.read_csv('data/processed/manifest.csv')
train_df  = manifest[manifest['split'] == 'train']
val_df    = manifest[manifest['split'] == 'val']
print(f'Train: {len(train_df)} | Val: {len(val_df)}')


def load_split(df):
    X, ys = [], []
    for _, row in df.iterrows():
        feat = np.load(row['features_path'])
        feat = pad_or_truncate(feat)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X.append(feat); ys.append(row['sound_label'])
    return np.array(X, np.float32), np.array(ys, np.int32)


print('Loading val...'); X_val, y_val = load_split(val_df)
print('Loading train...'); X_tr, y_tr = load_split(train_df)

y_tr_oh  = tf.keras.utils.to_categorical(y_tr,  NUM_SOUND)
y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_SOUND)

cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
cw_dict = {i: cw[i] for i in range(NUM_SOUND)}
print(f'Class weights: {cw_dict}')


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
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(NUM_SOUND, activation='softmax')(x)
    return tf.keras.Model(inp, out)


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

best_icbhi = 0.0
icbhi_history = []
train_acc_history = []
val_acc_history   = []
train_loss_history = []
val_loss_history   = []

print('\nStarting training...')
for epoch in range(EPOCHS):
    lr = float(LR_MIN + 0.5*(LR - LR_MIN)*(1 + np.cos(np.pi * epoch / EPOCHS)))
    model.optimizer.learning_rate.assign(lr)

    result = model.fit(
        X_tr, y_tr_oh,
        validation_data=(X_val, y_val_oh),
        batch_size=BATCH_SIZE,
        epochs=1, verbose=1,
        class_weight=cw_dict
    )

    train_acc_history.append(result.history['accuracy'][0])
    val_acc_history.append(result.history['val_accuracy'][0])
    train_loss_history.append(result.history['loss'][0])
    val_loss_history.append(result.history['val_loss'][0])

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    score  = icbhi_score(y_val, y_pred)
    icbhi_history.append(score)

    val_acc = result.history['val_accuracy'][0]
    print(f'  -> val_icbhi: {score:.4f} | val_acc: {val_acc:.4f}')

    if score > best_icbhi:
        best_icbhi = score
        model.save(f'{CKPT_DIR}/baseline_best.keras')
        print(f'  -> NEW BEST: {best_icbhi:.4f} saved')

    # early stop if no icbhi improvement for 15 epochs
    if len(icbhi_history) > 15 and max(icbhi_history[-15:]) <= max(icbhi_history[:-15]):
        print(f'Early stopping at epoch {epoch+1}')
        break

print(f'\nBest ICBHI: {best_icbhi:.4f}')
model_best = tf.keras.models.load_model(f'{CKPT_DIR}/baseline_best.keras', compile=False)
y_pred = np.argmax(model_best.predict(X_val, verbose=0), axis=1)

print('\nSound Classification Report:')
print(classification_report(y_val, y_pred, labels=list(range(NUM_SOUND)), target_names=SOUND_NAMES))

# ── Training curves ──────────────────────────────────────────
epochs_ran = len(icbhi_history)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_acc_history, label='Train Acc')
axes[0].plot(val_acc_history, label='Val Acc')
axes[0].set_title('Sound Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(train_loss_history, label='Train Loss')
axes[1].plot(val_loss_history, label='Val Loss')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()

axes[2].plot(icbhi_history, color='green', label='Val ICBHI')
axes[2].axhline(y=best_icbhi, color='red', linestyle='--', label=f'Best: {best_icbhi:.4f}')
axes[2].set_title('ICBHI Score'); axes[2].set_xlabel('Epoch'); axes[2].legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/baseline_training_curves.png', dpi=150)
print(f'Saved -> {RESULTS_DIR}/baseline_training_curves.png')

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_val, y_pred, labels=list(range(NUM_SOUND)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=SOUND_NAMES, yticklabels=SOUND_NAMES)
plt.title(f'Baseline CNN - Confusion Matrix (ICBHI: {best_icbhi:.4f})')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/baseline_confusion_matrix.png', dpi=150)
print(f'Saved -> {RESULTS_DIR}/baseline_confusion_matrix.png')
