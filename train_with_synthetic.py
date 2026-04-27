"""
Multitask training with synthetic data merged into the training split.
Architecture and hyperparameters identical to multitask_model.py.

Run after generate_synthetic.py:
    python train_with_synthetic.py
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

SEED          = 42
NUM_SOUND     = 4
NUM_DIAGNOSIS = 7
BATCH_SIZE    = 32
EPOCHS        = 80
LR            = 1e-3
LR_MIN        = 1e-6
TARGET_FRAMES = 63
N_MELS        = 128
DIAG_WEIGHT   = 0.1
CKPT_DIR      = 'data/checkpoints'
RESULTS_DIR   = 'data/results'

os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SOUND_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']
DIAG_NAMES  = ['Healthy', 'COPD', 'URTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis', 'Other']


def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def focal_loss(gamma=2.0, class_weights=None):
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        p_t     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal   = tf.pow(1.0 - p_t, gamma) * ce
        if class_weights is not None:
            weights = tf.reduce_sum(
                y_true * tf.constant(class_weights, dtype=tf.float32),
                axis=-1, keepdims=True,
            )
            focal = focal * weights
        return tf.reduce_mean(focal)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def asymmetric_focal_loss(class_gammas, class_weights=None):
    """Per-class focal gamma: hammer hard on the classes that matter most.
    class_gammas: list of gamma per class [Normal, Crackle, Wheeze, Both]
    """
    def loss_fn(y_true, y_pred):
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce       = -y_true * tf.math.log(y_pred)
        p_t      = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        gammas_t = tf.reduce_sum(
            y_true * tf.constant(class_gammas, dtype=tf.float32),
            axis=-1, keepdims=True,
        )
        focal    = tf.pow(1.0 - p_t, gammas_t) * ce
        if class_weights is not None:
            weights = tf.reduce_sum(
                y_true * tf.constant(class_weights, dtype=tf.float32),
                axis=-1, keepdims=True,
            )
            focal = focal * weights
        return tf.reduce_mean(focal)
    loss_fn.__name__ = 'asymmetric_focal_loss'
    return loss_fn


def icbhi_score(y_true, y_pred, n=NUM_SOUND):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        se.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        sp.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
    return (np.mean(se) + np.mean(sp)) / 2.0


class ICBHICallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X    = X
        self.y    = y
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        pred  = np.argmax(self.model.predict(self.X, verbose=0)[0], axis=1)
        score = icbhi_score(self.y, pred)
        logs['val_icbhi'] = score
        if score > self.best:
            self.best = score
            self.model.save(f'{CKPT_DIR}/synth_best.keras')
            print(f'  val_icbhi: {score:.4f} *** NEW BEST — saved ***')
        else:
            print(f'  val_icbhi: {score:.4f}')


# ── Load manifests ─────────────────────────────────────────────
manifest      = pd.read_csv('data/processed/manifest.csv')
synth_path    = 'data/processed/manifest_synthetic.csv'

if not os.path.exists(synth_path):
    print("ERROR: manifest_synthetic.csv not found.")
    print("Run:  python generate_synthetic.py  first.")
    raise SystemExit(1)

synth_df  = pd.read_csv(synth_path)
# Synthetic rows have no 'source' column in original manifest — add it
if 'source' not in manifest.columns:
    manifest['source'] = 'real'

manifest  = pd.concat([manifest, synth_df], ignore_index=True)
train_df  = manifest[manifest['split'] == 'train']
val_df    = manifest[(manifest['split'] == 'val') & (manifest['source'] == 'real')]

print(f"Train: {len(train_df)} ({(train_df.get('source', 'real') == 'synthetic').sum()} synthetic)")
print(f"Val:   {len(val_df)} (real only)")
print("Train distribution:")
print(train_df['sound_label'].value_counts().sort_index())


def load_split(df):
    X, ys, yd = [], [], []
    for _, row in df.iterrows():
        try:
            feat = np.load(row['features_path'])
        except Exception:
            continue
        feat = pad_or_truncate(feat)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X.append(feat)
        ys.append(row['sound_label'])
        yd.append(row['diagnosis_label'])
    return np.array(X, np.float32), np.array(ys, np.int32), np.array(yd, np.int32)


print('Loading val...')
X_val, y_sv, y_dv = load_split(val_df)
print(f'Val shape: {X_val.shape}')

print('Loading train...')
X_tr, y_str, y_dtr = load_split(train_df)
print(f'Train shape: {X_tr.shape}')

y_str_oh = tf.keras.utils.to_categorical(y_str, NUM_SOUND)
y_dtr_oh = tf.keras.utils.to_categorical(y_dtr, NUM_DIAGNOSIS)
y_sv_oh  = tf.keras.utils.to_categorical(y_sv,  NUM_SOUND)
y_dv_oh  = tf.keras.utils.to_categorical(y_dv,  NUM_DIAGNOSIS)


# ── Model (identical architecture to multitask_model.py) ───────
def build_model():
    inp = tf.keras.Input(shape=(N_MELS, TARGET_FRAMES, 1))
    x   = tf.keras.layers.Conv2D(32,  3, padding='same', activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.MaxPooling2D(2)(x)
    x   = tf.keras.layers.Conv2D(64,  3, padding='same', activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.MaxPooling2D(2)(x)
    x   = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.MaxPooling2D(2)(x)
    x   = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)

    shared = tf.keras.layers.Dense(256, activation='relu')(x)
    shared = tf.keras.layers.Dropout(0.55)(shared)   # slightly higher than baseline

    s         = tf.keras.layers.Dense(128, activation='relu')(shared)
    s         = tf.keras.layers.Dropout(0.35)(s)
    sound_out = tf.keras.layers.Dense(NUM_SOUND, activation='softmax', name='sound')(s)

    d        = tf.keras.layers.Dense(128, activation='relu')(shared)
    d        = tf.keras.layers.Dropout(0.35)(d)
    diag_out = tf.keras.layers.Dense(NUM_DIAGNOSIS, activation='softmax', name='diagnosis')(d)

    return tf.keras.Model(inp, [sound_out, diag_out])


model = build_model()


def lr_schedule(epoch):
    cos = np.cos(np.pi * epoch / EPOCHS)
    return float(LR_MIN + 0.5 * (LR - LR_MIN) * (1 + cos))


callbacks = [
    ICBHICallback(X_val, y_sv),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_icbhi', mode='max',
        patience=20, restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
]

cw      = compute_class_weight('balanced', classes=np.unique(y_str), y=y_str)
cw_list = [float(cw[i]) for i in range(NUM_SOUND)]
print(f'Class weights: {cw_list}')

# Asymmetric focal gamma: Normal=2 (easy), Crackle=4.5, Wheeze=3.5, Both=4.0
# Gamma=6 caused below-random instability in early epochs — dialled back
CLASS_GAMMAS = [2.0, 4.5, 3.5, 4.0]
print(f'Asymmetric gammas: {dict(zip(SOUND_NAMES, CLASS_GAMMAS))}')

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss={
        'sound':     asymmetric_focal_loss(CLASS_GAMMAS, cw_list),
        'diagnosis': focal_loss(2.0),
    },
    loss_weights={'sound': 1.0, 'diagnosis': DIAG_WEIGHT},
    metrics={'sound': 'accuracy', 'diagnosis': 'accuracy'},
)

print('\nStarting training with synthetic data...')
history = model.fit(
    X_tr, {'sound': y_str_oh, 'diagnosis': y_dtr_oh},
    validation_data=(X_val, {'sound': y_sv_oh, 'diagnosis': y_dv_oh}),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

best_icbhi = max(history.history['val_icbhi'])
print(f'\nBest val_icbhi: {best_icbhi:.4f}')

model_best = tf.keras.models.load_model(f'{CKPT_DIR}/synth_best.keras', compile=False)
val_preds  = model_best(X_val, training=False)
y_pred     = np.argmax(val_preds[0].numpy(), axis=1)

print('\nSound Classification Report:')
print(classification_report(y_sv, y_pred, target_names=SOUND_NAMES))

y_pred_d = np.argmax(val_preds[1].numpy(), axis=1)
print('Diagnosis Classification Report:')
print(classification_report(y_dv, y_pred_d, labels=list(range(NUM_DIAGNOSIS)), target_names=DIAG_NAMES))


# ── Plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['sound_accuracy'],     label='Train Acc')
axes[0].plot(history.history['val_sound_accuracy'], label='Val Acc')
axes[0].set_title('Sound Accuracy (w/ synthetic)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Total Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

axes[2].plot(history.history['val_icbhi'], label='Val ICBHI', color='green')
axes[2].axhline(
    y=best_icbhi, color='red', linestyle='--',
    label=f'Best: {best_icbhi:.4f}',
)
axes[2].set_title('ICBHI Score (w/ synthetic)')
axes[2].set_xlabel('Epoch')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/synth_training_curves.png', dpi=150)
print(f'Saved → {RESULTS_DIR}/synth_training_curves.png')

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_sound = confusion_matrix(y_sv, y_pred, labels=list(range(NUM_SOUND)))
sns.heatmap(cm_sound, annot=True, fmt='d', ax=axes[0],
            xticklabels=SOUND_NAMES, yticklabels=SOUND_NAMES, cmap='Blues')
axes[0].set_title('Sound — w/ Synthetic Data')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

cm_diag = confusion_matrix(y_dv, y_pred_d, labels=list(range(NUM_DIAGNOSIS)))
sns.heatmap(cm_diag, annot=True, fmt='d', ax=axes[1],
            xticklabels=DIAG_NAMES, yticklabels=DIAG_NAMES, cmap='Oranges')
axes[1].set_title('Diagnosis — w/ Synthetic Data')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/synth_confusion_matrices.png', dpi=150)
print(f'Saved → {RESULTS_DIR}/synth_confusion_matrices.png')
