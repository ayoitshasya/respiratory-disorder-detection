"""
train_teacher.py
----------------
Train the large teacher model on ICBHI data.
Save soft probability outputs for distillation.
Run this BEFORE train_distillation.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from teacher_model import build_teacher_model

# ── Config ──────────────────────────────────────────────
SEED         = 42
NUM_SOUND    = 4
NUM_DIAG     = 7
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 1e-3
LR_MIN       = 1e-6
TARGET_FRAMES = 126
N_MELS       = 128
DIAG_WEIGHT  = 0.1
CKPT_DIR     = 'data/checkpoints'
RESULTS_DIR  = 'data/results'
SOFT_LABELS_DIR = 'data/soft_labels'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SOFT_LABELS_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

SOUND_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']
DIAG_NAMES  = ['Healthy', 'COPD', 'URTI',
               'Bronchiectasis', 'Pneumonia', 'Bronchiolitis', 'Other']


# ── Helper functions ──────────────────────────────────────
def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def focal_loss(gamma=2.0, class_weights=None):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce  = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if class_weights is not None:
            weights = tf.reduce_sum(
                y_true * tf.constant(class_weights, dtype=tf.float32),
                axis=-1, keepdims=True
            )
            focal = focal * weights
        return tf.reduce_mean(focal)
    loss_fn.__name__ = 'focal_loss'
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
        self.X = X
        self.y = y
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        pred  = np.argmax(
            self.model.predict(self.X, verbose=0)[0], axis=1
        )
        score = icbhi_score(self.y, pred)
        logs['val_icbhi'] = score
        if score > self.best:
            self.best = score
            self.model.save(f'{CKPT_DIR}/teacher_best.keras')
            print(f'  val_icbhi: {score:.4f} *** NEW BEST — saved ***')
        else:
            print(f'  val_icbhi: {score:.4f}')


def load_split(df):
    X, ys, yd = [], [], []

    for _, row in df.iterrows():

        # Get only the filename from the old absolute path
        filename = os.path.basename(row['features_path'])

        # Determine which split the file belongs to
        split = row['split']

        # Build the path using THIS project
        feat_path = os.path.join(
            'data',
            'processed',
            split,
            filename
        )

        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Feature file not found: {feat_path}"
            )

        feat = np.load(feat_path)

        feat = pad_or_truncate(feat)

        if feat.ndim == 2:
            feat = feat[..., np.newaxis]

        X.append(feat)
        ys.append(row['sound_label'])
        yd.append(row['diagnosis_label'])

    return (
        np.array(X, np.float32),
        np.array(ys, np.int32),
        np.array(yd, np.int32)
    )


def lr_schedule(epoch):
    cos = np.cos(np.pi * epoch / EPOCHS)
    return float(LR_MIN + 0.5 * (LR - LR_MIN) * (1 + cos))


# ── Load data ──────────────────────────────────────────────
print("Loading manifest...")
manifest  = pd.read_csv('data/processed/manifest.csv')
train_df  = manifest[manifest['split'] == 'train']
val_df    = manifest[manifest['split'] == 'val']
print(f'Train: {len(train_df)} | Val: {len(val_df)}')

print("Loading val data...")
X_val, y_sv, y_dv = load_split(val_df)

print("Loading train data...")
X_tr, y_str, y_dtr = load_split(train_df)

# One-hot encode
y_str_oh = tf.keras.utils.to_categorical(y_str, NUM_SOUND)
y_dtr_oh = tf.keras.utils.to_categorical(y_dtr, NUM_DIAG)
y_sv_oh  = tf.keras.utils.to_categorical(y_sv,  NUM_SOUND)
y_dv_oh  = tf.keras.utils.to_categorical(y_dv,  NUM_DIAG)

# Class weights
cw      = compute_class_weight('balanced',
                               classes=np.unique(y_str), y=y_str)
cw_list = [float(cw[i]) for i in range(NUM_SOUND)]
print(f'Class weights: {cw_list}')

# ── Build and train teacher ────────────────────────────────
model = build_teacher_model()
model.summary()
print(f"\nTeacher parameters: {model.count_params():,}")

callbacks = [
    ICBHICallback(X_val, y_sv),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_icbhi', mode='max',
        patience=20, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss={
        'sound':     focal_loss(4.0, cw_list),
        'diagnosis': focal_loss(2.0)
    },
    loss_weights={'sound': 1.0, 'diagnosis': DIAG_WEIGHT},
    metrics={'sound': 'accuracy', 'diagnosis': 'accuracy'}
)

print('\nTraining teacher model...')
history = model.fit(
    X_tr,
    {'sound': y_str_oh, 'diagnosis': y_dtr_oh},
    validation_data=(X_val, {'sound': y_sv_oh, 'diagnosis': y_dv_oh}),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

best_icbhi = max(history.history['val_icbhi'])
print(f'\nTeacher best val_icbhi: {best_icbhi:.4f}')

# ── Generate soft labels from trained teacher ──────────────
print('\nGenerating soft labels for distillation...')
teacher = tf.keras.models.load_model(
    f'{CKPT_DIR}/teacher_best.keras', compile=False
)

# Soft labels on training set — these supervise the student
train_sound_soft, train_diag_soft = teacher.predict(X_tr, verbose=1)
val_sound_soft,   val_diag_soft   = teacher.predict(X_val, verbose=1)

np.save(f'{SOFT_LABELS_DIR}/train_sound_soft.npy', train_sound_soft)
np.save(f'{SOFT_LABELS_DIR}/train_diag_soft.npy',  train_diag_soft)
np.save(f'{SOFT_LABELS_DIR}/val_sound_soft.npy',   val_sound_soft)
np.save(f'{SOFT_LABELS_DIR}/val_diag_soft.npy',    val_diag_soft)

print(f'Soft labels saved to {SOFT_LABELS_DIR}/')
print(f'Train sound soft shape : {train_sound_soft.shape}')
print(f'Train diag soft shape  : {train_diag_soft.shape}')

# Check teacher quality
y_pred_teacher = np.argmax(train_sound_soft, axis=1)
print('\nTeacher train classification report:')
print(classification_report(y_str, y_pred_teacher,
                            target_names=SOUND_NAMES))

print('\nTeacher training complete.')
print(f'Best ICBHI : {best_icbhi:.4f}')
print(f'Soft labels: saved and ready for distillation')
print(f'Next step  : run train_distillation.py')