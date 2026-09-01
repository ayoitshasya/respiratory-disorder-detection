"""
train_distillation.py
---------------------
Train student model using Hinton knowledge distillation.
Student learns from teacher soft probability outputs
rather than hard one-hot labels.
Run AFTER train_teacher.py has generated soft labels.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Config ──────────────────────────────────────────────
SEED          = 42
NUM_SOUND     = 4
NUM_DIAG      = 7
BATCH_SIZE    = 32
EPOCHS        = 80
LR            = 1e-3
LR_MIN        = 1e-6
TARGET_FRAMES = 63
N_MELS        = 128
DIAG_WEIGHT   = 0.1
TEMPERATURE   = 4.0   # Hinton temperature — softens probability distributions
ALPHA         = 0.7   # weight for soft label loss (0=hard only, 1=soft only)
CKPT_DIR      = 'data/checkpoints'
RESULTS_DIR   = 'data/results'
SOFT_DIR      = 'data/soft_labels'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def distillation_loss(y_true_hard, y_soft_teacher,
                      y_pred_student, temperature, alpha,
                      class_weights=None):
    """
    Hinton distillation loss.
    Total loss = alpha * soft_loss + (1-alpha) * hard_loss

    soft_loss: KL divergence between teacher and student
               soft probabilities at high temperature
    hard_loss: focal loss against hard ground truth labels
    """
    # Soft loss — teacher supervises student at temperature T
    teacher_soft = tf.nn.softmax(
        tf.math.log(tf.clip_by_value(y_soft_teacher, 1e-7, 1.0))
        / temperature
    )
    student_soft = tf.nn.softmax(
        tf.math.log(tf.clip_by_value(y_pred_student, 1e-7, 1.0))
        / temperature
    )
    soft_loss = tf.keras.losses.KLDivergence()(
        teacher_soft, student_soft
    ) * (temperature ** 2)

    # Hard loss — standard focal loss against ground truth
    y_pred_c = tf.clip_by_value(y_pred_student, 1e-7, 1.0)
    ce    = -y_true_hard * tf.math.log(y_pred_c)
    p_t   = tf.reduce_sum(y_true_hard * y_pred_c,
                          axis=-1, keepdims=True)
    focal = tf.pow(1.0 - p_t, 4.0) * ce
    if class_weights is not None:
        w     = tf.reduce_sum(
            y_true_hard * tf.constant(class_weights, dtype=tf.float32),
            axis=-1, keepdims=True
        )
        focal = focal * w
    hard_loss = tf.reduce_mean(focal)

    return alpha * soft_loss + (1 - alpha) * hard_loss


class ICBHICallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, name='student'):
        super().__init__()
        self.X    = X
        self.y    = y
        self.name = name
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        pred  = np.argmax(
            self.model.predict(self.X, verbose=0)[0], axis=1
        )
        score = icbhi_score(self.y, pred)
        logs['val_icbhi'] = score
        if score > self.best:
            self.best = score
            self.model.save(
                f'{CKPT_DIR}/{self.name}_distilled_best.keras'
            )
            print(f'  val_icbhi: {score:.4f} *** NEW BEST — saved ***')
        else:
            print(f'  val_icbhi: {score:.4f}')


def build_student_model(input_shape=(128, 126, 1)):
    """
    Same deployable student model as our Sem 6 multitask CNN.
    280K parameters, ~150KB int8 — fits ESP32.
    """
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding='same',
                               use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    shared = tf.keras.layers.Dense(256, activation='relu')(x)
    shared = tf.keras.layers.Dropout(0.5)(shared)

    s = tf.keras.layers.Dense(128, activation='relu')(shared)
    s = tf.keras.layers.Dropout(0.3)(s)
    sound_out = tf.keras.layers.Dense(
        NUM_SOUND, activation='softmax', name='sound'
    )(s)

    d = tf.keras.layers.Dense(128, activation='relu')(shared)
    d = tf.keras.layers.Dropout(0.3)(d)
    diag_out = tf.keras.layers.Dense(
        NUM_DIAG, activation='softmax', name='diagnosis'
    )(d)

    return tf.keras.Model(inp, [sound_out, diag_out],
                          name='student_distilled')


def load_split(df):
    X, ys, yd = [], [], []
    for _, row in df.iterrows():
        feat = np.load(row['features_path'])
        feat = pad_or_truncate(feat)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X.append(feat)
        ys.append(row['sound_label'])
        yd.append(row['diagnosis_label'])
    return (np.array(X, np.float32),
            np.array(ys, np.int32),
            np.array(yd, np.int32))


def lr_schedule(epoch):
    cos = np.cos(np.pi * epoch / EPOCHS)
    return float(LR_MIN + 0.5 * (LR - LR_MIN) * (1 + cos))


# ── Load data ──────────────────────────────────────────────
print("Loading manifest...")
manifest = pd.read_csv('data/processed/manifest.csv')
train_df = manifest[manifest['split'] == 'train']
val_df   = manifest[manifest['split'] == 'val']

print("Loading features...")
X_tr,  y_str, y_dtr = load_split(train_df)
X_val, y_sv,  y_dv  = load_split(val_df)

# Hard one-hot labels
y_str_oh = tf.keras.utils.to_categorical(y_str, NUM_SOUND)
y_sv_oh  = tf.keras.utils.to_categorical(y_sv,  NUM_SOUND)
y_dtr_oh = tf.keras.utils.to_categorical(y_dtr, NUM_DIAG)
y_dv_oh  = tf.keras.utils.to_categorical(y_dv,  NUM_DIAG)

# Load teacher soft labels
print("Loading teacher soft labels...")
train_sound_soft = np.load(f'{SOFT_DIR}/train_sound_soft.npy')
train_diag_soft  = np.load(f'{SOFT_DIR}/train_diag_soft.npy')
val_sound_soft   = np.load(f'{SOFT_DIR}/val_sound_soft.npy')
val_diag_soft    = np.load(f'{SOFT_DIR}/val_diag_soft.npy')

print(f'Teacher soft labels loaded — shape: {train_sound_soft.shape}')

# Class weights
cw      = compute_class_weight('balanced',
                               classes=np.unique(y_str), y=y_str)
cw_list = [float(cw[i]) for i in range(NUM_SOUND)]
print(f'Class weights: {cw_list}')

# ── Build student ──────────────────────────────────────────
student = build_student_model()
student.summary()
print(f"\nStudent parameters : {student.count_params():,}")
print(f"Student float32 KB : {student.count_params()*4/1024:.1f}")
print(f"Student int8 est KB: {student.count_params()/1024:.1f}")

optimizer = tf.keras.optimizers.Adam(LR)


# ── Custom training loop with distillation loss ────────────
@tf.function
def train_step(X_batch, y_hard_batch, y_soft_batch,
               y_diag_hard, y_diag_soft):
    with tf.GradientTape() as tape:
        sound_pred, diag_pred = student(X_batch, training=True)

        # Sound distillation loss
        sound_loss = distillation_loss(
            y_hard_batch, y_soft_batch,
            sound_pred, TEMPERATURE, ALPHA, cw_list
        )

        # Diagnosis focal loss (no distillation for diagnosis)
        diag_pred_c = tf.clip_by_value(diag_pred, 1e-7, 1.0)
        diag_loss   = tf.reduce_mean(
            -y_diag_hard * tf.math.log(diag_pred_c)
        )

        total_loss = sound_loss + DIAG_WEIGHT * diag_loss

    grads = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, student.trainable_variables)
    )
    return total_loss, sound_loss


print('\nStarting distillation training...')
best_icbhi = 0.0

for epoch in range(EPOCHS):
    # Update learning rate
    new_lr = lr_schedule(epoch)
    optimizer.learning_rate.assign(new_lr)

    # Shuffle training data
    idx = np.random.permutation(len(X_tr))
    X_s = X_tr[idx]
    y_h = y_str_oh[idx]
    y_sf = train_sound_soft[idx]
    y_dh = y_dtr_oh[idx]
    y_dsf = train_diag_soft[idx]

    # Mini-batch training
    total_losses = []
    for start in range(0, len(X_s), BATCH_SIZE):
        end   = start + BATCH_SIZE
        loss, sloss = train_step(
            X_s[start:end], y_h[start:end],
            y_sf[start:end], y_dh[start:end],
            y_dsf[start:end]
        )
        total_losses.append(float(loss))

    mean_loss = np.mean(total_losses)

    # ICBHI score on validation
    val_preds = student.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_preds[0], axis=1)
    score = icbhi_score(y_sv, val_pred_classes)

    print(f'Epoch {epoch+1:3d}/{EPOCHS} — '
          f'loss: {mean_loss:.4f} — '
          f'val_icbhi: {score:.4f}', end='')

    if score > best_icbhi:
        best_icbhi = score
        student.save(f'{CKPT_DIR}/student_distilled_best.keras')
        print(' *** NEW BEST — saved ***')
    else:
        print()

    # Early stopping
    if epoch > 20 and score < best_icbhi - 0.05:
        print('Early stopping triggered')
        break

print(f'\nDistillation complete.')
print(f'Best student ICBHI : {best_icbhi:.4f}')

# ── Final evaluation ───────────────────────────────────────
print('\nFinal evaluation on validation set...')
best_student = tf.keras.models.load_model(
    f'{CKPT_DIR}/student_distilled_best.keras', compile=False
)
preds    = best_student.predict(X_val, verbose=0)
y_pred   = np.argmax(preds[0], axis=1)
y_pred_d = np.argmax(preds[1], axis=1)

print('\nSound Classification Report:')
print(classification_report(y_sv, y_pred, target_names=SOUND_NAMES))
print('\nDiagnosis Classification Report:')
print(classification_report(y_dv, y_pred_d,
                            target_names=DIAG_NAMES))

final_score = icbhi_score(y_sv, y_pred)
print(f'\nFinal distilled student ICBHI: {final_score:.4f}')
print(f'Baseline (Sem 6)             : 0.6226')
print(f'Improvement                  : '
      f'{(final_score - 0.6226)*100:+.2f}%')

# TFLite size check
print(f'\nModel size check:')
print(f'Parameters  : {best_student.count_params():,}')
print(f'Float32 size: {best_student.count_params()*4/1024:.1f} KB')
print(f'Int8 est.   : {best_student.count_params()/1024:.1f} KB')
print(f'ESP32 limit : 520 KB')
print(f'Deployable? : '
      f'{"YES" if best_student.count_params()/1024 < 400 else "NO"}')