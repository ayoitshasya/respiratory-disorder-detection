import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEED=42; NUM_SOUND=4; NUM_DIAGNOSIS=7; BATCH_SIZE=32
EPOCHS=50; LR=3e-4; LR_MIN=1e-6; TARGET_FRAMES=63; N_MELS=128
DIAG_WEIGHT=0.3; CKPT_DIR='data/checkpoints'; RESULTS_DIR='data/results'
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
SOUND_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']


def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def focal_loss(gamma=2.0, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        n = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1 - label_smoothing) + label_smoothing / n
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_mean(tf.pow(1.0 - p_t, gamma) * ce)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def mixup(X, y_sound, y_diag, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    return (lam * X + (1 - lam) * X[idx],
            lam * y_sound + (1 - lam) * y_sound[idx],
            lam * y_diag  + (1 - lam) * y_diag[idx])


def icbhi_score(y_true, y_pred, n=NUM_SOUND):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        se.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        sp.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
    return (np.mean(se) + np.mean(sp)) / 2.0


# ── Load data ─────────────────────────────────────────────────
manifest = pd.read_csv('data/processed/manifest_aug.csv')
train_df = manifest[manifest['split'] == 'train']
val_df   = manifest[manifest['split'] == 'val']
print(f'Train: {len(train_df)} | Val: {len(val_df)}')


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
    return np.array(X, dtype=np.float32), np.array(ys, dtype=np.int32), np.array(yd, dtype=np.int32)


print('Loading val into RAM...')
X_val, y_sv, y_dv = load_split(val_df)
print(f'Val: {X_val.shape}')

print('Loading train into RAM...')
X_tr, y_str, y_dtr = load_split(train_df)
print(f'Train: {X_tr.shape}')

y_str_oh = tf.keras.utils.to_categorical(y_str, NUM_SOUND)
y_dtr_oh = tf.keras.utils.to_categorical(y_dtr, NUM_DIAGNOSIS)
y_sv_oh  = tf.keras.utils.to_categorical(y_sv,  NUM_SOUND)
y_dv_oh  = tf.keras.utils.to_categorical(y_dv,  NUM_DIAGNOSIS)


# ── ResNet model ──────────────────────────────────────────────
def resnet_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation('relu')(x)


def build_model():
    reg = tf.keras.regularizers.l2(1e-4)
    inp = tf.keras.Input(shape=(N_MELS, TARGET_FRAMES, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = resnet_block(x, 32)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = resnet_block(x, 64)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = resnet_block(x, 128)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = resnet_block(x, 128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    shared = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    shared = tf.keras.layers.Dropout(0.6)(shared)
    s = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg)(shared)
    s = tf.keras.layers.Dropout(0.4)(s)
    sound_out = tf.keras.layers.Dense(NUM_SOUND, activation='softmax', name='sound')(s)
    d = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg)(shared)
    d = tf.keras.layers.Dropout(0.4)(d)
    diag_out = tf.keras.layers.Dense(NUM_DIAGNOSIS, activation='softmax', name='diagnosis')(d)
    return tf.keras.Model(inp, [sound_out, diag_out])


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss={'sound': focal_loss(2.0, 0.1), 'diagnosis': focal_loss(2.0, 0.1)},
    loss_weights={'sound': 1.0, 'diagnosis': DIAG_WEIGHT},
    metrics={'sound': 'accuracy', 'diagnosis': 'accuracy'}
)


def lr_schedule(epoch):
    cos = np.cos(np.pi * epoch / EPOCHS)
    return float(LR_MIN + 0.5 * (LR - LR_MIN) * (1 + cos))


# ── Apply mixup to training data ──────────────────────────────
print('Applying mixup to training data...')
X_mix, ys_mix, yd_mix = mixup(X_tr, y_str_oh, y_dtr_oh)


best_icbhi   = 0.0
patience_ctr = 0
PATIENCE     = 12
icbhi_history = []

print('Starting training...')
for epoch in range(EPOCHS):
    lr = lr_schedule(epoch)
    model.optimizer.learning_rate.assign(lr)

    # Apply mixup only after warmup epochs
    if epoch >= 5:
        X_mix, ys_mix, yd_mix = mixup(X_tr, y_str_oh, y_dtr_oh)
    else:
        X_mix, ys_mix, yd_mix = X_tr, y_str_oh, y_dtr_oh

    result = model.fit(
        X_mix, {'sound': ys_mix, 'diagnosis': yd_mix},
        validation_data=(X_val, {'sound': y_sv_oh, 'diagnosis': y_dv_oh}),
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=1
    )

    # Compute ICBHI after each epoch
    val_preds = model(X_val, training=False)
    y_pred    = np.argmax(val_preds[0].numpy(), axis=1)
    icbhi     = icbhi_score(y_sv, y_pred)
    icbhi_history.append(icbhi)

    val_acc = result.history['val_sound_accuracy'][0]
    print(f'  -> val_icbhi: {icbhi:.4f} | val_sound_acc: {val_acc:.4f}')

    if icbhi > best_icbhi:
        best_icbhi = icbhi
        model.save(f'{CKPT_DIR}/resnet_multitask_best.keras')
        print(f'  -> NEW BEST ICBHI: {best_icbhi:.4f} — saved')
        patience_ctr = 0
    else:
        patience_ctr += 1
        print(f'  -> No improvement. Patience: {patience_ctr}/{PATIENCE}')
        if patience_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

print(f'\nBest val_icbhi: {best_icbhi:.4f}')
model_best = tf.keras.models.load_model(f'{CKPT_DIR}/resnet_multitask_best.keras', compile=False)
val_preds  = model_best(X_val, training=False)
y_pred     = np.argmax(val_preds[0].numpy(), axis=1)
print(classification_report(y_sv, y_pred, target_names=SOUND_NAMES))
