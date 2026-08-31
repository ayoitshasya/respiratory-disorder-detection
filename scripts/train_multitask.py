"""
Final multitask training — baseline CNN architecture that works,
trained on original 2,891 samples with both sound + diagnosis heads.
Using original data (not augmented) to avoid overfitting collapse.
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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEED=42; NUM_SOUND=4; NUM_DIAGNOSIS=7; BATCH_SIZE=32
EPOCHS=80; LR=1e-3; LR_MIN=1e-6; TARGET_FRAMES=63; N_MELS=128
DIAG_WEIGHT=0.1; CKPT_DIR='data/checkpoints'; RESULTS_DIR='data/results'
DIAG_MISSING = -1  # sentinel: no diagnosis label available (e.g. HF_Lung rows) — masked out of diagnosis loss
MANIFEST_PATH = 'data/processed/manifest_merged.csv'
# RUN_SUFFIX keeps this run's checkpoint/plots distinct from the original
# ICBHI-only baseline (multitask_final_best.keras etc.) so both are on disk
# for comparison — never overwrite the baseline.
RUN_SUFFIX = 'sourceweighted'
CKPT_NAME = f'multitask_{RUN_SUFFIX}_best.keras'
# Batch size for post-training inference only (val/test eval) — separate from the
# training BATCH_SIZE above. model_best(X) (direct __call__) runs the WHOLE array
# as one forward pass with no batching, which OOMs on GPU for large eval sets;
# model.predict(X, batch_size=...) batches internally and avoids that.
PREDICT_BATCH_SIZE = 128
BASELINE_ICBHI_SCORE = 0.6226            # sem-6 ICBHI-only baseline, val split
BASELINE_NORMAL_TO_CRACKLE = (171, 506)  # ditto — true-Normal count / Normal->Crackle count
# Unweighted HF_Lung+ICBHI combined run (multitask_combined_best.keras) — regressed
# badly on ICBHI-only test (51.96%, below the 62.26% sem-6 baseline) despite a strong
# combined score, because HF_Lung is ~90% of training volume by row count and the
# model specialized on its acoustic characteristics at ICBHI's expense. This run adds
# source-aware sample weighting to correct that. Numbers are on the TEST split.
UNWEIGHTED_COMBINED_SCORES = {'full': 0.7164, 'icbhi': 0.5196, 'hf_lung': 0.7608}
# ICBHI's per-sample sound-loss weight is multiplied by this on top of the existing
# per-class weight, so it gets proportionally more training influence despite being
# the minority by volume (measured hf_lung/icbhi train-row ratio: 8.62x — this starts
# at the round number that roughly equalizes total per-epoch gradient contribution;
# tune here, not inline, if the val/test tradeoff needs adjusting.
ICBHI_SOURCE_WEIGHT = 9.0
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
SOUND_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']
DIAG_NAMES  = ['Healthy', 'COPD', 'URTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis', 'Other']


def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0,0), (0, t-c)])
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
            weights = tf.reduce_sum(y_true * tf.constant(class_weights, dtype=tf.float32), axis=-1, keepdims=True)
            focal = focal * weights
        # Per-sample loss (batch,), NOT a pre-reduced scalar: Keras' sample_weight
        # (used below for source-aware weighting) multiplies BEFORE the final batch
        # reduction, so it needs a per-sample tensor to multiply against. Verified
        # numerically equivalent to the old tf.reduce_mean(focal) (both axes at once)
        # whenever sample_weight is uniformly 1: mean_b[mean_c[x]] == mean_over_both[x].
        return tf.reduce_mean(focal, axis=-1)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def icbhi_score(y_true, y_pred, n=NUM_SOUND):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=cm.sum()-tp-fn-fp
        se.append(tp/(tp+fn) if tp+fn>0 else 0.0)
        sp.append(tn/(tn+fp) if tn+fp>0 else 0.0)
    return (np.mean(se)+np.mean(sp))/2.0


class ICBHICallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__(); self.X=X; self.y=y
        self.best = 0.0
    def on_epoch_end(self, epoch, logs=None):
        pred  = np.argmax(self.model.predict(self.X, verbose=0)[0], axis=1)
        score = icbhi_score(self.y, pred)
        logs['val_icbhi'] = score
        if score > self.best:
            self.best = score
            self.model.save(f'{CKPT_DIR}/{CKPT_NAME}')
            print(f'  val_icbhi: {score:.4f} *** NEW BEST — saved ***')
        else:
            print(f'  val_icbhi: {score:.4f}')


# ── Load data ──────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST_PATH)
train_df  = manifest[manifest['split'] == 'train']
val_df    = manifest[manifest['split'] == 'val']
print(f'Train: {len(train_df)} | Val: {len(val_df)}')
print('Train distribution:')
print(train_df['sound_label'].value_counts().sort_index())


def load_split(df):
    X, ys, yd, src = [], [], [], []
    for _, row in df.iterrows():
        feat = np.load(row['features_path'])
        feat = pad_or_truncate(feat)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X.append(feat); ys.append(row['sound_label']); yd.append(row['diagnosis_label'])
        src.append(row['source_dataset'])
    return (np.array(X, np.float32), np.array(ys, np.int32), np.array(yd, np.int32), np.array(src))


print('Loading val...')
X_val, y_sv, y_dv, src_val = load_split(val_df)
print(f'Val: {X_val.shape}')

print('Loading train...')
X_tr, y_str, y_dtr, src_tr = load_split(train_df)
print(f'Train: {X_tr.shape}')

y_str_oh = tf.keras.utils.to_categorical(y_str, NUM_SOUND)
y_sv_oh  = tf.keras.utils.to_categorical(y_sv,  NUM_SOUND)

# Diagnosis masking: rows with no real diagnosis label (diagnosis_label ==
# DIAG_MISSING, e.g. HF_Lung rows) get an all-zero one-hot target instead of
# a real class. With y_true all-zero, focal_loss's `ce = -y_true*log(y_pred)`
# and `p_t = sum(y_true*y_pred)` are both 0 for that row regardless of
# y_pred, so `focal` is exactly 0 there — zero diagnosis loss, zero gradient,
# with NO change to focal_loss() itself. The sound head is untouched by any
# of this, so it keeps full weight on every row, HF_Lung included.
# When diag_mask is all-True (no DIAG_MISSING present, e.g. ICBHI-only data),
# np.where's fallback branch and the [~mask]=0 assignment both select zero
# rows — this block is then a byte-identical no-op vs. the original
# to_categorical(y_dtr, NUM_DIAGNOSIS) call.
diag_mask_train = (y_dtr != DIAG_MISSING)
diag_mask_val   = (y_dv  != DIAG_MISSING)

y_dtr_oh = tf.keras.utils.to_categorical(np.where(diag_mask_train, y_dtr, 0), NUM_DIAGNOSIS)
y_dv_oh  = tf.keras.utils.to_categorical(np.where(diag_mask_val,   y_dv,  0), NUM_DIAGNOSIS)
y_dtr_oh[~diag_mask_train] = 0.0
y_dv_oh[~diag_mask_val]    = 0.0

print(f'Diagnosis-labeled rows: train {diag_mask_train.sum()}/{len(diag_mask_train)}, '
      f'val {diag_mask_val.sum()}/{len(diag_mask_val)}')


# ── Model — same proven architecture as baseline CNN ───────────
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
    sound_out = tf.keras.layers.Dense(NUM_SOUND, activation='softmax', name='sound')(s)

    d = tf.keras.layers.Dense(128, activation='relu')(shared)
    d = tf.keras.layers.Dropout(0.3)(d)
    diag_out = tf.keras.layers.Dense(NUM_DIAGNOSIS, activation='softmax', name='diagnosis')(d)

    return tf.keras.Model(inp, [sound_out, diag_out])


model = build_model()


def lr_schedule(epoch):
    cos = np.cos(np.pi * epoch / EPOCHS)
    return float(LR_MIN + 0.5*(LR - LR_MIN)*(1 + cos))


callbacks = [
    ICBHICallback(X_val, y_sv),   # must be first so val_icbhi is in logs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_icbhi', mode='max',
        patience=20, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
]

# Per-class weights (unchanged: still computed from the merged train set's sound
# label distribution). Previously baked directly into focal_loss(4.0, cw_list) for
# the sound head; now applied via sample_weight instead (see below) so it can be
# COMBINED with the new source weighting rather than replaced by it.
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_str), y=y_str)
cw_list = [float(cw[i]) for i in range(NUM_SOUND)]
print(f'Class weights: {cw_list}')

# Source-aware sample weighting: sound_sample_weight[i] = class_weight[label_i] *
# source_weight[source_i]. ICBHI rows get ICBHI_SOURCE_WEIGHT (default 9.0, ~matches
# the measured 8.62x hf_lung/icbhi train-row ratio) so their total per-epoch gradient
# influence is roughly on par with HF_Lung's, despite being ~10% of train rows by
# volume. HF_Lung rows keep source_weight=1.0 (baseline, unchanged influence).
SOURCE_WEIGHT_MAP = {'icbhi': ICBHI_SOURCE_WEIGHT, 'hf_lung': 1.0}
class_weight_per_sample  = np.array([cw_list[label] for label in y_str], dtype=np.float32)
source_weight_per_sample = np.array([SOURCE_WEIGHT_MAP[s] for s in src_tr], dtype=np.float32)
sound_sample_weight = class_weight_per_sample * source_weight_per_sample
print(f'Source weight map: {SOURCE_WEIGHT_MAP}')
print(f'Sound sample_weight: min={sound_sample_weight.min():.3f}, '
      f'max={sound_sample_weight.max():.3f}, mean={sound_sample_weight.mean():.3f}')

# Keras requires a sample_weight entry for EVERY output present in y (a partial dict
# raises "You should provide one sample_weight array per output" — verified). The
# diagnosis head keeps its existing masking mechanism (zeroed one-hot rows for
# DIAG_MISSING, above) entirely unchanged, so its sample_weight is all-ones — a
# mathematical no-op, not a second weighting scheme.
diag_sample_weight = np.ones(len(y_str), dtype=np.float32)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss={'sound': focal_loss(4.0), 'diagnosis': focal_loss(2.0)},
    loss_weights={'sound': 1.0, 'diagnosis': DIAG_WEIGHT},
    metrics={'sound': 'accuracy', 'diagnosis': 'accuracy'}
)

print('\nStarting training...')
history = model.fit(
    X_tr, {'sound': y_str_oh, 'diagnosis': y_dtr_oh},
    sample_weight={'sound': sound_sample_weight, 'diagnosis': diag_sample_weight},
    validation_data=(X_val, {'sound': y_sv_oh, 'diagnosis': y_dv_oh}),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f'\nBest val_icbhi: {max(history.history["val_icbhi"]):.4f}')
model_best = tf.keras.models.load_model(f'{CKPT_DIR}/{CKPT_NAME}', compile=False)
val_preds  = model_best.predict(X_val, batch_size=PREDICT_BATCH_SIZE, verbose=0)
y_pred     = np.argmax(val_preds[0], axis=1)
print('\nSound Classification Report:')
print(classification_report(y_sv, y_pred, target_names=SOUND_NAMES))
y_pred_d   = np.argmax(val_preds[1], axis=1)
print('Diagnosis Classification Report:')
# Exclude rows with no real diagnosis label (e.g. HF_Lung val rows) — their
# diagnosis_label is the DIAG_MISSING sentinel, not a class to score against.
# No-op when diag_mask_val is all-True (ICBHI-only data).
print(classification_report(y_dv[diag_mask_val], y_pred_d[diag_mask_val],
                             labels=list(range(NUM_DIAGNOSIS)), target_names=DIAG_NAMES))


# ── Held-out TEST evaluation: full combined / icbhi-only / hf_lung-only ──
def load_split_with_source(df):
    X, ys, src = [], [], []
    for _, row in df.iterrows():
        feat = np.load(row['features_path'])
        feat = pad_or_truncate(feat)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X.append(feat); ys.append(row['sound_label']); src.append(row['source_dataset'])
    return np.array(X, np.float32), np.array(ys, np.int32), np.array(src)


print('\nLoading test...')
test_df = manifest[manifest['split'] == 'test']
X_test, y_s_test, src_test = load_split_with_source(test_df)
n_icbhi_test, n_hf_test = (src_test == 'icbhi').sum(), (src_test == 'hf_lung').sum()
print(f'Test: {X_test.shape}  ({n_icbhi_test} icbhi, {n_hf_test} hf_lung)')

test_preds  = model_best.predict(X_test, batch_size=PREDICT_BATCH_SIZE, verbose=0)
y_pred_test = np.argmax(test_preds[0], axis=1)


def report_subset(name, mask):
    yt, yp = y_s_test[mask], y_pred_test[mask]
    score = icbhi_score(yt, yp)
    cm = confusion_matrix(yt, yp, labels=list(range(NUM_SOUND)))
    print(f'\n=== TEST subset: {name} (n={mask.sum()}) ===')
    print(f'  ICBHI score: {score:.4f} ({score*100:.2f}%)')
    print(f'  Normal->Crackle confusions: {cm[0, 1]} / {cm[0].sum()} true Normal')
    print('  Confusion matrix (rows=true, cols=pred), order Normal/Crackle/Wheeze/Both:')
    print(cm)
    print(classification_report(yt, yp, target_names=SOUND_NAMES, zero_division=0))
    return score, cm


full_score, full_cm   = report_subset('FULL (icbhi+hf_lung)', np.ones(len(y_s_test), dtype=bool))
icbhi_score_, icbhi_cm = report_subset("source_dataset=='icbhi'", src_test == 'icbhi')
hf_score, hf_cm        = report_subset("source_dataset=='hf_lung'", src_test == 'hf_lung')

print('\n=== Three-way comparison: sem-6 baseline vs. unweighted combined vs. this (source-weighted) run ===')
print('  NOTE: sem-6 baseline (62.26%, 171/506) was measured on the ORIGINAL ICBHI-only')
print('  VAL split (799 rows). The unweighted-combined and this-run numbers are both on')
print('  the (larger, merged) TEST split. Flagging so this is read as three comparable')
print('  TEST-split numbers plus one differently-sourced reference point, not four')
print('  identical-split numbers.')
print(f"  {'subset':22s} {'sem-6 baseline':>15s} {'unweighted combined':>21s} {'this run (source-wt)':>22s}")
print(f"  {'FULL (icbhi+hf_lung)':22s} {'—':>15s} "
      f"{UNWEIGHTED_COMBINED_SCORES['full']*100:20.2f}% {full_score*100:21.2f}%")
print(f"  {'icbhi-only':22s} {BASELINE_ICBHI_SCORE*100:14.2f}% "
      f"{UNWEIGHTED_COMBINED_SCORES['icbhi']*100:20.2f}% {icbhi_score_*100:21.2f}%")
print(f"  {'hf_lung-only':22s} {'—':>15s} "
      f"{UNWEIGHTED_COMBINED_SCORES['hf_lung']*100:20.2f}% {hf_score*100:21.2f}%")
print()
print(f"  icbhi-only delta vs sem-6 baseline:        {(icbhi_score_-BASELINE_ICBHI_SCORE)*100:+.2f} pts")
print(f"  icbhi-only delta vs unweighted combined:   {(icbhi_score_-UNWEIGHTED_COMBINED_SCORES['icbhi'])*100:+.2f} pts")
print(f"  full delta vs unweighted combined:         {(full_score-UNWEIGHTED_COMBINED_SCORES['full'])*100:+.2f} pts")
print(f"  hf_lung-only delta vs unweighted combined: {(hf_score-UNWEIGHTED_COMBINED_SCORES['hf_lung'])*100:+.2f} pts")
print()
print(f'  Baseline Normal->Crackle (val, icbhi-only, sem-6):  {BASELINE_NORMAL_TO_CRACKLE[0]}/{BASELINE_NORMAL_TO_CRACKLE[1]}')
print(f'  This run — FULL test Normal->Crackle:               {full_cm[0,1]}/{full_cm[0].sum()}')
print(f'  This run — icbhi-only test Normal->Crackle:         {icbhi_cm[0,1]}/{icbhi_cm[0].sum()}')
print(f'  This run — hf_lung-only test Normal->Crackle:       {hf_cm[0,1]}/{hf_cm[0].sum()}')

# Confusion matrix plot for the three test subsets (sound head only).
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for ax, (name, cm) in zip(axes, [('Full test', full_cm), ('ICBHI-only test', icbhi_cm),
                                  ('HF_Lung-only test', hf_cm)]):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=SOUND_NAMES,
                yticklabels=SOUND_NAMES, cmap='Blues')
    ax.set_title(name); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/multitask_{RUN_SUFFIX}_test_confusion_matrices.png', dpi=150)
print(f'\nSaved test confusion matrices -> {RESULTS_DIR}/multitask_{RUN_SUFFIX}_test_confusion_matrices.png')


# ── Plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curves — sound accuracy
ax = axes[0]
ax.plot(history.history['sound_accuracy'], label='Train Acc')
ax.plot(history.history['val_sound_accuracy'], label='Val Acc')
ax.set_title('Sound Accuracy'); ax.set_xlabel('Epoch'); ax.legend()

# Training curves — loss
ax = axes[1]
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('Total Loss'); ax.set_xlabel('Epoch'); ax.legend()

# ICBHI score per epoch
ax = axes[2]
ax.plot(history.history['val_icbhi'], label='Val ICBHI', color='green')
ax.axhline(y=max(history.history['val_icbhi']), color='red', linestyle='--',
           label=f'Best: {max(history.history["val_icbhi"]):.4f}')
ax.set_title('ICBHI Score'); ax.set_xlabel('Epoch'); ax.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/multitask_{RUN_SUFFIX}_training_curves.png', dpi=150)
print(f'Saved training curves -> {RESULTS_DIR}/multitask_{RUN_SUFFIX}_training_curves.png')

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_sound = confusion_matrix(y_sv, y_pred, labels=list(range(NUM_SOUND)))
sns.heatmap(cm_sound, annot=True, fmt='d', ax=axes[0],
            xticklabels=SOUND_NAMES, yticklabels=SOUND_NAMES, cmap='Blues')
axes[0].set_title('Sound Confusion Matrix')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# Same DIAG_MISSING filtering as the printed report above — excludes rows
# with no real diagnosis label. No-op when diag_mask_val is all-True.
cm_diag = confusion_matrix(y_dv[diag_mask_val], y_pred_d[diag_mask_val], labels=list(range(NUM_DIAGNOSIS)))
sns.heatmap(cm_diag, annot=True, fmt='d', ax=axes[1],
            xticklabels=DIAG_NAMES, yticklabels=DIAG_NAMES, cmap='Oranges')
axes[1].set_title('Diagnosis Confusion Matrix')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/multitask_{RUN_SUFFIX}_confusion_matrices.png', dpi=150)
print(f'Saved confusion matrices -> {RESULTS_DIR}/multitask_{RUN_SUFFIX}_confusion_matrices.png')
