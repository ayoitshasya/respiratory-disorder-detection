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
# RUN_SUFFIX keeps this run's checkpoint/plots distinct from prior runs
# (multitask_final_best.keras, multitask_combined_best.keras,
# multitask_sourceweighted_best.keras) so all stay on disk for comparison —
# never overwrite an earlier baseline.
RUN_SUFFIX = 'batchstratified'
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
# model specialized on its acoustic characteristics at ICBHI's expense. Numbers on TEST split.
UNWEIGHTED_COMBINED_SCORES = {'full': 0.7164, 'icbhi': 0.5196, 'hf_lung': 0.7608}
# Source-aware sample weighting (multitask_sourceweighted_best.keras, ICBHI_SOURCE_WEIGHT=9.0
# multiplied onto class_weight) FAILED and was abandoned: ICBHI-only barely moved (52.05%,
# +0.09pt vs unweighted) while hf_lung-only and full both got WORSE (-9.38pt, -6.27pt).
# Root cause (confirmed from the confusion matrix): class_weight x source_weight compounds
# in the extreme cell -- ICBHI's "Both" class (class_weight=16.55) x source_weight=9.0 =~149x,
# an outlier weight that taught the model to over-predict "Both" on ICBHI-like inputs rather
# than learn ICBHI's general acoustics better. Multiplicative weighting was the wrong
# mechanism; replaced by stratified batch sampling below (no per-sample weight changes,
# no extreme-weight interaction possible). Numbers on TEST split.
SOURCE_WEIGHTED_SCORES = {'full': 0.6537, 'icbhi': 0.5205, 'hf_lung': 0.6670}
# Stratified batch sampling (this run): every training batch draws a fixed number of ICBHI
# rows and a fixed number of HF_Lung rows (round(BATCH_SIZE*ICBHI_BATCH_FRACTION) and the
# remainder) from two separately-shuffled, indefinitely-repeating pools, so ICBHI gets
# guaranteed, consistent representation in every gradient update -- without touching
# per-sample loss magnitude at all. Existing per-class weighting (focal loss + class_weight)
# is UNCHANGED and still applies within each pool; this only rebalances SOURCE composition
# per batch, not class composition. Verified in isolation before wiring in (see the batch
# composition test): every batch is deterministically icbhi_batch_size/BATCH_SIZE ICBHI,
# not just close on average. Tune here, not inline, if the eval tradeoff needs adjusting.
ICBHI_BATCH_FRACTION = 0.3
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
        # (used below for per-class weighting) multiplies BEFORE the final batch
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
# label distribution, still applied via sample_weight the same way the Keras-crash
# fix already required). NO source multiplier this time -- sound_sample_weight is
# purely class_weight_per_sample, exactly like the original working ICBHI-only run.
# Source balance is now handled entirely by batch composition (below), not by
# distorting any individual sample's loss magnitude.
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_str), y=y_str)
cw_list = [float(cw[i]) for i in range(NUM_SOUND)]
print(f'Class weights: {cw_list}')

sound_sample_weight = np.array([cw_list[label] for label in y_str], dtype=np.float32)
print(f'Sound sample_weight (class-weight only, no source multiplier): '
      f'min={sound_sample_weight.min():.3f}, max={sound_sample_weight.max():.3f}, '
      f'mean={sound_sample_weight.mean():.3f}')

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

# ── Stratified batch sampling ──────────────────────────────────
# Every training batch draws a FIXED count from each source pool (not just close
# on average -- verified deterministic in isolation, see the batch-composition
# check run before this was wired in). Two separately-shuffled, indefinitely-
# repeating tf.data pools (icbhi, hf_lung), zipped and gathered into one batch.
# ICBHI rows get reused (repeat()) several times per epoch since the pool is
# smaller than its per-epoch draw count -- that's the intended oversampling
# effect, not a bug. Within each pool, existing per-class sample_weight still
# applies unchanged -- this only rebalances SOURCE composition, not class balance.
icbhi_mask = (src_tr == 'icbhi')
hf_mask    = (src_tr == 'hf_lung')
icbhi_batch_size = round(BATCH_SIZE * ICBHI_BATCH_FRACTION)
hf_batch_size    = BATCH_SIZE - icbhi_batch_size
print(f'Stratified batches: {icbhi_batch_size} icbhi + {hf_batch_size} hf_lung per '
      f'{BATCH_SIZE}-sample batch (realized fraction {icbhi_batch_size/BATCH_SIZE:.4f}, '
      f'target {ICBHI_BATCH_FRACTION}) -- pools: {icbhi_mask.sum()} icbhi / {hf_mask.sum()} hf_lung rows')


def make_pool_dataset(ids, batch_size, seed):
    ds = tf.data.Dataset.from_tensor_slices(ids)
    ds = ds.shuffle(buffer_size=len(ids), seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat()
    return ds.batch(batch_size, drop_remainder=True)


def concat_pool_ids(icbhi_ids, hf_ids):
    return tf.concat([icbhi_ids, hf_ids], axis=0)


icbhi_pool_ds = make_pool_dataset(np.where(icbhi_mask)[0], icbhi_batch_size, seed=SEED)
hf_pool_ds    = make_pool_dataset(np.where(hf_mask)[0],    hf_batch_size,    seed=SEED + 1)
batch_id_ds = tf.data.Dataset.zip((icbhi_pool_ds, hf_pool_ds)).map(
    concat_pool_ids, num_parallel_calls=tf.data.AUTOTUNE)

# Constant tensors captured once, gathered per batch by index -- avoids re-copying
# the full train arrays on every step.
_X_tr_t   = tf.constant(X_tr)
_y_s_t    = tf.constant(y_str_oh)
_y_d_t    = tf.constant(y_dtr_oh)
_sw_s_t   = tf.constant(sound_sample_weight)
_sw_d_t   = tf.constant(diag_sample_weight)


def gather_batch(idx):
    x  = tf.gather(_X_tr_t, idx)
    ys = tf.gather(_y_s_t, idx)
    yd = tf.gather(_y_d_t, idx)
    sws = tf.gather(_sw_s_t, idx)
    swd = tf.gather(_sw_d_t, idx)
    # Tuple structure (not dict) -- required to avoid the same Keras KeyError: 0
    # dict-resolution bug the sample_weight fix above already worked around;
    # applies identically whether data comes from raw arrays or a Dataset.
    return x, (ys, yd), (sws, swd)


train_ds = (batch_id_ds
            .map(gather_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

# Pools repeat() indefinitely, so steps_per_epoch must be set explicitly --
# matches the original (non-stratified) epoch length so wall-clock/epoch stays
# comparable across runs: one epoch = one pass over the full merged train set.
steps_per_epoch = -(-len(train_df) // BATCH_SIZE)
print(f'steps_per_epoch = {steps_per_epoch} (matches ceil({len(train_df)}/{BATCH_SIZE}))')

print('\nStarting training...')
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, {'sound': y_sv_oh, 'diagnosis': y_dv_oh}),
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

print('\n=== Four-way comparison: sem-6 baseline / unweighted combined / source-weighted (failed) / batch-stratified (this run) ===')
print('  NOTE: sem-6 baseline (62.26%, 171/506) was measured on the ORIGINAL ICBHI-only')
print('  VAL split (799 rows). The other three columns are all on the (larger, merged)')
print('  TEST split. Flagging so this is read as three same-split comparable numbers')
print('  plus one differently-sourced reference point, not four identical-split numbers.')
col = "  {:22s} {:>15s} {:>21s} {:>19s} {:>22s}"
print(col.format('subset', 'sem-6 baseline', 'unweighted comb.', 'source-wt (failed)', 'batch-strat (this run)'))
print(col.format('FULL (icbhi+hf_lung)', '—',
                  f"{UNWEIGHTED_COMBINED_SCORES['full']*100:.2f}%",
                  f"{SOURCE_WEIGHTED_SCORES['full']*100:.2f}%",
                  f"{full_score*100:.2f}%"))
print(col.format('icbhi-only', f"{BASELINE_ICBHI_SCORE*100:.2f}%",
                  f"{UNWEIGHTED_COMBINED_SCORES['icbhi']*100:.2f}%",
                  f"{SOURCE_WEIGHTED_SCORES['icbhi']*100:.2f}%",
                  f"{icbhi_score_*100:.2f}%"))
print(col.format('hf_lung-only', '—',
                  f"{UNWEIGHTED_COMBINED_SCORES['hf_lung']*100:.2f}%",
                  f"{SOURCE_WEIGHTED_SCORES['hf_lung']*100:.2f}%",
                  f"{hf_score*100:.2f}%"))
print()
print(f"  icbhi-only delta vs sem-6 baseline:            {(icbhi_score_-BASELINE_ICBHI_SCORE)*100:+.2f} pts")
print(f"  icbhi-only delta vs unweighted combined:       {(icbhi_score_-UNWEIGHTED_COMBINED_SCORES['icbhi'])*100:+.2f} pts")
print(f"  icbhi-only delta vs source-weighted (failed):  {(icbhi_score_-SOURCE_WEIGHTED_SCORES['icbhi'])*100:+.2f} pts")
print(f"  full delta vs unweighted combined:             {(full_score-UNWEIGHTED_COMBINED_SCORES['full'])*100:+.2f} pts")
print(f"  full delta vs source-weighted (failed):        {(full_score-SOURCE_WEIGHTED_SCORES['full'])*100:+.2f} pts")
print(f"  hf_lung-only delta vs unweighted combined:     {(hf_score-UNWEIGHTED_COMBINED_SCORES['hf_lung'])*100:+.2f} pts")
print(f"  hf_lung-only delta vs source-weighted (failed):{(hf_score-SOURCE_WEIGHTED_SCORES['hf_lung'])*100:+.2f} pts")
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
