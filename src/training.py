"""
training.py
-----------
Training loop with cosine annealing LR, ICBHI score tracking,
focal loss, class weights, and SpecAugment via tf.data.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

from augmentation import compute_class_weights, focal_loss, apply_spec_augment_to_batch
from preprocessing import load_feature, pad_or_truncate, TARGET_FRAMES

SEED = 42


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)


# ─────────────────────────────────────────────
# ICBHI score metric
# ─────────────────────────────────────────────

def icbhi_score(y_true, y_pred, num_classes=4):
    """
    ICBHI Challenge score = (Sensitivity + Specificity) / 2
    Computed as macro-average across classes.

    Args:
        y_true:      1D array of true labels
        y_pred:      1D array of predicted labels
        num_classes: Number of classes

    Returns:
        float — ICBHI score in [0, 1]
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    se_list, sp_list = [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        se_list.append(se)
        sp_list.append(sp)

    return (np.mean(se_list) + np.mean(sp_list)) / 2.0


class ICBHIScoreCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that computes ICBHI score on val set after each epoch.
    Tracked as 'val_icbhi' in history.
    """

    def __init__(self, val_data, num_classes=4):
        super().__init__()
        self.X_val       = val_data[0]
        self.y_val       = val_data[1]
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        score  = icbhi_score(self.y_val, y_pred, self.num_classes)
        logs['val_icbhi'] = score
        print(f"  val_icbhi: {score:.4f}", end='')


# ─────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────

def build_dataset(manifest_split, batch_size, is_training=False,
                  spec_augment=False, feature_col='features_path',
                  label_col='sound_label'):
    """
    Build a tf.data.Dataset from a manifest DataFrame.

    Args:
        manifest_split: DataFrame filtered to one split
        batch_size:     Batch size
        is_training:    If True, shuffle the dataset
        spec_augment:   If True, apply SpecAugment (training only)
        feature_col:    Column name for feature paths
        label_col:      Column name for labels

    Returns:
        tf.data.Dataset yielding (features, one_hot_labels)
    """
    paths  = manifest_split[feature_col].values
    labels = manifest_split[label_col].values
    num_classes = len(np.unique(labels))

    def load_npy(path, label):
        feat = np.load(path.numpy().decode())
        feat = pad_or_truncate(feat)
        # Add channel dim if 2D: (freq, time) → (freq, time, 1)
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        return feat.astype(np.float32), label

    def tf_load(path, label):
        feat, lbl = tf.py_function(load_npy, [path, label], [tf.float32, tf.int32])
        feat.set_shape([128, TARGET_FRAMES, 1])
        return feat, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if is_training and spec_augment:
        ds = ds.map(apply_spec_augment_to_batch, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def build_lr_schedule(lr_max, lr_min, total_epochs):
    """Cosine annealing learning rate schedule."""
    def schedule(epoch):
        cos = np.cos(np.pi * epoch / total_epochs)
        return float(lr_min + 0.5 * (lr_max - lr_min) * (1 + cos))
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


def train(model, manifest, config, checkpoints_dir, results_dir):
    """
    Full training pipeline.

    Args:
        model:            Uncompiled Keras model
        manifest:         Full manifest DataFrame (all splits)
        config:           Experiment config dict (from experiment_config.json)
        checkpoints_dir:  Directory to save best model weights
        results_dir:      Directory to save training history CSV

    Returns:
        (trained model, history dict)
    """
    set_seeds()
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    exp_name    = config['experiment_name']
    batch_size  = config['batch_size']
    epochs      = config['epochs']
    lr          = config['learning_rate']
    lr_min      = config.get('lr_min', 1e-6)
    num_classes = config['num_classes']
    label_col   = config.get('label_col', 'sound_label')
    use_spec_aug = config.get('augmentation', {}).get('spec_augment', True)

    train_manifest = manifest[manifest['split'] == 'train']
    val_manifest   = manifest[manifest['split'] == 'val']

    print(f"\nTrain: {len(train_manifest)} | Val: {len(val_manifest)}")

    # Class weights / focal loss
    train_labels = train_manifest[label_col].values
    loss_fn = config.get('loss', 'categorical_crossentropy')
    # Class imbalance handled by focal loss — class_weight not used with one-hot labels
    class_weight_dict = None
    if config.get('class_weights') == 'auto':
        cw = compute_class_weights(train_labels, num_classes)
        print(f"Class weights (for reference, handled via focal loss): {cw}")

    if loss_fn == 'focal_loss':
        gamma = config.get('focal_gamma', 2.0)
        loss_fn = focal_loss(gamma=gamma)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=config.get('weight_decay', 1e-4)
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Load all val features into memory for ICBHI callback
    X_val, y_val = [], []
    for _, row in val_manifest.iterrows():
        feat = load_feature(row['features_path'])
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X_val.append(feat)
        y_val.append(row[label_col])
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    # tf.data datasets
    train_ds = build_dataset(train_manifest, batch_size, is_training=True,
                             spec_augment=use_spec_aug, label_col=label_col)
    val_ds   = build_dataset(val_manifest,   batch_size, is_training=False,
                             label_col=label_col)

    ckpt_path = os.path.join(checkpoints_dir, f'{exp_name}_best.keras')
    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_icbhi', mode='max',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_icbhi', mode='max',
            patience=config.get('early_stopping_patience', 15),
            restore_best_weights=True, verbose=1
        ),
        build_lr_schedule(lr, lr_min, epochs),
        ICBHIScoreCallback(val_data=(X_val, y_val), num_classes=num_classes),
    ]

    print(f"\nTraining: {exp_name}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save history
    hist_df   = pd.DataFrame(history.history)
    hist_path = os.path.join(results_dir, f'{exp_name}_history.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f"\nHistory saved to: {hist_path}")

    return model, history.history
