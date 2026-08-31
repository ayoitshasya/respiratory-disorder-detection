"""
multitask_model.py
------------------
Multi-task CNN: shared backbone + sound-type head + diagnosis head.
Build this AFTER single-task models are working and baseline results look reasonable.

Head 1: Sound type  — 4 classes (normal, crackle, wheeze, both)
Head 2: Diagnosis   — 6 classes (Healthy, COPD, URTI, Bronchiectasis, LRTI, Asthma)

Samples without a diagnosis label only contribute to Head 1 loss.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from models import get_model
from augmentation import focal_loss, compute_class_weights
from preprocessing import load_feature
from training import icbhi_score, build_lr_schedule, set_seeds

SOUND_NAMES = ['normal', 'crackle', 'wheeze', 'both']
DIAG_NAMES  = ['Healthy', 'COPD', 'URTI', 'Bronchiectasis', 'LRTI', 'Asthma']

NUM_SOUND = 4
NUM_DIAG  = 6

LOSS_WEIGHT_SOUND = 0.6
LOSS_WEIGHT_DIAG  = 0.4


def build_multitask_model(backbone_name, input_shape):
    """
    Build multi-task model from a backbone architecture.

    The backbone's final Dense layers are stripped and replaced with
    two independent task heads sharing the same CNN feature extractor.

    Args:
        backbone_name: Name of backbone from models.py ('cnn_baseline', 'cnn_mobilenet')
        input_shape:   Input shape tuple (freq, time) — channel added internally

    Returns:
        Keras model with two outputs: ['sound_type', 'diagnosis']
    """
    inputs = layers.Input(shape=(*input_shape, 1))

    # Build backbone and extract shared features up to GlobalAveragePooling
    backbone = get_model(backbone_name, input_shape, num_classes=4)

    # Find the GAP layer and use its output as shared features
    gap_layer = None
    for layer in backbone.layers:
        if isinstance(layer, (layers.GlobalAveragePooling2D, layers.GlobalAveragePooling1D)):
            gap_layer = layer
            break

    if gap_layer is None:
        raise ValueError(f"Could not find GlobalAveragePooling in backbone '{backbone_name}'")

    # Rebuild backbone up to GAP
    shared_model = models.Model(
        inputs=backbone.input,
        outputs=gap_layer.output,
        name='shared_backbone'
    )
    shared_features = shared_model(inputs)

    # Sound Type Head (4 classes)
    s = layers.Dropout(0.4)(shared_features)
    s = layers.Dense(64, activation='relu')(s)
    s = layers.Dropout(0.3)(s)
    sound_output = layers.Dense(NUM_SOUND, activation='softmax', name='sound_type')(s)

    # Diagnosis Head (6 classes — slightly deeper since harder task)
    d = layers.Dense(128, activation='relu')(shared_features)
    d = layers.Dropout(0.4)(d)
    d = layers.Dense(64, activation='relu')(d)
    d = layers.Dropout(0.3)(d)
    diag_output = layers.Dense(NUM_DIAG, activation='softmax', name='diagnosis')(d)

    model = models.Model(
        inputs=inputs,
        outputs=[sound_output, diag_output],
        name=f'multitask_{backbone_name}'
    )
    return model


def train_multitask(model, manifest, config, checkpoints_dir, results_dir):
    """
    Train the multi-task model.
    Samples with diagnosis_label == -1 are excluded from diagnosis loss.

    Args:
        model:           Built multi-task Keras model
        manifest:        Full manifest DataFrame
        config:          Config dict
        checkpoints_dir: Directory to save checkpoints
        results_dir:     Directory to save outputs

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

    train_manifest = manifest[manifest['split'] == 'train']
    val_manifest   = manifest[manifest['split'] == 'val']

    # Compute class weights for sound labels
    train_sound_labels = train_manifest['sound_label'].values
    sound_class_weights = compute_class_weights(train_sound_labels, NUM_SOUND)
    print(f"Sound class weights: {sound_class_weights}")

    gamma    = config.get('focal_gamma', 2.0)
    loss_fns = {
        'sound_type': focal_loss(gamma=gamma),
        'diagnosis':  focal_loss(gamma=gamma),
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=1e-4),
        loss=loss_fns,
        loss_weights={'sound_type': LOSS_WEIGHT_SOUND, 'diagnosis': LOSS_WEIGHT_DIAG},
        metrics={'sound_type': 'accuracy', 'diagnosis': 'accuracy'}
    )

    # Build numpy arrays for training (simpler than tf.data for multi-output)
    def load_split(df):
        X, y_sound, y_diag = [], [], []
        for _, row in df.iterrows():
            feat = load_feature(row['features_path'])
            if feat.ndim == 2:
                feat = feat[..., np.newaxis]
            X.append(feat)
            y_sound.append(row['sound_label'])
            y_diag.append(row['diagnosis_label'])
        return (np.array(X, dtype=np.float32),
                np.array(y_sound, dtype=np.int32),
                np.array(y_diag,  dtype=np.int32))

    print("Loading training data...")
    X_train, y_s_train, y_d_train = load_split(train_manifest)
    print("Loading validation data...")
    X_val,   y_s_val,   y_d_val   = load_split(val_manifest)

    y_s_train_oh = tf.keras.utils.to_categorical(y_s_train, NUM_SOUND)
    y_s_val_oh   = tf.keras.utils.to_categorical(y_s_val,   NUM_SOUND)
    y_d_train_oh = tf.keras.utils.to_categorical(y_d_train, NUM_DIAG)
    y_d_val_oh   = tf.keras.utils.to_categorical(y_d_val,   NUM_DIAG)

    ckpt_path = os.path.join(checkpoints_dir, f'{exp_name}_best.keras')
    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_sound_type_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.get('early_stopping_patience', 15),
            restore_best_weights=True, verbose=1
        ),
        build_lr_schedule(lr, lr_min, epochs),
    ]

    print(f"\nTraining multitask model: {exp_name}")
    history = model.fit(
        X_train,
        {'sound_type': y_s_train_oh, 'diagnosis': y_d_train_oh},
        validation_data=(X_val, {'sound_type': y_s_val_oh, 'diagnosis': y_d_val_oh}),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )

    return model, history.history
