# ============================================================
# SPEC AUGMENTATION EXPERIMENT
# Respiratory Disorder Detection - ICBHI 2017
#
# IMPORTANT:
# This file is a separate experiment.
# The original train_multitask.py is NOT modified.
#
# Only change from baseline:
#   SpecAugment applied to TRAINING spectrograms
#
# Time masking  : maximum 10 frames
# Frequency mask: maximum 10 bins
#
# Validation and Test data are NOT augmented.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. CONFIGURATION
# ============================================================

# Spectrogram dimensions
N_MELS = 128
TARGET_FRAMES = 63

# Training configuration
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3

# Diagnosis loss weight
DIAG_LOSS_WEIGHT = 0.1

# Focal loss parameters
SOUND_GAMMA = 4.0
DIAG_GAMMA = 2.0

# SpecAugment limits
MAX_TIME_MASK = 10
MAX_FREQ_MASK = 10

# Dataset
MANIFEST_PATH = "/content/processed/processed/manifest.csv"

# Separate output folders for this experiment
CKPT_DIR = "data/checkpoints/specaugment"
RESULTS_DIR = "data/results/specaugment"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 4. SPEC AUGMENTATION
# ============================================================

def spec_augment(
    feat,
    max_time_mask=10,
    max_freq_mask=10
):
    """
    Apply SpecAugment to one spectrogram.

    Input shape:
        [128, 63, 1]

    Time masking:
        Randomly masks 0 to 10 time frames.

    Frequency masking:
        Randomly masks 0 to 10 frequency bins.

    This function is applied ONLY to training samples.
    """

    # --------------------------------------------------------
    # TIME MASKING
    # --------------------------------------------------------

    time_width = tf.random.uniform(
        [],
        minval=0,
        maxval=max_time_mask + 1,
        dtype=tf.int32
    )

    time_start = tf.random.uniform(
        [],
        minval=0,
        maxval=TARGET_FRAMES - time_width + 1,
        dtype=tf.int32
    )

    time_mask = tf.concat(
        [
            tf.ones([time_start]),
            tf.zeros([time_width]),
            tf.ones(
                [
                    TARGET_FRAMES
                    - time_start
                    - time_width
                ]
            )
        ],
        axis=0
    )

    # Shape:
    # [1, 63, 1]
    time_mask = time_mask[
        tf.newaxis,
        :,
        tf.newaxis
    ]

    feat = feat * time_mask


    # --------------------------------------------------------
    # FREQUENCY MASKING
    # --------------------------------------------------------

    freq_width = tf.random.uniform(
        [],
        minval=0,
        maxval=max_freq_mask + 1,
        dtype=tf.int32
    )

    freq_start = tf.random.uniform(
        [],
        minval=0,
        maxval=N_MELS - freq_width + 1,
        dtype=tf.int32
    )

    freq_mask = tf.concat(
        [
            tf.ones([freq_start]),
            tf.zeros([freq_width]),
            tf.ones(
                [
                    N_MELS
                    - freq_start
                    - freq_width
                ]
            )
        ],
        axis=0
    )

    # Shape:
    # [128, 1, 1]
    freq_mask = freq_mask[
        :,
        tf.newaxis,
        tf.newaxis
    ]

    feat = feat * freq_mask

    return feat


# ============================================================
# 5. FOCAL LOSS
# ============================================================

def focal_loss(gamma=2.0, class_weights=None):
    """
    Multi-class focal loss.
    """

    def loss_fn(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        ce = -y_true * tf.math.log(y_pred)

        focal_factor = tf.pow(
            1.0 - y_pred,
            gamma
        )

        loss = focal_factor * ce

        if class_weights is not None:
            weights = tf.reduce_sum(
                y_true * class_weights,
                axis=-1,
                keepdims=True
            )

            loss = loss * weights

        return tf.reduce_sum(
            loss,
            axis=-1
        )

    return loss_fn


# ============================================================
# 6. ICBHI SCORE
# ============================================================

def icbhi_score(y_true, y_pred):
    """
    Original project ICBHI validation metric.

    NOTE:
    This is kept unchanged for training/early stopping
    so that the baseline training procedure is preserved.

    The FINAL TEST result will be calculated separately
    using the standard ICBHI Average Score:
        Sensitivity = abnormal recall
        Specificity = normal recall
        Score = (Sensitivity + Specificity) / 2
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    sensitivities = []
    specificities = []

    for i in range(4):

        tp = cm[i, i]

        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp

        tn = np.sum(cm) - tp - fn - fp

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificities)

    return (sensitivity + specificity) / 2


# ============================================================
# 7. CUSTOM CALLBACK
# ============================================================

class ICBHICallback(Callback):

    def __init__(self, val_x, val_y):
        super().__init__()

        self.val_x = val_x
        self.val_y = val_y

        self.best_score = -np.inf


    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        # Predict validation sound class
        predictions = self.model.predict(
            self.val_x,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        sound_predictions = predictions["sound"]

        y_pred = np.argmax(
            sound_predictions,
            axis=1
        )

        y_true = np.argmax(
            self.val_y["sound"],
            axis=1
        )

        score = icbhi_score(
            y_true,
            y_pred
        )

        logs["val_icbhi"] = score

        print(
            f" - val_icbhi: {score:.4f}",
            end=""
        )

        # Save best model
        if score > self.best_score:

            self.best_score = score

            self.model.save(
                f"{CKPT_DIR}/specaugment_best.keras"
            )

            print(
                " *** NEW BEST — saved ***",
                end=""
            )


# ============================================================
# 8. LOAD MANIFEST
# ============================================================

print("\n")
print("=" * 60)
print("           LOADING DATASET")
print("=" * 60)

manifest = pd.read_csv(
    MANIFEST_PATH
)


# ============================================================
# 9. FIX MANIFEST FEATURE PATHS
# ============================================================

# The manifest contains Windows-style paths such as:
#
# data\processed\train\xxxx.npy
#
# Colab uses:
#
# /content/processed/processed/train/xxxx.npy
#
# Convert paths in memory only.

manifest["features_path"] = manifest[
    "features_path"
].apply(
    lambda p:
        "/content/processed/processed/"
        + p.replace("\\", "/")
        .split("/data/processed/")[-1]
)


# ============================================================
# 10. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset loaded successfully!")

print("\nManifest shape:")
print(manifest.shape)

print("\nColumns:")
print(manifest.columns.tolist())


# ============================================================
# 11. DETERMINE TRAIN / VALIDATION / TEST DATA
# ============================================================

train_df = manifest[
    manifest["split"] == "train"
].copy()

val_df = manifest[
    manifest["split"] == "val"
].copy()

test_df = manifest[
    manifest["split"] == "test"
].copy()


print("\nTrain:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))


# ============================================================
# 12. LOAD FEATURES
# ============================================================

def load_features(df):
    X = []
    sound_labels = []
    diagnosis_labels = []

    for _, row in df.iterrows():

        feature_path = row["features_path"]

        feature = np.load(feature_path).astype(np.float32)

        # Ensure feature has shape [N_MELS, TIME]
        if feature.ndim == 3:
            feature = np.squeeze(feature)

        # Pad or truncate time dimension to TARGET_FRAMES
        if feature.shape[1] < TARGET_FRAMES:
            pad_width = TARGET_FRAMES - feature.shape[1]
            feature = np.pad(
                feature,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )
        else:
            feature = feature[:, :TARGET_FRAMES]

        # Add channel dimension -> [128, 63, 1]
        feature = feature[:, :, np.newaxis]

        X.append(feature)

        sound_labels.append(int(row["sound_label"]))
        diagnosis_labels.append(int(row["diagnosis_label"]))

    X = np.asarray(X, dtype=np.float32)

    sound_labels = np.asarray(sound_labels, dtype=np.int32)
    diagnosis_labels = np.asarray(diagnosis_labels, dtype=np.int32)

    return X, sound_labels, diagnosis_labels


# ============================================================
# 13. LOAD TRAINING DATA
# ============================================================

print("\nLoading train...")

(
    X_tr,
    y_str,
    y_dtr
) = load_features(
    train_df
)

print(
    "Train:",
    X_tr.shape
)


# ============================================================
# 14. LOAD VALIDATION DATA
# ============================================================

print("\nLoading val...")

(
    X_val,
    y_sv,
    y_dv
) = load_features(
    val_df
)

print(
    "Val:",
    X_val.shape
)


# ============================================================
# 15. LOAD TEST DATA
# ============================================================

print("\nTest data is loaded only for final evaluation.")

(
    X_test,
    y_st,
    y_dt
) = load_features(
    test_df
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# 16. ONE-HOT ENCODING
# ============================================================

NUM_SOUND = 4
NUM_DIAG = 7

y_str_oh = tf.keras.utils.to_categorical(
    y_str,
    NUM_SOUND
)

y_dtr_oh = tf.keras.utils.to_categorical(
    y_dtr,
    NUM_DIAG
)

y_sv_oh = tf.keras.utils.to_categorical(
    y_sv,
    NUM_SOUND
)

y_dv_oh = tf.keras.utils.to_categorical(
    y_dv,
    NUM_DIAG
)


# ============================================================
# 17. CLASS WEIGHTS
# ============================================================

# Same class weights as baseline.
# DO NOT CHANGE for SpecAugment experiment.

class_counts = np.bincount(
    y_str,
    minlength=NUM_SOUND
)

total_samples = len(y_str)

class_weights = (
    total_samples
    /
    (
        NUM_SOUND
        * class_counts
    )
)

print("\nClass weights:")
print(
    class_weights.tolist()
)

class_weights_tf = tf.constant(
    class_weights,
    dtype=tf.float32
)


# ============================================================
# 18. BUILD BASELINE CNN ARCHITECTURE
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            N_MELS,
            TARGET_FRAMES,
            1
        )
    )


    # --------------------------------------------------------
    # CONV BLOCK 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # CONV BLOCK 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # CONV BLOCK 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # CONV BLOCK 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # SHARED DENSE LAYER
    # --------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.5
    )(x)


    # --------------------------------------------------------
    # SOUND CLASSIFICATION HEAD
    # --------------------------------------------------------

    sound = layers.Dense(
        128,
        activation="relu"
    )(x)

    sound = layers.Dropout(
        0.5
    )(sound)

    sound = layers.Dense(
        NUM_SOUND,
        activation="softmax",
        name="sound"
    )(sound)


    # --------------------------------------------------------
    # DIAGNOSIS CLASSIFICATION HEAD
    # --------------------------------------------------------

    diagnosis = layers.Dense(
        128,
        activation="relu"
    )(x)

    diagnosis = layers.Dropout(
        0.5
    )(diagnosis)

    diagnosis = layers.Dense(
        NUM_DIAG,
        activation="softmax",
        name="diagnosis"
    )(diagnosis)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = models.Model(
        inputs=inputs,
        outputs={
            "sound": sound,
            "diagnosis": diagnosis
        }
    )

    return model


# ============================================================
# 19. BUILD MODEL
# ============================================================

model = build_model()

print("\n")
print("=" * 60)
print("              MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# 20. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=LR
    ),

    loss={
        "sound": focal_loss(
            gamma=SOUND_GAMMA,
            class_weights=class_weights_tf
        ),

        "diagnosis": focal_loss(
            gamma=DIAG_GAMMA
        )
    },

    loss_weights={
        "sound": 1.0,
        "diagnosis": DIAG_LOSS_WEIGHT
    },

    metrics={
        "sound": "accuracy",
        "diagnosis": "accuracy"
    }
)


# ============================================================
# 21. CREATE TRAINING DATASET
# ============================================================

print("\n")
print("=" * 60)
print("       CREATING TRAINING DATASET")
print("=" * 60)

train_ds = tf.data.Dataset.from_tensor_slices(

    (
        X_tr,

        {
            "sound": y_str_oh,
            "diagnosis": y_dtr_oh
        }
    )
)


# Shuffle training samples
train_ds = train_ds.shuffle(
    len(X_tr),
    seed=SEED,
    reshuffle_each_iteration=True
)


# ------------------------------------------------------------
# APPLY SPECAUGMENT
# ------------------------------------------------------------
#
# IMPORTANT:
# SpecAugment is applied only to training data.
#
# Because this is inside Dataset.map(), the random masks
# are generated during dataset iteration.
#
# Validation and test datasets below do NOT use this map.
# ------------------------------------------------------------

train_ds = train_ds.map(

    lambda x, y:
        (
            spec_augment(
                x,
                max_time_mask=MAX_TIME_MASK,
                max_freq_mask=MAX_FREQ_MASK
            ),
            y
        ),

    num_parallel_calls=tf.data.AUTOTUNE
)


train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 22. CREATE VALIDATION DATASET
# ============================================================
#
# NO SPEC AUGMENTATION HERE.
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(

    (
        X_val,

        {
            "sound": y_sv_oh,
            "diagnosis": y_dv_oh
        }
    )
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 23. CALLBACKS
# ============================================================

icbhi_callback = ICBHICallback(

    val_x=X_val,

    val_y={
        "sound": y_sv_oh,
        "diagnosis": y_dv_oh
    }
)


early_stopping = EarlyStopping(

    monitor="val_icbhi",

    mode="max",

    patience=20,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# 24. TRAIN MODEL
# ============================================================

print("\n")
print("=" * 60)
print("       STARTING TRAINING WITH SPECAUGMENT")
print("=" * 60)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        icbhi_callback,
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 25. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    f"{CKPT_DIR}/"
    "specaugment_final.keras"
)

model.save(
    final_model_path
)

print("\nFinal model saved to:")
print(final_model_path)


# ============================================================
# 26. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = (
    f"{RESULTS_DIR}/"
    "specaugment_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved to:")
print(history_path)


# ============================================================
# 27. LOAD BEST MODEL
# ============================================================

best_model_path = (
    f"{CKPT_DIR}/"
    "specaugment_best.keras"
)

if os.path.exists(
    best_model_path
):

    print("\nLoading best SpecAugment model...")

    best_model = tf.keras.models.load_model(
        best_model_path,
        compile=False
    )

else:

    print(
        "\nBest checkpoint not found."
    )

    best_model = model


# ============================================================
# 28. FINAL TEST PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("          FINAL TEST EVALUATION")
print("=" * 60)

test_predictions = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

sound_probabilities = (
    test_predictions["sound"]
)

y_test_pred = np.argmax(
    sound_probabilities,
    axis=1
)

y_test_true = y_st


# ============================================================
# 29. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test_true,

    y_test_pred,

    labels=[
        0,
        1,
        2,
        3
    ]
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 30. CLASSIFICATION REPORT
# ============================================================

class_names = [
    "Normal",
    "Crackle",
    "Wheeze",
    "Both"
]

print("\nClassification Report:")

print(
    classification_report(
        y_test_true,
        y_test_pred,
        labels=[
            0,
            1,
            2,
            3
        ],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 31. STANDARD ICBHI AVERAGE SCORE
# ============================================================
#
# ICBHI definition:
#
# Abnormal classes:
#   Crackle
#   Wheeze
#   Both
#
# Normal class:
#   Normal
#
# Sensitivity:
#   Correct abnormal predictions /
#   Total abnormal samples
#
# Specificity:
#   Correct Normal predictions /
#   Total Normal samples
#
# ICBHI Score:
#   (Sensitivity + Specificity) / 2
# ============================================================

normal_class = 0

abnormal_classes = [
    1,
    2,
    3
]


# ------------------------------------------------------------
# True abnormal samples
# ------------------------------------------------------------

true_abnormal = np.isin(
    y_test_true,
    abnormal_classes
)

pred_abnormal = np.isin(
    y_test_pred,
    abnormal_classes
)


# ------------------------------------------------------------
# Sensitivity
# ------------------------------------------------------------

abnormal_correct = np.sum(
    true_abnormal & pred_abnormal
)

total_abnormal = np.sum(
    true_abnormal
)

sensitivity = (
    abnormal_correct
    /
    total_abnormal
)


# ------------------------------------------------------------
# Specificity
# ------------------------------------------------------------

true_normal = (
    y_test_true == normal_class
)

pred_normal = (
    y_test_pred == normal_class
)

normal_correct = np.sum(
    true_normal & pred_normal
)

total_normal = np.sum(
    true_normal
)

specificity = (
    normal_correct
    /
    total_normal
)


# ------------------------------------------------------------
# STANDARD ICBHI SCORE
# ------------------------------------------------------------

standard_icbhi_score = (
    sensitivity
    +
    specificity
) / 2


# ============================================================
# 32. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("       STANDARD ICBHI TEST RESULTS")
print("=" * 60)

print(
    f"Abnormal Sensitivity : "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Normal Specificity   : "
    f"{specificity * 100:.2f}%"
)

print(
    f"ICBHI Average Score  : "
    f"{standard_icbhi_score * 100:.2f}%"
)


# ============================================================
# 33. SAVE FINAL RESULTS
# ============================================================

results = pd.DataFrame({

    "metric": [
        "Abnormal Sensitivity",
        "Normal Specificity",
        "Standard ICBHI Average Score"
    ],

    "value": [
        sensitivity,
        specificity,
        standard_icbhi_score
    ]
})

results_path = (
    f"{RESULTS_DIR}/"
    "specaugment_test_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nTest results saved to:")
print(results_path)


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 60)
print("       SPECAUGMENT EXPERIMENT COMPLETE")
print("=" * 60)
