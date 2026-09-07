
# SpecAugment Experiment

## Overview

This experiment evaluates the effect of **SpecAugment** as a data augmentation technique for respiratory sound classification using the ICBHI 2017 respiratory sound dataset.

The experiment is implemented as a separate training script so that the original baseline implementation remains unchanged.

---

## Objective

To investigate whether applying SpecAugment during training can improve respiratory sound classification performance while keeping the following unchanged:

- Model architecture
- Loss functions
- Class weights
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Dataset split
- Validation preprocessing
- Test preprocessing

Only the training-time preprocessing is modified by introducing SpecAugment.

---

## Dataset

The experiment uses the **ICBHI 2017 Respiratory Sound Database**.

### Sound Classes

The four sound classes are:

1. Normal
2. Crackle
3. Wheeze
4. Both

The dataset used in this experiment contains:

| Dataset Split | Samples |
|---|---:|
| Training | 2,891 |
| Validation | 799 |
| Test | 1,629 |
| **Total** | **5,319** |

### Feature Representation

The respiratory audio samples are represented using Mel-spectrogram features.

Feature dimensions used by the model:

128 Mel-frequency bins × 63 time frames × 1 channel

## What is SpecAugment?

SpecAugment is a data augmentation technique designed for audio and speech-related machine learning tasks.

Instead of modifying the original audio signal, SpecAugment modifies its spectrogram representation.

The main techniques include:

* Time masking
* Frequency masking
* Time warping

In this experiment, only **time masking and frequency masking** are used.

---

## SpecAugment Configuration

The augmentation parameters were deliberately restricted to satisfy the experiment requirements.

### Time Masking

A randomly selected continuous region along the time axis is masked.

Maximum time mask:

10 frames


### Frequency Masking

A randomly selected continuous region along the frequency axis is masked.

Maximum frequency mask:

10 frequency bins

Therefore:

Maximum time mask = 10 frames
Maximum frequency mask = 10 bins


The mask width is randomly selected between `0` and the specified maximum for every sample.

---

## Implementation

SpecAugment is applied using TensorFlow's `tf.data.Dataset.map()`.

The augmentation is applied only to the training dataset:

train_ds = train_ds.map(
    lambda x, y: (spec_augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


This ensures that the validation and test datasets remain unchanged.

The augmentation is randomized, meaning that different masks can be applied to the same training sample during different training passes.

### SpecAugment Function

def spec_augment(
    feat,
    max_time_mask=10,
    max_freq_mask=10
):
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

    time_mask = tf.concat([
        tf.ones([time_start]),
        tf.zeros([time_width]),
        tf.ones([TARGET_FRAMES - time_start - time_width])
    ], axis=0)

    time_mask = time_mask[tf.newaxis, :, tf.newaxis]
    feat = feat * time_mask

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

    freq_mask = tf.concat([
        tf.ones([freq_start]),
        tf.zeros([freq_width]),
        tf.ones([N_MELS - freq_start - freq_width])
    ], axis=0)

    freq_mask = freq_mask[:, tf.newaxis, tf.newaxis]
    feat = feat * freq_mask

    return feat
---

## Training Pipeline

The overall pipeline is:

ICBHI Dataset
      ↓
Preprocessed Mel-spectrogram Features
      ↓
Train / Validation / Test Split
      ↓
      ├── Training → SpecAugment → CNN
      │
      ├── Validation → CNN
      │
      └── Test → CNN

SpecAugment is applied **only during training**.

Validation and test samples are not augmented.

---

## Model Architecture

The original multitask CNN architecture is retained.

The convolutional layers are:

Conv2D: 32 filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D: 64 filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D: 128 filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D: 256 filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Global Average Pooling
      ↓
Dense: 256
      ↓
Dropout
      ↓
      ├── Sound Classification Head
      │
      └── Diagnosis Classification Head

The model has approximately **522,763 parameters** in the SpecAugment training configuration.

---

## Training Configuration

The training configuration was kept consistent with the baseline experiment.

| Parameter             | Value                 |
| --------------------- | --------------------- |
| Batch Size            | 32                    |
| Maximum Epochs        | 80                    |
| Optimizer             | Adam                  |
| Learning Rate         | 0.001                 |
| Random Seed           | 42                    |
| Sound Loss            | Focal Loss, gamma = 4 |
| Diagnosis Loss        | Focal Loss, gamma = 2 |
| Diagnosis Loss Weight | 0.1                   |
| Sound Class Weighting | Same as baseline      |
| Time Mask             | Maximum 10 frames     |
| Frequency Mask        | Maximum 10 bins       |

Early stopping was used during training.

The best model was restored after training.

---

## Important Experimental Constraint

The original baseline training file was **not modified**.

Baseline file:

scripts/train_multitask.py


SpecAugment has its own training file:

scripts/train_specaugment.py


This makes the experiment easier to reproduce and ensures that the baseline implementation remains unchanged.

---

## Results

The final SpecAugment model was evaluated using the **standard ICBHI Average Score**.

### Test Performance

| Metric               |     Result |
| -------------------- | ---------: |
| Abnormal Sensitivity | **56.64%** |
| Normal Specificity   | **43.32%** |
| ICBHI Average Score  | **49.98%** |

The standard ICBHI Average Score is calculated as:

ICBHI Score = (Sensitivity + Specificity) / 2


For this experiment:

ICBHI Score = (56.64 + 43.32) / 2
            = 49.98%

---

## Test Confusion Matrix

The test confusion matrix was:

                 Predicted
              N     C     W     B
Actual N     334   169   154   114
       C     273   144    61    94
       W      53    25    24    81
       B      46    15    15    27


Where:

* `N` = Normal
* `C` = Crackle
* `W` = Wheeze
* `B` = Both

---

## Classification Report

              precision    recall  f1-score   support

Normal          0.4731    0.4332    0.4523       771
Crackle         0.4079    0.2517    0.3114       572
Wheeze          0.0945    0.1311    0.1098       183
Both            0.0854    0.2621    0.1289       103

accuracy                            0.3247      1629
macro avg       0.2652    0.2696    0.2506      1629
weighted avg    0.3832    0.3247    0.3439      1629


---

## Generated Files

The experiment generates the following files:

### Training Script

scripts/train_specaugment.py


### Saved Model

data/checkpoints/specaugment/specaugment_final.keras

### Training History

data/results/specaugment/specaugment_training_history.csv


### Test Results

data/results/specaugment/specaugment_test_results.csv


---

## Reproducibility

The experiment uses:

Random Seed = 42


The same dataset split, model architecture, loss configuration, class weighting, optimizer, learning rate, batch size, and training configuration as the baseline are used.

The only experimental change is the addition of SpecAugment to the training pipeline.

---

## Branch

All SpecAugment work is maintained on the dedicated Git branch: specaugment


The `main` branch is not modified by this experiment.

---

## Experiment Summary

The SpecAugment experiment introduces randomized time and frequency masking to the training spectrograms.

The maximum masking limits are:

Time Mask      → 10 frames
Frequency Mask → 10 bins


The augmentation is applied only to the training dataset using TensorFlow's data pipeline.

The final test performance using the standard ICBHI metric was:

Sensitivity = 56.64%
Specificity = 43.32%

ICBHI Average Score = 49.98%


This experiment provides a controlled evaluation of SpecAugment while keeping the original baseline model and training configuration unchanged.
