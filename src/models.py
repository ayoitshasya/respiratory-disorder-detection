"""
models.py
---------
Three CNN architectures for respiratory sound classification.

Factory function: get_model(name, input_shape, num_classes)

Model A — Baseline CNN 2D          (mel-spectrogram input, <300K params)
Model B — 1D CNN on MFCC           (MFCC sequence input,   <150K params)
Model C — MobileNet-style CNN 2D   (mel-spectrogram input, <200K params)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ─────────────────────────────────────────────
# Model A — Baseline CNN 2D
# ─────────────────────────────────────────────

def build_cnn_baseline(input_shape, num_classes):
    """
    4-block 2D CNN on mel-spectrogram.
    Target: <300K parameters.

    Args:
        input_shape: (freq_bins, time_steps) — 2D, no channel dim (added internally)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(*input_shape, 1))  # add channel dim

    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='cnn_baseline')


# ─────────────────────────────────────────────
# Model B — 1D CNN on MFCC
# ─────────────────────────────────────────────

def build_cnn_1d_mfcc(input_shape, num_classes):
    """
    1D CNN operating on MFCC time sequences.
    Input shape: (time_steps, 39) — 39 MFCC features per time step.
    Target: <150K parameters.

    Args:
        input_shape: (time_steps, 39)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='cnn_1d_mfcc')


# ─────────────────────────────────────────────
# Model C — MobileNet-style (Depthwise Separable)
# ─────────────────────────────────────────────

def _depthwise_sep_block(x, filters, strides=(1, 1)):
    """Depthwise separable conv block: depthwise → BN → ReLU → pointwise → BN → ReLU."""
    x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_cnn_mobilenet(input_shape, num_classes):
    """
    Lightweight CNN with depthwise separable convolutions (MobileNet-style).
    Target: <200K parameters.

    Args:
        input_shape: (freq_bins, time_steps)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(*input_shape, 1))

    # Stem
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable blocks
    x = _depthwise_sep_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    x = _depthwise_sep_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = _depthwise_sep_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='cnn_mobilenet')


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

_MODEL_REGISTRY = {
    'cnn_baseline':  build_cnn_baseline,
    'cnn_1d_mfcc':   build_cnn_1d_mfcc,
    'cnn_mobilenet': build_cnn_mobilenet,
}


def get_model(name, input_shape, num_classes):
    """
    Factory function to build a model by name.

    Args:
        name:        Model name: 'cnn_baseline', 'cnn_1d_mfcc', 'cnn_mobilenet'
        input_shape: Tuple of input dimensions (without batch or channel)
        num_classes: Number of output classes

    Returns:
        Uncompiled Keras model
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{name}'. Choose from: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](input_shape, num_classes)


def print_model_summary(model):
    """Print model summary with param count and estimated size."""
    model.summary()
    total_params = model.count_params()
    size_mb      = total_params * 4 / 1024 / 1024   # float32 = 4 bytes
    size_int8_kb = total_params / 1024               # int8 = 1 byte
    print(f"\nTotal parameters  : {total_params:,}")
    print(f"Estimated size    : {size_mb:.2f} MB (float32)")
    print(f"Estimated TFLite  : ~{size_int8_kb:.0f} KB (int8 quantized)")


if __name__ == '__main__':
    print("=" * 55)
    print("Model A — Baseline CNN 2D")
    print("=" * 55)
    m = get_model('cnn_baseline', (128, 251), 4)
    print_model_summary(m)

    print("\n" + "=" * 55)
    print("Model B — 1D CNN on MFCC")
    print("=" * 55)
    m = get_model('cnn_1d_mfcc', (251, 39), 4)
    print_model_summary(m)

    print("\n" + "=" * 55)
    print("Model C — MobileNet-style CNN")
    print("=" * 55)
    m = get_model('cnn_mobilenet', (128, 251), 4)
    print_model_summary(m)
