"""
augmentation.py
---------------
Audio-level augmentation (applied before feature extraction, training split only)
and SpecAugment (applied on-the-fly via tf.data during training).

RULE: NEVER apply augmentation to val or test splits.
"""

import numpy as np
import librosa
import tensorflow as tf


# ─────────────────────────────────────────────
# Audio-level augmentation
# ─────────────────────────────────────────────

def time_stretch(audio, rate=None):
    """
    Stretch audio in time without changing pitch.

    Args:
        audio: 1D numpy array
        rate:  Stretch factor (0.8 = slower, 1.2 = faster). Random if None.

    Returns:
        Stretched audio array
    """
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps=None):
    """
    Shift pitch by n semitones without changing tempo.

    Args:
        audio:   1D numpy array
        sr:      Sample rate
        n_steps: Semitones to shift (±2). Random if None.

    Returns:
        Pitch-shifted audio array
    """
    if n_steps is None:
        n_steps = np.random.choice([-2, -1, 1, 2])
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_gaussian_noise(audio, snr_db=None):
    """
    Add Gaussian noise at a target SNR.

    Args:
        audio:  1D numpy array
        snr_db: Signal-to-noise ratio in dB (15–25). Random if None.

    Returns:
        Noisy audio array
    """
    if snr_db is None:
        snr_db = np.random.uniform(15, 25)
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def augment_audio(audio, sr, aug_types=None):
    """
    Apply one or more audio-level augmentations.

    Args:
        audio:     1D numpy array
        sr:        Sample rate
        aug_types: List of augmentation names to apply. Random subset if None.
                   Options: 'time_stretch', 'pitch_shift', 'noise'

    Returns:
        Augmented audio array
    """
    all_augs = ['time_stretch', 'pitch_shift', 'noise']
    if aug_types is None:
        # Apply 1–2 random augmentations
        n    = np.random.randint(1, 3)
        aug_types = list(np.random.choice(all_augs, size=n, replace=False))

    for aug in aug_types:
        if aug == 'time_stretch':
            audio = time_stretch(audio)
        elif aug == 'pitch_shift':
            audio = pitch_shift(audio, sr)
        elif aug == 'noise':
            audio = add_gaussian_noise(audio)

    return audio


# ─────────────────────────────────────────────
# SpecAugment (applied on-the-fly during training)
# ─────────────────────────────────────────────

def spec_augment(spectrogram, time_mask_pct=0.20, freq_mask_pct=0.15):
    """
    Apply SpecAugment: random time and frequency masking.

    Args:
        spectrogram:   2D tensor (freq_bins, time_steps)
        time_mask_pct: Max fraction of time steps to mask
        freq_mask_pct: Max fraction of frequency bins to mask

    Returns:
        Augmented spectrogram tensor (same shape)
    """
    spec = tf.identity(spectrogram)
    freq_bins, time_steps = tf.shape(spec)[0], tf.shape(spec)[1]

    # Frequency masking
    max_freq_mask = tf.cast(tf.cast(freq_bins, tf.float32) * freq_mask_pct, tf.int32)
    f_mask_size   = tf.random.uniform([], 0, max_freq_mask + 1, dtype=tf.int32)
    f_start       = tf.random.uniform([], 0, freq_bins - f_mask_size + 1, dtype=tf.int32)
    freq_mask     = tf.concat([
        tf.ones([f_start, time_steps]),
        tf.zeros([f_mask_size, time_steps]),
        tf.ones([freq_bins - f_start - f_mask_size, time_steps])
    ], axis=0)
    spec = spec * tf.cast(freq_mask, spec.dtype)

    # Time masking
    max_time_mask = tf.cast(tf.cast(time_steps, tf.float32) * time_mask_pct, tf.int32)
    t_mask_size   = tf.random.uniform([], 0, max_time_mask + 1, dtype=tf.int32)
    t_start       = tf.random.uniform([], 0, time_steps - t_mask_size + 1, dtype=tf.int32)
    time_mask     = tf.concat([
        tf.ones([freq_bins, t_start]),
        tf.zeros([freq_bins, t_mask_size]),
        tf.ones([freq_bins, time_steps - t_start - t_mask_size])
    ], axis=1)
    spec = spec * tf.cast(time_mask, spec.dtype)

    return spec


def apply_spec_augment_to_batch(X_batch, y_batch):
    """
    tf.data-compatible function to apply SpecAugment to a batch.
    Pass this to dataset.map() for the training pipeline.

    Args:
        X_batch: Tensor of shape (batch, freq, time, channels)
        y_batch: Labels (passed through unchanged)

    Returns:
        (augmented_X_batch, y_batch)
    """
    augmented = tf.map_fn(
        lambda x: tf.expand_dims(spec_augment(x[..., 0]), axis=-1),
        X_batch
    )
    return augmented, y_batch


# ─────────────────────────────────────────────
# Class imbalance utilities
# ─────────────────────────────────────────────

def compute_class_weights(labels, num_classes):
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        labels:      1D array of integer class labels
        num_classes: Total number of classes

    Returns:
        dict mapping class_index → weight
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)   # avoid division by zero
    total  = counts.sum()
    weights = total / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def focal_loss(gamma=2.0, alpha=None):
    """
    Focal loss for multi-class classification.
    Reduces loss contribution from easy examples, focusing on hard ones.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Per-class weight array (None = uniform)

    Returns:
        Keras loss function
    """
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        p_t     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_w = tf.pow(1.0 - p_t, gamma)

        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true * tf.constant(alpha, dtype=tf.float32), axis=-1, keepdims=True)
            focal_w = focal_w * alpha_t

        return tf.reduce_mean(focal_w * ce)

    loss_fn.__name__ = f'focal_loss_g{gamma}'
    return loss_fn
