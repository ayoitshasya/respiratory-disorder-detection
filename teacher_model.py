"""
teacher_model.py
----------------
Large teacher model for knowledge distillation.
This model is too large for ESP32 but trained to get
maximum ICBHI score. Its soft outputs will supervise
the student (our deployable multitask CNN).
DO NOT deploy this model — training use only.
"""

import numpy as np
import tensorflow as tf

NUM_SOUND = 4
NUM_DIAG  = 7

def build_teacher_model(input_shape=(128, 126, 1)):
    """
    Large CNN with 128->256->512->1024 filters.
    ~10x more parameters than student.
    Only used during training, never deployed.
    """
    inp = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 4
    x = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Shared dense
    shared = tf.keras.layers.Dense(512, activation='relu')(x)
    shared = tf.keras.layers.Dropout(0.5)(shared)

    # Sound head
    s = tf.keras.layers.Dense(256, activation='relu')(shared)
    s = tf.keras.layers.Dropout(0.3)(s)
    sound_out = tf.keras.layers.Dense(
        NUM_SOUND, activation='softmax', name='sound'
    )(s)

    # Diagnosis head
    d = tf.keras.layers.Dense(256, activation='relu')(shared)
    d = tf.keras.layers.Dropout(0.3)(d)
    diag_out = tf.keras.layers.Dense(
        NUM_DIAG, activation='softmax', name='diagnosis'
    )(d)

    model = tf.keras.Model(inp, [sound_out, diag_out], name='teacher')
    return model


if __name__ == '__main__':
    model = build_teacher_model()
    model.summary()
    total = model.count_params()
    print(f"\nTeacher parameters : {total:,}")
    print(f"Teacher float32 size: {total*4/1024:.1f} KB")
    print(f"ESP32 limit         : 520 KB")
    print(f"Deployable?         : NO — teacher is training only")