import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no window needed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\processed_data"
MODEL_SAVE_PATH     = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\model"

CLASS_NAMES   = ['normal', 'crackle', 'wheeze', 'both']
INPUT_SHAPE   = (128, 216, 2)   # (freq_bins, time_frames, channels)
NUM_CLASSES   = 4
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    X, y = [], []

    print("Loading .npy files...")
    for label, cls_name in enumerate(CLASS_NAMES):
        cls_path = os.path.join(PROCESSED_DATA_PATH, cls_name)
        files    = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        print(f"  {cls_name:10s}: {len(files)} files")

        for f in files:
            feat = np.load(os.path.join(cls_path, f))
            X.append(feat)
            y.append(label)

    X = np.array(X, dtype=np.float32)   # (N, 128, 216, 2)
    y = np.array(y, dtype=np.int32)     # (N,)

    print(f"\nDataset shape : {X.shape}")
    print(f"Labels shape  : {y.shape}")
    return X, y

# ============================================================
# CNN MODEL
# ============================================================

def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN architecture:
      - 4 convolutional blocks (Conv → BN → ReLU → MaxPool → Dropout)
      - Global Average Pooling (avoids overfitting vs Flatten)
      - 2 Dense layers
      - Softmax output
    """
    inputs = layers.Input(shape=input_shape)

    # --- Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # --- Block 4 ---
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # --- Classifier head ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='respiratory_cnn')
    return model

# ============================================================
# TRAINING
# ============================================================

def train():

    # --- load data ---
    X, y = load_dataset()

    # --- split: train / val / test ---
    X_train, X_test,  y_train, y_test  = train_test_split(X,       y,       test_size=TEST_SPLIT,  random_state=42, stratify=y)
    X_train, X_val,   y_train, y_val   = train_test_split(X_train, y_train, test_size=VAL_SPLIT,   random_state=42, stratify=y_train)

    print(f"\nTrain : {X_train.shape[0]} samples")
    print(f"Val   : {X_val.shape[0]}   samples")
    print(f"Test  : {X_test.shape[0]}  samples")

    # --- one-hot encode ---
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)

    # --- build model ---
    model = build_model()
    model.summary()

    # --- compile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- callbacks ---
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    # --- train ---
    print("\nTraining...")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb_list
    )

    # --- evaluate on test set ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    y_pred_probs = model.predict(X_test)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # --- plots ---
    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_pred)

    # --- save final model ---
    model.save(os.path.join(MODEL_SAVE_PATH, 'respiratory_cnn_final.keras'))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_PATH, 'training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix — Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

# ============================================================
# INFERENCE  —  single audio clip (for IoT device use)
# ============================================================

def predict_single(npy_path, model_path=None):
    """
    Load a single .npy feature file and predict the class.
    Use this function on your IoT device after feature extraction.
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.keras')

    model = tf.keras.models.load_model(model_path)

    feat = np.load(npy_path)               # (128, 216, 2)
    feat = np.expand_dims(feat, axis=0)    # (1, 128, 216, 2)

    probs     = model.predict(feat)[0]
    pred_idx  = np.argmax(probs)
    pred_cls  = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"\nPrediction : {pred_cls.upper()}  ({confidence:.1f}% confidence)")
    print("Class probabilities:")
    for cls, prob in zip(CLASS_NAMES, probs):
        bar = '█' * int(prob * 30)
        print(f"  {cls:10s}: {prob:.4f}  {bar}")

    return pred_cls, probs

# ============================================================

if __name__ == "__main__":
    train()
