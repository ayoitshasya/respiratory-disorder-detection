import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\processed_data"
MODEL_SAVE_PATH     = r"C:\Users\Hasya Abburi\Desktop\respiratory disorder detection\model"
CLASS_NAMES         = ['normal', 'crackle', 'wheeze', 'both']
VAL_SPLIT           = 0.15
TEST_SPLIT          = 0.15

def load_dataset():
    X, y = [], []
    print("Loading .npy files...")
    for label, cls_name in enumerate(CLASS_NAMES):
        cls_path = os.path.join(PROCESSED_DATA_PATH, cls_name)
        files    = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        print(f"  {cls_name:10s}: {len(files)} files")
        for f in files:
            X.append(np.load(os.path.join(cls_path, f)))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

X, y = load_dataset()

# reproduce same split with same random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=VAL_SPLIT, random_state=42, stratify=y_train)

print(f"\nTest set: {X_test.shape[0]} samples")

# load best model
model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'best_model.keras'))
print("Model loaded.")

# predict
y_pred = np.argmax(model.predict(X_test), axis=1)

# --- Classification Report ---
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")
