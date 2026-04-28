import os
import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Model config — must match training hyperparameters exactly ────────────────
MODEL_PATH   = "multitask_best.keras"   # ← put your .keras file path here
TARGET_FRAMES = 63
N_MELS        = 128
SR_MODEL      = 22050   # sample rate used during training

SOUND_CLASSES     = ["Normal", "Crackle", "Wheeze", "Both"]
DIAGNOSIS_CLASSES = ["Healthy", "COPD", "URTI", "Bronchiectasis",
                     "Pneumonia", "Bronchiolitis", "Other"]

# ── Load model once at startup ────────────────────────────────────────────────
_model = None

def build_model():
    """Rebuild the exact same architecture from training."""
    import tensorflow as tf
    NUM_SOUND = 4; NUM_DIAGNOSIS = 7

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


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Place multitask_final_best.keras in the same directory as app.py."
            )
        _model = build_model()
        _model.load_weights(MODEL_PATH)   # ← skip config, load weights only
        print(f"[INFO] Model weights loaded from {MODEL_PATH}")
    return _model

MODEL_READY = os.path.exists(MODEL_PATH)   # auto-detected; no manual toggle needed

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — mirrors the training pipeline exactly
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_truncate(feat, t=TARGET_FRAMES):
    """Pad / truncate the time axis to exactly TARGET_FRAMES columns."""
    c = feat.shape[-1]          # feat shape: (N_MELS, time)
    if c < t:
        feat = np.pad(feat, [(0, 0), (0, t - c)])
    else:
        feat = feat[..., :t]
    return feat


def preprocess_for_model(y, sr):
    """
    Replicate the feature extraction used in the training manifest.
    Adjust this if your training used a different mel pipeline.
    """
    # Resample if needed
    if sr != SR_MODEL:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_MODEL)
        sr = SR_MODEL

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    feat = pad_or_truncate(mel_db, TARGET_FRAMES)   # (128, 63)
    feat = feat[..., np.newaxis]                     # (128, 63, 1)
    feat = np.expand_dims(feat, 0).astype(np.float32)  # (1, 128, 63, 1)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def run_model(y, sr):
    """
    Returns a dict:
        sound     → sorted list of {label, probability}
        diagnosis → sorted list of {label, probability}
    """
    model = get_model()
    inp = preprocess_for_model(y, sr)

    sound_probs, diag_probs = model.predict(inp, verbose=0)
    sound_probs = sound_probs[0].tolist()
    diag_probs  = diag_probs[0].tolist()

    sound_preds = sorted(
        [{"label": cls, "probability": round(float(p), 4)}
         for cls, p in zip(SOUND_CLASSES, sound_probs)],
        key=lambda x: x["probability"], reverse=True,
    )
    diag_preds = sorted(
        [{"label": cls, "probability": round(float(p), 4)}
         for cls, p in zip(DIAGNOSIS_CLASSES, diag_probs)],
        key=lambda x: x["probability"], reverse=True,
    )
    return {"sound": sound_preds, "diagnosis": diag_preds}


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr


def downsample(arr, n=1000):
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist(), idx.tolist()


def compute_raw_waveform(y, sr, n=1200):
    idx = np.linspace(0, len(y) - 1, n, dtype=int)
    return {"times": (idx / sr).tolist(), "amplitudes": y[idx].tolist()}


def compute_rms_envelope(y, sr, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return {"times": times.tolist(), "values": rms.tolist()}


def compute_spectral_centroid(y, sr, hop_length=512):
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    frames = np.arange(len(sc))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
    return {"times": times.tolist(), "values": sc_norm.tolist(), "values_hz": sc.tolist()}


def compute_mfcc_mean(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return {
        "coefficients": list(range(1, n_mfcc + 1)),
        "means": np.mean(mfcc, axis=1).tolist(),
        "stds": np.std(mfcc, axis=1).tolist(),
    }


def compute_mfcc_over_time(y, sr, n_mfcc=13, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    frames = np.arange(mfcc.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    step = max(1, mfcc.shape[1] // 200)
    mfcc_small = mfcc[:, ::step]
    mfcc_norm = (mfcc_small - mfcc_small.min()) / (mfcc_small.max() - mfcc_small.min() + 1e-8)
    return {"times": times[::step].tolist(), "matrix": mfcc_norm.tolist(), "n_mfcc": n_mfcc}



def compute_zcr(y, sr, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    frames = np.arange(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return {"times": times.tolist(), "values": zcr.tolist()}


def _fig_to_b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_axes(figsize=(10, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8fa3bf")
    ax.xaxis.label.set_color("#8fa3bf")
    ax.yaxis.label.set_color("#8fa3bf")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222d42")
    return fig, ax


def compute_mel_spectrogram_image(y, sr):
    fig, ax = make_axes()
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                   fmax=8000, ax=ax, cmap="inferno")
    try:
        cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cb.ax.tick_params(colors="#8fa3bf")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8fa3bf")
    except Exception:
        pass
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return _fig_to_b64(fig)


def compute_mfcc_image(y, sr):
    fig, ax = make_axes()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
    try:
        cb = fig.colorbar(img, ax=ax)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8fa3bf")
    except Exception:
        pass
    ax.set_ylabel("MFCC coefficient"); ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return _fig_to_b64(fig)


def compute_chroma_image(y, sr):
    fig, ax = make_axes()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma",
                                   ax=ax, cmap="viridis")
    try:
        cb = fig.colorbar(img, ax=ax)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8fa3bf")
    except Exception:
        pass
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch class")
    plt.tight_layout()
    return _fig_to_b64(fig)


def audio_metadata(y, sr, filepath):
    duration = float(len(y)) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(sc))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    return {
        "filename": os.path.basename(filepath),
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "rms_energy": round(rms, 4),
        "zero_crossing_rate": round(zcr_mean, 4),
        "spectral_centroid_hz": round(sc_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/visualize", methods=["POST"])
def visualize():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file — only .wav files accepted"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        y, sr = load_audio(filepath)
        return jsonify({
            "success": True,
            "metadata": audio_metadata(y, sr, filepath),
            "waveforms": {
                "raw":               compute_raw_waveform(y, sr),
                "rms_envelope":      compute_rms_envelope(y, sr),
                "spectral_centroid": compute_spectral_centroid(y, sr),
                "zcr":               compute_zcr(y, sr),
                "mfcc_mean":         compute_mfcc_mean(y, sr),
                "mfcc_over_time":    compute_mfcc_over_time(y, sr),
                "chroma":            compute_chroma(y, sr),
            },
            "spectrograms": {
                "mel":    compute_mel_spectrogram_image(y, sr),
                "mfcc":   compute_mfcc_image(y, sr),
                "chroma": compute_chroma_image(y, sr),
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Returns dual-head predictions:
      - sound      → 4 classes (Normal / Crackle / Wheeze / Both)
      - diagnosis  → 7 classes (Healthy / COPD / URTI / Bronchiectasis /
                                Pneumonia / Bronchiolitis / Other)
    """
    if not MODEL_READY:
        return jsonify({
            "success": True,
            "model_pending": True,
            "message": (
                f"Model file '{MODEL_PATH}' not found. "
                "Place the .keras file in the same directory as app.py."
            ),
        })

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file — only .wav files accepted"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        y, sr = load_audio(filepath)
        results = run_model(y, sr)

        sound = results["sound"]
        diag  = results["diagnosis"]

        return jsonify({
            "success": True,
            "model_pending": False,
            # Keys PredictionPanel.jsx reads
            "top_prediction": {"disease": sound[0]["label"], "probability": sound[0]["probability"]},
            "predictions":    [{"disease": p["label"], "probability": p["probability"]} for p in sound],
            # Diagnosis (bonus — available for future frontend use)
            "top_diagnosis":         {"disease": diag[0]["label"], "probability": diag[0]["probability"]},
            "diagnosis_predictions": [{"disease": p["label"], "probability": p["probability"]} for p in diag],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_READY,
        "model_path": MODEL_PATH,
        "sound_classes": SOUND_CLASSES,
        "diagnosis_classes": DIAGNOSIS_CLASSES,
    })


if __name__ == "__main__":
    # Eagerly load the model on startup so the first request isn't slow
    if MODEL_READY:
        try:
            get_model()
        except Exception as e:
            print(f"[WARN] Could not pre-load model: {e}")
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)

