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

# ── Set to True once your model is ready and run_model() is implemented ───────
MODEL_READY = False

# ── Update these to match your model's output label order ─────────────────────
DISEASE_CLASSES = [
    "Healthy",
    "COPD",
    "Pneumonia",
    "Bronchitis",
    "Asthma",
    "Pulmonary Fibrosis",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_audio(filepath, sr=22050):
    """Load full audio at target sample rate (no fixed duration cap)."""
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr


def downsample(arr, n=1000):
    """Downsample 1-D array to n points for lightweight transfer."""
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist(), idx.tolist()


# ── Waveform representations ──────────────────────────────────────────────────

def compute_raw_waveform(y, sr, n=1200):
    idx = np.linspace(0, len(y) - 1, n, dtype=int)
    return {
        "times": (idx / sr).tolist(),
        "amplitudes": y[idx].tolist(),
    }


def compute_rms_envelope(y, sr, frame_length=2048, hop_length=512):
    """RMS energy envelope — shows breathing rhythm / loudness curve."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return {
        "times": times.tolist(),
        "values": rms.tolist(),
    }


def compute_spectral_centroid(y, sr, hop_length=512):
    """Spectral centroid over time — brightness / wheeze indicator."""
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    frames = np.arange(len(sc))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
    return {
        "times": times.tolist(),
        "values": sc_norm.tolist(),
        "values_hz": sc.tolist(),
    }


def compute_mfcc_mean(y, sr, n_mfcc=13):
    """Mean MFCC coefficients — compact spectral shape fingerprint."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return {
        "coefficients": list(range(1, n_mfcc + 1)),
        "means": np.mean(mfcc, axis=1).tolist(),
        "stds": np.std(mfcc, axis=1).tolist(),
    }


def compute_mfcc_over_time(y, sr, n_mfcc=13, hop_length=512):
    """Full MFCC matrix over time (downsampled) for heatmap display."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    frames = np.arange(mfcc.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    step = max(1, mfcc.shape[1] // 200)
    mfcc_small = mfcc[:, ::step]
    mfcc_norm = (mfcc_small - mfcc_small.min()) / (mfcc_small.max() - mfcc_small.min() + 1e-8)
    return {
        "times": times[::step].tolist(),
        "matrix": mfcc_norm.tolist(),
        "n_mfcc": n_mfcc,
    }


def compute_chroma(y, sr, hop_length=512):
    """Chroma features — harmonic content / pitch class distribution."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    frames = np.arange(chroma.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    step = max(1, chroma.shape[1] // 200)
    return {
        "times": times[::step].tolist(),
        "matrix": chroma[:, ::step].tolist(),
        "pitch_classes": ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'],
    }


def compute_zcr(y, sr, hop_length=512):
    """Zero-crossing rate — useful for detecting crackles/fricatives."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    frames = np.arange(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return {
        "times": times.tolist(),
        "values": zcr.tolist(),
    }


# ── Spectrogram images ────────────────────────────────────────────────────────

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


# REPLACE the entire compute_mel_spectrogram_image function with:
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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return _fig_to_b64(fig)


# REPLACE the entire compute_mfcc_image function with:
def compute_mfcc_image(y, sr):
    fig, ax = make_axes()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
    try:
        cb = fig.colorbar(img, ax=ax)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8fa3bf")
    except Exception:
        pass
    ax.set_ylabel("MFCC coefficient")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return _fig_to_b64(fig)

# REPLACE the entire compute_chroma_image function with:
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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch class")
    plt.tight_layout()
    return _fig_to_b64(fig)

# ── Audio metadata ────────────────────────────────────────────────────────────

# REPLACE the entire audio_metadata function with:
def audio_metadata(y, sr, filepath):
    duration = float(len(y)) / sr          # avoids librosa.get_duration() version issues
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


# ── Model inference ───────────────────────────────────────────────────────────

def run_model(y, sr):
    """
    ══════════════════════════════════════════════════════════════════
    REPLACE THIS with your trained model. Set MODEL_READY = True above.

    PyTorch example:
        import torch
        _model = torch.load("model.pt", map_location="cpu"); _model.eval()
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        t = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
        with torch.no_grad():
            probs = torch.softmax(_model(t), dim=1).squeeze().numpy()
        return probs.tolist()

    Keras example:
        import tensorflow as tf
        _model = tf.keras.models.load_model("model.h5")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        inp = np.expand_dims(np.expand_dims(mel_db, 0), -1)  # (1,128,T,1)
        return _model.predict(inp, verbose=0)[0].tolist()
    ══════════════════════════════════════════════════════════════════
    """
    raise NotImplementedError("Model not yet connected — set MODEL_READY = True after implementing run_model()")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/visualize", methods=["POST"])
def visualize():
    """
    Always-available endpoint: returns all waveforms + spectrograms.
    Works regardless of MODEL_READY.
    """
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

    # AFTER
    except Exception as e:
        import traceback
        traceback.print_exc()          # prints full stack trace to your terminal
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint — returns model_pending when MODEL_READY is False.
    Frontend polls / calls this separately so visualizations never block on it.
    """
    if not MODEL_READY:
        return jsonify({
            "success": True,
            "model_pending": True,
            "message": "Model is still in training. Visualizations are fully available.",
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
        probabilities = run_model(y, sr)
        predictions = sorted(
            [{"disease": cls, "probability": round(float(p), 4)}
             for cls, p in zip(DISEASE_CLASSES, probabilities)],
            key=lambda x: x["probability"], reverse=True,
        )
        return jsonify({
            "success": True,
            "model_pending": False,
            "predictions": predictions,
            "top_prediction": predictions[0],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_READY,
        "disease_classes": DISEASE_CLASSES,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)
