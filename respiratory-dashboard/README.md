# PulmoScan — Respiratory Disease Detection Dashboard

React frontend + Flask REST API for lung auscultation analysis.

## Architecture

```
respiratory-dashboard/
├── backend/
│   ├── app.py              ← Flask API (audio processing + model inference)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx         ← Main dashboard shell
    │   ├── App.css         ← Dark clinical theme
    │   └── components/
    │       ├── UploadZone.jsx      ← Drag-and-drop .wav upload
    │       ├── WaveformViewer.jsx  ← Canvas waveform animation
    │       ├── SpectrogramViewer.jsx ← Mel spectrogram image
    │       ├── PredictionPanel.jsx ← Disease probabilities + circular gauge
    │       ├── MetadataBar.jsx     ← Audio stats bar
    │       └── StatusBadge.jsx     ← Live pipeline status
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## API Flow

```
POST /api/analyze (multipart .wav)
  └─ librosa.load()
  └─ melspectrogram() → PNG image (base64) + raw matrix
  └─ extract_features() → model input tensor
  └─ run_model() → probabilities per disease class
  └─ JSON response → React frontend
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # starts on http://localhost:5000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                       # starts on http://localhost:3000
```

## Plugging in Your Model

Open `backend/app.py` and replace the `run_model()` function:

### PyTorch example
```python
import torch

_model = None
def get_model():
    global _model
    if _model is None:
        _model = torch.load("your_model.pt", map_location="cpu")
        _model.eval()
    return _model

def run_model(features):
    model = get_model()
    # features shape: (128, time_frames)
    tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    return probs.tolist()
```

### TensorFlow/Keras example
```python
import tensorflow as tf
import numpy as np

_model = None
def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model("your_model.h5")
    return _model

def run_model(features):
    model = get_model()
    inp = np.expand_dims(np.expand_dims(features, 0), -1)  # (1,128,T,1)
    probs = model.predict(inp, verbose=0)[0]
    return probs.tolist()
```

## Updating Disease Classes

Edit the `DISEASE_CLASSES` list in `app.py` to match your model's output order:

```python
DISEASE_CLASSES = [
    "Healthy",
    "COPD",
    "Pneumonia",
    # ... your classes
]
```

Then update the `SEVERITY` map in `PredictionPanel.jsx` to add colors and risk labels.

## API Response Schema

```json
{
  "success": true,
  "metadata": {
    "filename": "patient_01.wav",
    "duration_s": 10.0,
    "sample_rate": 22050,
    "rms_energy": 0.0412,
    "zero_crossing_rate": 0.0831
  },
  "waveform": {
    "times": [0.0, 0.01, ...],
    "amplitudes": [0.003, -0.012, ...]
  },
  "mel_spectrogram_image": "<base64 PNG>",
  "predictions": [
    { "disease": "COPD", "probability": 0.72 },
    { "disease": "Healthy", "probability": 0.15 },
    ...
  ],
  "top_prediction": { "disease": "COPD", "probability": 0.72 }
}
```

## Disclaimer

For research and development use only. Not a substitute for clinical diagnosis.
