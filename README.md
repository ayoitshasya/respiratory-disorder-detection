# Respiratory Disorder Detection

ML pipeline for classifying respiratory sounds from the ICBHI 2017 dataset using CNNs.

## Problem Statement

Classify lung sound recordings into 4 categories:
- **Normal** — healthy lungs
- **Crackle** — indicates fluid or infection (discontinuous, explosive sounds)
- **Wheeze** — indicates narrowed airways (continuous, musical sounds)
- **Both** — crackle + wheeze combined

Secondary task: predict patient diagnosis (COPD, Pneumonia, URTI, etc.) from the same audio input.

## Dataset

**ICBHI 2017** — 101 patients, 5,319 respiratory cycles recorded across multiple chest locations.

> **Data is not included in this repo** (too large for GitHub).
> Preprocessed `.npy` features and manifests are available on Google Drive (contact repo owner).
> Raw dataset available at: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

### Class Distribution (Training Set)

| Sound Class | Samples | % of Total |
|-------------|---------|------------|
| Normal | 1,643 | 56.8% |
| Crackle | 626 | 21.7% |
| Wheeze | 419 | 14.5% |
| Both | 203 | 7.0% |

> Heavy class imbalance — Normal has 8x more samples than Both. This is why accuracy is a misleading metric and ICBHI score is used instead.

### Label Maps

| Sound Label | Class |
|-------------|-------|
| 0 | Normal |
| 1 | Crackle |
| 2 | Wheeze |
| 3 | Both |

| Diagnosis Label | Class |
|-----------------|-------|
| 0 | Healthy |
| 1 | COPD |
| 2 | URTI |
| 3 | Bronchiectasis |
| 4 | Pneumonia |
| 5 | Bronchiolitis |
| 6 | Other |

## Project Structure

```
resp-detection/
├── train_baseline.py           # Baseline CNN — sound classification only
├── multitask_model.py          # Multitask CNN — sound + diagnosis heads (proven baseline)
├── train_with_synthetic.py     # Multitask CNN + synthetic data (experimental)
├── train_highres.py            # Multitask CNN + hop_length=256 + modest aug (current best attempt)
├── generate_synthetic.py       # Generate synthetic crackle/wheeze/both samples
├── augment_minority.py         # Original heavy augmentation (abandoned — caused collapse)
├── augment_modest.py           # Modest real-data augmentation (2–5x minority classes)
├── reextract_features.py       # Re-extract features after hop_length change
├── configs/
│   └── experiment_config.json  # Hyperparameters
├── src/
│   ├── data_loader.py          # Patient-level train/val/test splits
│   ├── preprocessing.py        # Butterworth filter, mel-spectrogram (hop_length=256)
│   ├── augmentation.py         # Audio augmentation + SpecAugment
│   ├── synthetic_generator.py  # Mathematical synthesis of wheeze/crackle/both
│   ├── models.py               # CNN architectures
│   ├── multitask_model.py      # Shared backbone + sound + diagnosis heads
│   ├── training.py             # Focal loss, cosine LR, ICBHI score tracking
│   ├── evaluation.py           # Confusion matrix, metrics, training curves
│   └── export_tflite.py        # TFLite int8 quantization export
└── data/                       # Not tracked in git — see Google Drive
    ├── checkpoints/            # Saved .keras model files
    ├── results/                # Training curves, confusion matrices (PNG)
    └── processed/
        ├── manifest.csv              # Original splits (5,319 samples)
        ├── manifest_aug_modest.csv   # Modest augmented splits (6,415 train samples)
        ├── manifest_synthetic.csv    # Synthetic samples manifest
        ├── train/                    # Mel spectrogram .npy (hop_length=256)
        ├── val/                      # Mel spectrogram .npy files (val)
        ├── test/                     # Mel spectrogram .npy files (test)
        ├── train_aug_modest/         # Modest augmented .npy files
        └── train_synthetic/          # Synthetic .npy files
```

## Pipeline

### 1. Patient-Level Splitting (63/19/19 patients)
Recordings from the same patient stay in one split only. Prevents data leakage — without this, the model memorises a patient's breathing style rather than learning disease patterns.

### 2. Preprocessing
- **Butterworth bandpass filter** (4th order, 100–2000 Hz) — removes heart noise (<100 Hz) and equipment noise (>2000 Hz). Butterworth chosen for maximally flat frequency response (no ripple distortion).
- **Mel spectrogram** (128 mel bins, n_fft=2048, hop_length=256) — converts audio to a 2D frequency×time image. Mel scale mimics human hearing (logarithmic), better suited to medical audio than linear spectrograms.
- **Fixed shape** — all clips padded or truncated to 126 time frames → final shape (128, 126, 1).

> **hop_length change (512→256):** Crackle sounds are short explosive bursts (5–100ms). At hop_length=512 a 50ms crackle spans only 1–2 spectrogram frames. At hop_length=256 it spans 3+ frames, giving the CNN more signal to learn from. This doubles temporal resolution at no architectural cost since GlobalAveragePooling handles variable input sizes.

### 3. Data Augmentation (minority classes only)
Applied to training set only. Val/test never augmented.

| Technique | Effect | Why |
|-----------|--------|-----|
| Time stretch (0.80–0.92x, 1.08–1.20x) | Slows/speeds audio | Duration variation is irrelevant to class |
| Pitch shift (±1.5–3.0 semitones) | Shifts frequency | Patient chest size varies pitch slightly |
| Gaussian noise (SNR 18–28 dB) | Adds microphone noise | Simulates real hospital recording variation |
| Stretch + noise combined | Both above | Extra diversity |

**Modest copies:** Crackle×2, Wheeze×3, Both×5, Normal unchanged → 6,415 total training samples.

| Class | Original | After Aug |
|-------|----------|-----------|
| Normal | 1,643 | 1,643 |
| Crackle | 626 | 1,878 |
| Wheeze | 419 | 1,676 |
| Both | 203 | 1,218 |

> **Previous augmentation attempts and what failed:**
> - *21k heavy augmentation* (Normal×3, Crackle×8, Wheeze×11, Both×20): ICBHI collapsed to 0.50. Too many near-identical copies caused the model to memorise augmentation transforms rather than learn acoustic features.
> - *Mathematical synthesis* (`src/synthetic_generator.py`): Generated wheeze (vibrato sinusoids), crackle (bandpassed noise bursts), and both (layered) sounds mixed into real normal breathing recordings. 1,500 samples at alpha 0.25–0.70 caused Normal recall to collapse from 97%→51% (model over-detected pathology). 700 samples at lower alpha (0.10–0.28) caused training instability. Root cause: synthetic acoustic patterns don't match real recordings closely enough — model learns synthetic-specific features that don't transfer.

### 4. Model Training

#### Baseline CNN
- 4 Conv2D blocks (32→64→128→256 filters), BatchNorm, MaxPooling
- GlobalAveragePooling → Dense(256) → Dense(128) → Dense(4, softmax)
- Single output: sound classification only
- Loss: categorical crossentropy + class weights
- LR: cosine decay 1e-3 → 1e-6

#### Multitask CNN (Final Model)
- Same 4-block CNN backbone (shared)
- Two output heads branching from shared Dense(256):
  - **Sound head**: Dense(128) → Dense(4, softmax)
  - **Diagnosis head**: Dense(128) → Dense(7, softmax)
- Loss: focal loss (γ=4.0 + class weights for sound, γ=2.0 for diagnosis)
- Loss weights: sound=1.0, diagnosis=0.1
- Early stopping on val_icbhi (patience=20)

#### Why Multitask
The shared backbone learns features useful for both tasks simultaneously. A feature useful for detecting COPD (which causes crackles) also helps the sound head detect crackles. This acts as regularisation — critical with only 2,891 training samples.

#### Why Focal Loss (not crossentropy)
Focal loss = `-(1 - p_t)^γ × log(p_t)`. The `(1-p_t)^γ` factor down-weights easy examples (Normal, which the model predicts correctly with high confidence) and focuses training on hard, rare examples (Wheeze, Both). Combined with class weights for additional imbalance correction.

#### Why ResNet was abandoned
A ResNet multitask model (with skip connections, mixup augmentation, label smoothing) was tested (`train_resnet.py`) but ICBHI collapsed to 0.50. Skip connections and mixup added too much complexity for a 2,891-sample dataset. The simpler baseline CNN architecture generalises better at this data scale.

### 5. Evaluation
- **ICBHI Score** = (mean Sensitivity + mean Specificity) / 2, averaged across all 4 sound classes
- Per-class precision, recall, F1
- Confusion matrix
- Training curves (accuracy, loss, ICBHI per epoch)

### 6. Export
TFLite int8 quantization for deployment on mobile/IoT devices.

## Results

| Model | ICBHI | Crackle Recall | Normal Recall | Notes |
|-------|-------|---------------|---------------|-------|
| Baseline CNN | 59.80% | — | — | Sound only, 2,891 samples |
| ResNet Multitask | ~50% | — | — | Collapsed — abandoned |
| **Multitask CNN** | **62.26%** | 21% | 97% | Sound + diagnosis, 2,891 samples |
| Multitask + Synth v1 | 61.05% | 48% | 51% | 1,500 synthetic samples — Normal collapsed |
| Multitask + Synth v2 | 56.28% | — | — | Asymmetric focal loss — unstable |
| Highres CNN (raw) | 58.00% | 71% | 43% | hop_length=256, 6,415 samples — crackle recall 3.4x improvement |
| **Highres CNN + Threshold Tuning** | **62.01%** | 59% | 49% | Post-hoc class weight tuning — all 4 classes detected |

> **Final result:** 62.01% ICBHI with balanced per-class detection. Crackle recall improved from 21% (midterm) to 59% while maintaining competitive overall score. All 4 sound classes now detected vs midterm where Both recall was 0%.

> **Threshold tuning:** After training, class probability weights [Normal=0.8, Crackle=0.5, Wheeze=2.0, Both=10.0] applied before argmax to rebalance predictions without retraining.

> **Metric:** ICBHI Score = (Sensitivity + Specificity) / 2, averaged across all classes. Accuracy is misleading due to class imbalance (57% Normal). A model predicting Normal for everything gets 57% accuracy but ICBHI of 50%.

> **State of the art** on ICBHI 2017 is ~65–72%, achieved with pretrained audio transformers (PANNs, AST) trained on millions of clips. Our CNN from scratch on 2,891 samples achieving 62.26% is competitive.

## Setup

```bash
pip install tensorflow librosa soundfile scikit-learn pandas numpy matplotlib seaborn
```

### Train baseline CNN
```bash
python train_baseline.py
```

### Train multitask CNN (proven baseline — 62.26% ICBHI)
```bash
python multitask_model.py
```

### Train high-resolution model (current best attempt)
```bash
# Step 1: Re-extract features with hop_length=256
python reextract_features.py

# Step 2: Generate modest augmentation (2-5x minority classes)
python augment_modest.py

# Step 3: Train
python train_highres.py
```

### Generate synthetic data (experimental)
```bash
python generate_synthetic.py       # generates 700 synthetic samples
python train_with_synthetic.py     # trains with synthetic + real data
```

### Run on Google Colab
Mount Drive with preprocessed data, then run training scripts in a Colab cell for GPU acceleration (~3x faster than CPU).

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Patient-level splits | Prevents data leakage — same patient's recordings stay in one split |
| Butterworth filter | Maximally flat response, no ripple distortion of lung sounds |
| Mel spectrogram | Logarithmic frequency scale matches human auditory perception |
| GlobalAveragePooling | Reduces overfitting vs Flatten — fewer parameters |
| Focal loss + class weights | Dual mechanism for class imbalance — sample-level and class-level correction |
| Multitask learning | Shared backbone regularised by two objectives simultaneously |
| diagnosis weight=0.1 | Sound is primary task — diagnosis acts as regulariser, not co-equal objective |
| Modest augmentation (2–5x) | 21k augmented data caused collapse; 2–5x provides diversity without duplication |
| val_icbhi early stopping | Accuracy is misleading metric; ICBHI directly measures per-class balance |
| hop_length=256 | Doubles temporal resolution — crackle bursts span 3+ frames instead of 1 |
| Real augmentation over synthetic | Mathematical synthesis creates out-of-distribution features that don't transfer to real audio |

## Team

TY Engineering Group Project
