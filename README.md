# Respiratory Disorder Detection

ML pipeline for classifying respiratory sounds from the ICBHI 2017 dataset using CNNs.

## Problem Statement

Classify lung sound recordings into 4 categories:
- **Normal** — healthy lungs
- **Crackle** — indicates fluid or infection
- **Wheeze** — indicates narrowed airways
- **Both** — crackle + wheeze combined

Secondary task: predict patient diagnosis (COPD, Pneumonia, URTI, etc.)

## Dataset

**ICBHI 2017** — 101 patients, 5,319 respiratory cycles recorded across multiple chest locations.

> **Data is not included in this repo** (too large for GitHub).
> Preprocessed `.npy` features and manifests are available on Google Drive (contact repo owner).
> Raw dataset available at: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

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
├── main.py                     # CLI entrypoint
├── configs/
│   └── experiment_config.json  # Hyperparameters
├── src/
│   ├── data_loader.py          # Patient-level train/val/test splits
│   ├── preprocessing.py        # Butterworth filter, mel-spectrogram, MFCC
│   ├── augmentation.py         # Audio augmentation + SpecAugment
│   ├── models.py               # CNN architectures (baseline, 1D, MobileNet-style)
│   ├── multitask_model.py      # Shared backbone + sound + diagnosis heads
│   ├── training.py             # Focal loss, cosine LR, ICBHI score tracking
│   ├── evaluation.py           # Confusion matrix, metrics, training curves
│   └── export_tflite.py        # TFLite int8 quantization export
├── augment_minority.py         # Audio augmentation script for minority classes
└── data/                       # Not tracked in git — see Google Drive
    └── processed/
        ├── manifest.csv        # Original splits (5,319 samples)
        ├── manifest_aug.csv    # Augmented splits (~21k samples)
        ├── train/              # Mel spectrogram .npy files (train)
        ├── val/                # Mel spectrogram .npy files (val)
        ├── test/               # Mel spectrogram .npy files (test)
        └── train_aug_v2/       # Augmented .npy files (~18k extra)
```

## Pipeline

1. **Patient-level splitting** — 63/19/19 patients for train/val/test (no data leakage)
2. **Feature extraction** — Butterworth bandpass filter (100–2000 Hz) → mel-spectrogram (128×63)
3. **Augmentation** — Time stretch, pitch shift, gaussian noise on minority classes
4. **Training** — Focal loss (γ=2.0), cosine LR decay, ICBHI score tracking
5. **Evaluation** — Per-class metrics, confusion matrix, ICBHI score
6. **Export** — TFLite int8 quantization for mobile deployment

## Results

| Model | ICBHI Score |
|-------|-------------|
| Baseline CNN (no augmentation) | 58.61% |
| Multitask CNN (5k samples) | 57.68% |
| Multitask CNN (21k augmented) | In progress |

> **Metric:** ICBHI Score = (Sensitivity + Specificity) / 2, averaged across all classes.
> Accuracy alone is misleading due to class imbalance (68% of samples are Normal).

## Setup

```bash
pip install tensorflow librosa soundfile scikit-learn pandas numpy matplotlib seaborn
```

### Run locally
```bash
python main.py --phase train --config configs/experiment_config.json
```

### Run on Google Colab
Mount Drive with preprocessed data, then use the self-contained training cells (see Colab notebook).

## Key Design Decisions

- **Patient-level splits** — prevents data leakage (same patient's recordings stay in one split)
- **Focal loss** — handles class imbalance without needing class weights
- **Multitask learning** — joint sound + diagnosis prediction forces richer feature learning
- **Augmentation on train only** — val/test sets are never augmented

## Team

TY Engineering Group Project
