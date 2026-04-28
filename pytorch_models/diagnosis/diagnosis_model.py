"""
Respiratory Diagnosis Pipeline  (revised)
==========================================
Changes from original
----------------------
  1. CSV has NO header — first col = Patient ID, second = Diagnosis
  2. Single audio directory containing both .wav and .txt annotation files
     (same base name, e.g. 102_1b1_Ar_sc_Meditron.wav / .txt)
  3. Ground-truth annotation files used to build cycle metadata during
     training; frozen Cycle CNN (v2) used only at inference time
  4. Rare classes merged:
       LRTI + Asthma + Bronchiolitis + Bronchiectasis  →  'Other'
  5. Stratified 80/20 train/test split performed in code
  6. Annotation columns:  start  end  crackle  wheeze
       (1,0)→crackle  (0,1)→wheeze  (1,1)→both  (0,0)→normal

Architecture
-------------
  Full audio WAV
    │
    ├── Annotation .txt  (train)  OR  Cycle CNN v2  (inference)
    │       └── [n_normal, n_crackle, n_wheeze, n_both]  — normalised fractions
    │
    └── Full-recording mel spectrogram
            └── EfficientNet-B0 backbone → 1280-d
                    ↓ concat with 4-d metadata → 1284-d
                    MLP → N diagnoses

Requirements
------------
    pip install torch torchvision torchaudio librosa scikit-learn matplotlib seaborn tqdm pandas
"""

import os, random, shutil
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Single folder that holds BOTH .wav and .txt files
    "audio_dir":        "D:/LY_Project/Respiratory_Sound_Database/audio_and_txt_files",
    "diagnosis_csv":    "D:/LY_Project/Respiratory_Sound_Database/patient_diagnosis.csv",           # NO header; col0=PatientID, col1=Diagnosis
    "cycle_model_path": r"best_respiratory_v2.pt",  # frozen cycle CNN (inference only)

    # ── Audio ──────────────────────────────────────────────────────────────
    "sample_rate":      22050,
    "full_duration":    20.0,   # max seconds of full recording to use
    "cycle_duration":   5.0,    # fixed length each cycle is padded/trimmed to
    "n_mels":           128,
    "n_fft":            1024,
    "hop_length":       512,
    "f_min":            50,
    "f_max":            2000,

    # ── Cycle CNN (v2) class order — must match training ──────────────────
    "cycle_classes":    ["normal", "crackle", "wheeze", "both"],

    # ── Diagnosis classes (merged) ─────────────────────────────────────────
    # LRTI, Asthma, Bronchiolitis, Bronchiectasis → 'Other'
    "diag_classes": ["Healthy", "COPD", "URTI", "Pneumonia", "Other"],

    # Which raw CSV labels map to 'Other'
    "other_labels": {"LRTI", "Asthma", "Bronchiolitis", "Bronchiectasis"},

    # ── Train / test split ─────────────────────────────────────────────────
    "test_size":        0.20,   # stratified 80/20
    "split_seed":       42,

    # ── Model ──────────────────────────────────────────────────────────────
    "dropout":          0.5,
    "mlp_hidden":       256,

    # ── Training ───────────────────────────────────────────────────────────
    "batch_size":       8,
    "num_epochs":       60,
    "lr":               5e-4,
    "weight_decay":     1e-3,
    "grad_clip":        2.0,
    "patience":         15,
    "kfold":            5,
    "num_workers":      0,
    "seed":             42,

    # ── Augmentation (full spectrogram) ────────────────────────────────────
    "time_mask_pct":    0.10,
    "freq_mask_pct":    0.10,
    "noise_std":        0.01,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_DIAG   = len(CONFIG["diag_classes"])
N_CYCLE  = len(CONFIG["cycle_classes"])  # 4
META_DIM = N_CYCLE                       # [n_normal, n_crackle, n_wheeze, n_both]


# ─────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])


# ─────────────────────────────────────────────
# 3. CSV READER
# ─────────────────────────────────────────────

def load_diagnosis_map(csv_path: str, cfg: dict) -> dict:
    """
    CSV has NO header.
    Column 0 = Patient ID (int or string)
    Column 1 = Diagnosis string

    Rare classes are merged into 'Other'.
    Returns {zero_padded_patient_id_str: label_int}
    """
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["patient_id", "diagnosis"]
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.zfill(3)
    df["diagnosis"]  = df["diagnosis"].str.strip()

    # Merge rare classes
    df["diagnosis"] = df["diagnosis"].apply(
        lambda d: "Other" if d in cfg["other_labels"] else d
    )

    diag_lower  = {d.lower(): i for i, d in enumerate(cfg["diag_classes"])}
    mapping     = {}
    unknown     = set()

    for _, row in df.iterrows():
        dx = row["diagnosis"].lower()
        if dx in diag_lower:
            mapping[row["patient_id"]] = diag_lower[dx]
        else:
            unknown.add(row["diagnosis"])

    if unknown:
        print(f"  WARNING — unrecognised diagnoses (skipped): {unknown}")

    print(f"  Loaded {len(mapping)} patient→diagnosis mappings")
    counts = {}
    for pid, lbl in mapping.items():
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, n in sorted(counts.items()):
        print(f"    {cfg['diag_classes'][lbl]:16s}: {n}")

    return mapping


def patient_id_from_filename(filename: str) -> str:
    """'102_1b1_Ar_sc_Meditron.wav' → '102'"""
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[0].strip().zfill(3)


# ─────────────────────────────────────────────
# 4. ANNOTATION READER  (ground-truth cycles)
# ─────────────────────────────────────────────

def parse_annotation_file(txt_path: str) -> np.ndarray:
    """
    Reads an ICBHI-style annotation file.

    Each row:  start_sec  end_sec  crackle(0/1)  wheeze(0/1)

    Returns a normalised 4-d vector:
        [frac_normal, frac_crackle, frac_wheeze, frac_both]

    If the file is missing or empty, returns zeros (handled gracefully).
    """
    counts = np.zeros(N_CYCLE, dtype=np.float32)

    if not os.path.isfile(txt_path):
        return counts  # no annotation → all-zero metadata

    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            crackle = int(parts[2])
            wheeze  = int(parts[3])
        except ValueError:
            continue

        if   crackle == 1 and wheeze == 1:
            counts[3] += 1   # both
        elif crackle == 1:
            counts[1] += 1   # crackle
        elif wheeze  == 1:
            counts[2] += 1   # wheeze
        else:
            counts[0] += 1   # normal

    total = counts.sum()
    if total > 0:
        counts /= total   # normalise to fractions

    return counts


# ─────────────────────────────────────────────
# 5. AUDIO UTILITIES
# ─────────────────────────────────────────────

def load_audio(path: str, sr: int, duration: float) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * duration)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    else:
        wav = wav[:n]
    return wav


def wav_to_melspec(wav: np.ndarray, cfg: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav, sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        fmin=cfg["f_min"], fmax=cfg["f_max"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    lo, hi  = log_mel.min(), log_mel.max()
    return (log_mel - lo) / (hi - lo + 1e-8)


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    return torch.tensor(spec).unsqueeze(0).repeat(3, 1, 1)


def augment_spec(spec: np.ndarray, cfg: dict) -> np.ndarray:
    spec = spec.copy()
    _, T  = spec.shape
    ml = int(T * cfg["time_mask_pct"] * random.random())
    if ml > 0:
        t0 = random.randint(0, T - ml)
        spec[:, t0:t0+ml] = 0.0
    F, _ = spec.shape
    ml = int(F * cfg["freq_mask_pct"] * random.random())
    if ml > 0:
        f0 = random.randint(0, F - ml)
        spec[f0:f0+ml, :] = 0.0
    spec += cfg["noise_std"] * np.random.randn(*spec.shape).astype(np.float32)
    return np.clip(spec, 0.0, 1.0)


# ─────────────────────────────────────────────
# 6. CYCLE CNN  (frozen v2 — inference only)
# ─────────────────────────────────────────────

class CycleCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        w  = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        bb = models.efficientnet_b0(weights=w)
        in_f = bb.classifier[1].in_features
        bb.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self.net = bb

    def forward(self, x):
        return self.net(x)


def load_cycle_cnn(model_path: str, num_classes: int, device) -> CycleCNN:
    model = CycleCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print(f"Cycle CNN loaded from {model_path}  (frozen, inference only)")
    return model


# ─────────────────────────────────────────────
# 7. CYCLE SPLITTER  (used only at inference)
# ─────────────────────────────────────────────

def split_into_cycles(wav: np.ndarray, sr: int, cfg: dict) -> list:
    """Energy-based breathing cycle segmentation (inference fallback)."""
    frame_len = 512
    hop       = 256
    rms       = librosa.feature.rms(y=wav, frame_length=frame_len, hop_length=hop)[0]
    threshold = 0.02 * rms.max()
    active    = (rms > threshold).astype(int)

    min_sil_frames = int(0.15 * sr / hop)
    min_cyc_frames = int(1.0  * sr / hop)
    max_cyc_frames = int(8.0  * sr / hop)

    i = 0
    while i < len(active):
        if active[i] == 0:
            j = i
            while j < len(active) and active[j] == 0:
                j += 1
            if (j - i) < min_sil_frames:
                active[i:j] = 1
            i = j
        else:
            i += 1

    segments = []
    in_seg, start = False, 0
    for idx, val in enumerate(active):
        if val == 1 and not in_seg:
            in_seg, start = True, idx
        elif val == 0 and in_seg:
            in_seg = False
            dur = idx - start
            if min_cyc_frames <= dur <= max_cyc_frames:
                s = librosa.frames_to_samples(start, hop_length=hop)
                e = librosa.frames_to_samples(idx,   hop_length=hop)
                segments.append(wav[s:e])
    if in_seg:
        dur = len(active) - start
        if min_cyc_frames <= dur <= max_cyc_frames:
            s = librosa.frames_to_samples(start, hop_length=hop)
            segments.append(wav[s:])

    if len(segments) < 2:
        cs = int(cfg["cycle_duration"] * sr)
        segments = [wav[i:i+cs] for i in range(0, len(wav) - cs + 1, cs)]

    return segments


@torch.no_grad()
def extract_cycle_metadata_inference(wav_full: np.ndarray,
                                     cycle_cnn: CycleCNN,
                                     cfg: dict, device) -> np.ndarray:
    """
    Used at inference time only.
    Splits recording into cycles via energy segmentation,
    classifies each with frozen v2 CNN.
    Returns normalised 4-d fraction vector.
    """
    cycles = split_into_cycles(wav_full, cfg["sample_rate"], cfg)
    counts = np.zeros(N_CYCLE, dtype=np.float32)
    n      = int(cfg["sample_rate"] * cfg["cycle_duration"])

    for seg in cycles:
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        else:
            seg = seg[:n]
        spec   = wav_to_melspec(seg, cfg)
        tensor = spec_to_tensor(spec).unsqueeze(0).to(device)
        pred   = int(cycle_cnn(tensor).argmax(1).item())
        counts[pred] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ─────────────────────────────────────────────
# 8. DIAGNOSIS MODEL
# ─────────────────────────────────────────────

class DiagnosisCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        w  = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        bb = models.efficientnet_b0(weights=w)
        self.features = bb.features
        self.avgpool  = bb.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)   # (B, 1280)


class DiagnosisModel(nn.Module):
    def __init__(self, num_diag: int, mlp_hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.backbone   = DiagnosisCNNBackbone()
        feat_dim        = 1280 + META_DIM  # 1284

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, num_diag),
        )

    def forward(self, spec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        audio_feat = self.backbone(spec)
        combined   = torch.cat([audio_feat, meta], dim=1)
        return self.classifier(combined)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  Backbone unfrozen (lr ×0.05 applied via param groups)")


# ─────────────────────────────────────────────
# 9. DATASET
# ─────────────────────────────────────────────

class DiagnosisDataset(Dataset):
    """
    Each item = (mel_spec_tensor, cycle_meta_tensor, label_int)

    cycle_meta is read from the ground-truth .txt annotation file
    (same directory as the .wav, same base name).

    If a .txt file is missing, the metadata vector is all-zeros
    (treated as 'all normal') — a warning is printed once.
    """

    def __init__(self, wav_paths: list, diag_map: dict, audio_dir: str,
                 cfg: dict, augment: bool = False):
        self.cfg     = cfg
        self.augment = augment
        self.samples = []   # (wav_path, meta_np, label_int)
        missing_txt  = []

        for wav_path in wav_paths:
            fname = os.path.basename(wav_path)
            pid   = patient_id_from_filename(fname)
            if pid not in diag_map:
                continue
            label    = diag_map[pid]
            txt_path = os.path.join(audio_dir,
                                    os.path.splitext(fname)[0] + ".txt")
            if not os.path.isfile(txt_path):
                missing_txt.append(fname)
            meta = parse_annotation_file(txt_path)
            self.samples.append((wav_path, meta, label))

        if missing_txt:
            print(f"  WARNING — {len(missing_txt)} files missing .txt annotation "
                  f"(metadata set to zeros): {missing_txt[:5]}{'...' if len(missing_txt)>5 else ''}")
        print(f"  Dataset built: {len(self.samples)} samples  "
              f"(augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, meta, label = self.samples[idx]
        wav  = load_audio(path, self.cfg["sample_rate"], self.cfg["full_duration"])
        spec = wav_to_melspec(wav, self.cfg)
        if self.augment:
            spec = augment_spec(spec, self.cfg)
        return (spec_to_tensor(spec),
                torch.tensor(meta, dtype=torch.float32),
                label)

    def labels(self):
        return [l for _, _, l in self.samples]

    def class_weights(self):
        lbls   = self.labels()
        counts = np.bincount(lbls, minlength=N_DIAG).astype(float)
        w      = 1.0 / (counts + 1e-6)
        return torch.tensor(w / w.sum(), dtype=torch.float)


# ─────────────────────────────────────────────
# 10. BUILD TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def build_splits(audio_dir: str, diag_map: dict, cfg: dict):
    """
    Collects all .wav files in audio_dir that have a matching diagnosis,
    then performs a stratified 80/20 split at the FILE level.

    Returns (train_wav_paths, test_wav_paths).
    """
    all_wavs  = sorted(
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    )
    # Keep only files with a known patient ID
    valid_wavs = []
    valid_lbls = []
    for wp in all_wavs:
        pid = patient_id_from_filename(os.path.basename(wp))
        if pid in diag_map:
            valid_wavs.append(wp)
            valid_lbls.append(diag_map[pid])

    print(f"\nTotal labelled .wav files found: {len(valid_wavs)}")

    # Stratified split
    tr_wavs, te_wavs, _, _ = train_test_split(
        valid_wavs, valid_lbls,
        test_size=cfg["test_size"],
        stratify=valid_lbls,
        random_state=cfg["split_seed"],
    )
    print(f"Train: {len(tr_wavs)}  |  Test: {len(te_wavs)}")

    # Print per-class counts
    tr_counts = {}
    te_counts = {}
    for wp in tr_wavs:
        lbl = diag_map[patient_id_from_filename(os.path.basename(wp))]
        tr_counts[lbl] = tr_counts.get(lbl, 0) + 1
    for wp in te_wavs:
        lbl = diag_map[patient_id_from_filename(os.path.basename(wp))]
        te_counts[lbl] = te_counts.get(lbl, 0) + 1

    print(f"\n{'Diagnosis':20s}  {'Train':>6}  {'Test':>6}")
    print("─" * 36)
    for i, name in enumerate(cfg["diag_classes"]):
        print(f"  {name:18s}  {tr_counts.get(i,0):6d}  {te_counts.get(i,0):6d}")

    return tr_wavs, te_wavs


# ─────────────────────────────────────────────
# 11. FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits, labels):
        log_p   = F.log_softmax(logits, dim=1)
        pt      = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)
        log_pt  = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[labels]
        loss    = -alpha_t * (1 - pt) ** self.gamma * log_pt
        return loss.mean()


# ─────────────────────────────────────────────
# 12. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, metas, labels in loader:
        specs, metas, labels = specs.to(device), metas.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs, metas)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(specs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(specs)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for specs, metas, labels in loader:
        specs, metas = specs.to(device), metas.to(device)
        preds = model(specs, metas).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# 13. K-FOLD TRAINING
# ─────────────────────────────────────────────

def train_kfold(train_ds: DiagnosisDataset, cfg: dict):
    """
    Stratified k-fold cross-validation on the training set.
    The best-fold weights are saved as 'best_diagnosis_model.pt'.
    """
    labels    = np.array(train_ds.labels())
    skf       = StratifiedKFold(n_splits=cfg["kfold"], shuffle=True,
                                random_state=cfg["seed"])
    fold_accs = []
    best_acc  = 0.0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold+1}/{cfg['kfold']}  "
              f"(train={len(tr_idx)}, val={len(va_idx)})")
        print(f"{'─'*60}")

        tr_loader = DataLoader(Subset(train_ds, tr_idx),
                               batch_size=cfg["batch_size"],
                               shuffle=True,
                               num_workers=cfg["num_workers"])
        va_loader = DataLoader(Subset(train_ds, va_idx),
                               batch_size=cfg["batch_size"],
                               shuffle=False,
                               num_workers=cfg["num_workers"])

        model = DiagnosisModel(
            num_diag=N_DIAG,
            mlp_hidden=cfg["mlp_hidden"],
            dropout=cfg["dropout"],
        ).to(DEVICE)

        # Stage 1: head only
        model.freeze_backbone()
        alpha     = train_ds.class_weights().to(DEVICE)
        criterion = FocalLoss(alpha, gamma=2.0)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["num_epochs"], eta_min=1e-6
        )

        best_fold_acc  = 0.0
        patience_count = 0
        stage2_started = False

        for epoch in range(1, cfg["num_epochs"] + 1):

            # Stage 2: unfreeze backbone at epoch 10
            if epoch == 10 and not stage2_started:
                model.unfreeze_backbone()
                stage2_started = True
                optimizer = optim.AdamW([
                    {"params": model.backbone.parameters(),
                     "lr": cfg["lr"] * 0.05},
                    {"params": model.classifier.parameters(),
                     "lr": cfg["lr"]},
                ], weight_decay=cfg["weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["num_epochs"] - 10, eta_min=1e-6
                )

            tr_loss, tr_acc = train_epoch(
                model, tr_loader, criterion, optimizer, DEVICE, cfg["grad_clip"]
            )
            va_preds, va_labels = eval_epoch(model, va_loader, DEVICE)
            va_acc = (va_preds == va_labels).mean()
            scheduler.step()

            flag = ""
            if va_acc > best_fold_acc:
                best_fold_acc  = va_acc
                patience_count = 0
                torch.save(model.state_dict(), f"fold_{fold+1}_best.pt")
                flag = " ✓"
            else:
                patience_count += 1

            if epoch % 5 == 0 or flag:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Ep {epoch:03d}  lr={lr:.1e}  "
                      f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                      f"va_acc={va_acc:.3f}{flag}")

            if patience_count >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

        fold_accs.append(best_fold_acc)
        print(f"  Fold {fold+1} best val acc: {best_fold_acc:.4f}")

        if best_fold_acc > best_acc:
            best_acc = best_fold_acc
            shutil.copy(f"fold_{fold+1}_best.pt", "best_diagnosis_model.pt")

    print(f"\nK-fold results  : {[f'{a:.3f}' for a in fold_accs]}")
    print(f"Mean ± std      : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"Best model saved: best_diagnosis_model.pt")
    return best_acc


# ─────────────────────────────────────────────
# 14. PLOTS
# ─────────────────────────────────────────────

def plot_confusion(labels, preds, class_names, title="Test set"):
    active = sorted(set(labels) | set(preds))
    names  = [class_names[i] for i in active]
    cm     = confusion_matrix(labels, preds, labels=active)
    cm_n   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, t in zip(
        axes, [cm, cm_n], ["d", ".2f"],
        [f"Count — {title}", f"Recall — {title}"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(t)
    plt.tight_layout()
    fname = f"confusion_diagnosis_{title.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved → {fname}")


def plot_cycle_metadata(dataset: DiagnosisDataset, class_names: list):
    """Bar chart: average cycle-type fractions per diagnosis class."""
    from collections import defaultdict
    sums = defaultdict(lambda: np.zeros(N_CYCLE))
    cnts = defaultdict(int)
    for _, meta, label in dataset.samples:
        sums[label] += meta
        cnts[label] += 1
    avgs = {lbl: sums[lbl] / cnts[lbl] for lbl in sums}

    labels_present = sorted(avgs.keys())
    diag_names     = [class_names[l] for l in labels_present]
    cycle_names    = CONFIG["cycle_classes"]
    colors         = ["#1D9E75", "#D85A30", "#378ADD", "#7F77DD"]

    x   = np.arange(len(labels_present))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(labels_present) * 1.8), 5))
    for i, (cname, col) in enumerate(zip(cycle_names, colors)):
        vals = [avgs[l][i] for l in labels_present]
        ax.bar(x + i * w, vals, w, label=cname, color=col, alpha=0.85)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(diag_names, rotation=30, ha="right")
    ax.set_ylabel("Mean fraction of cycles")
    ax.set_title("Cycle type distribution per diagnosis (ground-truth annotations)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("cycle_metadata_by_diagnosis.png", dpi=150); plt.close()
    print("Saved → cycle_metadata_by_diagnosis.png")


# ─────────────────────────────────────────────
# 15. MAIN
# ─────────────────────────────────────────────

def main(cfg=CONFIG):
    # ── Load diagnosis CSV (no header) ────────────────────────────────────
    diag_map = load_diagnosis_map(cfg["diagnosis_csv"], cfg)

    # ── Stratified 80/20 split ────────────────────────────────────────────
    tr_wavs, te_wavs = build_splits(cfg["audio_dir"], diag_map, cfg)

    # ── Build datasets (metadata from ground-truth .txt files) ────────────
    train_ds = DiagnosisDataset(
        tr_wavs, diag_map, cfg["audio_dir"], cfg, augment=True
    )
    test_ds  = DiagnosisDataset(
        te_wavs, diag_map, cfg["audio_dir"], cfg, augment=False
    )

    # ── Visualise cycle metadata ───────────────────────────────────────────
    plot_cycle_metadata(train_ds, cfg["diag_classes"])

    # ── K-fold training ───────────────────────────────────────────────────
    train_kfold(train_ds, cfg)

    # ── Final test evaluation ─────────────────────────────────────────────
    model = DiagnosisModel(N_DIAG, cfg["mlp_hidden"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(
        torch.load("best_diagnosis_model.pt", map_location=DEVICE)
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"]
    )
    all_preds, all_labels = eval_epoch(model, test_loader, DEVICE)

    present = sorted(set(all_labels))
    names   = [cfg["diag_classes"][i] for i in present]

    print("\n" + "═" * 55)
    print("FINAL TEST RESULTS")
    print("═" * 55)
    print(classification_report(all_labels, all_preds,
                                labels=present, target_names=names))
    plot_confusion(all_labels, all_preds, cfg["diag_classes"])


# ─────────────────────────────────────────────
# 16. SINGLE-PATIENT INFERENCE
# ─────────────────────────────────────────────

def predict_patient(wav_path: str,
                    cycle_model_path: str = CONFIG["cycle_model_path"],
                    diag_model_path:  str = "best_diagnosis_model.pt",
                    cfg: dict = CONFIG):
    """
    Full inference for one new patient recording.

    At inference time we do NOT have a .txt annotation file, so the
    frozen Cycle CNN v2 predicts the cycle breakdown instead.

    Usage
    -----
        pred, meta, probs = predict_patient("new_patient.wav")
    """
    device    = torch.device("cpu")
    cycle_cnn = load_cycle_cnn(cycle_model_path, N_CYCLE, device)

    wav_full  = load_audio(wav_path, cfg["sample_rate"], cfg["full_duration"])
    meta      = extract_cycle_metadata_inference(wav_full, cycle_cnn, cfg, device)

    print(f"\nFile: {os.path.basename(wav_path)}")
    print("Cycle breakdown (v2 CNN predictions):")
    for cls, frac in zip(cfg["cycle_classes"], meta):
        bar = "█" * int(frac * 30)
        print(f"  {cls:8s} {frac:.3f}  {bar}")

    diag_model = DiagnosisModel(N_DIAG, cfg["mlp_hidden"], cfg["dropout"])
    diag_model.load_state_dict(torch.load(diag_model_path, map_location="cpu"))
    diag_model.eval()

    spec  = wav_to_melspec(wav_full, cfg)
    s_ten = spec_to_tensor(spec).unsqueeze(0)
    m_ten = torch.tensor(meta).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(diag_model(s_ten, m_ten), dim=1).squeeze().numpy()

    pred = cfg["diag_classes"][probs.argmax()]
    print(f"\nDiagnosis prediction: {pred}")
    print("Top probabilities:")
    for cls, p in sorted(zip(cfg["diag_classes"], probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:16s} {p:.4f}  {bar}")

    return pred, meta, probs


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()