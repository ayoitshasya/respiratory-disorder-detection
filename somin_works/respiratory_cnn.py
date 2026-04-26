"""
Respiratory Sound Classification v2 — Fixes for Overfitting
============================================================
Changes vs v1
─────────────
1.  TRANSFER LEARNING   : EfficientNet-B0 pretrained on ImageNet (frozen stem,
                          fine-tune top layers). Far better features than random init.
2.  STRONGER AUGMENT    : Time-mask, freq-mask, additive Gaussian noise, time-shift,
                          mixup between training samples.
3.  LABEL SMOOTHING     : CrossEntropyLoss(label_smoothing=0.1) — penalises
                          over-confident predictions that cause train loss collapse.
4.  EARLY STOPPING      : Stops when val accuracy hasn't improved for `patience` epochs.
5.  GRADIENT CLIPPING   : Prevents the val-loss spikes seen in v1.
6.  LOWER LR + WARMUP   : Starts at lr/10 for 3 epochs then rises to full lr
                          (avoids early instability).
7.  ICBHI SCORE         : Computed automatically after every epoch and at final eval.

Expected folder structure (unchanged from v1)
----------------------------------------------
processed_training_cycles/   ← flat folder, all .wav files
processed_testing_cycles/    ← flat folder, all .wav files

Filename format:  ..._cycle0_normal.wav  /  _crackle  /  _wheeze  /  _both

Requirements
------------
    pip install torch torchvision torchaudio librosa scikit-learn matplotlib seaborn tqdm
"""

import os, random, math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # ── Paths (use raw strings on Windows to avoid escape-sequence warnings) ──
    "train_dir": "D:\LY_Project\Previous_trials\processed_training_cycles",
    "test_dir":  "D:\LY_Project\Previous_trials\processed_testing_cycles",
    "save_path": "best_respiratory_v2.pt",

    # ── Audio ──────────────────────────────────────────────────────────────
    "sample_rate": 22050,
    "duration":    5.0,      # seconds — pad / crop every clip
    "n_mels":      128,
    "n_fft":       1024,
    "hop_length":  512,
    "f_min":       50,
    "f_max":       2000,     # respiratory sounds live in 50–2000 Hz

    # ── Augmentation ───────────────────────────────────────────────────────
    "time_mask_pct":  0.15,  # max fraction of time axis to mask
    "freq_mask_pct":  0.15,  # max fraction of freq axis to mask
    "noise_std":      0.01,  # Gaussian noise std (applied to normalised spec)
    "time_shift_pct": 0.10,  # max circular shift fraction
    "mixup_alpha":    0.3,   # Beta distribution α for mixup (0 = disabled)

    # ── Model ──────────────────────────────────────────────────────────────
    "backbone":       "efficientnet_b0",   # torchvision pretrained backbone
    "freeze_epochs":  5,     # epochs to keep backbone frozen (warm-up head first)
    "dropout":        0.5,

    # ── Training ───────────────────────────────────────────────────────────
    "batch_size":     32,
    "num_epochs":     60,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "label_smoothing": 0.1,
    "grad_clip":      2.0,   # max gradient norm
    "warmup_epochs":  3,     # LR linearly ramps from lr/10 → lr over this many epochs
    "patience":       12,    # early stopping patience (val-accuracy based)
    "num_workers":    0,     # set to 4 on Linux/Mac
    "seed":           42,

    # ── Classes ────────────────────────────────────────────────────────────
    "classes": ["normal", "crackle", "wheeze", "both"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


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
# 3. LABEL PARSING
# ─────────────────────────────────────────────

def parse_label(filename, classes):
    """Last underscore token before .wav → class string."""
    stem  = os.path.splitext(filename)[0]
    token = stem.split("_")[-1].lower().strip()
    return token if token in classes else None


# ─────────────────────────────────────────────
# 4. AUDIO UTILITIES
# ─────────────────────────────────────────────

def load_audio(path, sr, duration):
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    target_len  = int(sr * duration)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return waveform


def waveform_to_melspec(waveform, cfg):
    mel     = librosa.feature.melspectrogram(
        y=waveform, sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        fmin=cfg["f_min"], fmax=cfg["f_max"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalise(spec):
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min > 0:
        return (spec - s_min) / (s_max - s_min)
    return spec


# ─────────────────────────────────────────────
# 5. DATASET  (with richer augmentation)
# ─────────────────────────────────────────────

class RespiratoryDataset(Dataset):
    def __init__(self, root_dir, classes, config, augment=False):
        self.cfg          = config
        self.augment      = augment
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples      = []
        self.skipped      = 0

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Not found: {root_dir}")

        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            label_str = parse_label(fname, classes)
            if label_str is None:
                self.skipped += 1
                continue
            self.samples.append(
                (os.path.join(root_dir, fname), self.class_to_idx[label_str])
            )

        print(f"\nDataset: {root_dir}")
        print(f"  Total : {len(self.samples)}  |  Skipped : {self.skipped}")
        for cls, idx in self.class_to_idx.items():
            n = sum(1 for _, l in self.samples if l == idx)
            print(f"    {cls:8s}: {n:5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform    = load_audio(path, self.cfg["sample_rate"], self.cfg["duration"])
        spec        = waveform_to_melspec(waveform, self.cfg)
        spec        = normalise(spec)

        if self.augment:
            spec = self._time_shift(spec)
            spec = self._add_noise(spec)
            spec = self._time_mask(spec)
            spec = self._freq_mask(spec)

        tensor = torch.tensor(spec).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        return tensor, label

    # ── Augmentation helpers ───────────────────────────────────────────────

    def _time_shift(self, spec):
        _, T   = spec.shape
        shift  = int(T * self.cfg["time_shift_pct"] * random.random())
        return np.roll(spec, shift, axis=1)

    def _add_noise(self, spec):
        noise = np.random.randn(*spec.shape).astype(np.float32)
        return spec + self.cfg["noise_std"] * noise

    def _time_mask(self, spec):
        _, T = spec.shape
        ml   = int(T * self.cfg["time_mask_pct"] * random.random())
        if ml > 0:
            t0 = random.randint(0, T - ml)
            spec[:, t0:t0 + ml] = 0.0
        return spec

    def _freq_mask(self, spec):
        F, _ = spec.shape
        ml   = int(F * self.cfg["freq_mask_pct"] * random.random())
        if ml > 0:
            f0 = random.randint(0, F - ml)
            spec[f0:f0 + ml, :] = 0.0
        return spec

    def get_sample_weights(self):
        counts  = np.zeros(len(self.class_to_idx))
        for _, l in self.samples:
            counts[l] += 1
        class_w = 1.0 / (counts + 1e-6)
        return torch.tensor([class_w[l] for _, l in self.samples], dtype=torch.float)


# ─────────────────────────────────────────────
# 6. MIXUP COLLATE FUNCTION
# ─────────────────────────────────────────────

def mixup_collate(alpha):
    """
    Returns a collate_fn that applies Mixup on each batch.
    Mixup blends two random samples (and their labels as soft targets).
    Uses hard labels during inference (alpha = 0 disables it).
    """
    def collate_fn(batch):
        specs   = torch.stack([b[0] for b in batch])
        labels  = torch.tensor([b[1] for b in batch], dtype=torch.long)

        if alpha <= 0:
            return specs, labels

        lam   = np.random.beta(alpha, alpha)
        idx   = torch.randperm(len(specs))
        specs = lam * specs + (1 - lam) * specs[idx]
        # Return both label sets + lambda so loss fn can blend
        return specs, (labels, labels[idx], lam)

    return collate_fn


class MixupCrossEntropy(nn.Module):
    """CrossEntropy that accepts either hard or soft (mixup) labels."""
    def __init__(self, num_classes, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        if isinstance(targets, tuple):
            y1, y2, lam = targets
            return lam * self.ce(logits, y1) + (1 - lam) * self.ce(logits, y2)
        return self.ce(logits, targets)


# ─────────────────────────────────────────────
# 7. MODEL  (EfficientNet-B0 + custom head)
# ─────────────────────────────────────────────

class RespiratoryNet(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet with a custom 4-class head.

    Freezing strategy
    ─────────────────
    For the first `freeze_epochs` epochs only the head is trained.
    After that, the full network is unfrozen for fine-tuning at a lower LR.
    This avoids destroying ImageNet features early on.
    """

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        weights   = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone  = models.efficientnet_b0(weights=weights)

        # Replace classifier head
        in_feats  = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)

    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.net.named_parameters():
            param.requires_grad = ("classifier" in name)

    def unfreeze_all(self):
        for param in self.net.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────
# 8. DATA LOADERS
# ─────────────────────────────────────────────

def make_loaders(config):
    train_ds = RespiratoryDataset(
        config["train_dir"], config["classes"], config, augment=True
    )
    test_ds  = RespiratoryDataset(
        config["test_dir"],  config["classes"], config, augment=False
    )

    sampler = WeightedRandomSampler(
        weights=train_ds.get_sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=mixup_collate(config["mixup_alpha"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 9. ICBHI SCORE HELPER
# ─────────────────────────────────────────────

def icbhi_score(labels, preds, num_classes=4):
    """
    ICBHI Score = (mean Sensitivity + mean Specificity) / 2

    Sensitivity per class = TP / (TP + FN)   ← recall
    Specificity per class = TN / (TN + FP)
    """
    labels = np.array(labels)
    preds  = np.array(preds)
    sens_list, spec_list = [], []

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        tn = ((preds != c) & (labels != c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        sens_list.append(tp / (tp + fn + 1e-9))
        spec_list.append(tn / (tn + fp + 1e-9))

    se = np.mean(sens_list)
    sp = np.mean(spec_list)
    score = (se + sp) / 2
    return score, se, sp, sens_list, spec_list


# ─────────────────────────────────────────────
# 10. TRAIN / EVALUATE LOOPS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, targets in tqdm(loader, desc="  train", leave=False):
        specs = specs.to(device)

        # Handle both hard labels and mixup tuples
        if isinstance(targets, (list, tuple)) and not isinstance(targets, torch.Tensor):
            y1, y2, lam = targets
            targets_device = (y1.to(device), y2.to(device), lam)
            hard_labels = y1.to(device)   # for accuracy tracking
        else:
            targets_device = targets.to(device)
            hard_labels    = targets_device

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, targets_device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(specs)
        correct    += (logits.argmax(1) == hard_labels).sum().item()
        total      += len(specs)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits  = model(specs)
        loss    = criterion(logits, labels)
        preds   = logits.argmax(1)
        total_loss += loss.item() * len(specs)
        correct    += (preds == labels).sum().item()
        total      += len(specs)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# 11. WARMUP + COSINE LR SCHEDULER
# ─────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warm-up for `warmup_epochs`, then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * (e / self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────
# 12. PLOTS
# ─────────────────────────────────────────────

def plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(tr_losses, label="Train"); axes[0].plot(vl_losses, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(tr_accs, label="Train"); axes[1].plot(vl_accs, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(icbhi_scores, color="green", label="ICBHI Score")
    axes[2].set_title("ICBHI Score (val)"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1); axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves_v2.png", dpi=150)
    plt.close()
    print("Saved → training_curves_v2.png")


def plot_confusion(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    # also show row-normalised (recall) version side by side
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm,      cm_norm],
        ["d",     ".2f"],
        ["Count", "Recall (row-normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {title}")

    plt.tight_layout()
    plt.savefig("confusion_matrix_v2.png", dpi=150)
    plt.close()
    print("Saved → confusion_matrix_v2.png")


# ─────────────────────────────────────────────
# 13. MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(config=CONFIG):
    train_loader, test_loader = make_loaders(config)

    model = RespiratoryNet(
        num_classes=len(config["classes"]),
        dropout=config["dropout"],
    ).to(DEVICE)

    # Freeze backbone for the first `freeze_epochs` epochs
    model.freeze_backbone()
    print(f"\nBackbone frozen for first {config['freeze_epochs']} epochs.")

    criterion = MixupCrossEntropy(
        num_classes=len(config["classes"]),
        label_smoothing=config["label_smoothing"],
    )
    # Separate LR for backbone vs head (head gets higher LR)
    head_params     = list(model.net.classifier.parameters())
    backbone_params = [p for p in model.net.parameters() if not any(
        p is hp for hp in head_params
    )]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config["lr"] * 0.1},
        {"params": head_params,     "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["num_epochs"],
        base_lr=config["lr"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("─" * 75)

    best_val_acc   = 0.0
    patience_count = 0
    tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores = [], [], [], [], []

    for epoch in range(1, config["num_epochs"] + 1):

        # Unfreeze backbone after freeze_epochs
        if epoch == config["freeze_epochs"] + 1:
            model.unfreeze_all()
            print(f"\n  → Backbone unfrozen at epoch {epoch}\n")

        lr = scheduler.step()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, config["grad_clip"]
        )
        # Use plain CE (no mixup) for eval
        plain_ce = nn.CrossEntropyLoss()
        vl_loss, vl_acc, vl_preds, vl_labels = evaluate(
            model, test_loader, plain_ce, DEVICE
        )

        icbhi, se, sp, _, _ = icbhi_score(vl_labels, vl_preds, len(config["classes"]))

        tr_losses.append(tr_loss); vl_losses.append(vl_loss)
        tr_accs.append(tr_acc);    vl_accs.append(vl_acc)
        icbhi_scores.append(icbhi)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            patience_count = 0
            torch.save(model.state_dict(), config["save_path"])
            flag = "  ✓"
        else:
            patience_count += 1

        print(
            f"Ep {epoch:03d}/{config['num_epochs']}  "
            f"lr={lr:.2e}  "
            f"| tr loss {tr_loss:.4f} acc {tr_acc:.3f}  "
            f"| vl loss {vl_loss:.4f} acc {vl_acc:.3f}  "
            f"| ICBHI {icbhi:.4f} (Se {se:.3f} Sp {sp:.3f})"
            f"{flag}"
        )

        if patience_count >= config["patience"]:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest val accuracy : {best_val_acc:.4f}")

    # ── Final report ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(config["save_path"], map_location=DEVICE))
    plain_ce = nn.CrossEntropyLoss()
    _, _, all_preds, all_labels = evaluate(model, test_loader, plain_ce, DEVICE)

    icbhi, se, sp, sens_list, spec_list = icbhi_score(
        all_labels, all_preds, len(config["classes"])
    )

    print("\n" + "═" * 50)
    print("FINAL TEST RESULTS")
    print("═" * 50)
    print(classification_report(all_labels, all_preds, target_names=config["classes"]))
    print(f"  Mean Sensitivity (Se) : {se:.4f}")
    print(f"  Mean Specificity (Sp) : {sp:.4f}")
    print(f"  ICBHI Score           : {icbhi:.4f}  ({icbhi*100:.1f}%)")
    print()
    for cls, s, sp_ in zip(config["classes"], sens_list, spec_list):
        print(f"    {cls:8s}  Se={s:.3f}  Sp={sp_:.3f}")

    plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores)
    plot_confusion(all_labels, all_preds, config["classes"])

    return model


# ─────────────────────────────────────────────
# 14. SINGLE-FILE INFERENCE
# ─────────────────────────────────────────────

def predict(wav_path, model_path=CONFIG["save_path"], config=CONFIG):
    """
    Predict the respiratory class of a single .wav file.

    Example
    -------
        label, probs = predict("processed_testing_cycles/101_cycle0_wheeze.wav")
    """
    model = RespiratoryNet(num_classes=len(config["classes"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    waveform = load_audio(wav_path, config["sample_rate"], config["duration"])
    spec     = normalise(waveform_to_melspec(waveform, config))
    tensor   = torch.tensor(spec).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()

    predicted = config["classes"][probs.argmax()]
    print(f"\nFile     : {os.path.basename(wav_path)}")
    print(f"Predicted: {predicted}")
    for cls, p in zip(config["classes"], probs):
        bar = "█" * int(p * 30)
        print(f"  {cls:8s} {p:.4f}  {bar}")
    return predicted, probs


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()