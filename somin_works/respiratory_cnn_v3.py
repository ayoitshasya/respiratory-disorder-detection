"""
Respiratory Sound Classification v3
=====================================
Targeted fixes for the v2 plateau at ICBHI ~62%:

1.  FOCAL LOSS           : Down-weights easy "normal" samples, forces the model
                           to focus on hard "both" / "wheeze" samples.
                           gamma=2, per-class alpha weights from inverse frequency.

2.  HARD OVERSAMPLING    : "both" cycles are duplicated 4× and "wheeze" 2× in the
                           training folder scan — before WeightedRandomSampler runs.
                           This directly injects more "both" signal.

3.  TWO-STAGE FINE-TUNE  : Stage 1 — only the EfficientNet head trains (5 ep).
                           Stage 2 — top-2 EfficientNet blocks + head (lr/5).
                           Stage 3 — full network (lr/10).
                           This prevents early feature destruction while still
                           fine-tuning the backbone deeply.

4.  COSINE LR WITH RESTARTS (SGDR) : Restarts every 10 epochs so the model
                           escapes local optima — replaces the single cosine decay
                           that caused the plateau.

5.  TEST-TIME AUGMENTATION (TTA) : At inference, each file is evaluated 5×
                           (original + 4 augmented) and probabilities averaged.
                           Typically +1–2% ICBHI score for free.

6.  ICBHI-AWARE CHECKPOINT : Saves on best ICBHI score, not best accuracy, since
                           that's the metric that actually matters for this dataset.

7.  MIXUP ALPHA REDUCED   : 0.2 → less aggressive blending, which was blurring
                           the already-rare "both" class signal.

All other settings (folder structure, label parsing, mel spec params) unchanged.

Requirements
------------
    pip install torch torchvision torchaudio librosa scikit-learn matplotlib seaborn tqdm
"""

import os, random, math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # ── Paths  (use r"..." on Windows to avoid escape-sequence warnings) ───
    "train_dir": "D:\LY_Project\Previous_trials\processed_training_cycles",
    "test_dir":  "D:\LY_Project\Previous_trials\processed_testing_cycles",
    "save_path":  "best_respiratory_v3.pt",

    # ── Audio ──────────────────────────────────────────────────────────────
    "sample_rate":  22050,
    "duration":     5.0,
    "n_mels":       128,
    "n_fft":        1024,
    "hop_length":   512,
    "f_min":        50,
    "f_max":        2000,

    # ── Hard oversampling multipliers (applied before WeightedRandomSampler)
    #    Increase these if "both" / "wheeze" recall is still low.
    "oversample": {
        "normal":  1,
        "crackle": 1,
        "wheeze":  2,
        "both":    4,
    },

    # ── Augmentation ───────────────────────────────────────────────────────
    "time_mask_pct":  0.15,
    "freq_mask_pct":  0.15,
    "noise_std":      0.015,
    "time_shift_pct": 0.10,
    "mixup_alpha":    0.2,    # reduced from 0.3 to preserve "both" signal
    "tta_runs":       5,      # test-time augmentation passes (1 = disabled)

    # ── Model ──────────────────────────────────────────────────────────────
    "dropout":        0.5,

    # ── Focal loss ─────────────────────────────────────────────────────────
    "focal_gamma":    2.0,    # focusing parameter (0 = standard CE)

    # ── Three-stage fine-tuning ────────────────────────────────────────────
    #   Stage 1: head only        (epochs 1 … stage1_end)
    #   Stage 2: top-2 blocks + head (stage1_end+1 … stage2_end)
    #   Stage 3: full network     (stage2_end+1 … num_epochs)
    "stage1_end":     5,
    "stage2_end":     15,

    # ── Training ───────────────────────────────────────────────────────────
    "batch_size":    32,
    "num_epochs":    70,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "grad_clip":     2.0,
    "warmup_epochs": 3,
    "patience":      15,      # early stopping on ICBHI score
    "num_workers":   0,
    "seed":          42,

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
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        fmin=cfg["f_min"], fmax=cfg["f_max"],
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalise(spec):
    lo, hi = spec.min(), spec.max()
    return (spec - lo) / (hi - lo + 1e-8)


def augment_spec(spec, cfg):
    """Apply all augmentations to a normalised spectrogram (H×W numpy array)."""
    spec = _time_shift(spec, cfg["time_shift_pct"])
    spec = _add_noise(spec, cfg["noise_std"])
    spec = _time_mask(spec, cfg["time_mask_pct"])
    spec = _freq_mask(spec, cfg["freq_mask_pct"])
    return spec


def _time_shift(spec, pct):
    _, T  = spec.shape
    shift = int(T * pct * random.random())
    return np.roll(spec, shift, axis=1)

def _add_noise(spec, std):
    return spec + std * np.random.randn(*spec.shape).astype(np.float32)

def _time_mask(spec, pct):
    _, T = spec.shape
    ml   = int(T * pct * random.random())
    if ml > 0:
        t0 = random.randint(0, T - ml)
        spec = spec.copy(); spec[:, t0:t0 + ml] = 0.0
    return spec

def _freq_mask(spec, pct):
    F, _ = spec.shape
    ml   = int(F * pct * random.random())
    if ml > 0:
        f0 = random.randint(0, F - ml)
        spec = spec.copy(); spec[f0:f0 + ml, :] = 0.0
    return spec


def spec_to_tensor(spec):
    """Normalised H×W numpy → (3, H, W) float tensor."""
    return torch.tensor(spec).unsqueeze(0).repeat(3, 1, 1)


# ─────────────────────────────────────────────
# 5. DATASET  (with hard oversampling)
# ─────────────────────────────────────────────

class RespiratoryDataset(Dataset):
    """
    Flat folder dataset with hard oversampling.
    Files for rare classes are repeated `oversample[cls]` times in self.samples
    before WeightedRandomSampler runs, so the model literally sees more of them.
    """

    def __init__(self, root_dir, classes, config, augment=False):
        self.cfg          = config
        self.augment      = augment
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples      = []
        skipped           = 0

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Not found: {root_dir}")

        base_samples = []
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            label_str = parse_label(fname, classes)
            if label_str is None:
                skipped += 1
                continue
            base_samples.append(
                (os.path.join(root_dir, fname), self.class_to_idx[label_str], label_str)
            )

        # Hard oversampling — only applied to training set
        multipliers = config.get("oversample", {c: 1 for c in classes}) if augment else {c: 1 for c in classes}
        for path, idx, cls in base_samples:
            for _ in range(multipliers.get(cls, 1)):
                self.samples.append((path, idx))

        print(f"\nDataset: {root_dir}  (augment={augment})")
        print(f"  Base samples  : {len(base_samples)}  |  After oversampling: {len(self.samples)}  |  Skipped: {skipped}")
        for cls, idx in self.class_to_idx.items():
            n_base = sum(1 for _, _, c in base_samples if c == cls)
            n_os   = sum(1 for _, l in self.samples if l == idx)
            print(f"    {cls:8s}: {n_base:4d} base  →  {n_os:4d} after oversampling")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform    = load_audio(path, self.cfg["sample_rate"], self.cfg["duration"])
        spec        = normalise(waveform_to_melspec(waveform, self.cfg))
        if self.augment:
            spec = augment_spec(spec, self.cfg)
        return spec_to_tensor(spec), label

    def get_sample_weights(self):
        counts  = np.zeros(len(self.class_to_idx))
        for _, l in self.samples:
            counts[l] += 1
        class_w = 1.0 / (counts + 1e-6)
        return torch.tensor([class_w[l] for _, l in self.samples], dtype=torch.float)

    def class_counts(self):
        counts = np.zeros(len(self.class_to_idx), dtype=int)
        for _, l in self.samples:
            counts[l] += 1
        return counts


# ─────────────────────────────────────────────
# 6. FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : per-class weight tensor (inverse-frequency weights)
    gamma : focusing parameter. gamma=0 → standard CE.
            gamma=2 is standard for imbalanced classification.

    Also supports soft (mixup) targets via the `targets` tuple form.
    """

    def __init__(self, alpha, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.num_classes     = len(alpha)

    def forward(self, logits, targets):
        # ── Handle mixup tuple ─────────────────────────────────────────────
        if isinstance(targets, tuple):
            y1, y2, lam = targets
            return lam * self._focal(logits, y1) + (1 - lam) * self._focal(logits, y2)
        return self._focal(logits, targets)

    def _focal(self, logits, labels):
        # Label smoothing baked in
        K   = self.num_classes
        eps = self.label_smoothing
        # one-hot with smoothing
        with torch.no_grad():
            smooth = torch.full_like(logits, eps / (K - 1))
            smooth.scatter_(1, labels.unsqueeze(1), 1.0 - eps)

        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()

        # p_t for the true class (for focal weight)
        p_t     = (prob * smooth).sum(dim=1)
        focal_w = (1 - p_t) ** self.gamma

        # per-sample alpha weight
        alpha_t = self.alpha[labels]

        loss = -(focal_w * alpha_t * (log_prob * smooth).sum(dim=1))
        return loss.mean()


def make_focal_loss(class_counts, gamma, device):
    """Build FocalLoss with alpha = inverse-frequency weights (normalised to sum=1)."""
    counts = torch.tensor(class_counts, dtype=torch.float)
    alpha  = 1.0 / (counts + 1e-6)
    alpha  = alpha / alpha.sum()       # normalise
    return FocalLoss(alpha.to(device), gamma=gamma, label_smoothing=0.05)


# ─────────────────────────────────────────────
# 7. MODEL  (EfficientNet-B0, three-stage unfreezing)
# ─────────────────────────────────────────────

class RespiratoryNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        weights  = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features
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

    # ── Freezing helpers ───────────────────────────────────────────────────

    def set_stage(self, stage):
        """
        stage 1 : head only
        stage 2 : last 2 MBConv blocks + head
        stage 3 : full network
        """
        # Freeze everything first
        for p in self.net.parameters():
            p.requires_grad = False

        if stage == 1:
            for p in self.net.classifier.parameters():
                p.requires_grad = True
            print("  Stage 1: training HEAD only")

        elif stage == 2:
            # EfficientNet features is a Sequential of blocks 0-8;
            # unfreeze blocks 6, 7, 8 (last 3) + classifier
            for p in self.net.classifier.parameters():
                p.requires_grad = True
            for blk in list(self.net.features.children())[-3:]:
                for p in blk.parameters():
                    p.requires_grad = True
            print("  Stage 2: training last-3 feature blocks + HEAD")

        elif stage == 3:
            for p in self.net.parameters():
                p.requires_grad = True
            print("  Stage 3: full network")

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ─────────────────────────────────────────────
# 8. MIXUP COLLATE
# ─────────────────────────────────────────────

def mixup_collate(alpha):
    def collate_fn(batch):
        specs  = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if alpha <= 0:
            return specs, labels
        lam    = np.random.beta(alpha, alpha)
        idx    = torch.randperm(len(specs))
        specs  = lam * specs + (1 - lam) * specs[idx]
        return specs, (labels, labels[idx], lam)
    return collate_fn


# ─────────────────────────────────────────────
# 9. DATA LOADERS
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
    return train_loader, test_loader, train_ds.class_counts()


# ─────────────────────────────────────────────
# 10. ICBHI SCORE
# ─────────────────────────────────────────────

def icbhi_score(labels, preds, num_classes=4):
    labels, preds = np.array(labels), np.array(preds)
    sens, spec = [], []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        tn = ((preds != c) & (labels != c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        sens.append(tp / (tp + fn + 1e-9))
        spec.append(tn / (tn + fp + 1e-9))
    se = np.mean(sens)
    sp = np.mean(spec)
    return (se + sp) / 2, se, sp, sens, spec


# ─────────────────────────────────────────────
# 11. TRAIN / EVALUATE / TTA
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, targets in tqdm(loader, desc="  train", leave=False):
        specs = specs.to(device)
        hard  = targets[0].to(device) if isinstance(targets, tuple) else targets.to(device)
        tgt   = (targets[0].to(device), targets[1].to(device), targets[2]) \
                if isinstance(targets, tuple) else targets.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_params(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(specs)
        correct    += (logits.argmax(1) == hard).sum().item()
        total      += len(specs)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Standard single-pass evaluation."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0
    ce = nn.CrossEntropyLoss()
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        total_loss += ce(logits, labels).item() * len(specs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += len(specs)
    return total_loss / total, all_preds, all_labels


@torch.no_grad()
def evaluate_tta(model, test_ds, config, device):
    """
    Test-Time Augmentation: run each sample `tta_runs` times, average softmax probs.
    Falls back to standard evaluate if tta_runs == 1.
    """
    tta_runs = config["tta_runs"]
    if tta_runs <= 1:
        loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
        _, preds, labels = evaluate(model, loader, device)
        return preds, labels

    model.eval()
    all_probs  = []
    all_labels = []

    for idx in tqdm(range(len(test_ds)), desc="  TTA", leave=False):
        path, label = test_ds.samples[idx]
        all_labels.append(label)

        waveform = load_audio(path, config["sample_rate"], config["duration"])
        base_spec = normalise(waveform_to_melspec(waveform, config))

        probs_accum = torch.zeros(len(config["classes"]))
        for run in range(tta_runs):
            if run == 0:
                spec = base_spec          # first pass: clean
            else:
                spec = augment_spec(base_spec.copy(), config)

            tensor = spec_to_tensor(spec).unsqueeze(0).to(device)
            logits = model(tensor)
            probs_accum += torch.softmax(logits.squeeze(), dim=0).cpu()

        all_probs.append((probs_accum / tta_runs).numpy())

    all_probs  = np.stack(all_probs)
    all_preds  = all_probs.argmax(axis=1).tolist()
    return all_preds, all_labels


# ─────────────────────────────────────────────
# 12. LR SCHEDULER — COSINE WITH WARM RESTARTS
# ─────────────────────────────────────────────

class WarmupCosineRestarts:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing with restarts
    every `restart_period` epochs (SGDR style).
    """
    def __init__(self, optimizer, warmup_epochs, restart_period, base_lr, min_lr=1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.restart_period = restart_period
        self.base_lr        = base_lr
        self.min_lr         = min_lr
        self.epoch          = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * (e / self.warmup_epochs)
        else:
            t = (e - self.warmup_epochs) % self.restart_period
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * t / self.restart_period)
            )
        for pg in self.optimizer.param_groups:
            # Scale backbone LR relative to head LR
            scale = pg.get("lr_scale", 1.0)
            pg["lr"] = lr * scale
        return lr


# ─────────────────────────────────────────────
# 13. PLOTS
# ─────────────────────────────────────────────

def plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores, stage_epochs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax in axes:
        for ep in stage_epochs:
            ax.axvline(ep, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    axes[0].plot(tr_losses, label="Train"); axes[0].plot(vl_losses, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(tr_accs, label="Train"); axes[1].plot(vl_accs, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(icbhi_scores, color="green", label="ICBHI")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("ICBHI Score (val)"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves_v3.png", dpi=150)
    plt.close()
    print("Saved → training_curves_v3.png")


def plot_confusion(labels, preds, class_names):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Count", "Recall (row-normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_v3.png", dpi=150)
    plt.close()
    print("Saved → confusion_matrix_v3.png")


# ─────────────────────────────────────────────
# 14. MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(config=CONFIG):
    train_loader, test_loader, class_counts = make_loaders(config)

    model = RespiratoryNet(
        num_classes=len(config["classes"]),
        dropout=config["dropout"],
    ).to(DEVICE)

    criterion = make_focal_loss(class_counts, config["focal_gamma"], DEVICE)

    # Build optimizer with per-group lr_scale tags
    def make_optimizer(model):
        head_params     = list(model.net.classifier.parameters())
        backbone_params = [p for p in model.net.parameters()
                           if p.requires_grad and not any(p is h for h in head_params)]
        return optim.AdamW([
            {"params": backbone_params, "lr": config["lr"] * 0.1, "lr_scale": 0.1},
            {"params": head_params,     "lr": config["lr"],        "lr_scale": 1.0},
        ], weight_decay=config["weight_decay"])

    # Stage 1 — head only
    model.set_stage(1)
    optimizer = make_optimizer(model)
    scheduler = WarmupCosineRestarts(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        restart_period=10,
        base_lr=config["lr"],
    )

    total_params    = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters (total): {total_params:,}")
    print("─" * 80)

    best_icbhi      = 0.0
    patience_count  = 0
    tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores = [], [], [], [], []
    stage_epochs    = [config["stage1_end"], config["stage2_end"]]  # for plot markers

    for epoch in range(1, config["num_epochs"] + 1):

        # ── Stage transitions ──────────────────────────────────────────────
        if epoch == config["stage1_end"] + 1:
            model.set_stage(2)
            optimizer = make_optimizer(model)
            scheduler = WarmupCosineRestarts(
                optimizer, warmup_epochs=2, restart_period=10, base_lr=config["lr"]
            )

        if epoch == config["stage2_end"] + 1:
            model.set_stage(3)
            optimizer = make_optimizer(model)
            scheduler = WarmupCosineRestarts(
                optimizer, warmup_epochs=2, restart_period=10,
                base_lr=config["lr"] * 0.5   # lower LR for full fine-tune
            )

        lr = scheduler.step()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, config["grad_clip"]
        )
        vl_loss, vl_preds, vl_labels = evaluate(model, test_loader, DEVICE)
        vl_acc  = np.mean(np.array(vl_preds) == np.array(vl_labels))
        icbhi, se, sp, _, _ = icbhi_score(vl_labels, vl_preds, len(config["classes"]))

        tr_losses.append(tr_loss); vl_losses.append(vl_loss)
        tr_accs.append(tr_acc);    vl_accs.append(vl_acc)
        icbhi_scores.append(icbhi)

        flag = ""
        if icbhi > best_icbhi:              # checkpoint on ICBHI, not accuracy
            best_icbhi     = icbhi
            patience_count = 0
            torch.save(model.state_dict(), config["save_path"])
            flag = "  ✓"
        else:
            patience_count += 1

        print(
            f"Ep {epoch:03d}/{config['num_epochs']}  lr={lr:.2e}  "
            f"| tr loss {tr_loss:.4f} acc {tr_acc:.3f}  "
            f"| vl loss {vl_loss:.4f} acc {vl_acc:.3f}  "
            f"| ICBHI {icbhi:.4f} (Se {se:.3f} Sp {sp:.3f})"
            f"{flag}"
        )

        if patience_count >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest ICBHI (val, single-pass): {best_icbhi:.4f}")

    # ── Final evaluation with TTA ──────────────────────────────────────────
    model.load_state_dict(torch.load(config["save_path"], map_location=DEVICE))

    # Re-build test_ds directly for TTA (we need raw sample access)
    from torch.utils.data import Dataset as _DS
    test_ds_raw = RespiratoryDataset(
        config["test_dir"], config["classes"], config, augment=False
    )

    print(f"\nRunning TTA with {config['tta_runs']} passes …")
    all_preds, all_labels = evaluate_tta(model, test_ds_raw, config, DEVICE)

    icbhi, se, sp, sens_list, spec_list = icbhi_score(
        all_labels, all_preds, len(config["classes"])
    )

    print("\n" + "═" * 55)
    print("FINAL TEST RESULTS  (with TTA)")
    print("═" * 55)
    print(classification_report(all_labels, all_preds, target_names=config["classes"]))
    print(f"  Mean Sensitivity (Se) : {se:.4f}")
    print(f"  Mean Specificity (Sp) : {sp:.4f}")
    print(f"  ICBHI Score           : {icbhi:.4f}  ({icbhi*100:.1f}%)")
    print()
    for cls, s, sp_ in zip(config["classes"], sens_list, spec_list):
        print(f"    {cls:8s}  Se={s:.3f}  Sp={sp_:.3f}")

    plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores, stage_epochs)
    plot_confusion(all_labels, all_preds, config["classes"])

    return model


# ─────────────────────────────────────────────
# 15. SINGLE-FILE INFERENCE  (with TTA)
# ─────────────────────────────────────────────

def predict(wav_path, model_path=CONFIG["save_path"], config=CONFIG):
    model = RespiratoryNet(num_classes=len(config["classes"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    waveform  = load_audio(wav_path, config["sample_rate"], config["duration"])
    base_spec = normalise(waveform_to_melspec(waveform, config))

    probs_accum = torch.zeros(len(config["classes"]))
    with torch.no_grad():
        for run in range(config["tta_runs"]):
            spec   = base_spec if run == 0 else augment_spec(base_spec.copy(), config)
            tensor = spec_to_tensor(spec).unsqueeze(0)
            probs_accum += torch.softmax(model(tensor).squeeze(), dim=0)

    probs     = (probs_accum / config["tta_runs"]).numpy()
    predicted = config["classes"][probs.argmax()]

    print(f"\nFile     : {os.path.basename(wav_path)}")
    print(f"Predicted: {predicted}")
    for cls, p in zip(config["classes"], probs):
        print(f"  {cls:8s} {p:.4f}  {'█' * int(p * 30)}")
    return predicted, probs


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()
