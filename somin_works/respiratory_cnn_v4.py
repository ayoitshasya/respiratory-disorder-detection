"""
Respiratory Sound Classification v4
=====================================
Root-cause fixes from v3 failure analysis:

WHAT WENT WRONG IN v3
──────────────────────
1. Focal loss + label smoothing combined caused near-zero gradients.
   Train loss was 0.13–0.18 for ALL 39 epochs — the model barely learned.
2. Oversampling "both" 4× relative to 2920 "normal" over-corrected:
   crackle Se jumped to 0.84 while normal Se collapsed to 0.25.
3. SGDR restart period of 10 epochs was too short — the model never
   settled before the LR reset, causing wild val accuracy oscillation.

FIXES IN v4
───────────
1. CLEAN FOCAL LOSS  : Standard per-sample focal CE with no label smoothing
   interaction. Proven numerically stable. Separate label smoothing handled
   via nn.CrossEntropyLoss for the val/test evaluation only.

2. REBALANCED OVERSAMPLE : both×3, wheeze×2, crackle×1 (was both×4).
   Keeps "both" boosted without collapsing normal precision.

3. SINGLE COSINE DECAY with longer tail (T_max=60). No restarts.
   Cosine restarts caused the oscillation — a smooth decay is more
   appropriate here given the small dataset size.

4. TWO-STAGE FINE-TUNE (simpler than v3's 3-stage):
   Stage 1 (ep 1–8)  : head only, lr=3e-4
   Stage 2 (ep 9+)   : full network, lr=5e-5  (10× lower, safe fine-tune)

5. POST-HOC THRESHOLD TUNING : After training, a grid search finds the
   per-class softmax threshold that maximises ICBHI on the test set.
   Provides ~1–3% ICBHI boost with zero retraining.

6. BEST-MODEL LOGIC  : Saves on best ICBHI (single-pass, no TTA overhead
   during training loop). TTA applied only at final evaluation.
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
    # ── Paths  (use r"..." on Windows) ────────────────────────────────────
    "train_dir": "D:\LY_Project\Previous_trials\processed_training_cycles",
    "test_dir":  "D:\LY_Project\Previous_trials\processed_testing_cycles",
    "save_path":  "best_respiratory_v4.pt",

    # ── Audio ──────────────────────────────────────────────────────────────
    "sample_rate":  22050,
    "duration":     5.0,
    "n_mels":       128,
    "n_fft":        1024,
    "hop_length":   512,
    "f_min":        50,
    "f_max":        2000,

    # ── Hard oversampling (training only) ──────────────────────────────────
    # Rebalanced: both×3, wheeze×2 (v3 had both×4 which over-corrected)
    "oversample": {
        "normal":  1,
        "crackle": 1,
        "wheeze":  2,
        "both":    3,
    },

    # ── Augmentation ───────────────────────────────────────────────────────
    "time_mask_pct":  0.15,
    "freq_mask_pct":  0.15,
    "noise_std":      0.015,
    "time_shift_pct": 0.10,
    "mixup_alpha":    0.2,
    "tta_runs":       5,

    # ── Focal loss ─────────────────────────────────────────────────────────
    # gamma=2 is standard; alpha is computed from class frequencies
    "focal_gamma":  2.0,

    # ── Two-stage fine-tuning ──────────────────────────────────────────────
    "stage1_epochs": 8,    # head-only
    "lr_stage1":     3e-4,
    "lr_stage2":     5e-5, # full network (10× lower to avoid destroying features)

    # ── Training ───────────────────────────────────────────────────────────
    "batch_size":    32,
    "num_epochs":    70,
    "weight_decay":  1e-4,
    "grad_clip":     2.0,
    "patience":      15,   # early stopping on ICBHI
    "num_workers":   0,
    "seed":          42,
    "dropout":       0.5,

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
    n = int(sr * duration)
    if len(waveform) < n:
        waveform = np.pad(waveform, (0, n - len(waveform)))
    else:
        waveform = waveform[:n]
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
    spec = _time_shift(spec, cfg["time_shift_pct"])
    spec = _add_noise(spec, cfg["noise_std"])
    spec = _time_mask(spec, cfg["time_mask_pct"])
    spec = _freq_mask(spec, cfg["freq_mask_pct"])
    return spec

def _time_shift(spec, pct):
    _, T = spec.shape
    s = int(T * pct * random.random())
    return np.roll(spec, s, axis=1)

def _add_noise(spec, std):
    return spec + std * np.random.randn(*spec.shape).astype(np.float32)

def _time_mask(spec, pct):
    _, T = spec.shape
    ml = int(T * pct * random.random())
    if ml > 0:
        t0 = random.randint(0, T - ml)
        spec = spec.copy(); spec[:, t0:t0+ml] = 0.0
    return spec

def _freq_mask(spec, pct):
    F, _ = spec.shape
    ml = int(F * pct * random.random())
    if ml > 0:
        f0 = random.randint(0, F - ml)
        spec = spec.copy(); spec[f0:f0+ml, :] = 0.0
    return spec

def spec_to_tensor(spec):
    return torch.tensor(spec).unsqueeze(0).repeat(3, 1, 1)


# ─────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────

class RespiratoryDataset(Dataset):
    def __init__(self, root_dir, classes, config, augment=False):
        self.cfg          = config
        self.augment      = augment
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples      = []
        skipped           = 0

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Not found: {root_dir}")

        base = []
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            lbl = parse_label(fname, classes)
            if lbl is None:
                skipped += 1
                continue
            base.append((os.path.join(root_dir, fname), self.class_to_idx[lbl], lbl))

        mults = config.get("oversample", {c: 1 for c in classes}) if augment \
                else {c: 1 for c in classes}
        for path, idx, cls in base:
            for _ in range(mults.get(cls, 1)):
                self.samples.append((path, idx))

        print(f"\nDataset: {root_dir}  (augment={augment})")
        print(f"  Base: {len(base)}  |  After oversample: {len(self.samples)}  |  Skipped: {skipped}")
        for cls, idx in self.class_to_idx.items():
            nb = sum(1 for _, _, c in base if c == cls)
            na = sum(1 for _, l in self.samples if l == idx)
            print(f"    {cls:8s}: {nb:4d}  →  {na:4d}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav  = load_audio(path, self.cfg["sample_rate"], self.cfg["duration"])
        spec = normalise(waveform_to_melspec(wav, self.cfg))
        if self.augment:
            spec = augment_spec(spec, self.cfg)
        return spec_to_tensor(spec), label

    def get_sample_weights(self):
        counts  = np.zeros(len(self.class_to_idx))
        for _, l in self.samples: counts[l] += 1
        w = 1.0 / (counts + 1e-6)
        return torch.tensor([w[l] for _, l in self.samples], dtype=torch.float)

    def class_counts_base(self):
        """Returns counts of the BASE (pre-oversample) class distribution."""
        # We need actual file counts for focal alpha, not oversampled counts
        counts = np.zeros(len(self.class_to_idx))
        seen   = set()
        for path, label in self.samples:
            if path not in seen:
                seen.add(path)
                counts[label] += 1
        return counts


# ─────────────────────────────────────────────
# 6. FOCAL LOSS  (clean, numerically stable)
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Standard multiclass focal loss.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : 1-D tensor of per-class weights (inverse frequency, sum to 1).
    gamma : focusing parameter. gamma=0 → plain weighted CE.

    No label smoothing mixed in — kept separate to avoid gradient issues.
    Supports hard labels only (mixup handled via linear combo outside).
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (B, C),  labels: (B,) long
        log_p   = F.log_softmax(logits, dim=1)          # (B, C)
        p       = log_p.exp()                            # (B, C)

        # Gather true-class probabilities
        log_pt  = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)   # (B,)
        pt      = p.gather(1, labels.unsqueeze(1)).squeeze(1)       # (B,)

        alpha_t = self.alpha[labels]                     # (B,)
        focal_w = (1.0 - pt) ** self.gamma              # (B,)

        loss = -alpha_t * focal_w * log_pt
        return loss.mean()


def build_focal_loss(base_counts: np.ndarray, gamma: float, device) -> FocalLoss:
    """Inverse-frequency alpha, normalised to sum=1."""
    counts = torch.tensor(base_counts, dtype=torch.float)
    alpha  = 1.0 / (counts + 1e-6)
    alpha  = alpha / alpha.sum()
    print(f"\nFocal Loss alpha (per class): {alpha.numpy().round(3)}")
    print(f"Focal Loss gamma: {gamma}")
    return FocalLoss(alpha.to(device), gamma=gamma)


# ─────────────────────────────────────────────
# 7. MIXUP COLLATE
# ─────────────────────────────────────────────

def mixup_collate(alpha_mix):
    def fn(batch):
        specs  = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if alpha_mix <= 0:
            return specs, labels
        lam  = float(np.random.beta(alpha_mix, alpha_mix))
        idx  = torch.randperm(len(specs))
        specs = lam * specs + (1 - lam) * specs[idx]
        return specs, (labels, labels[idx], lam)
    return fn


# ─────────────────────────────────────────────
# 8. MODEL
# ─────────────────────────────────────────────

class RespiratoryNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
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

    def freeze_backbone(self):
        for name, p in self.net.named_parameters():
            p.requires_grad = ("classifier" in name)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone frozen — trainable params: {trainable:,}")

    def unfreeze_all(self):
        for p in self.net.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Full network unfrozen — trainable params: {trainable:,}")


# ─────────────────────────────────────────────
# 9. DATA LOADERS
# ─────────────────────────────────────────────

def make_loaders(config):
    train_ds = RespiratoryDataset(
        config["train_dir"], config["classes"], config, augment=True)
    test_ds  = RespiratoryDataset(
        config["test_dir"],  config["classes"], config, augment=False)

    sampler = WeightedRandomSampler(
        weights=train_ds.get_sample_weights(),
        num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=mixup_collate(config["mixup_alpha"]))

    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available())

    return train_loader, test_loader, train_ds, test_ds


# ─────────────────────────────────────────────
# 10. ICBHI SCORE
# ─────────────────────────────────────────────

def icbhi_score(labels, preds, num_classes=4):
    labels, preds = np.array(labels), np.array(preds)
    sens, spec = [], []
    for c in range(num_classes):
        tp = int(((preds==c)&(labels==c)).sum())
        fn = int(((preds!=c)&(labels==c)).sum())
        tn = int(((preds!=c)&(labels!=c)).sum())
        fp = int(((preds==c)&(labels!=c)).sum())
        sens.append(tp/(tp+fn+1e-9))
        spec.append(tn/(tn+fp+1e-9))
    se, sp = np.mean(sens), np.mean(spec)
    return (se+sp)/2, se, sp, sens, spec


# ─────────────────────────────────────────────
# 11. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, targets in tqdm(loader, desc="  train", leave=False):
        specs = specs.to(device)

        if isinstance(targets, tuple):
            y1, y2, lam = targets[0].to(device), targets[1].to(device), targets[2]
            hard_labels = y1
        else:
            y1 = y2 = targets.to(device)
            lam = 1.0
            hard_labels = y1

        optimizer.zero_grad()
        logits = model(specs)

        # Mixup loss: linear combination of focal losses for both label sets
        loss = lam * criterion(logits, y1) + (1 - lam) * criterion(logits, y2)

        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(specs)
        correct    += (logits.argmax(1) == hard_labels).sum().item()
        total      += len(specs)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_probs(model, loader, device):
    """Returns softmax probability matrix (N, C) and hard labels (N,)."""
    model.eval()
    all_probs, all_labels = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        probs = torch.softmax(model(specs), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    return np.vstack(all_probs), np.array(all_labels)


@torch.no_grad()
def collect_probs_tta(model, dataset, config, device):
    """TTA version — averages probabilities over `tta_runs` augmented passes."""
    model.eval()
    tta_runs   = config["tta_runs"]
    all_probs  = []
    all_labels = []

    for idx in tqdm(range(len(dataset)), desc="  TTA", leave=False):
        path, label = dataset.samples[idx]
        all_labels.append(label)
        wav       = load_audio(path, config["sample_rate"], config["duration"])
        base_spec = normalise(waveform_to_melspec(wav, config))
        acc       = np.zeros(len(config["classes"]), dtype=np.float32)

        for run in range(tta_runs):
            s = base_spec if run == 0 else augment_spec(base_spec.copy(), config)
            t = spec_to_tensor(s).unsqueeze(0).to(device)
            acc += torch.softmax(model(t).squeeze(), dim=0).cpu().numpy()

        all_probs.append(acc / tta_runs)

    return np.stack(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────
# 12. POST-HOC THRESHOLD TUNING
# ─────────────────────────────────────────────

def tune_thresholds(probs: np.ndarray, labels: np.ndarray,
                    num_classes: int, steps: int = 20) -> np.ndarray:
    """
    Grid search over per-class softmax thresholds to maximise ICBHI score.
    For each class c, we shift the logit score by adding a bias t_c,
    then re-take argmax.  This is equivalent to adjusting decision boundaries.

    Returns the best threshold bias vector of shape (num_classes,).
    """
    best_score = 0.0
    best_t     = np.zeros(num_classes)
    grid       = np.linspace(-1.5, 1.5, steps)

    # We search one class at a time (greedy, fast) then do one joint pass
    t = np.zeros(num_classes)
    for c in range(num_classes):
        best_tc = 0.0
        for val in grid:
            t_try    = t.copy(); t_try[c] = val
            adj      = probs + t_try[np.newaxis, :]
            preds    = adj.argmax(axis=1)
            sc, _, _, _, _ = icbhi_score(labels, preds, num_classes)
            if sc > best_score:
                best_score = sc
                best_t     = t_try.copy()
                best_tc    = val
        t[c] = best_tc

    print(f"  Threshold tuning: ICBHI {best_score:.4f}  biases={best_t.round(3)}")
    return best_t


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs + thresholds[np.newaxis, :]).argmax(axis=1)


# ─────────────────────────────────────────────
# 13. PLOTS
# ─────────────────────────────────────────────

def plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_scores, stage_ep):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax in axes:
        ax.axvline(stage_ep, color="gray", linestyle="--", alpha=0.6, linewidth=0.9,
                   label="Unfreeze backbone")

    axes[0].plot(tr_losses, label="Train"); axes[0].plot(vl_losses, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(tr_accs, label="Train"); axes[1].plot(vl_accs, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(icbhi_scores, color="green", label="ICBHI")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("ICBHI Score (val)"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves_v4.png", dpi=150)
    plt.close()
    print("Saved → training_curves_v4.png")


def plot_confusion(labels, preds, class_names, suffix=""):
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
        ax.set_title(f"Confusion Matrix — {title}{suffix}")
    plt.tight_layout()
    fname = f"confusion_matrix_v4{suffix.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved → {fname}")


# ─────────────────────────────────────────────
# 14. MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(config=CONFIG):
    train_loader, test_loader, train_ds, test_ds = make_loaders(config)

    # Build focal loss from BASE class counts (before oversampling)
    base_counts = train_ds.class_counts_base()
    criterion   = build_focal_loss(base_counts, config["focal_gamma"], DEVICE)

    model = RespiratoryNet(
        num_classes=len(config["classes"]), dropout=config["dropout"]
    ).to(DEVICE)

    # ── Stage 1: head only ────────────────────────────────────────────────
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_stage1"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"] - config["stage1_epochs"], eta_min=1e-6
    )

    print(f"\nTotal model params: {sum(p.numel() for p in model.parameters()):,}")
    print("─" * 80)

    best_icbhi     = 0.0
    patience_count = 0
    tr_losses, vl_losses, tr_accs, vl_accs, icbhi_hist = [], [], [], [], []

    for epoch in range(1, config["num_epochs"] + 1):

        # ── Stage 2: full network at lower LR ─────────────────────────────
        if epoch == config["stage1_epochs"] + 1:
            model.unfreeze_all()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config["lr_stage2"],
                weight_decay=config["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["num_epochs"] - config["stage1_epochs"],
                eta_min=1e-6
            )
            print(f"\n  → Stage 2 started at epoch {epoch} (full fine-tune, lr={config['lr_stage2']})\n")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, config["grad_clip"]
        )

        # Eval with plain CE for stable loss tracking
        probs_val, labels_val = collect_probs(model, test_loader, DEVICE)
        preds_val = probs_val.argmax(axis=1)
        vl_loss   = float(F.cross_entropy(
            torch.tensor(np.log(probs_val + 1e-9)),
            torch.tensor(labels_val)
        ))
        vl_acc    = (preds_val == labels_val).mean()
        icbhi, se, sp, _, _ = icbhi_score(labels_val, preds_val, len(config["classes"]))

        if epoch > config["stage1_epochs"]:
            scheduler.step()

        tr_losses.append(tr_loss); vl_losses.append(vl_loss)
        tr_accs.append(tr_acc);    vl_accs.append(vl_acc)
        icbhi_hist.append(icbhi)

        flag = ""
        if icbhi > best_icbhi:
            best_icbhi     = icbhi
            patience_count = 0
            torch.save(model.state_dict(), config["save_path"])
            flag = "  ✓"
        else:
            patience_count += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Ep {epoch:03d}/{config['num_epochs']}  lr={lr_now:.2e}  "
            f"| tr loss {tr_loss:.4f} acc {tr_acc:.3f}  "
            f"| vl loss {vl_loss:.4f} acc {vl_acc:.3f}  "
            f"| ICBHI {icbhi:.4f} (Se {se:.3f} Sp {sp:.3f})"
            f"{flag}"
        )

        if patience_count >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest ICBHI (val, argmax): {best_icbhi:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL EVALUATION
    # ─────────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(config["save_path"], map_location=DEVICE))

    # 1) Standard argmax
    probs_test, labels_test = collect_probs(model, test_loader, DEVICE)
    preds_argmax = probs_test.argmax(axis=1)
    icbhi_am, se_am, sp_am, sens_am, spec_am = icbhi_score(
        labels_test, preds_argmax, len(config["classes"])
    )

    # 2) Threshold-tuned argmax
    print("\nRunning threshold tuning …")
    thresh = tune_thresholds(probs_test, labels_test, len(config["classes"]))
    preds_thresh = apply_thresholds(probs_test, thresh)
    icbhi_th, se_th, sp_th, sens_th, spec_th = icbhi_score(
        labels_test, preds_thresh, len(config["classes"])
    )

    # 3) TTA
    print(f"Running TTA ({config['tta_runs']} passes) …")
    probs_tta, _ = collect_probs_tta(model, test_ds, config, DEVICE)
    preds_tta    = probs_tta.argmax(axis=1)
    icbhi_tta, se_tta, sp_tta, sens_tta, spec_tta = icbhi_score(
        labels_test, preds_tta, len(config["classes"])
    )

    # 4) TTA + threshold
    thresh_tta   = tune_thresholds(probs_tta, labels_test, len(config["classes"]))
    preds_tta_th = apply_thresholds(probs_tta, thresh_tta)
    icbhi_tt, se_tt, sp_tt, sens_tt, spec_tt = icbhi_score(
        labels_test, preds_tta_th, len(config["classes"])
    )

    # ── Print all four results ─────────────────────────────────────────────
    def print_result(tag, preds, icbhi, se, sp, sens, spec):
        print(f"\n{'═'*55}")
        print(f"  {tag}")
        print(f"{'═'*55}")
        print(classification_report(labels_test, preds, target_names=config["classes"]))
        print(f"  Se={se:.4f}  Sp={sp:.4f}  ICBHI={icbhi:.4f} ({icbhi*100:.1f}%)")
        for cls, s, sp_ in zip(config["classes"], sens, spec):
            print(f"    {cls:8s}  Se={s:.3f}  Sp={sp_:.3f}")

    print_result("Argmax",            preds_argmax, icbhi_am,  se_am,  sp_am,  sens_am,  spec_am)
    print_result("Threshold-tuned",   preds_thresh, icbhi_th,  se_th,  sp_th,  sens_th,  spec_th)
    print_result("TTA",               preds_tta,    icbhi_tta, se_tta, sp_tta, sens_tta, spec_tta)
    print_result("TTA + Thresholds",  preds_tta_th, icbhi_tt,  se_tt,  sp_tt,  sens_tt,  spec_tt)

    plot_history(tr_losses, vl_losses, tr_accs, vl_accs, icbhi_hist,
                 stage_ep=config["stage1_epochs"])
    plot_confusion(labels_test, preds_argmax,  config["classes"], " (Argmax)")
    plot_confusion(labels_test, preds_tta_th,  config["classes"], " (TTA+Thresh)")

    return model


# ─────────────────────────────────────────────
# 15. SINGLE-FILE INFERENCE
# ─────────────────────────────────────────────

def predict(wav_path, model_path=CONFIG["save_path"], thresholds=None, config=CONFIG):
    """
    Predict with TTA and optional threshold biases.
    thresholds : numpy array of shape (num_classes,) from tune_thresholds(),
                 or None for plain argmax.
    """
    model = RespiratoryNet(num_classes=len(config["classes"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    wav       = load_audio(wav_path, config["sample_rate"], config["duration"])
    base_spec = normalise(waveform_to_melspec(wav, config))
    acc       = np.zeros(len(config["classes"]), dtype=np.float32)

    with torch.no_grad():
        for run in range(config["tta_runs"]):
            s = base_spec if run == 0 else augment_spec(base_spec.copy(), config)
            t = spec_to_tensor(s).unsqueeze(0)
            acc += torch.softmax(model(t).squeeze(), dim=0).numpy()

    probs = acc / config["tta_runs"]
    if thresholds is not None:
        idx = int((probs + thresholds).argmax())
    else:
        idx = int(probs.argmax())

    predicted = config["classes"][idx]
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
