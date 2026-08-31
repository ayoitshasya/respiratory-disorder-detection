"""
evaluation.py
-------------
Model evaluation: confusion matrix, per-class metrics, ICBHI score,
training curves, and model size reporting.
All outputs saved to results/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from training import icbhi_score
from preprocessing import load_feature


SOUND_NAMES = ['normal', 'crackle', 'wheeze', 'both']
DIAG_NAMES  = ['Healthy', 'COPD', 'URTI', 'Bronchiectasis', 'LRTI', 'Asthma']


def evaluate_model(model, manifest, config, results_dir, class_names=None):
    """
    Full evaluation on the test split.

    Args:
        model:       Trained Keras model
        manifest:    Full manifest DataFrame
        config:      Experiment config dict
        results_dir: Directory to save outputs
        class_names: List of class name strings

    Returns:
        dict with all metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    exp_name    = config['experiment_name']
    num_classes = config['num_classes']
    label_col   = config.get('label_col', 'sound_label')

    if class_names is None:
        class_names = SOUND_NAMES if num_classes == 4 else DIAG_NAMES

    test_manifest = manifest[manifest['split'] == 'test']

    # Load test features
    X_test, y_test = [], []
    for _, row in test_manifest.iterrows():
        feat = load_feature(row['features_path'])
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X_test.append(feat)
        y_test.append(row[label_col])
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # ICBHI score
    score = icbhi_score(y_test, y_pred, num_classes)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=class_names)

    print(f"\n{'='*60}")
    print(f"TEST SET — {exp_name}")
    print(f"{'='*60}")
    print(report_str)
    print(f"ICBHI Score (SE+SP)/2 : {score:.4f}")

    # Save report CSV
    report_df   = pd.DataFrame(report).transpose()
    report_path = os.path.join(results_dir, f'{exp_name}_report.csv')
    report_df.to_csv(report_path)

    # Save ICBHI score
    metrics = {'icbhi_score': score, 'accuracy': report['accuracy']}
    for cls in class_names:
        if cls in report:
            metrics[f'{cls}_f1']      = report[cls]['f1-score']
            metrics[f'{cls}_recall']  = report[cls]['recall']
            metrics[f'{cls}_precision'] = report[cls]['precision']

    metrics_path = os.path.join(results_dir, f'{exp_name}_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrices
    plot_confusion_matrix(y_test, y_pred, class_names, exp_name, results_dir)

    # Model size
    print_model_size(model, exp_name, results_dir)

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, exp_name, results_dir):
    """Save raw and normalised confusion matrix plots."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{exp_name} — Confusion Matrix (Counts)')
    axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{exp_name} — Confusion Matrix (Normalised)')
    axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(results_dir, f'{exp_name}_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {path}")


def plot_training_curves(history, exp_name, results_dir):
    """Save training/validation loss, accuracy, and ICBHI score curves."""
    os.makedirs(results_dir, exist_ok=True)

    has_icbhi = 'val_icbhi' in history

    n_plots = 3 if has_icbhi else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    axes[0].plot(history['accuracy'],     label='Train')
    axes[0].plot(history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['loss'],     label='Train')
    axes[1].plot(history['val_loss'], label='Val')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    if has_icbhi:
        axes[2].plot(history['val_icbhi'], label='Val ICBHI', color='green')
        axes[2].set_title('ICBHI Score (Val)'); axes[2].set_xlabel('Epoch')
        axes[2].legend(); axes[2].grid(True)

    plt.suptitle(exp_name)
    plt.tight_layout()
    path = os.path.join(results_dir, f'{exp_name}_training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved: {path}")


def print_model_size(model, exp_name, results_dir):
    """Print and save model size information."""
    total_params   = model.count_params()
    float32_mb     = total_params * 4 / 1024 / 1024
    int8_kb        = total_params / 1024

    print(f"\nModel size — {exp_name}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  float32 size  : {float32_mb:.2f} MB")
    print(f"  ~int8 TFLite  : {int8_kb:.0f} KB (estimate)")

    size_info = {
        'experiment':     exp_name,
        'parameters':     total_params,
        'float32_mb':     round(float32_mb, 3),
        'estimated_int8_kb': round(int8_kb, 1),
    }

    import json
    path = os.path.join(results_dir, f'{exp_name}_model_size.json')
    with open(path, 'w') as f:
        json.dump(size_info, f, indent=2)


def compare_models(results_dir):
    """
    Print a comparison table of all models evaluated so far.
    Reads all *_metrics.json files in results_dir.
    """
    import glob, json
    files   = glob.glob(os.path.join(results_dir, '*_metrics.json'))
    records = []
    for fp in sorted(files):
        with open(fp) as f:
            m = json.load(f)
        exp = os.path.basename(fp).replace('_metrics.json', '')
        records.append({'experiment': exp, **m})

    if not records:
        print("No metrics files found.")
        return

    df = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
