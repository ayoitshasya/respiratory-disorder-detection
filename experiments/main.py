"""
main.py
-------
CLI entrypoint for the respiratory disorder detection pipeline.

Usage:
    python main.py --phase split
    python main.py --phase preprocess
    python main.py --phase train --config configs/experiment_config.json
    python main.py --phase evaluate --config configs/experiment_config.json
    python main.py --phase export --config configs/experiment_config.json
    python main.py --phase all --config configs/experiment_config.json
"""

import os
import sys
import json
import argparse
import random
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, 'data')
METADATA_CSV  = os.path.join(DATA_DIR, 'raw', 'diagnosis_metadata', 'training_metadata_with_diagnosis.csv')
CYCLES_DIR    = os.path.join(DATA_DIR, 'cycles')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SPLITS_DIR    = os.path.join(DATA_DIR, 'splits')
CHECKPTS_DIR  = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def phase_split():
    """Phase 1a: Patient-level splitting + manifest generation."""
    from data_loader import run_split_pipeline
    run_split_pipeline(METADATA_CSV, SPLITS_DIR, CYCLES_DIR, PROCESSED_DIR)


def phase_preprocess(config):
    """Phase 1b: Feature extraction and caching."""
    from data_loader import load_manifest
    from preprocessing import cache_features

    feature_type = config.get('input_type', 'mel')
    print(f"\nPreprocessing with feature type: {feature_type}")

    manifest = load_manifest(PROCESSED_DIR)
    manifest = cache_features(manifest, feature_type=feature_type, overwrite=False)

    # Verify a sample
    from preprocessing import get_feature_shape
    shape = get_feature_shape(manifest, feature_type)
    print(f"\nFeature shape: {shape}")


def phase_train(config):
    """Phase 2+3: Build model and train."""
    import tensorflow as tf
    tf.random.set_seed(SEED)

    from data_loader import load_manifest
    from models import get_model, print_model_summary
    from training import train
    from evaluation import plot_training_curves

    manifest     = load_manifest(PROCESSED_DIR)
    model_name   = config['model']
    feature_type = config.get('input_type', 'mel')

    # Determine input shape from cached features
    from preprocessing import get_feature_shape
    feat_shape = get_feature_shape(manifest, feature_type)
    # feat_shape is (freq, time) — models add channel internally
    input_shape = feat_shape[:2]

    print(f"\nBuilding model: {model_name}")
    print(f"Input shape: {input_shape}")
    model = get_model(model_name, input_shape, config['num_classes'])
    print_model_summary(model)

    model, history = train(model, manifest, config, CHECKPTS_DIR, RESULTS_DIR)
    plot_training_curves(history, config['experiment_name'], RESULTS_DIR)

    return model


def phase_evaluate(config, model=None):
    """Phase 4: Evaluate on test set."""
    import tensorflow as tf
    from data_loader import load_manifest
    from evaluation import evaluate_model

    manifest = load_manifest(PROCESSED_DIR)
    exp_name = config['experiment_name']

    if model is None:
        ckpt_path = os.path.join(CHECKPTS_DIR, f'{exp_name}_best.keras')
        print(f"Loading model from: {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)

    metrics = evaluate_model(model, manifest, config, RESULTS_DIR)
    return model, metrics


def phase_export(config, model=None):
    """Phase 6: TFLite export + quantization."""
    import tensorflow as tf
    from data_loader import load_manifest
    from export_tflite import export_and_report

    manifest = load_manifest(PROCESSED_DIR)
    exp_name = config['experiment_name']

    if model is None:
        ckpt_path = os.path.join(CHECKPTS_DIR, f'{exp_name}_best.keras')
        print(f"Loading model from: {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)

    report = export_and_report(
        model, manifest, exp_name, RESULTS_DIR,
        num_classes=config['num_classes']
    )
    return report


def main():
    parser = argparse.ArgumentParser(description='Respiratory Disorder Detection Pipeline')
    parser.add_argument('--phase', required=True,
                        choices=['split', 'preprocess', 'train', 'evaluate', 'export', 'all'],
                        help='Pipeline phase to run')
    parser.add_argument('--config', default='configs/experiment_config.json',
                        help='Path to experiment config JSON')
    args = parser.parse_args()

    config = load_config(args.config) if args.phase != 'split' else {}

    if args.phase == 'split':
        phase_split()

    elif args.phase == 'preprocess':
        phase_preprocess(config)

    elif args.phase == 'train':
        phase_train(config)

    elif args.phase == 'evaluate':
        phase_evaluate(config)

    elif args.phase == 'export':
        phase_export(config)

    elif args.phase == 'all':
        print("\n" + "="*60)
        print("RUNNING FULL PIPELINE")
        print("="*60)
        phase_split()
        phase_preprocess(config)
        model = phase_train(config)
        model, _ = phase_evaluate(config, model)
        phase_export(config, model)
        print("\nPipeline complete!")


if __name__ == '__main__':
    main()
