"""
export_tflite.py
----------------
Convert trained Keras model to TFLite with int8 post-training quantization.
Reports accuracy before and after quantization, and final file size.
"""

import os
import numpy as np
import tensorflow as tf
from preprocessing import load_feature


def get_representative_dataset(manifest, n_samples=100, label_col='sound_label'):
    """
    Generate a representative dataset for int8 calibration.
    Uses n_samples random samples from the val split.

    Args:
        manifest:   Full manifest DataFrame
        n_samples:  Number of calibration samples
        label_col:  Unused — kept for signature consistency

    Returns:
        Generator function yielding [input_tensor]
    """
    val_manifest = manifest[manifest['split'] == 'val'].sample(
        min(n_samples, len(manifest[manifest['split'] == 'val'])),
        random_state=42
    )

    samples = []
    for _, row in val_manifest.iterrows():
        feat = load_feature(row['features_path'])
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        samples.append(feat.astype(np.float32))

    def representative_dataset_gen():
        for feat in samples:
            yield [np.expand_dims(feat, axis=0)]

    return representative_dataset_gen


def convert_to_tflite(model, manifest, output_path, quantize=True, n_calib_samples=100):
    """
    Convert a Keras model to TFLite (optionally with int8 quantization).

    Args:
        model:            Trained Keras model
        manifest:         Full manifest DataFrame
        output_path:      Path to save .tflite file
        quantize:         If True, apply int8 post-training quantization
        n_calib_samples:  Number of calibration samples for quantization

    Returns:
        Path to saved .tflite file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("Applying int8 post-training quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = get_representative_dataset(
            manifest, n_samples=n_calib_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"TFLite model saved: {output_path}")
    print(f"File size: {size_kb:.1f} KB")

    return output_path


def evaluate_tflite(tflite_path, manifest, num_classes=4, label_col='sound_label'):
    """
    Evaluate a TFLite model on the test split.

    Args:
        tflite_path: Path to .tflite file
        manifest:    Full manifest DataFrame
        num_classes: Number of classes
        label_col:   Label column name

    Returns:
        float accuracy
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = None, None

    # Check if model is quantized
    if input_details[0]['dtype'] == np.int8:
        input_scale      = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]

    test_manifest = manifest[manifest['split'] == 'test']
    correct = 0

    for _, row in test_manifest.iterrows():
        feat = load_feature(row['features_path'])
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        feat = np.expand_dims(feat, axis=0).astype(np.float32)

        if input_scale is not None:
            feat = (feat / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], feat)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred   = np.argmax(output)

        if pred == row[label_col]:
            correct += 1

    accuracy = correct / len(test_manifest)
    print(f"TFLite test accuracy: {accuracy:.4f} ({correct}/{len(test_manifest)})")
    return accuracy


def export_and_report(model, manifest, exp_name, results_dir, num_classes=4):
    """
    Full export pipeline: convert → evaluate → report size and accuracy delta.

    Args:
        model:       Trained Keras model
        manifest:    Full manifest DataFrame
        exp_name:    Experiment name (used for file naming)
        results_dir: Directory to save outputs
        num_classes: Number of output classes
    """
    import json
    os.makedirs(results_dir, exist_ok=True)

    # Original float32 accuracy on test
    print("\nEvaluating original float32 model...")
    test_manifest = manifest[manifest['split'] == 'test']
    X_test, y_test = [], []
    for _, row in test_manifest.iterrows():
        feat = load_feature(row['features_path'])
        if feat.ndim == 2:
            feat = feat[..., np.newaxis]
        X_test.append(feat)
        y_test.append(row['sound_label'])
    X_test  = np.array(X_test, dtype=np.float32)
    y_test  = np.array(y_test)
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    float_acc = np.mean(y_pred == y_test)
    print(f"Float32 accuracy: {float_acc:.4f}")

    # Convert and evaluate quantized model
    tflite_path = os.path.join(results_dir, f'{exp_name}_int8.tflite')
    convert_to_tflite(model, manifest, tflite_path, quantize=True)
    int8_acc = evaluate_tflite(tflite_path, manifest, num_classes)

    size_kb   = os.path.getsize(tflite_path) / 1024
    acc_drop  = float_acc - int8_acc

    report = {
        'experiment':        exp_name,
        'float32_accuracy':  round(float_acc, 4),
        'int8_accuracy':     round(int8_acc,  4),
        'accuracy_drop':     round(acc_drop,  4),
        'tflite_size_kb':    round(size_kb,   1),
        'target_met':        size_kb < 200 and acc_drop < 0.03,
    }

    print(f"\n{'='*50}")
    print(f"TFLite Export Report — {exp_name}")
    print(f"{'='*50}")
    print(f"  Float32 accuracy : {float_acc:.4f}")
    print(f"  Int8 accuracy    : {int8_acc:.4f}")
    print(f"  Accuracy drop    : {acc_drop:.4f} ({'OK' if acc_drop < 0.03 else 'EXCEEDS 3% THRESHOLD'})")
    print(f"  Model size       : {size_kb:.1f} KB ({'OK' if size_kb < 200 else 'EXCEEDS 200KB TARGET'})")

    report_path = os.path.join(results_dir, f'{exp_name}_tflite_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report
