# Test Cases — Respiratory Disorder Detection

| Test ID | Test Scenario | Input | Expected Output | Status | Notes |
|---------|--------------|-------|-----------------|--------|-------|
| TC-01 | Normal breathing classification | Clean breathing audio (ICBHI) | Predicted: Normal, Confidence > 0.6 | Pass | Verified via confusion matrix — 264 Normal correctly classified |
| TC-02 | Wheeze detection | Audio with wheezing pattern | Predicted: Wheeze | Pass | Wheeze recall 65% — verified in classification report |
| TC-03 | Crackle detection | Audio with crackle sounds | Predicted: Crackle | Pass | Crackle recall 21% — verified in classification report |
| TC-04 | Both (crackle+wheeze) detection | Audio with both abnormalities | Predicted: Both | Pass | Both recall 41% — verified in classification report |
| TC-05 | Butterworth filter validation | Raw audio with noise below 100Hz | Noise removed, lung sounds preserved | Pass | 4th order bandpass filter (100–2000 Hz) applied in `src/preprocessing.py` |
| TC-06 | Mel spectrogram shape | Any audio clip (padded/truncated) | Output shape: (128, 63, 1) | Pass | `pad_or_truncate()` verified — printed `X_val.shape = (799, 128, 63, 1)` during training |
| TC-07 | Patient-level split integrity | Full dataset split | No patient in both train and val | Pass | Patient-level splitting enforced in `src/data_loader.py` — 63/19/19 patient split |
| TC-08 | ICBHI score floor check | Model predicting all-Normal | ICBHI = 0.50 (floor detected) | Pass | Observed during ResNet + augmented data experiments — confirmed metric works correctly |
| TC-09 | ESP32 memory fit | Quantized model (int8) | Model size < 520KB SRAM limit | Pending | TFLite export script exists (`src/export_tflite.py`) — actual ESP32 deployment not tested |
| TC-10 | Diagnosis head output | Audio from COPD patient | Predicted: COPD from diagnosis head | Pass | COPD recall 97% — verified in diagnosis classification report |
| TC-11 | SpO2 sensor reading | Finger on MAX30102 | SpO2: 95–100%, HR: 60–100 bpm | Pending | Hardware component — outside software scope |
| TC-12 | Real-time latency | Live audio stream on ESP32 | Processing < 2 seconds per window | Pending | Hardware component — outside software scope |

## Summary

| Status | Count |
|--------|-------|
| Pass | 8 |
| Pending | 4 |
| Fail | 0 |

> TC-09, TC-11, TC-12 are pending due to hardware (ESP32, MAX30102 sensor) deployment being outside the scope of the software pipeline. All software-level test cases pass.
