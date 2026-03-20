# TC-10: Diagnosis Head Output

**Status:** Pass

## Output
See diagnosis confusion matrix image: `TC-10_diagnosis_confusion_matrix.png`

- COPD row: **569 correctly classified** out of 587
- COPD Precision: 0.77, Recall: 0.97
- Diagnosis head successfully predicts patient condition from audio alone

## Code Reference
[`multitask_model.py` — diagnosis output head (line 130)](../multitask_model.py#L130)
```python
diag_out = tf.keras.layers.Dense(NUM_DIAGNOSIS, activation='softmax', name='diagnosis')(d)
```
7-class softmax output: Healthy / COPD / URTI / Bronchiectasis / Pneumonia / Bronchiolitis / Other

The diagnosis head shares the CNN backbone with the sound head — no extra computation needed.
