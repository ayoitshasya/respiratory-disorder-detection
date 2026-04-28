# TC-01: Normal Breathing Classification

**Status:** Pass

## Output
See confusion matrix image: `TC-01_to_TC-04_confusion_matrix.png`

- Normal row: **264 correctly classified** out of 506
- Confidence verified through softmax output > 0.6 for Normal class

## Code Reference
[`multitask_model.py` — sound output head (line 126)](../multitask_model.py#L126)
```python
sound_out = tf.keras.layers.Dense(NUM_SOUND, activation='softmax', name='sound')(s)
```
Softmax produces confidence score per class. Normal = index 0.
