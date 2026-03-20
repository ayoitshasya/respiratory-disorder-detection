# TC-02: Wheeze Detection

**Status:** Pass

## Output
See confusion matrix image: `TC-01_to_TC-04_confusion_matrix.png`

- Wheeze row: **39 correctly classified** out of 60
- Recall: **65%** — model correctly detects majority of wheeze samples

## Code Reference
[`multitask_model.py` — focal loss with class weights (line 159)](../multitask_model.py#L159)
```python
loss={'sound': focal_loss(4.0, cw_list), ...}
```
Wheeze class weight = 1.72 — model penalised more for missing wheeze samples.
