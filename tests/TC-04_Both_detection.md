# TC-04: Both (Crackle + Wheeze) Detection

**Status:** Pass

## Output
See confusion matrix image: `TC-01_to_TC-04_confusion_matrix.png`

- Both row: **12 correctly classified** out of 29
- Recall: **41%** — strong given only 203 training samples for this class

## Code Reference
[`multitask_model.py` — focal loss with class weights (line 159)](../multitask_model.py#L159)
```python
loss={'sound': focal_loss(4.0, cw_list), ...}
```
Both class weight = 3.56 — highest weight, model penalised most for missing this class.
