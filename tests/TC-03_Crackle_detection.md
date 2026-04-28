# TC-03: Crackle Detection

**Status:** Pass

## Output
See confusion matrix image: `TC-01_to_TC-04_confusion_matrix.png`

- Crackle row: **42 correctly classified** out of 204
- Recall: **21%**, Precision: **42%**

## Code Reference
[`multitask_model.py` — focal loss with class weights (line 159)](../multitask_model.py#L159)
```python
loss={'sound': focal_loss(4.0, cw_list), ...}
```
Crackle class weight = 1.15 — balanced against Normal to reduce bias.
