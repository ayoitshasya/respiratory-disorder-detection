# TC-08: ICBHI Score Floor Check

**Status:** Pass

## Output
See training curves image: `TC-08_icbhi_floor_training_curves.png`

During ResNet + 21k augmented data experiment, ICBHI was stuck at **0.50** for all epochs — confirming the metric correctly detects when a model predicts only the majority class (Normal).

Training log output:
```
val_icbhi: 0.5000
val_icbhi: 0.5000
val_icbhi: 0.5000
...
```

## Code Reference
[`multitask_model.py` — icbhi_score function (line 51)](../multitask_model.py#L51)
```python
def icbhi_score(y_true, y_pred, n=NUM_SOUND):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    se, sp = [], []
    for i in range(n):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=cm.sum()-tp-fn-fp
        se.append(tp/(tp+fn) if tp+fn>0 else 0.0)
        sp.append(tn/(tn+fp) if tn+fp>0 else 0.0)
    return (np.mean(se)+np.mean(sp))/2.0
```
When model predicts all-Normal: SE=0 for Crackle/Wheeze/Both, SP=1 for all → average = 0.50.
