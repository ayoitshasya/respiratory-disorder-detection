# TC-06: Mel Spectrogram Shape

**Status:** Pass

## Output
Printed during training run:
```
Val: (799, 128, 63, 1)
Train: (2891, 128, 63, 1)
```
Shape per sample = **(128, 63, 1)** — confirmed.

## Code Reference
[`multitask_model.py` — pad_or_truncate function (line 23)](../multitask_model.py#L23)
```python
def pad_or_truncate(feat, t=TARGET_FRAMES):
    c = feat.shape[-1]
    if c < t:
        feat = np.pad(feat, [(0,0), (0, t-c)])
    else:
        feat = feat[..., :t]
    return feat
```
- 128 mel frequency bins (N_MELS=128)
- 63 time frames (TARGET_FRAMES=63)
- 1 channel (grayscale mel spectrogram)
