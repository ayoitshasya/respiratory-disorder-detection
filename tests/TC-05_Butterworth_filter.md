# TC-05: Butterworth Filter Validation

**Status:** Pass

## Output
Filter applied to all audio during preprocessing. Frequencies outside 100–2000 Hz are removed.

## Code Reference
[`src/preprocessing.py` — Butterworth bandpass filter](../src/preprocessing.py)
```python
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(audio, lowcut=100, highcut=2000, sr=22050, order=4):
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)
```
- Removes heart noise below 100 Hz
- Removes equipment/electrical noise above 2000 Hz
- Lung sounds (crackles, wheezes) preserved in 100–2000 Hz range
