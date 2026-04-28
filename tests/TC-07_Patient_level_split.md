# TC-07: Patient-Level Split Integrity

**Status:** Pass

## Output
Train/Val/Test splits contain no overlapping patients:
- Train: 63 patients, 2,891 samples
- Val: 19 patients, 799 samples
- Test: 19 patients, 1,629 samples

No patient ID appears in more than one split — verified by `data_loader.py`.

## Code Reference
[`src/data_loader.py` — patient-level splitting](../src/data_loader.py)

Splitting is done at the **patient level**, not the sample level. All recordings from one patient stay in the same split. This prevents data leakage — without this, the model could memorise a specific patient's breathing pattern rather than learning disease features.
