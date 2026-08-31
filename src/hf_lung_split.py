"""
hf_lung_split.py
------------------
Size-aware, date-grouped train/val/test split for HF_Lung_V1.

Unlike src/data_loader.py's make_patient_splits() (random shuffle then slice
by fraction of PATIENT COUNT), this uses a greedy largest-first bin-packing
over SAMPLE COUNT per date-group, because HF_Lung's date-group sizes are
extremely skewed (min 1 file, max 676 files per date — see Step 0 findings).
Random shuffling of date-group order would let the ~60/20/20 ratio drift
significantly from the actual per-window sample ratio purely by chance of
which huge date lands where. Greedy assignment (largest date first, each
date going to whichever split is currently furthest below its target share)
keeps the realized sample ratio close to the target regardless of skew.

The date key is a PSEUDO-PATIENT PROXY, not a real patient ID (see
hf_lung_labels.extract_date_key) — grouping by it prevents date-level leakage
(same likely-subject's clips split across train/val/test), but it is an
approximation per the HF_Lung README, not a guarantee.
"""
import collections


def compute_date_sizes(per_file_windows, date_key_fn):
    """
    date_key -> total window count (Normal + adventitious) across all files
    sharing that date.
    """
    sizes = collections.Counter()
    file_to_date = {}
    for wav_name, info in per_file_windows.items():
        date_key = date_key_fn(wav_name)
        file_to_date[wav_name] = date_key
        sizes[date_key] += len(info['adventitious']) + len(info['normal'])
    return sizes, file_to_date


def greedy_size_aware_split(date_sizes, ratios=(('train', 0.6), ('val', 0.2), ('test', 0.2))):
    """
    Largest-date-first greedy bin-packing across splits, targeting `ratios`
    as SAMPLE-COUNT proportions (not date-count proportions).

    At each step, the next (largest remaining) date is assigned to whichever
    split currently has the lowest count/target ratio — i.e. whichever split
    is proportionally furthest behind its target share so far. This is the
    standard greedy heuristic for balanced multiway partitioning and keeps
    the realized ratio close to target even with heavily skewed group sizes.

    Returns:
        dict date_key -> split name
        dict split name -> assigned sample count (for reporting)
    """
    total = sum(date_sizes.values())
    targets = {name: frac for name, frac in ratios}
    assert abs(sum(targets.values()) - 1.0) < 1e-9, "ratios must sum to 1.0"

    counts = {name: 0 for name, _ in ratios}
    assignment = {}

    for date_key, size in sorted(date_sizes.items(), key=lambda kv: -kv[1]):
        # Ratio of current fill to target share; pick the most under-filled split.
        def fill_ratio(name):
            target_count = targets[name] * total
            return counts[name] / target_count if target_count > 0 else float('inf')

        chosen = min(counts.keys(), key=fill_ratio)
        assignment[date_key] = chosen
        counts[chosen] += size

    return assignment, counts


def apply_split(per_file_windows, assignment, file_to_date):
    """Attach a 'split' key to every window dict in per_file_windows, in place."""
    for wav_name, info in per_file_windows.items():
        split = assignment[file_to_date[wav_name]]
        for w in info['adventitious']:
            w['split'] = split
        for w in info['normal']:
            w['split'] = split
    return per_file_windows


def verify_no_date_leakage(assignment):
    """Each date_key maps to exactly one split by construction; sanity-assert it."""
    # assignment is date_key -> single split string, so leakage is structurally
    # impossible here — this just double-checks the invariant explicitly,
    # mirroring data_loader.make_patient_splits()'s explicit overlap asserts.
    splits_seen = collections.defaultdict(set)
    for date_key, split in assignment.items():
        splits_seen[date_key].add(split)
    bad = {k: v for k, v in splits_seen.items() if len(v) > 1}
    assert not bad, f"Date(s) assigned to multiple splits: {bad}"
