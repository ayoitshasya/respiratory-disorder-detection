"""
Step 2 driver — run the HF_Lung label mapping across the full dataset and
report window counts per class (Crackle/Wheeze/Both/Normal), plus a separate
Stridor/Rhonchi occurrence log. No audio is touched here — labels only.
"""
import os
import sys
import csv
import glob
import collections

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hf_lung_labels import parse_label_file, classify_events, sample_normal_windows

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = {
    'train': os.path.join(BASE, 'data', 'raw', 'hf_lung_v1', 'train'),
    'test': os.path.join(BASE, 'data', 'raw', 'hf_lung_v1', 'test'),
}
STRIDOR_RHONCHI_CSV = os.path.join(BASE, 'data', 'raw', 'hf_lung_v1', 'stridor_rhonchi_log.csv')


def build_all_windows():
    """
    Run the label parsing + classification across the full HF_Lung dataset.
    Pure data-building step, no printing — reused by Step 3's sample picker.

    Returns:
        per_file_windows: dict wav_name -> {split, adventitious: [...], normal: [...]}
        class_counts: collections.Counter over {Normal, Crackle, Wheeze, Both}
        stridor_rhonchi_rows: list of dicts (logged occurrences, not windowed)
        normal_shortfall_files: int
        malformed_files: list of (path, error) tuples
    """
    class_counts = collections.Counter()
    per_file_windows = {}   # filename -> list of window dicts (label, times, source split)
    stridor_rhonchi_rows = []
    normal_shortfall_files = 0
    malformed_files = []

    for split, root in ROOTS.items():
        label_files = sorted(glob.glob(os.path.join(root, '*_label.txt')))
        for lf in label_files:
            wav_name = os.path.basename(lf).replace('_label.txt', '.wav')
            try:
                events = parse_label_file(lf)
            except ValueError as e:
                malformed_files.append((lf, str(e)))
                continue

            result = classify_events(events)
            adv_windows = result['windows']
            n_needed = len(adv_windows)
            normal_windows = sample_normal_windows(events, n_needed)
            if len(normal_windows) < n_needed:
                normal_shortfall_files += 1

            for w in adv_windows:
                class_counts[w['label']] += 1
            class_counts['Normal'] += len(normal_windows)

            for row in result['stridor_rhonchi']:
                stridor_rhonchi_rows.append({
                    'split': split, 'source_file': wav_name, **row,
                })

            per_file_windows[wav_name] = {
                'split': split,
                'adventitious': adv_windows,
                'normal': normal_windows,
            }

    return per_file_windows, class_counts, stridor_rhonchi_rows, normal_shortfall_files, malformed_files


def main():
    (per_file_windows, class_counts, stridor_rhonchi_rows,
     normal_shortfall_files, malformed_files) = build_all_windows()

    os.makedirs(os.path.dirname(STRIDOR_RHONCHI_CSV), exist_ok=True)
    with open(STRIDOR_RHONCHI_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'source_file', 'event_type', 'start', 'end'])
        writer.writeheader()
        writer.writerows(stridor_rhonchi_rows)

    print("=== Step 2: window counts per class (full dataset) ===")
    total = sum(class_counts.values())
    for label in ['Normal', 'Crackle', 'Wheeze', 'Both']:
        c = class_counts[label]
        print(f"  {label:10s}: {c:6d}  ({c/total*100:.1f}%)")
    print(f"  {'TOTAL':10s}: {total:6d}")
    print()
    print(f"Files with fewer Normal windows than adventitious-event windows needed "
          f"(clip too densely labeled to fit a 1:1 match): {normal_shortfall_files}")
    print(f"Malformed label files skipped: {len(malformed_files)}")
    for lf, err in malformed_files[:10]:
        print(f"  {lf}: {err}")
    print()
    print(f"Stridor/Rhonchi occurrences logged (not windowed, not force-mapped): {len(stridor_rhonchi_rows)}")
    sr_counts = collections.Counter(r['event_type'] for r in stridor_rhonchi_rows)
    for t, c in sr_counts.items():
        print(f"  {t}: {c}")
    print(f"  -> saved to {STRIDOR_RHONCHI_CSV}")

    files_with_events = sum(1 for v in per_file_windows.values() if v['adventitious'])
    files_total = len(per_file_windows)
    print()
    print(f"Files with >=1 adventitious event: {files_with_events} / {files_total}")

    return per_file_windows, class_counts


if __name__ == '__main__':
    main()
