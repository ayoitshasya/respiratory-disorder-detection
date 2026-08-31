"""
hf_lung_labels.py
------------------
Parsing and 4-class label mapping for HF_Lung_V1 event-timestamped labels.

Label file format (one event per line): "<TYPE> <start HH:MM:SS.mmm> <end HH:MM:SS.mmm>"
TYPE in {I, E, D, Wheeze, Stridor, Rhonchi}. I/E are breath-phase markers
(inhalation/exhalation) and are NOT part of the 4-class sound label set —
they're used only to confirm a window overlaps active breathing, never to
assign a class.

4-class mapping (matches ICBHI's Normal/Crackle/Wheeze/Both):
    D (discontinuous adventitious sound) alone -> Crackle
    Wheeze alone                               -> Wheeze
    D overlapping a Wheeze event                -> Both
    event-free region (no D/Wheeze/Stridor/Rhonchi overlap) -> Normal
    Stridor, Rhonchi -> NOT force-mapped into the 4 classes. Logged
        separately (see log_stridor_rhonchi) for awareness only.

Window = 1.0s (16,000 samples @ 16kHz), centered on each event's midpoint,
shifted inward at clip boundaries so it always stays within [0, 15]s.
See scripts/hf_lung_step2_label_counts.py for the math derivation.
"""

import os
import re
import bisect

WINDOW_DUR = 1.0          # seconds — derived to match TARGET_FRAMES=63 @ hop_length=256, SR=16000
CLIP_DUR = 15.0            # seconds — every HF_Lung clip is exactly 15s

ADVENTITIOUS_TYPES = {'D', 'Wheeze', 'Stridor', 'Rhonchi'}
BREATH_PHASE_TYPES = {'I', 'E'}
LOGGED_ONLY_TYPES = {'Stridor', 'Rhonchi'}

_TS_RE = re.compile(r'^(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$')


def _parse_timestamp(ts):
    """'HH:MM:SS.mmm' -> seconds (float)."""
    m = _TS_RE.match(ts)
    if not m:
        raise ValueError(f"Unrecognised timestamp format: {ts!r}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def parse_label_file(path):
    """
    Parse a HF_Lung _label.txt file.

    Returns:
        list of (event_type: str, start: float, end: float) sorted by start.
    """
    events = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed label line in {path}: {line!r}")
            etype, start_s, end_s = parts
            start, end = _parse_timestamp(start_s), _parse_timestamp(end_s)
            if end < start:
                raise ValueError(f"End before start in {path}: {line!r}")
            events.append((etype, start, end))
    events.sort(key=lambda e: e[1])
    return events


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def _center_window(center, clip_dur=CLIP_DUR, window_dur=WINDOW_DUR):
    """Center a fixed-length window on `center`, shifted inward at clip edges."""
    half = window_dur / 2.0
    start = center - half
    end = center + half
    if start < 0:
        start, end = 0.0, window_dur
    elif end > clip_dur:
        start, end = clip_dur - window_dur, clip_dur
    # Defensive fallback — should never trigger given valid [0, clip_dur] timestamps.
    if start < 0 or end > clip_dur:
        start, end = max(0.0, start), min(clip_dur, end)
    return start, end


def classify_events(events):
    """
    Classify each D/Wheeze event into a windowed candidate, per-event overlap
    check (NOT a merged-intersection window): each D or Wheeze event gets its
    own window centered on ITS OWN midpoint, relabeled 'Both' if it overlaps
    an event of the other type. This means an overlapping D+Wheeze region can
    produce two 'Both' windows (one centered on the D event, one on the
    Wheeze event) rather than one window at the intersection — documented
    here so it isn't mistaken for the alternative design later.

    Returns:
        dict with keys:
          'windows': list of dicts {label, start, end, window_start, window_end,
                     source_type, source_start, source_end}
          'stridor_rhonchi': list of dicts {event_type, start, end} (logged only)
    """
    d_events = [(s, e) for t, s, e in events if t == 'D']
    wheeze_events = [(s, e) for t, s, e in events if t == 'Wheeze']
    logged = [{'event_type': t, 'start': s, 'end': e}
              for t, s, e in events if t in LOGGED_ONLY_TYPES]

    windows = []

    for s, e in d_events:
        overlaps_wheeze = any(_overlaps(s, e, ws, we) for ws, we in wheeze_events)
        label = 'Both' if overlaps_wheeze else 'Crackle'
        center = (s + e) / 2.0
        ws, we = _center_window(center)
        windows.append({'label': label, 'window_start': ws, 'window_end': we,
                         'source_type': 'D', 'source_start': s, 'source_end': e})

    for s, e in wheeze_events:
        overlaps_d = any(_overlaps(s, e, ds, de) for ds, de in d_events)
        label = 'Both' if overlaps_d else 'Wheeze'
        center = (s + e) / 2.0
        ws, we = _center_window(center)
        windows.append({'label': label, 'window_start': ws, 'window_end': we,
                         'source_type': 'Wheeze', 'source_start': s, 'source_end': e})

    return {'windows': windows, 'stridor_rhonchi': logged}


def find_normal_gaps(events, clip_dur=CLIP_DUR, window_dur=WINDOW_DUR):
    """
    Find time regions free of ANY adventitious event (D/Wheeze/Stridor/Rhonchi —
    all four, since Stridor/Rhonchi regions aren't silence either even though
    they're not part of the 4-class set). I/E overlap is fine and ignored.

    Returns:
        list of (gap_start, gap_end) tuples, only gaps >= window_dur, sorted
        by descending length (biggest gaps first — best sampling candidates).
    """
    occupied = sorted([(s, e) for t, s, e in events if t in ADVENTITIOUS_TYPES])
    merged = []
    for s, e in occupied:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gaps = []
    cursor = 0.0
    for s, e in merged:
        if s - cursor >= window_dur:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if clip_dur - cursor >= window_dur:
        gaps.append((cursor, clip_dur))

    gaps.sort(key=lambda g: -(g[1] - g[0]))
    return gaps


def sample_normal_windows(events, n_needed, clip_dur=CLIP_DUR, window_dur=WINDOW_DUR):
    """
    Greedily tile up to n_needed non-overlapping window_dur windows into the
    event-free gaps (biggest gaps first). Returns fewer than n_needed if the
    clip doesn't have enough event-free time — callers should track shortfall.

    Returns:
        list of dicts {label: 'Normal', window_start, window_end}
    """
    gaps = find_normal_gaps(events, clip_dur, window_dur)
    windows = []
    for gap_start, gap_end in gaps:
        cursor = gap_start
        while cursor + window_dur <= gap_end and len(windows) < n_needed:
            windows.append({'label': 'Normal', 'window_start': cursor,
                             'window_end': cursor + window_dur})
            cursor += window_dur
        if len(windows) >= n_needed:
            break
    return windows


def extract_date_key(filename):
    """
    Extract the date-grouping proxy key from a HF_Lung filename (pseudo-patient
    ID — NOT a real patient ID, see README: files sharing a date are LIKELY
    from the same subject, but this is an approximation, not a guarantee).

    steth_yyyymmdd_HH_MM_ss.wav      -> 'yyyymmdd'
    trunc_yyyy-mm-dd-HH-MM-ss-LX_N.wav -> 'yyyy-mm-dd'
    """
    base = os.path.basename(filename)
    m = re.match(r'^steth_(\d{8})_', base)
    if m:
        return m.group(1)
    m = re.match(r'^trunc_(\d{4}-\d{2}-\d{2})-', base)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract date key from filename: {filename!r}")
