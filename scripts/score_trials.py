#!/usr/bin/env python3
"""Score every trial of the study from its per-tick record.

Reads the per-tick files pa<N>_<channel>_<task>_raw_data.csv (pilot:
<N>_<channel>_<task>_raw_data.csv) and writes, for each block, the
per-trial file with the same name ending in _data.csv. This is done for
the main study ("participant tests") and the pilot ("pilot tests"). For
the main study it also writes per-participant-summary_data.csv.

Usage:
    python scripts/score_trials.py          (from DATA_DIR)
    python score_trials.py                  (from DATA_DIR/scripts)
    python score_trials.py [DATA_DIR] [OUT_DIR]

DATA_DIR contains the "participant tests" and "pilot tests" folders.
With no arguments it is the parent of the folder holding this script, so
the script belongs in DATA_DIR/scripts/. OUT_DIR defaults to DATA_DIR.

Only the *_raw_data.csv files are read. The per-trial *_data.csv files
and per-participant-summary_data.csv are overwritten; no other file is
read or changed. Every block of both studies is scored before anything
is written, so a bad input file leaves every existing output as it was.

A trial with fewer than 20 ticks in the scoring window keeps its row,
with every metric empty, and does not enter the per-participant means.

Column meanings are given in COLUMNS.md in DATA_DIR. Standard library
only.
"""
import csv
import io
import math
import os
import re
import sys
from collections import defaultdict

# The first second is the target-acquisition phase, in which the force rises
# from rest toward the target. The metrics describe holding and tracking, so
# every metric uses the window from 1000 ms to the end of the trial.
WINDOW_START_MS = 1000
HOLD_MS = 3000          # continuous in-band time that counts as a success
BAND = 15.0             # tolerance band, +/- points around the target
MAX_LAG = 10            # best-lag search range, in ticks
MIN_SAMPLES = 20        # a window with fewer ticks is not scored
TARGET_AMP = 30.0       # amplitude of the tracking target, in points

# (folder, participant-id pattern, write a per-participant summary)
STUDIES = (
    ("participant tests", r"pa\d+", True),
    ("pilot tests", r"\d+", False),
)
CHANNELS = ("pitch", "intensity")
TASKS = ("maintenance", "tracking")

# Columns a per-tick file must have. SampleIndex only locates a bad row in
# error messages, so a file without it can still be scored.
MAINT_RAW_COLS = ["Trial", "TimeMs", "Target", "Force"]
TRACK_RAW_COLS = ["Trial", "Scenario", "SpeedPercent", "TimeMs", "Actual",
                  "Target"]

MAINT_COLS = ["Trial", "Target", "RMSE", "SD", "TIB", "Success",
              "CompletionTime", "Bias", "LongestStreak", "TimeToFirstEntry",
              "NSamples"]
TRACK_COLS = ["Trial", "Scenario", "SpeedPercent", "RMSE", "PearsonR",
              "BestLagR", "BestLagMs", "Gain", "Bias", "TIB", "NSamples"]

SUMMARY_FILE = "per-participant-summary_data.csv"
SUMMARY_COLS = ["Participant", "Channel", "Task", "BestLagMs", "BestLagR",
                "Bias", "CompletionTime", "Gain", "LongestStreak", "PearsonR",
                "RMSE", "SD", "Success", "TIB", "TimeToFirstEntry"]
# Per-trial columns averaged into the per-participant summary.
SUMMARY_KEYS = {
    "maintenance": ["Success", "TIB", "RMSE", "SD", "LongestStreak", "Bias",
                    "CompletionTime", "TimeToFirstEntry"],
    "tracking": ["PearsonR", "BestLagR", "BestLagMs", "TIB", "RMSE", "Gain",
                 "Bias"],
}


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
# Means, variances and correlations are summed left to right with sum();
# the least-squares sums in sinusoid_gain use math.fsum. The summation
# method can change the last bit of a result, which can move a value across
# a fourth-decimal boundary, so it is part of the definition of the
# published values.
def pearson(a, b):
    """Pearson r; NaN if fewer than 3 points or either series is constant."""
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def best_lag(actual, target, maxlag=MAX_LAG):
    """Highest r of actual[i] against target[i - lag], |lag| <= maxlag ticks.

    Returns (r, lag); a positive lag means the force followed the target.
    If no lag gives a defined correlation it returns (None, None), which is
    written as an empty cell.
    """
    best_r, best_l = None, None
    n = len(actual)
    for lag in range(-maxlag, maxlag + 1):
        a, t = [], []
        for i in range(n):
            j = i - lag
            if 0 <= j < len(target):
                a.append(actual[i])
                t.append(target[j])
        # A shift is used only if the overlap covers at least half of the
        # window (n // 2 ticks): a correlation over a short overlap is noisy
        # and could win the search by chance.
        if len(a) < n // 2:
            continue
        r = pearson(a, t)
        if math.isnan(r):
            continue
        if best_r is None or r > best_r:
            best_r, best_l = r, lag
    return best_r, best_l


def solve3(m, v):
    """Solve the 3x3 system m x = v by Gaussian elimination.

    Partial pivoting keeps the elimination stable when a leading entry is
    small.
    """
    a = [list(row) + [rhs] for row, rhs in zip(m, v)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0.0:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, 3):
            k = a[r][col] / a[col][col]
            for c in range(col, 4):
                a[r][c] -= k * a[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = ((a[r][3] - sum(a[r][c] * x[c] for c in range(r + 1, 3)))
                / a[r][r])
    return x


def sinusoid_gain(times_s, force, freq_hz):
    """Amplitude of the produced force at the target frequency, divided by
    the target amplitude.

    Least-squares fit of a + b*cos(2*pi*f*t) + c*sin(2*pi*f*t), returning
    sqrt(b^2 + c^2) / TARGET_AMP. Only the component at the target frequency
    counts. A ratio of standard deviations would also count power at other
    frequencies (tremor, jitter, dropouts) as tracking amplitude.
    """
    w = 2 * math.pi * freq_hz
    cols = ([1.0] * len(times_s),
            [math.cos(w * t) for t in times_s],
            [math.sin(w * t) for t in times_s])
    ata = [[math.fsum(p * q for p, q in zip(ci, cj)) for cj in cols]
           for ci in cols]
    aty = [math.fsum(p * y for p, y in zip(ci, force)) for ci in cols]
    coef = solve3(ata, aty)
    if coef is None:
        return None
    return math.hypot(coef[1], coef[2]) / TARGET_AMP


# ----------------------------------------------------------------------------
# Reading the per-tick files
# ----------------------------------------------------------------------------
def plain_path(path):
    """path without the Windows extended-length prefix added by io_path."""
    if os.name == "nt":
        if path.startswith("\\\\?\\UNC\\"):
            return "\\\\" + path[8:]
        if path.startswith("\\\\?\\"):
            return path[4:]
    return path


def read_rows(path, required):
    """Rows of a per-tick file as dicts, after checking its header."""
    try:
        with io.open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            absent = [c for c in required if c not in header]
            if absent:
                raise ValueError("%s: column(s) not found: %s"
                                 % (plain_path(path), ", ".join(absent)))
            return list(reader)
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError("%s: not a readable UTF-8 CSV file: %s"
                         % (plain_path(path), e)) from None


def cell_text(d, col):
    v = d.get(col)
    return v.strip() if v is not None and v.strip() else "?"


def where(path, d):
    """File, trial and SampleIndex of a raw row, for error messages."""
    return "%s, trial %s, SampleIndex %s" % (
        plain_path(path), cell_text(d, "Trial"), cell_text(d, "SampleIndex"))


def num(d, col, conv, path):
    """Numeric value of one cell of a raw row, converted with int or float.

    A missing, unparsable or non-finite value raises ValueError naming the
    file, trial and SampleIndex. A NaN in particular would pass through
    every comparison and sum without complaint and surface only as an
    unexplained empty or implausible metric.
    """
    text = d.get(col)
    if text is None or not text.strip():
        raise ValueError("%s: %s is missing" % (where(path, d), col))
    try:
        v = conv(text)
    except ValueError:
        raise ValueError("%s: %s is not a valid %s: %r"
                         % (where(path, d), col,
                            "integer" if conv is int else "number",
                            text)) from None
    if not math.isfinite(v):
        raise ValueError("%s: %s is not finite: %r"
                         % (where(path, d), col, text))
    return v


# ----------------------------------------------------------------------------
# Per-trial scoring
# ----------------------------------------------------------------------------
def unscored_row(path, trial, cols, n_samples, **ids):
    """Row for a trial with too few ticks in the window to be scored.

    The trial keeps its row, with its identifying columns and NSamples and
    every metric empty, so the file still lists every trial that was run
    and a short trial cannot be mistaken for one that was never recorded.
    """
    print("  warning: %s: trial %d has %d ticks in the scoring window "
          "(fewer than %d); its metrics are left empty"
          % (plain_path(path), trial, n_samples, MIN_SAMPLES))
    row = dict.fromkeys(cols)
    row.update(ids, Trial=trial, NSamples=n_samples)
    return row


def score_maintenance(path):
    """Per-trial metrics for holding a constant target."""
    trials = defaultdict(list)
    targets = {}
    for d in read_rows(path, MAINT_RAW_COLS):
        tr = num(d, "Trial", int, path)
        tg = num(d, "Target", float, path)
        trials[tr].append(
            (num(d, "TimeMs", int, path), tg, num(d, "Force", float, path)))
        # Every metric is taken against a single target per trial. If the
        # rows of a trial disagreed, the trial would silently be scored
        # against whichever value sorts first.
        if targets.setdefault(tr, tg) != tg:
            raise ValueError("%s: Target changes within the trial "
                             "(%r, then %r)"
                             % (where(path, d), targets[tr], tg))

    out, skipped = [], 0
    for tr in sorted(trials):
        rows = sorted(trials[tr])
        target = rows[0][1]
        win = [(t, f) for t, _, f in rows if t >= WINDOW_START_MS]
        if len(win) < MIN_SAMPLES:
            skipped += 1
            out.append(unscored_row(path, tr, MAINT_COLS, len(win),
                                    Target=int(target)))
            continue

        # Ticks are not evenly spaced, so each tick is weighted by the real
        # time elapsed since the tick before it; counting ticks would bias
        # the time measures toward stretches where ticks happen to be
        # denser. A tick stands for the interval that ends at it, and the
        # first tick of the window stands for the time since the window
        # opened.
        prev = WINDOW_START_MS
        in_band_ms = 0.0
        consec = 0.0
        longest = 0.0
        first_entry = None
        completion = None
        forces = []

        for t, f in win:
            dt = t - prev
            prev = t
            forces.append(f)
            inb = abs(f - target) <= BAND
            if inb:
                in_band_ms += dt
                if first_entry is None:
                    first_entry = (t - dt) - WINDOW_START_MS
                consec += dt
                if consec > longest:
                    longest = consec
                if completion is None and consec >= HOLD_MS:
                    # The streak reached HOLD_MS (consec - HOLD_MS) ms before
                    # this tick, i.e. inside the tick's own interval.
                    completion = (t - WINDOW_START_MS) - (consec - HOLD_MS)
            else:
                consec = 0.0

        # Population statistics (divide by n), so that RMSE^2 = SD^2 + Bias^2
        # holds exactly for the constant target.
        n = len(forces)
        mean = sum(forces) / n
        rmse = math.sqrt(sum((f - target) ** 2 for f in forces) / n)
        sd = math.sqrt(sum((f - mean) ** 2 for f in forces) / n)
        span = max(1.0, win[-1][0] - WINDOW_START_MS)

        out.append({
            "Trial": tr, "Target": int(target),
            "RMSE": rmse, "SD": sd, "TIB": in_band_ms / span,
            "Success": 1 if longest >= HOLD_MS else 0,
            "CompletionTime": (None if completion is None
                               else completion / 1000.0),
            "Bias": mean - target,
            "LongestStreak": longest / 1000.0,
            "TimeToFirstEntry": (None if first_entry is None
                                 else first_entry / 1000.0),
            "NSamples": n,
        })
    return out, skipped


def score_tracking(path):
    """Per-trial metrics for following the sinusoidal target."""
    trials = defaultdict(list)
    meta = {}
    for d in read_rows(path, TRACK_RAW_COLS):
        tr = num(d, "Trial", int, path)
        trials[tr].append((num(d, "TimeMs", int, path),
                           num(d, "Actual", float, path),
                           num(d, "Target", float, path)))
        # Scenario and SpeedPercent describe the whole trial, and the gain
        # fit takes SpeedPercent as the target frequency. If the rows of a
        # trial disagreed, the fit would use an arbitrary one of them.
        key = (num(d, "Scenario", int, path),
               num(d, "SpeedPercent", int, path))
        if meta.setdefault(tr, key) != key:
            raise ValueError("%s: Scenario/SpeedPercent change within the "
                             "trial (%d/%d, then %d/%d)"
                             % ((where(path, d),) + meta[tr] + key))

    out, skipped = [], 0
    for tr in sorted(trials):
        rows = sorted(trials[tr])
        sc, sp = meta[tr]
        win = [(a, t) for ms, a, t in rows if ms >= WINDOW_START_MS]
        if len(win) < MIN_SAMPLES:
            skipped += 1
            out.append(unscored_row(path, tr, TRACK_COLS, len(win),
                                    Scenario=sc, SpeedPercent=sp))
            continue
        times = [ms for ms, _, _ in rows if ms >= WINDOW_START_MS]
        actual = [a for a, _ in win]
        target = [t for _, t in win]
        n = len(actual)

        rmse = math.sqrt(sum((a - t) ** 2 for a, t in win) / n)
        r = pearson(actual, target)
        blr, shift = best_lag(actual, target)
        ma, mt = sum(actual) / n, sum(target) / n

        # The lag is found as a whole number of ticks. Ticks are not uniform
        # (about 62.7 ms on average, with jitter), so the shift is converted
        # to ms with this trial's own mean tick interval inside the window.
        tick_ms = (times[-1] - times[0]) / (n - 1)

        # The target was computed from the same clock that times the ticks,
        # so the fit uses the logged tick times (seconds from trial onset),
        # which keeps it on the target's own time base.
        gain = sinusoid_gain([ms / 1000.0 for ms in times], actual,
                             sp / 100.0)

        # Time in band, weighted by real inter-tick time as in maintenance,
        # with the band centred on the moving target.
        prev, in_band_ms = WINDOW_START_MS, 0.0
        for ms, a, t in rows:
            if ms < WINDOW_START_MS:
                continue
            dt = ms - prev
            prev = ms
            if abs(a - t) <= BAND:
                in_band_ms += dt
        tib = in_band_ms / max(1.0, rows[-1][0] - WINDOW_START_MS)

        out.append({
            "Trial": tr, "Scenario": sc, "SpeedPercent": sp,
            "RMSE": rmse, "PearsonR": r, "BestLagR": blr,
            "BestLagMs": None if shift is None else shift * tick_ms,
            "Gain": gain,
            "Bias": ma - mt, "TIB": tib, "NSamples": n,
        })
    return out, skipped


SCORERS = {"maintenance": (score_maintenance, MAINT_COLS),
           "tracking": (score_tracking, TRACK_COLS)}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def fmt(v, nd=4):
    """Cell text of a value; empty when the value is undefined.

    NaN and infinities count as undefined. Written out as 'nan' or 'inf'
    they would be read as numbers by some tools and would break the means
    taken over these files.
    """
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return ("%." + str(nd) + "f") % v
    return str(v)


def mean_of_cells(cells):
    """Mean of formatted cell values, leaving out empty and non-finite cells.

    An empty cell is an undefined value (for example CompletionTime of a
    trial that did not succeed, or any metric of a trial too short to be
    scored), so it must not count as zero.
    """
    vals = []
    for c in cells:
        try:
            v = float(c)
        except ValueError:
            continue
        if math.isfinite(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def participant_summary(pids, blocks):
    """One row per participant x channel x task: the mean of the trial values.

    The means are taken over the values as written to the per-trial files, so
    the summary can be reproduced from those files alone. CompletionTime is
    defined only on successful trials, so its mean covers those trials.
    """
    rows = []
    for pid in pids:
        for ch in CHANNELS:
            for task in TASKS:
                trials = blocks.get((pid, ch, task))
                if trials is None:
                    continue
                rec = {"Participant": pid, "Channel": ch, "Task": task}
                for k in SUMMARY_KEYS[task]:
                    rec[k] = fmt(mean_of_cells([t[k] for t in trials]))
                rows.append([rec.get(c, "") for c in SUMMARY_COLS])
    return rows


def participant_number(pid):
    return int(re.search(r"\d+", pid).group())


def io_path(path):
    """Absolute path usable for file access even when it is very long.

    Windows rejects paths over 260 characters unless they carry the
    extended-length prefix, and deeply nested data folders reach that limit.
    """
    path = os.path.abspath(path)
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def path_key(path):
    """Form of a path under which two spellings of one location are equal."""
    return os.path.normcase(os.path.abspath(plain_path(path)))


def check_output_path(path, raw_keys):
    """Raise ValueError if path could be one of the raw input files.

    The raw files are the only record of what was measured. The summaries
    can always be computed again from them; the raw files cannot be
    recovered from the summaries.
    """
    if os.path.basename(path).lower().endswith("_raw_data.csv"):
        raise ValueError("refusing to write a raw data file: %s"
                         % plain_path(path))
    if path_key(path) in raw_keys:
        raise ValueError("refusing to overwrite an input file: %s"
                         % plain_path(path))


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def make_dirs(path, created):
    """os.makedirs that records each folder it creates in created.

    A failed run removes those folders again, so that it leaves nothing
    behind, not even an empty folder.
    """
    if os.path.isdir(path):
        return
    parent = os.path.dirname(path)
    if parent and parent != path:
        make_dirs(parent, created)
    os.mkdir(path)
    created.append(path)


def stage_csv(path, header, rows):
    """Write a CSV to a new temporary file next to path and return its name.

    UTF-8 without BOM and with CRLF line endings. The line ending is set
    explicitly so the files come out byte-identical on every platform. The
    temporary file is in the destination folder because os.replace can
    swap a file in one step only within one file system.
    """
    folder, base = os.path.split(path)
    for _ in range(100):
        tmp = os.path.join(folder,
                           ".%s.%s.tmp" % (base, os.urandom(4).hex()))
        try:
            f = io.open(tmp, "x", encoding="utf-8", newline="\r\n")
        except FileExistsError:
            continue
        break
    else:
        raise OSError("no free temporary file name in %s"
                      % plain_path(folder))
    try:
        with f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        remove_quietly(tmp)
        raise
    return tmp


def write_all(outputs, raw_paths):
    """Write every output file (path, header, rows) into place.

    Each output is written first to a temporary file in its destination
    folder, and the destinations are replaced only when all of them are
    written. A failure while writing (disk full, no permission) leaves
    every destination as it was; a failure during the final replacements
    is reported with the number of files already replaced.
    """
    raw_keys = {path_key(p) for p in raw_paths}
    for path, _, _ in outputs:
        check_output_path(path, raw_keys)

    created, staged = [], []
    try:
        for path, header, rows in outputs:
            make_dirs(os.path.dirname(path), created)
            staged.append((stage_csv(path, header, rows), path))
        # os.replace fails on a destination that another program holds
        # open (a spreadsheet, typically) or that is read-only. Opening each
        # existing destination for writing, without writing to it, raises
        # that error now, before any file has been replaced, instead of
        # partway through the replacements.
        for _, path in staged:
            if os.path.exists(path):
                with io.open(path, "r+b"):
                    pass
    except BaseException as e:
        for tmp, _ in staged:
            remove_quietly(tmp)
        for folder in reversed(created):
            try:
                os.rmdir(folder)
            except OSError:
                pass
        if isinstance(e, OSError):
            raise OSError("%s; no output file was changed" % e) from e
        raise

    for i, (tmp, path) in enumerate(staged):
        try:
            os.replace(tmp, path)
        except OSError as e:
            for rest, _ in staged[i:]:
                remove_quietly(rest)
            raise OSError("cannot replace %s (%s); %d of %d output files "
                          "had already been replaced"
                          % (plain_path(path), e, i, len(staged))) from e


def score_study(src, dst, id_pattern, with_summary):
    """Score one study folder without writing anything.

    Returns the outputs to write, as (path, header, rows), and the paths of
    the raw files that were read.
    """
    name = re.compile(r"^(%s)_(%s)_(%s)_raw_data\.csv$"
                      % (id_pattern, "|".join(CHANNELS), "|".join(TASKS)))
    found = {}
    for fn in os.listdir(src):
        m = name.match(fn)
        if m:
            found[m.groups()] = os.path.join(src, fn)
    pids = sorted({pid for pid, _, _ in found}, key=participant_number)

    outputs, blocks, missing = [], {}, []
    n_trials = n_skipped = 0
    for pid in pids:
        for ch in CHANNELS:
            for task in TASKS:
                raw = found.get((pid, ch, task))
                if raw is None:
                    missing.append("%s_%s_%s_raw_data.csv" % (pid, ch, task))
                    continue
                scorer, cols = SCORERS[task]
                results, skipped = scorer(raw)
                cells = [{k: fmt(row[k]) for k in cols} for row in results]
                outputs.append(
                    (os.path.join(dst, "%s_%s_%s_data.csv" % (pid, ch, task)),
                     cols, [[c[k] for k in cols] for c in cells]))
                blocks[(pid, ch, task)] = cells
                n_trials += len(results) - skipped
                n_skipped += skipped

    if with_summary and blocks:
        outputs.append((os.path.join(dst, SUMMARY_FILE), SUMMARY_COLS,
                        participant_summary(pids, blocks)))

    print("  %d participants, %d blocks, %d trials scored, %d not scored"
          % (len(pids), len(blocks), n_trials, n_skipped))
    for fn in missing:
        print("  missing: %s" % fn)
    if with_summary and blocks:
        print("  %s: %d rows" % (SUMMARY_FILE, len(blocks)))
    print("  %d files to write" % len(outputs))
    return outputs, list(found.values())


def score_all(data_dir, out_dir):
    """Score every study folder in data_dir; nothing is written here."""
    outputs, raw_paths, studies = [], [], 0
    for folder, id_pattern, with_summary in STUDIES:
        src = io_path(os.path.join(data_dir, folder))
        if not os.path.isdir(src):
            print("%s: folder not found, skipped" % folder)
            continue
        print("%s:" % folder)
        studies += 1
        study_outputs, study_raw = score_study(
            src, io_path(os.path.join(out_dir, folder)), id_pattern,
            with_summary)
        outputs += study_outputs
        raw_paths += study_raw
    if not studies:
        sys.exit("no study folders found in %s" % data_dir)
    return outputs, raw_paths


def main(argv):
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(argv) > 3:
        sys.exit("usage: python score_trials.py [DATA_DIR] [OUT_DIR]")
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(
        argv[1] if len(argv) > 1 else os.path.dirname(here))
    out_dir = os.path.abspath(argv[2] if len(argv) > 2 else data_dir)

    print("data:   %s" % data_dir)
    print("output: %s" % out_dir)
    print("window: %d ms to trial end; band +/-%g; hold %d ms; "
          "lag search +/-%d ticks" % (WINDOW_START_MS, BAND, HOLD_MS, MAX_LAG))
    try:
        outputs, raw_paths = score_all(data_dir, out_dir)
    except (ValueError, OSError) as e:
        sys.exit("error: %s\nno files were written" % e)
    try:
        write_all(outputs, raw_paths)
    except ValueError as e:
        sys.exit("error: %s\nno files were written" % e)
    except OSError as e:
        sys.exit("error: %s" % e)
    print("total: %d files written" % len(outputs))


if __name__ == "__main__":
    main(sys.argv)
