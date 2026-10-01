"""Scoring-threshold sensitivity: all 33 configurations (Appendix E, Table 9).

Reads the per-tick files of the data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_raw_data.csv   (N = 1..24)

Usage:
    python sensitivity_sweep.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). Results are printed to stdout.

Every configuration re-scores the same recorded force with the scoring code of
score_trials.py; only the tolerance band, the hold requirement and the start of
the scoring window change.
  * Maintenance success: tolerance +/-10, 15, 20, 25 points x hold 2.0, 2.5,
    3.0, 3.5 s, window from 1000 ms (16 configurations).
  * Maintenance success: window from 0, 500, 1000, 1500, 2000 ms, tolerance
    +/-15, hold 3.0 s (5 configurations).
  * Tracking time in band: tolerance +/-10, 15, 20, 25 points x window from 0,
    1000, 2000 ms (12 configurations).
For each configuration and participant the trial values are averaged within
each channel. The paired difference (pitch minus intensity) over the 24
participants is given with its t-based 95 % CI, d_z = mean / SD of the
differences, the paired t-test p value, and the number of participants whose
difference is positive.
"""
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_trials as st  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
PT = os.path.join(DATA, "participant tests")
PIDS = list(range(1, 25))
CHANNELS = ("pitch", "intensity")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")


def participant_means(task, key, band, hold_ms, start_ms):
    st.BAND, st.HOLD_MS, st.WINDOW_START_MS = band, hold_ms, start_ms
    scorer = st.score_maintenance if task == "maintenance" else st.score_tracking
    out = {ch: [] for ch in CHANNELS}
    for pid in PIDS:
        for ch in CHANNELS:
            path = os.path.join(PT, "pa%d_%s_%s_raw_data.csv" % (pid, ch, task))
            rows, skipped = scorer(st.io_path(path))
            vals = [r[key] for r in rows if r.get(key) is not None]
            out[ch].append(float(np.mean(vals)))
    return np.array(out["pitch"]), np.array(out["intensity"])


def line(label, a, b):
    d = a - b
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.ttest_rel(a, b).pvalue
    print("   %-24s pitch %.3f  intensity %.3f  diff %+.3f  95%% CI [%+.3f, %+.3f]  d_z %.2f  p %s  diff > 0: %d/%d"
          % (label, a.mean(), b.mean(), m, m - half, m + half, m / sd, fmt_p(p), int((d > 0).sum()), n))
    return p


def fmt_p(p):
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def main():
    ps = []
    print("1. Maintenance success, tolerance x hold (window from 1000 ms)")
    for band in (10.0, 15.0, 20.0, 25.0):
        for hold in (2000, 2500, 3000, 3500):
            a, b = participant_means("maintenance", "Success", band, hold, 1000)
            ps.append(line("+/-%d, %.1f s" % (band, hold / 1000.0), a, b))
    print()
    print("2. Maintenance success, window onset (tolerance +/-15, hold 3.0 s)")
    for start in (0, 500, 1000, 1500, 2000):
        a, b = participant_means("maintenance", "Success", 15.0, 3000, start)
        ps.append(line("window from %d ms" % start, a, b))
    print()
    print("3. Tracking time in band, tolerance x window onset")
    for band in (10.0, 15.0, 20.0, 25.0):
        for start in (0, 1000, 2000):
            a, b = participant_means("tracking", "TIB", band, 3000, start)
            ps.append(line("+/-%d, from %d ms" % (band, start), a, b))
    print()
    ps = np.array(ps)
    print("configurations: %d; p < .05: %d; largest p: %.3f" % (len(ps), int((ps < 0.05).sum()), ps.max()))


if __name__ == "__main__":
    main()
