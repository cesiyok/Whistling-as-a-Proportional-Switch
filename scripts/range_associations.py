"""Calibrated intensity range and the channel differences (Section V-G).

Reads from the data package:
    <DATA_DIR>/participant tests/calibration-log_data.csv
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv   (N = 1..24)

Usage:
    python range_associations.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). Results are printed to stdout.

For each participant: the calibrated intensity range in dB (last accepted
intensity calibration row), and participant means of the per-trial values.
Channel differences are pitch minus intensity; success and TIB in percent.
Pearson correlations are given with Fisher-z 95 % CIs. Two correlations that
share the calibrated range are compared with Steiger's (1980) Z test for
dependent, overlapping correlations, and with a participant bootstrap of their
difference (10000 resamples, fixed seed).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
PT = os.path.join(DATA, "participant tests")
PIDS = list(range(1, 25))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")


def load():
    cal = pd.read_csv(os.path.join(PT, "calibration-log_data.csv"))
    rows = []
    for pid in PIDS:
        g = cal[(cal["Participant"] == "pa%d" % pid) & (cal["Accepted"] == 1)
                & cal["Type"].str.contains("Intensity")]
        r = {"range_db": float(g["IntensityRangeDb"].iloc[-1])}
        for ch in ("pitch", "intensity"):
            m = pd.read_csv(os.path.join(PT, "pa%d_%s_maintenance_data.csv" % (pid, ch)))
            t = pd.read_csv(os.path.join(PT, "pa%d_%s_tracking_data.csv" % (pid, ch)))
            r["maint_success_" + ch] = 100 * m["Success"].mean()
            r["maint_tib_" + ch] = 100 * m["TIB"].mean()
            r["track_tib_" + ch] = 100 * t["TIB"].mean()
            r["track_r_" + ch] = t["PearsonR"].mean()
        rows.append(r)
    d = pd.DataFrame(rows)
    d["maint_success_diff"] = d["maint_success_pitch"] - d["maint_success_intensity"]
    d["maint_tib_diff"] = d["maint_tib_pitch"] - d["maint_tib_intensity"]
    d["track_tib_diff"] = d["track_tib_pitch"] - d["track_tib_intensity"]
    return d


def corr_ci(x, y):
    r, p = stats.pearsonr(x, y)
    z, h = np.arctanh(r), 1.96 / np.sqrt(len(x) - 3)
    return r, np.tanh(z - h), np.tanh(z + h), p


def steiger(r12, r13, r23, n):
    """Steiger (1980), Eq. 15: H0 rho12 = rho13 with variable 1 shared."""
    rm = (r12 + r13) / 2.0
    psi = r23 * (1 - 2 * rm ** 2) - 0.5 * rm ** 2 * (1 - 2 * rm ** 2 - r23 ** 2)
    c = psi / (1 - rm ** 2) ** 2
    z = (np.arctanh(r12) - np.arctanh(r13)) * np.sqrt((n - 3) / (2 - 2 * c))
    return z, 2 * stats.norm.sf(abs(z))


def main():
    d = load()
    x = d["range_db"].values
    n = len(x)
    print("calibrated intensity range: M %.2f dB, range %.2f-%.2f dB, N = %d" % (x.mean(), x.min(), x.max(), n))
    print()
    print("1. Correlations with the calibrated range (Fisher-z 95 % CI)")
    for key in ("maint_success_diff", "maint_tib_diff", "maint_success_intensity", "maint_tib_intensity",
                "track_tib_diff", "track_tib_intensity", "track_r_intensity"):
        r, lo, hi, p = corr_ci(x, d[key].values)
        print("   %-24s r = %+.3f  95%% CI [%+.2f, %+.2f]  p = %.3f" % (key, r, lo, hi, p))
    print()
    print("2. Maintenance vs tracking: dependent correlations sharing the range")
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(10000, n))
    for a, b in (("maint_success_diff", "track_tib_diff"), ("maint_tib_diff", "track_tib_diff"),
                 ("maint_tib_intensity", "track_tib_intensity")):
        ya, yb = d[a].values, d[b].values
        r12, r13, r23 = np.corrcoef(x, ya)[0, 1], np.corrcoef(x, yb)[0, 1], np.corrcoef(ya, yb)[0, 1]
        z, p = steiger(r12, r13, r23, n)
        boot = np.array([np.corrcoef(x[i], ya[i])[0, 1] - np.corrcoef(x[i], yb[i])[0, 1] for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("   r(range, %s) - r(range, %s) = %+.3f  bootstrap 95%% CI [%+.2f, %+.2f];"
              "  r between the two = %+.3f;  Steiger Z = %+.2f  p = %.4f"
              % (a, b, r12 - r13, lo, hi, r23, z, p))


if __name__ == "__main__":
    main()
