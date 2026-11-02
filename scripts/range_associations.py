"""Calibrated intensity range and the channel differences (Section V-G).

Reads from the data package:
    <DATA_DIR>/participant tests/calibration-log_data.csv
    <DATA_DIR>/participant tests/openloop-log_data.csv
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv   (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_intensity_maintenance_raw_data.csv

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

Shared normalization: a variability in force points is the same variability in
dB multiplied by 100 / R, R being the calibrated range, so it is tied to R by
the mapping itself. Section 3 therefore repeats the associations with the
variability in dB, which does not depend on R:
  * open-loop amplitude SD: mean over the three open-loop segments of AmpSdDb
    (dB) and AmpSdPoints (force points);
  * maintenance amplitude SD: per intensity maintenance trial the SD (ddof 0)
    of 20 log10(Amplitude) over the valid ticks (Amplitude > 0) of the scoring
    window (TimeMs >= 1000), as in Table 11, averaged over the participant's
    trials; and the force-point SD of the per-trial files.
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


def amplitude_sd_db(pid):
    """Mean over the intensity maintenance trials of the within-trial SD of the dB reading."""
    raw = pd.read_csv(os.path.join(PT, "pa%d_intensity_maintenance_raw_data.csv" % pid),
                      float_precision="round_trip")
    sds = []
    for _, tr in raw.groupby("Trial"):
        a = tr.loc[(tr["TimeMs"] >= 1000) & (tr["Amplitude"] > 0), "Amplitude"].to_numpy(np.float64)
        sds.append(np.std(20.0 * np.log10(a)))
    return float(np.mean(sds))


def load():
    cal = pd.read_csv(os.path.join(PT, "calibration-log_data.csv"))
    ol = pd.read_csv(os.path.join(PT, "openloop-log_data.csv"))
    rows = []
    for pid in PIDS:
        g = cal[(cal["Participant"] == "pa%d" % pid) & (cal["Accepted"] == 1)
                & cal["Type"].str.contains("Intensity")]
        o = ol[ol["Participant"] == "pa%d" % pid]
        r = {"range_db": float(g["IntensityRangeDb"].iloc[-1]),
             "openloop_sd_db": o["AmpSdDb"].mean(), "openloop_sd_points": o["AmpSdPoints"].mean(),
             "maint_sd_db_intensity": amplitude_sd_db(pid)}
        for ch in ("pitch", "intensity"):
            m = pd.read_csv(os.path.join(PT, "pa%d_%s_maintenance_data.csv" % (pid, ch)))
            t = pd.read_csv(os.path.join(PT, "pa%d_%s_tracking_data.csv" % (pid, ch)))
            r["maint_success_" + ch] = 100 * m["Success"].mean()
            r["maint_tib_" + ch] = 100 * m["TIB"].mean()
            r["maint_sd_points_" + ch] = m["SD"].mean()
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
    print()
    print("3. Shared normalization: amplitude variability in force points (= dB x 100 / range) and in dB")
    print("   open-loop amplitude SD: M %.2f dB (SD %.2f), %.2f points; maintenance amplitude SD: M %.2f dB (SD %.2f)"
          % (d["openloop_sd_db"].mean(), d["openloop_sd_db"].std(), d["openloop_sd_points"].mean(),
             d["maint_sd_db_intensity"].mean(), d["maint_sd_db_intensity"].std()))
    for key in ("openloop_sd_points", "openloop_sd_db", "maint_sd_points_intensity", "maint_sd_db_intensity"):
        r, lo, hi, p = corr_ci(x, d[key].values)
        print("   r(range, %-26s) = %+.3f  95%% CI [%+.2f, %+.2f]  p = %.3f" % (key, r, lo, hi, p))
    for key in ("openloop_sd_points", "openloop_sd_db"):
        for y in ("maint_tib_intensity", "maint_sd_points_intensity", "maint_sd_db_intensity"):
            r, lo, hi, p = corr_ci(d[key].values, d[y].values)
            print("   r(%-18s, %-26s) = %+.3f  95%% CI [%+.2f, %+.2f]  p = %.3f" % (key, y, r, lo, hi, p))


if __name__ == "__main__":
    main()
