"""Maintenance and tracking analyses of the main study (Sections V-A to V-C, Tables 1-4, Appendix F-B).

Reproduces the data overview, the descriptive tables and repeated-measures
ANOVAs of the maintenance and tracking tasks, the channel comparisons and
simple effects quoted in the text, the rate slopes of the tracking measures,
and the measured update tick. The mixed-effects model of maintenance success
is in mixed_models.py and is not repeated here.

Reads only the public data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv      (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_raw_data.csv  tick intervals,
                                  output checks and the unrounded re-scoring
    <DATA_DIR>/participant tests/calibration-log_data.csv
    <DATA_DIR>/participant tests/session-log_data.csv
    <DATA_DIR>/participant tests/openloop-log_data.csv
and imports score_trials.py (scoring definitions) from <DATA_DIR>/scripts.

Usage:
    python maintenance_tracking.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/, next to score_trials.py). Nothing is written;
the results go to standard output, and every run prints the same text.

Definitions
  * Cell: participant x channel x target (maintenance, 5 trials per cell) or
    x rate (tracking, 3 trials per cell). The cell value is the mean of the
    trial values. CompletionTime and TimeToFirstEntry are defined only on the
    trials that met their criterion; their cell mean is over those trials,
    and a cell without any such trial is undefined.
  * Correlations (PearsonR, BestLagR): each trial's r is converted to
    z = arctanh(r) and the cell value is the mean z. The ANOVA, the paired
    tests and the rate slopes on correlations use these z cells; the tables
    show the raw r cells (mean of the three trial r values).
  * Tables 1 and 3: M +- SD (n - 1) across participants of the cell values.
    Overall: mean across participants of each participant's mean over the
    target (rate) cells; for CompletionTime over the defined cells only.
  * Repeated-measures ANOVA, 2 (Channel) x K (Target or Rate), from the
    sums-of-squares decomposition; partial eta^2 = SS / (SS + SS_error).
    Greenhouse-Geisser epsilon from the covariance matrix of orthonormal
    contrasts: of the channel-averaged scores for Target (Rate), of the
    pitch - intensity differences for the interaction. p_GG uses both
    degrees of freedom multiplied by epsilon. Mauchly's test is printed for
    reference. Channel has two levels, so its F equals the paired t squared.
  * Channel comparison: paired t on the participant means over the cells,
    pitch - intensity, with its 95% CI and d_z = mean / SD of the
    differences. Wilcoxon signed-rank test on the same means: zero
    differences dropped, average ranks for ties, Z from the normal
    approximation without tie correction, with the sign of pitch - intensity,
    r = |Z| / sqrt(24); p from scipy.stats.wilcoxon with its default method.
  * Simple effects: paired t per target (rate); Holm step-down over the
    targets (rates) of one metric.
  * Rate slopes (Appendix F-B): least-squares slope of each participant's
    five cell values on rate in Hz, per channel; paired t on the slopes.
  * Update tick: difference between consecutive TimeMs values within a trial,
    over every tick of every trial of both tasks (0-6000 ms). Also printed:
    the mean over trials of each trial's mean tick inside the scoring window,
    the factor score_trials.py uses to convert the best lag to ms.
  * Output checks, over the ticks of the scoring window (TimeMs >= 1000):
    force exactly 0 for any reason, and output gated off. Intensity: gated
    off when Amplitude = 0 (no valid estimate). Pitch: the per-tick files
    hold the last f0 while the gate is off, so a tick counts as gated off
    when force is 0 and f0 is 0 or above the participant's calibrated
    minimum (MinMidi of the last calibration row); force 0 with f0 at or
    below that minimum is the lower bound, not a gate. Durations weight
    each tick by the time since the tick before it, as in score_trials.py.
  * Unrounded check: every trial is scored again from its per-tick file
    with score_trials.py, without the 4-decimal rounding of the per-trial
    files, and the analyses are repeated on those values.
"""
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1:2] in (["-h"], ["--help"]):
    print(__doc__)
    sys.exit(0)
if len(sys.argv) > 2:
    sys.exit("usage: python maintenance_tracking.py [DATA_DIR]")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DATA, "scripts"))
import score_trials as ST  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")

# io_path adds the Windows extended-length prefix, so that a deeply nested
# package folder (paths over 260 characters) can still be read.
PT = ST.io_path(os.path.join(DATA, "participant tests"))

PIDS = ["pa%d" % i for i in range(1, 25)]
CHANNELS = ("pitch", "intensity")
TASKS = ("maintenance", "tracking")
TARGETS = (20, 40, 60, 80)
RATES = (20, 35, 50, 65, 80)              # SpeedPercent, i.e. rate x 100 Hz
RATE_HZ = np.array(RATES, np.float64) / 100.0
WINDOW_START_MS = ST.WINDOW_START_MS
FRAME_MS = 50.0                            # audio frame period by design
N_MAINT, N_TRACK = 20, 15                  # trials per block

# (column, label, scale, decimals, sign shown, lower is better)
MAINT_METRICS = (
    ("RMSE", "RMSE (points)", 1.0, 2, False, True),
    ("SD", "SD (points)", 1.0, 2, False, True),
    ("TIB", "TIB (%)", 100.0, 1, False, False),
    ("Success", "Success (%)", 100.0, 1, False, False),
    ("LongestStreak", "Longest streak (s)", 1.0, 2, False, False),
    ("CompletionTime", "Completion (s)", 1.0, 2, False, True),
    ("TimeToFirstEntry", "Time to first entry (s)", 1.0, 2, False, True),
    ("Bias", "Bias (points)", 1.0, 2, True, None),
)
MAINT_ANOVA = ("RMSE", "SD", "TIB", "Success", "LongestStreak", "Bias", "TimeToFirstEntry")

TRACK_METRICS = (
    ("RMSE", "RMSE (points)", 1.0, 2, False, True),
    ("PearsonR", "Pearson r", 1.0, 3, False, False),
    ("BestLagR", "Best-lag r", 1.0, 3, False, False),
    ("BestLagMs", "Lag (ms)", 1.0, 0, True, None),
    ("Gain", "Gain", 1.0, 3, False, False),
    ("TIB", "TIB (%)", 100.0, 1, False, False),
    ("Bias", "Bias (points)", 1.0, 2, True, None),
)
TRACK_ANOVA = ("PearsonR", "BestLagR", "RMSE", "TIB", "Gain", "BestLagMs", "Bias")
CORRELATIONS = ("PearsonR", "BestLagR")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fp(p):
    """p value as printed, to three significant digits."""
    return "%.3g" % p


def fnum(x, nd, sign=False):
    if x != x:
        return "n/a"
    return (("%+." if sign else "%.") + str(nd) + "f") % x


def msd(v, nd, sign=False):
    v = v[np.isfinite(v)]
    return "%s+-%s" % (fnum(v.mean(), nd, sign), fnum(v.std(ddof=1), nd))


class Report(object):
    """Collects the output lines, keyed, so that two runs can be compared."""

    def __init__(self):
        self.items = []

    def head(self, text):
        self.items.append((None, ""))
        self.items.append((None, text))
        self.items.append((None, "-" * len(text)))

    def line(self, key, text):
        self.items.append((key, text))

    def note(self, text):
        self.items.append((None, text))

    def show(self):
        for _, text in self.items:
            print(text)

    def keyed(self):
        return dict((k, t) for k, t in self.items if k is not None)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def orthonormal_contrasts(k):
    """k x (k - 1) matrix of orthonormal (Helmert) contrasts."""
    c = np.zeros((k, k - 1))
    for j in range(1, k):
        c[:j, j - 1] = 1.0
        c[j, j - 1] = -float(j)
        c[:, j - 1] /= math.sqrt(j * (j + 1.0))
    return c


def sphericity(m):
    """Greenhouse-Geisser epsilon and Mauchly's test for n x k scores m."""
    n, k = m.shape
    c = orthonormal_contrasts(k)
    s = c.T @ np.cov(m, rowvar=False) @ c
    q = k - 1
    eps = np.trace(s) ** 2 / (q * np.trace(s @ s))
    w = np.linalg.det(s) / (np.trace(s) / q) ** q
    chi2 = -(n - 1 - (2.0 * q * q + q + 2.0) / (6.0 * q)) * math.log(w)
    df = q * (q + 1) / 2 - 1
    return eps, w, chi2, df, stats.chi2.sf(chi2, df)


def rm_anova(y):
    """2 x K repeated-measures ANOVA on y[participant, channel, level].

    Returns a dict per effect ('ch', 'lv', 'int') with F, df1, df2, p,
    eta2p and, for the K-level effects, eps, p_gg and the Mauchly test.
    """
    n, a, b = y.shape
    gm = y.mean()
    m_a, m_b, m_s = y.mean(axis=(0, 2)), y.mean(axis=(0, 1)), y.mean(axis=(1, 2))
    m_ab, m_sa, m_sb = y.mean(axis=0), y.mean(axis=2), y.mean(axis=1)
    ss_a = n * b * np.sum((m_a - gm) ** 2)
    ss_b = n * a * np.sum((m_b - gm) ** 2)
    ss_ab = n * np.sum((m_ab - gm) ** 2) - ss_a - ss_b
    ss_s = a * b * np.sum((m_s - gm) ** 2)
    ss_sa = b * np.sum((m_sa - gm) ** 2) - ss_s - ss_a
    ss_sb = a * np.sum((m_sb - gm) ** 2) - ss_s - ss_b
    ss_sab = np.sum((y - gm) ** 2) - ss_a - ss_b - ss_ab - ss_s - ss_sa - ss_sb
    scores = {"lv": y.mean(axis=1), "int": y[:, 0, :] - y[:, 1, :]}
    out = {}
    for eff, ss, df1, sse, df2 in (
            ("ch", ss_a, a - 1, ss_sa, (n - 1) * (a - 1)),
            ("lv", ss_b, b - 1, ss_sb, (n - 1) * (b - 1)),
            ("int", ss_ab, (a - 1) * (b - 1), ss_sab, (n - 1) * (a - 1) * (b - 1))):
        f = (ss / df1) / (sse / df2)
        r = {"F": f, "df1": df1, "df2": df2, "p": stats.f.sf(f, df1, df2),
             "eta2p": ss / (ss + sse)}
        if eff in scores:
            eps, w, chi2, df, pm = sphericity(scores[eff])
            r.update(eps=eps, p_gg=stats.f.sf(f, df1 * eps, df2 * eps),
                     mauchly_w=w, mauchly_chi2=chi2, mauchly_df=df, mauchly_p=pm)
        out[eff] = r
    return out


def oneway_rm(m):
    """One-way repeated-measures ANOVA on n x k scores m."""
    n, k = m.shape
    gm = m.mean()
    ss_c = n * np.sum((m.mean(axis=0) - gm) ** 2)
    ss_s = k * np.sum((m.mean(axis=1) - gm) ** 2)
    ss_e = np.sum((m - gm) ** 2) - ss_c - ss_s
    df1, df2 = k - 1, (n - 1) * (k - 1)
    f = (ss_c / df1) / (ss_e / df2)
    eps = sphericity(m)[0]
    return {"F": f, "df1": df1, "df2": df2, "p": stats.f.sf(f, df1, df2),
            "eps": eps, "p_gg": stats.f.sf(f, df1 * eps, df2 * eps),
            "eta2p": ss_c / (ss_c + ss_e)}


def paired(a, b):
    """Paired t test of a - b with 95% CI and d_z."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    n = d.size
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    t = m / se
    return {"n": n, "diff": m, "lo": m - half, "hi": m + half, "t": t,
            "p": 2.0 * stats.t.sf(abs(t), n - 1), "dz": m / sd}


def signed_rank(a, b):
    """Wilcoxon signed-rank test of a - b: Z, p, r = |Z| / sqrt(N), N."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    nz = d[d != 0]
    n = nz.size
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - n * (n + 1) / 4.0) / math.sqrt(var)
    p = stats.wilcoxon(nz).pvalue
    return {"Z": z, "p": p, "r": abs(z) / math.sqrt(d.size), "n": n}


def holm(p):
    """Holm step-down adjusted p values."""
    p = np.asarray(p, np.float64)
    adj = np.empty(p.size)
    run = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, (p.size - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def slopes(y):
    """Least-squares slope on rate (Hz) of each participant x channel row."""
    out = np.empty(y.shape[:2])
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            out[i, j] = np.polyfit(RATE_HZ, y[i, j, :], 1)[0]
    return out


def test_text(r, nd):
    return ("diff %s  95%% CI [%s, %s]  t(%d) = %.2f  p = %s  d_z = %.2f"
            % (fnum(r["diff"], nd, True), fnum(r["lo"], nd, True), fnum(r["hi"], nd, True),
               r["n"] - 1, r["t"], fp(r["p"]), r["dz"]))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_trials(task):
    """Per-trial files of one task, all participants and channels."""
    parts = []
    for pid in PIDS:
        for ch in CHANNELS:
            d = pd.read_csv(os.path.join(PT, "%s_%s_%s_data.csv" % (pid, ch, task)),
                            float_precision="round_trip")
            d.insert(0, "pid", pid)
            d.insert(1, "ch", ch)
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def score_unrounded(task):
    """The same trials scored again from the per-tick files, unrounded."""
    scorer = ST.score_maintenance if task == "maintenance" else ST.score_tracking
    parts = []
    for pid in PIDS:
        for ch in CHANNELS:
            rows, _ = scorer(os.path.join(PT, "%s_%s_%s_raw_data.csv" % (pid, ch, task)))
            d = pd.DataFrame(rows).astype(float)
            d.insert(0, "pid", pid)
            d.insert(1, "ch", ch)
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def load_raw(pid, ch, task):
    d = pd.read_csv(os.path.join(PT, "%s_%s_%s_raw_data.csv" % (pid, ch, task)),
                    float_precision="round_trip")
    return d.sort_values(["Trial", "SampleIndex"], kind="stable")


def cells(d, col, level_col, levels, z=False):
    """y[participant, channel, level] = mean of the trial values of the cell."""
    v = d[col].astype(float)
    if z:
        assert (v.abs() < 1).all(), "correlation of +-1"
        v = np.arctanh(v)
    g = v.groupby([d["pid"], d["ch"], d[level_col].astype(int)]).mean()
    y = np.full((len(PIDS), len(CHANNELS), len(levels)), np.nan)
    for i, pid in enumerate(PIDS):
        for j, ch in enumerate(CHANNELS):
            for k, lv in enumerate(levels):
                if (pid, ch, lv) in g.index:
                    y[i, j, k] = g[(pid, ch, lv)]
    return y


def overall(y):
    """Participant x channel mean over the defined cells."""
    return np.nanmean(y, axis=2)


# ---------------------------------------------------------------------------
# A. Data overview
# ---------------------------------------------------------------------------
def midi(hz):
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def output_checks(min_midi):
    """Per trial: ticks in the window with force 0 and with the output gated off.

    Also returns every inter-tick interval of every trial (0-6000 ms) and the
    mean tick inside each trial's scoring window.
    """
    gaps, window_tick, rows = [], [], []
    for pid in PIDS:
        for ch in CHANNELS:
            for task in TASKS:
                raw = load_raw(pid, ch, task)
                fcol = "Force" if task == "maintenance" else "Actual"
                for trial, g in raw.groupby("Trial", sort=True):
                    t = g["TimeMs"].to_numpy(np.int64)
                    gaps.append(np.diff(t))
                    w = t >= WINDOW_START_MS
                    tw = t[w]
                    window_tick.append((tw[-1] - tw[0]) / (tw.size - 1.0))
                    dt = np.diff(np.concatenate(([WINDOW_START_MS], tw))).astype(np.float64)
                    zero = g[fcol].to_numpy(np.float64)[w] == 0.0
                    f0 = g["FrequencyHz"].to_numpy(np.float64)[w]
                    if ch == "intensity":
                        gated = g["Amplitude"].to_numpy(np.float64)[w] == 0.0
                    else:
                        above = midi(np.where(f0 > 0, f0, 440.0)) > min_midi[pid]
                        gated = zero & ((f0 == 0) | above)
                    run_n = best_n = 0
                    run_ms = best_ms = 0.0
                    for gi, di in zip(gated, dt):
                        run_n, run_ms = (run_n + 1, run_ms + di) if gi else (0, 0.0)
                        if run_n > best_n:
                            best_n, best_ms = run_n, run_ms
                    rows.append((pid, ch, task, int(trial), tw.size, int(zero.sum()),
                                 int((zero & (tw > 5000)).sum()),
                                 int(gated.sum()), int((gated & (tw >= 5000)).sum()),
                                 float(dt[gated].sum()), best_n, best_ms))
    pt = pd.DataFrame(rows, columns=["pid", "ch", "task", "trial", "n", "nzero", "nzero_last",
                                     "ngated", "ngated_last", "ms_gated", "run_n", "run_ms"])
    return pt, np.concatenate(gaps).astype(np.float64), np.array(window_tick)


def data_overview(maint, track):
    rep = Report()
    rep.head("A. Data overview (Section V-A)")
    nm, nt = len(maint), len(track)
    rep.line("n_trials", "trials: maintenance %d, tracking %d, total %d (expected %d)"
             % (nm, nt, nm + nt, len(PIDS) * 2 * (N_MAINT + N_TRACK)))
    for d, n_block, level, n_cell in ((maint, N_MAINT, "Target", 5),
                                      (track, N_TRACK, "SpeedPercent", 3)):
        size = d.groupby(["pid", "ch"]).size()
        assert (size == n_block).all() and len(size) == 2 * len(PIDS)
        assert (d.groupby(["pid", "ch", level]).size() == n_cell).all()
    rep.line("n_cell", "trials per cell: maintenance 5 per target, tracking 3 per rate "
             "(checked for every participant and channel)")
    unscored = int(maint["RMSE"].isna().sum() + track["RMSE"].isna().sum())
    rep.line("unscored", "trials with fewer than %d ticks in the window (unscored): %d"
             % (ST.MIN_SAMPLES, unscored))
    ns = np.concatenate([maint["NSamples"].to_numpy(float), track["NSamples"].to_numpy(float)])
    rep.line("nsamples", "ticks per scoring window: min %d, max %d, M = %.1f, SD = %.1f"
             % (ns.min(), ns.max(), ns.mean(), ns.std(ddof=1)))

    # Calibrated ranges: the bounds in effect after each participant's last row.
    cal = pd.read_csv(os.path.join(PT, "calibration-log_data.csv"), float_precision="round_trip")
    last = cal.groupby("Participant", sort=False).tail(1).set_index("Participant").loc[PIDS]

    pt, gaps, window_tick = output_checks(last["MinMidi"].astype(float).to_dict())
    rep.note("output in the scoring window (trials with at least one such tick, ticks):")
    for task in TASKS:
        for ch in CHANNELS:
            s = pt[(pt.task == task) & (pt.ch == ch)]
            rep.line("valid_%s_%s" % (task, ch),
                     "  %-11s %-9s force = 0: %3d trials, %4d ticks | gated off: %3d trials, "
                     "%4d ticks, %4d of them in the last second"
                     % (task, ch, (s.nzero > 0).sum(), s.nzero.sum(), (s.ngated > 0).sum(),
                        s.ngated.sum(), s.ngated_last.sum()))
    rep.line("valid_total", "  all 1680 trials: force = 0 on at least one tick in %d; output "
             "gated off on at least one tick in %d; gated off for more than half of the "
             "window in %d" % ((pt.nzero > 0).sum(), (pt.ngated > 0).sum(),
                                (pt.ngated > pt.n / 2.0).sum()))
    nzero, nzero_last = pt.nzero.sum(), pt.nzero_last.sum()
    rep.line("zero_total", "  force = 0 (any reason): %d trials, %d ticks, %d of them "
             "(%.1f%%) in the last second (TimeMs > 5000); force = 0 on more than half of "
             "the window ticks in %d trials" % ((pt.nzero > 0).sum(), nzero, nzero_last,
                                                100.0 * nzero_last / nzero,
                                                (pt.nzero > pt.n / 2.0).sum()))
    rep.note("  longest continuous gate-off in the window:")
    top = pt.sort_values(["run_ms", "pid", "trial"], ascending=[False, True, True]).head(3)
    for _, r in top.iterrows():
        rep.line("valid_top_%s_%s_%s_%d" % (r.pid, r.ch, r.task, r.trial),
                 "    %s %s %s trial %d: %d ticks, %.2f s (%.2f s at the nominal 62.5 ms tick);"
                 " gated off in total %d of %d ticks, %.2f s"
                 % (r.pid, r.ch, r.task, r.trial, r.run_n, r.run_ms / 1000.0, r.run_n * 0.0625,
                    r.ngated, r.n, r.ms_gated / 1000.0))

    span = last["PitchSpanSemitones"].to_numpy(float)
    rng = last["IntensityRangeDb"].to_numpy(float)
    rep.line("span", "pitch span (semitones): M = %.2f, SD = %.2f, range %.2f-%.2f"
             % (span.mean(), span.std(ddof=1), span.min(), span.max()))
    rep.line("range", "intensity range (dB): M = %.2f, SD = %.2f, range %.2f-%.2f"
             % (rng.mean(), rng.std(ddof=1), rng.min(), rng.max()))
    rep.line("s_pitch", "sensitivity s, pitch: 100 / mean span = %.2f points/semitone "
             "(mean of 100 / span = %.2f)" % (100.0 / span.mean(), np.mean(100.0 / span)))
    rep.line("s_int", "sensitivity s, intensity: 100 / mean range = %.2f points/dB "
             "(mean of 100 / range = %.2f); participants %.2f-%.2f points/dB"
             % (100.0 / rng.mean(), np.mean(100.0 / rng), 100.0 / rng.max(), 100.0 / rng.min()))
    counts = cal.groupby("Type")["Accepted"].agg(["size", "sum"])
    rep.line("cal_rows", "calibration measurements: %d, accepted %d (%s)"
             % (len(cal), cal["Accepted"].sum(),
                ", ".join("%s %d" % (k, counts.loc[k, "size"])
                          for k in ("MinPitch", "MaxPitch", "MinIntensity", "MaxIntensity"))))
    fr = cal[["Frames1", "Frames2", "Frames3"]].to_numpy(float).ravel()
    rep.line("cal_frames", "frames per calibration recording: n = %d, M = %.1f, SD = %.1f, min %d"
             % (fr.size, fr.mean(), fr.std(ddof=1), fr.min()))
    ses = pd.read_csv(os.path.join(PT, "session-log_data.csv"), float_precision="round_trip")
    ol = pd.read_csv(os.path.join(PT, "openloop-log_data.csv"), float_precision="round_trip")
    devices = sorted(set(cal["Device"]) | set(ses["Device"]) | set(ol["Device"]))
    mus = sorted(set("%.4f" % v for v in pd.concat([cal["SampleRateMultiplier"],
                                                    ses["SampleRateMultiplier"],
                                                    ol["SampleRateMultiplier"]])))
    rep.line("device", "capture devices: %d (%s); mu values: %s"
             % (len(devices), "; ".join(devices), ", ".join(mus)))
    typ = ses["RoomDbaTyp"].to_numpy(float)
    rep.line("dba", "room level, typical dB(A): M = %.3f (%.2f), SD = %.2f over %d sessions; "
             "range of typical %d-%d; lowest minimum %d, highest maximum %d"
             % (typ.mean(), typ.mean(), typ.std(ddof=1), len(ses), typ.min(), typ.max(),
                ses["RoomDbaMin"].min(), ses["RoomDbaMax"].max()))

    # Open-loop production noise, participant means of the three segments.
    pp = ol.groupby("Participant")[["PitchSdPoints", "AmpSdPoints"]].mean().loc[PIDS]
    psd, asd = pp["PitchSdPoints"].to_numpy(), pp["AmpSdPoints"].to_numpy()
    r = paired(asd, psd)
    rep.line("openloop", "open-loop SD (points): pitch %.2f+-%.2f, amplitude %.2f+-%.2f, "
             "ratio %.2f; amplitude - pitch: t(%d) = %.2f  p = %s  d_z = %.2f"
             % (psd.mean(), psd.std(ddof=1), asd.mean(), asd.std(ddof=1),
                asd.mean() / psd.mean(), r["n"] - 1, r["t"], fp(r["p"]), r["dz"]))

    vals, cnt = np.unique(gaps, return_counts=True)
    mean_tick = gaps.mean()
    rep.line("tick", "update tick (Appendix C): %d intervals, mean %.2f ms, SD %.2f, mode %d, "
             "1st-99th percentile %.0f-%.0f ms"
             % (gaps.size, mean_tick, gaps.std(ddof=1), vals[np.argmax(cnt)],
                np.percentile(gaps, 1), np.percentile(gaps, 99)))
    rep.line("tick_window", "  mean tick inside the scoring window, averaged over the 1680 "
             "trials: %.2f ms" % window_tick.mean())
    rep.line("tick2", "  lag search +-%d ticks = +-%.0f ms; frames not displayed: 1 - %.0f / %.2f "
             "= %.3f" % (ST.MAX_LAG, ST.MAX_LAG * mean_tick, FRAME_MS, mean_tick,
                          1.0 - FRAME_MS / mean_tick))
    return rep, mean_tick


# ---------------------------------------------------------------------------
# B. Maintenance
# ---------------------------------------------------------------------------
def descriptive_rows(rep, prefix, metrics, cube):
    for col, label, scale, nd, sign, _ in metrics:
        y = cube[col] * scale
        for j, ch in enumerate(CHANNELS):
            cellstr = "  ".join("%13s" % msd(y[:, j, k], nd, sign) for k in range(y.shape[2]))
            n = np.isfinite(y[:, j, :]).sum(axis=0)
            extra = "" if (n == len(PIDS)).all() else "   n = %s" % "/".join(str(v) for v in n)
            rep.line("%s_%s_%s" % (prefix, col, ch), "%-24s %-9s %s | %s%s"
                     % (label, ch, cellstr, fnum(overall(y)[:, j].mean(), nd, sign), extra))


def anova_rows(rep, prefix, names, cube, labels, level_name):
    r = rm_anova(cube[names[0]])
    rep.note("%-22s Channel F p eta2p | %s F eps p_GG eta2p | Channel x %s F eps p_GG eta2p"
             % ("", level_name, level_name))
    rep.note("%-22s df %d,%d | df %d,%d | df %d,%d"
             % ("", r["ch"]["df1"], r["ch"]["df2"], r["lv"]["df1"], r["lv"]["df2"],
                r["int"]["df1"], r["int"]["df2"]))
    for col in names:
        r = rm_anova(cube[col])
        c, l, i = r["ch"], r["lv"], r["int"]
        rep.line("%s_%s" % (prefix, col),
                 "%-22s %7.2f %7s %5.2f | %6.2f %4.2f %7s %5.2f | %6.2f %4.2f %7s %5.2f"
                 % (labels[col], c["F"], fp(c["p"]), c["eta2p"], l["F"], l["eps"], fp(l["p_gg"]),
                    l["eta2p"], i["F"], i["eps"], fp(i["p_gg"]), i["eta2p"]))
        rep.line("%s_%s_mauchly" % (prefix, col),
                 "%-22s Mauchly: main W = %.3f, p = %s; interaction W = %.3f, p = %s;"
                 " uncorrected p %s / %s"
                 % ("", l["mauchly_w"], fp(l["mauchly_p"]), i["mauchly_w"], fp(i["mauchly_p"]),
                    fp(l["p"]), fp(i["p"])))


def channel_rows(rep, prefix, metrics, cube, zcube=None):
    for col, label, scale, nd, _, lower_better in metrics:
        y = (zcube[col] if zcube is not None and col in zcube else cube[col]) * scale
        ov = overall(y)
        a, b = ov[:, 0], ov[:, 1]
        r = paired(a, b)
        w = signed_rank(a, b)
        if lower_better is None:
            cnt = "pitch higher %d/24" % np.sum(a > b)
        elif lower_better:
            cnt = "pitch lower %d/24" % np.sum(a < b)
        else:
            cnt = "pitch higher %d/24" % np.sum(a > b)
        tag = label + (" (z)" if zcube is not None and col in zcube else "")
        rep.line("%s_%s" % (prefix, col), "%-24s %s | Wilcoxon Z = %+.2f p = %s r = %.2f | %s"
                 % (tag, test_text(r, max(nd, 2) + 1), w["Z"], fp(w["p"]), w["r"], cnt))


def maintenance(maint, title_suffix=""):
    rep = Report()
    cube = dict((c[0], cells(maint, c[0], "Target", TARGETS)) for c in MAINT_METRICS)
    labels = dict((c[0], c[1].split(" (")[0]) for c in MAINT_METRICS)

    rep.head("B. Maintenance (Section V-B)" + title_suffix)
    rep.note("Table 1: M+-SD across participants; columns target 20, 40, 60, 80 | Overall")
    descriptive_rows(rep, "t1", MAINT_METRICS, cube)

    rep.note("")
    rep.note("Table 2: repeated-measures ANOVA")
    anova_rows(rep, "t2", MAINT_ANOVA, cube, labels, "Target")

    rep.note("")
    rep.note("Channel comparison on participant means (pitch - intensity):")
    channel_rows(rep, "ch", MAINT_METRICS, cube)
    summary = ("RMSE", "SD", "TIB", "Success", "LongestStreak")
    dzs, rs = [], []
    for col in summary:
        ov = overall(cube[col])
        dzs.append(abs(paired(ov[:, 0], ov[:, 1])["dz"]))
        rs.append(signed_rank(ov[:, 0], ov[:, 1])["r"])
    rep.line("ch_summary", "effect sizes over %s: |d_z| %.2f-%.2f, Wilcoxon r %.2f-%.2f"
             % (", ".join(summary), min(dzs), max(dzs), min(rs), max(rs)))

    rep.note("")
    rep.note("Simple channel effects per target (paired t, Holm over the 4 targets):")
    for col, label, scale, nd, sign, _ in MAINT_METRICS:
        if col == "CompletionTime":
            continue
        y = cube[col] * scale
        res = [paired(y[:, 0, k], y[:, 1, k]) for k in range(len(TARGETS))]
        adj = holm([r["p"] for r in res])
        rep.line("se_%s" % col, "%-24s %s" % (label, "  ".join(
            "%d: %s t = %.2f p_Holm = %s d_z = %.2f"
            % (tg, fnum(r["diff"], nd, True), r["t"], fp(q), r["dz"])
            for tg, r, q in zip(TARGETS, res, adj))))

    rep.note("")
    rep.note("Text statistics:")
    s = cube["Success"] * 100.0
    rep.line("succ_range", "success: intensity %.1f%% at 20 -> %.1f%% at 60; pitch %.1f-%.1f%%; "
             "channel gap %.1f at 20, %.1f at 60 points"
             % (s[:, 1, 0].mean(), s[:, 1, 2].mean(), s[:, 0].mean(axis=0).min(),
                s[:, 0].mean(axis=0).max(), s[:, 0, 0].mean() - s[:, 1, 0].mean(),
                s[:, 0, 2].mean() - s[:, 1, 2].mean()))
    b = cube["Bias"]
    rep.line("bias_pitch", "pitch bias cell means: %s (largest |value| %.2f)"
             % (", ".join("%+.2f" % v for v in b[:, 0].mean(axis=0)),
                np.abs(b[:, 0].mean(axis=0)).max()))
    r = oneway_rm(b[:, 1, :])
    rep.line("bias_int", "intensity bias, one-way Target effect within intensity: F(%d,%d) = %.2f  "
             "eps = %.2f  p_GG = %s  eta2p = %.2f" % (r["df1"], r["df2"], r["F"], r["eps"],
                                                      fp(r["p_gg"]), r["eta2p"]))
    ov = overall(cube["TimeToFirstEntry"])
    rep.line("ttfe", "time to first entry: pitch %.3f s, intensity %.3f s"
             % (ov[:, 0].mean(), ov[:, 1].mean()))
    ov = overall(cube["CompletionTime"])
    r = paired(ov[:, 0], ov[:, 1])
    rep.line("completion", "completion time: pitch %.3f s, intensity %.3f s, %s"
             % (ov[:, 0].mean(), ov[:, 1].mean(), test_text(r, 3)))
    ok = maint["Success"] == 1
    ct = maint.loc[ok].groupby(["pid", "ch"])["CompletionTime"].mean().unstack().loc[PIDS]
    rep.line("completion_trials", "completion time, mean over each participant's successful trials "
             "(per-participant summary): pitch %.3f s, intensity %.3f s; shortest trial %.3f s"
             % (ct["pitch"].mean(), ct["intensity"].mean(), maint.loc[ok, "CompletionTime"].min()))
    for col in ("CompletionTime", "TimeToFirstEntry"):
        n = maint.groupby(["ch", "Target", "pid"])[col].count()
        rep.line("cells_%s" % col, "%s: defined on %d of %d trials; trials per cell %s" % (
            col, maint[col].notna().sum(), len(maint), "; ".join(
                "%s %s" % (ch, " ".join("%d:%d-%d" % (tg, n.loc[(ch, tg)].min(), n.loc[(ch, tg)].max())
                                       for tg in TARGETS)) for ch in CHANNELS)))

    # Figure 4: the trial whose SD is closest to the channel median at target 60.
    for ch in CHANNELS:
        d = maint[(maint.ch == ch) & (maint.Target == 60)]
        med = d["SD"].median()
        k = (d["SD"] - med).abs().to_numpy().argmin()
        row = d.iloc[k]
        raw = load_raw(row.pid, ch, "maintenance")
        g = raw[raw.Trial == row.Trial]
        f, t = g["Force"].to_numpy(float), g["TimeMs"].to_numpy(float)
        first = t[np.argmax(f > 0)] if (f > 0).any() else float("nan")
        rep.line("fig4_%s" % ch, "Figure 4, %-9s: %s trial %d, SD %.2f (median %.2f), TIB %.2f; "
                 "force first above 0 at %.0f ms" % (ch, row.pid, row.Trial, row.SD, med,
                                                      row.TIB, first))
    return rep, cube


# ---------------------------------------------------------------------------
# C. Tracking and Appendix F-B
# ---------------------------------------------------------------------------
def tracking(track, mean_tick, title_suffix=""):
    rep = Report()
    cube = dict((c[0], cells(track, c[0], "SpeedPercent", RATES)) for c in TRACK_METRICS)
    zcube = dict((c, cells(track, c, "SpeedPercent", RATES, z=True)) for c in CORRELATIONS)
    labels = {"PearsonR": "Pearson r (z)", "BestLagR": "Best-lag r (z)", "RMSE": "RMSE",
              "TIB": "TIB", "Gain": "Gain", "BestLagMs": "Lag", "Bias": "Bias"}

    rep.head("C. Tracking (Section V-C)" + title_suffix)
    rep.note("Table 3: M+-SD across participants; columns 0.20, 0.35, 0.50, 0.65, 0.80 Hz | Overall"
             " (r rows: raw r)")
    descriptive_rows(rep, "t3", TRACK_METRICS, cube)
    for col in CORRELATIONS:
        ov = overall(zcube[col])
        rep.line("t3_%s_ztanh" % col, "%-24s back-transformed mean of z: pitch %.3f, intensity %.3f"
                 % (col, np.tanh(ov[:, 0].mean()), np.tanh(ov[:, 1].mean())))

    rep.note("")
    rep.note("Table 4: repeated-measures ANOVA (correlations in z)")
    anova_cube = dict(cube)
    anova_cube.update(zcube)
    anova_rows(rep, "t4", TRACK_ANOVA, anova_cube, labels, "Rate")

    rep.note("")
    rep.note("Channel comparison on participant means (pitch - intensity; correlations in z):")
    channel_rows(rep, "ch", TRACK_METRICS, cube, zcube)

    rep.note("")
    rep.note("Text statistics:")
    g = cube["Gain"]
    ov = overall(g)
    rep.line("gain", "gain: pitch %.3f, intensity %.3f; undershoot %.3f and %.3f, ratio %.2f; "
             "at 0.20 Hz %.3f and %.3f"
             % (ov[:, 0].mean(), ov[:, 1].mean(), 1 - ov[:, 0].mean(), 1 - ov[:, 1].mean(),
                (1 - ov[:, 1].mean()) / (1 - ov[:, 0].mean()), g[:, 0, 0].mean(), g[:, 1, 0].mean()))
    ms = cube["BestLagMs"]
    ov = overall(ms)
    rep.line("lag", "lag: pitch %+.1f ms, intensity %+.1f ms, pitch - intensity %+.1f ms; "
             "one tick = %.1f ms" % (ov[:, 0].mean(), ov[:, 1].mean(),
                                     ov[:, 0].mean() - ov[:, 1].mean(), mean_tick))
    rep.line("lag_cells", "lag cell means: pitch %s | intensity %s"
             % (" ".join("%+.1f" % v for v in ms[:, 0].mean(axis=0)),
                " ".join("%+.1f" % v for v in ms[:, 1].mean(axis=0))))
    for j, ch in enumerate(CHANNELS):
        r_raw = overall(cube["PearsonR"])[:, j].mean()
        r_lag = overall(cube["BestLagR"])[:, j].mean()
        rep.line("lag_gain_%s" % ch, "best-lag increase in r, %-9s: %.4f (from the rounded "
                 "table values %.3f - %.3f = %.3f)"
                 % (ch, r_lag - r_raw, r_lag, r_raw, round(r_lag, 3) - round(r_raw, 3)))
    rr = cube["PearsonR"]
    rep.line("r_every_rate", "raw r, pitch minus intensity per rate: %s"
             % " ".join("%+.3f" % v for v in (rr[:, 0] - rr[:, 1]).mean(axis=0)))
    rep.line("gain_effect_size", "gain d_z relative to d_z of r (z), RMSE, TIB: %s"
             % ", ".join("%.2f" % (paired(overall(anova_cube["Gain"])[:, 0],
                                          overall(anova_cube["Gain"])[:, 1])["dz"]
                                   / abs(paired(overall(anova_cube[c])[:, 0],
                                                overall(anova_cube[c])[:, 1])["dz"]))
                         for c in ("PearsonR", "RMSE", "TIB")))

    rep.head("Appendix F-B: rate dependence of tracking fidelity" + title_suffix)
    rep.note("per-participant slope on rate (units per Hz): pitch, intensity, intensity / pitch;"
             " paired t on the slopes (pitch - intensity)")
    for key, label, y in (("TIB", "TIB (points/Hz)", cube["TIB"] * 100.0),
                          ("RMSE", "RMSE (points/Hz)", cube["RMSE"]),
                          ("PearsonR_z", "Pearson r, z (/Hz)", zcube["PearsonR"]),
                          ("PearsonR_raw", "Pearson r, raw (/Hz)", cube["PearsonR"])):
        sl = slopes(y)
        r = paired(sl[:, 0], sl[:, 1])
        rep.line("fb_%s" % key, "%-22s %+.3f  %+.3f  ratio %.2f   t(%d) = %.2f  p = %s  d_z = %.2f"
                 % (label, sl[:, 0].mean(), sl[:, 1].mean(), sl[:, 1].mean() / sl[:, 0].mean(),
                    r["n"] - 1, r["t"], fp(r["p"]), r["dz"]))
    return rep, cube, zcube


def smallest_effects(mcube, tcube, tzcube):
    """Channel effects with p < .05 over both tasks, ordered by |d_z|."""
    out = []
    for task, cube, names in (("maintenance", mcube, MAINT_ANOVA + ("CompletionTime",)),
                              ("tracking", dict(tcube, **tzcube), TRACK_ANOVA)):
        for col in names:
            ov = overall(cube[col])
            r = paired(ov[:, 0], ov[:, 1])
            if r["p"] < 0.05:
                out.append((abs(r["dz"]), task, col, r["dz"], r["p"]))
    out.sort()
    return out


def compare(label, rounded, unrounded):
    a, b = rounded.keyed(), unrounded.keyed()
    changed = [k for k in a if a[k] != b.get(k)]
    print("%s: %d printed lines, %d change with the unrounded values" % (label, len(a), len(changed)))
    for k in changed:
        print("  per-trial files: %s" % a[k].strip())
        print("  unrounded:       %s" % b[k].strip())


def main():
    maint = load_trials("maintenance")
    track = load_trials("tracking")

    rep, mean_tick = data_overview(maint, track)
    rep.show()
    mrep, mcube = maintenance(maint)
    mrep.show()
    trep, tcube, tzcube = tracking(track, mean_tick)
    trep.show()

    print("")
    print("Significant channel effects (p < .05), smallest |d_z| first:")
    labels = dict((c[0], c[1]) for c in MAINT_METRICS + TRACK_METRICS)
    for adz, task, col, dz, p in smallest_effects(mcube, tcube, tzcube)[:4]:
        print("  %-11s %-24s d_z = %+.2f  p = %s" % (task, labels[col], dz, fp(p)))

    print("")
    title = "Unrounded check: trials re-scored from the per-tick files with score_trials.py"
    print(title)
    print("-" * len(title))
    um, ut = score_unrounded("maintenance"), score_unrounded("tracking")
    for name, d, u in (("maintenance", maint, um), ("tracking", track, ut)):
        assert len(d) == len(u)
        assert (d["Trial"].to_numpy() == u["Trial"].to_numpy()).all()
        cols = [c for c in d.columns if c not in ("pid", "ch")]
        dev = max(np.nanmax(np.abs(d[c].to_numpy(float) - u[c].to_numpy(float))) for c in cols)
        print("%s: per-trial files equal the re-scored values to within %.1e" % (name, dev))
    compare("maintenance", mrep, maintenance(um, " [unrounded]")[0])
    compare("tracking", trep, tracking(ut, mean_tick, " [unrounded]")[0])


if __name__ == "__main__":
    main()
