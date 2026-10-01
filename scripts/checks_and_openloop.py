"""Data-integrity, order, fatigue and recency checks, open-loop production noise,
session log and pilot measurements (Sections V-A and V-D, Appendix F-F and
Table 13, Appendix G).

Reproduces the numbers reported in the paper for
  * the data overview: trial counts, ticks per scoring window, and the trials
    in which the output was reset to zero by the voicing gate;
  * the session log: capture device, sampling-rate correction factor and room
    level;
  * open-loop production noise: channel comparison in force points, its
    association with task performance, and the association between tone
    amplitude and f0 in the open-loop recordings;
  * the order, fatigue and recency checks (the practice analysis is done by
    mixed_models.py and the trials closest to a break by near_break_analysis.py);
  * the pilot measurements from which the task parameters were derived.

Reads only the public data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv       (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_raw_data.csv
    <DATA_DIR>/participant tests/calibration-log_data.csv
    <DATA_DIR>/participant tests/openloop-log_data.csv
    <DATA_DIR>/participant tests/openloop-raw_data.csv
    <DATA_DIR>/participant tests/session-log_data.csv
    <DATA_DIR>/participant tests/subjective-measures.xlsx
    <DATA_DIR>/pilot tests/<N>_<channel>_<task>_data.csv / _raw_data.csv  (N = 1..3)
    <DATA_DIR>/pilot tests/calibration-log_data.csv, openloop-log_data.csv,
        openloop-raw_data.csv, session-log_data.csv
and imports score_trials.py from <DATA_DIR>/scripts for the scoring constants.

Usage:
    python checks_and_openloop.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). Nothing is written; the results go to standard
output, and every run prints the same text. The run takes a few seconds.

Definitions
  * Scoring window: TimeMs >= 1000. Each tick stands for the interval that ends
    at it; the first tick of the window stands for the time since 1000 ms (as
    in score_trials.py).
  * Output reset by the voicing gate ("invalid tick"): after six invalid frames
    (300 ms) the chain sets the force to 0. In intensity files such ticks have
    Amplitude = 0. In pitch files FrequencyHz keeps its last value on these
    ticks, so a zero force is counted as a reset when the logged FrequencyHz
    maps to more than 5 force points with the participant's pitch bounds, or
    when FrequencyHz is 0. A zero force at a lower mapped value is the clip at
    the lower bound (the mapped values of zero-force ticks fall either below
    5 or above 10 points, so the threshold is not critical).
  * Calibrated bounds: the last accepted row of each type in the calibration
    log, i.e. the bounds in effect after calibration.
  * Open-loop noise: per participant the mean over the three segments of
    PitchSdPoints and AmpSdPoints; ratio = mean amplitude noise / mean pitch
    noise over the participants; paired t over the 24 participants, d_z =
    mean / SD of the paired differences.
  * Tone amplitude and f0 in open loop: per participant the Pearson r between
    ln f0 and 20 log10 Amplitude over all frames of the three segments.
  * Participant performance: mean over the participant's trials of a block.
  * Sequences (Table 8 of the paper): participants 1-12 followed sequence A
    (pitch maintenance, intensity maintenance, pitch tracking, intensity
    tracking), participants 13-24 sequence B, the exact reverse. This
    assignment is documented in COLUMNS.md.
  * Order check: pitch minus intensity of participant means, sequence A vs B,
    Student t (pooled variance), Cohen's d with the pooled SD.
  * Fatigue check: first half (maintenance trials 1-10, tracking 1-7) minus
    second half (11-20, 8-15) of each block, paired t over participants.
  * Recency check: SUS scored from the ten items (odd items x - 1, even items
    5 - x, sum x 2.5); raw TLX = mean of the six subscales with Performance as
    load (100 - x), averaged over the two blocks of a channel. Two-sample tests
    as in the order check; preference by exact two-sided binomial test.
  * Pilot: open-loop values are means over the nine segments (dB, semitones)
    or means and SDs over the three participant means (force points).
    Acquisition time is the start of the interval of the first in-band tick,
    measured from trial onset (the first-entry rule of score_trials.py with the
    window opened at 0 ms), averaged over the maintenance trials of a channel.
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
    sys.exit("usage: python checks_and_openloop.py [DATA_DIR]")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DATA, "scripts"))
import score_trials as ST  # noqa: E402

PT = ST.io_path(os.path.join(DATA, "participant tests"))
PILOT = ST.io_path(os.path.join(DATA, "pilot tests"))

PIDS = ["pa%d" % i for i in range(1, 25)]
PILOT_IDS = ["1", "2", "3"]
CHANNELS = ("pitch", "intensity")
TASKS = ("maintenance", "tracking")
NTR = {"maintenance": 20, "tracking": 15}
SEQ_A = PIDS[:12]           # pitch maintenance first
SEQ_B = PIDS[12:]           # intensity tracking first (reverse of A)
WINDOW_START_MS = ST.WINDOW_START_MS
BAND = ST.BAND
NOMINAL_TICK_S = 0.0625     # 50 ms timer request, rounded up to 4 x 15.625 ms
RESET_MIN_POINTS = 5.0      # see the module docstring
FORCE_COL = {"maintenance": "Force", "tracking": "Actual"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def say(s=""):
    print(s)


def fp(p):
    """p as printed in the paper: <.001, or three decimals without the zero."""
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def ttest1(x):
    """One-sample (paired-difference) t test with d_z."""
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    t = m / (sd / math.sqrt(n))
    return dict(n=n, mean=m, sd=sd, t=t, df=n - 1, p=2 * stats.t.sf(abs(t), n - 1), dz=m / sd)


def ttest2(a, b):
    """Student two-sample t test (pooled variance) with Cohen's d."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ttest_ind(a, b)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return dict(ma=a.mean(), mb=b.mean(), t=float(r.statistic), df=na + nb - 2,
                p=float(r.pvalue), d=(a.mean() - b.mean()) / sp)


def read(folder, name):
    return pd.read_csv(os.path.join(folder, name), float_precision="round_trip")


def trial_file(folder, pid, ch, task, raw=False):
    return read(folder, "%s_%s_%s_%sdata.csv" % (pid, ch, task, "raw_" if raw else ""))


def bounds(cal):
    """Pitch and intensity bounds in effect after calibration, per participant."""
    acc = cal[cal.Accepted == 1]
    out = {}
    for pid, d in acc.groupby("Participant", sort=False):
        last = {t: g.iloc[-1] for t, g in d.groupby("Type")}
        out[str(pid)] = dict(min_midi=float(last["MinPitch"].MeasuredValue),
                             max_midi=float(last["MaxPitch"].MeasuredValue),
                             min_amp=float(last["MinIntensity"].MeasuredValue),
                             max_amp=float(last["MaxIntensity"].MeasuredValue))
    return out


def load_trials(folder, pids):
    parts = []
    for pid in pids:
        for ch in CHANNELS:
            for task in TASKS:
                d = trial_file(folder, pid, ch, task)
                d["pid"], d["channel"], d["task"] = pid, ch, task
                parts.append(d)
    return pd.concat(parts, ignore_index=True)


def block_means(tr, pids, key):
    """Participant x channel x task means of one per-trial column."""
    g = tr.groupby(["pid", "channel", "task"])[key].mean()
    return {(p, c, t): float(g.loc[(p, c, t)]) for p in pids for c in CHANNELS for t in TASKS
            if (p, c, t) in g.index}


# ---------------------------------------------------------------------------
# 1. Data overview
# ---------------------------------------------------------------------------
def invalid_ticks(w, ch, task, bnd):
    """Boolean mask of the window ticks on which the gate had reset the output."""
    if ch == "intensity":
        return (w.Amplitude.values == 0.0)
    f = w.FrequencyHz.values.astype(float)
    force = w[FORCE_COL[task]].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
    mapped = 100.0 * (midi - bnd["min_midi"]) / (bnd["max_midi"] - bnd["min_midi"])
    return (force <= 0.0) & ((f <= 0.0) | (mapped > RESET_MIN_POINTS))


def data_overview(tr, bnd):
    say("=" * 78)
    say("1. DATA OVERVIEW")
    say("=" * 78)
    for task in TASKS:
        n = int((tr.task == task).sum())
        say("  %-11s trials: %d (%d participants x 2 channels x %d)"
            % (task, n, tr.pid.nunique(), NTR[task]))
    per = tr.groupby("pid").size()
    say("  scored trials: %d; per participant %d..%d; trials with an empty metric: %d"
        % (len(tr), per.min(), per.max(), int(tr[["RMSE", "TIB"]].isna().any(axis=1).sum())))
    ns = tr.NSamples.astype(float)
    say("  ticks per scoring window: %d..%d, M = %.2f, SD = %.2f (over %d trials)"
        % (ns.min(), ns.max(), ns.mean(), ns.std(ddof=1), len(ns)))

    rows = []
    for pid in PIDS:
        for ch in CHANNELS:
            for task in TASKS:
                raw = trial_file(PT, pid, ch, task, raw=True)
                for trial, g in raw.groupby("Trial", sort=True):
                    g = g.sort_values("TimeMs")
                    w = g[g.TimeMs >= WINDOW_START_MS]
                    t = w.TimeMs.values.astype(float)
                    dt = np.diff(np.r_[WINDOW_START_MS, t])
                    bad = invalid_ticks(w, ch, task, bnd[pid])
                    zero = w[FORCE_COL[task]].values <= 0.0
                    run = best = best_ms = run_ms = 0
                    for b, d in zip(bad, dt):
                        run, run_ms = (run + 1, run_ms + d) if b else (0, 0.0)
                        if run > best:
                            best, best_ms = run, run_ms
                    rows.append(dict(pid=pid, channel=ch, task=task, trial=int(trial), n=len(w),
                                     n_bad=int(bad.sum()), bad_ms=float(dt[bad].sum()),
                                     longest=best, longest_ms=float(best_ms),
                                     n_zero=int(zero.sum()),
                                     bad_last_s_only=bool(bad.any() and (t[bad] > 5000).all())))
    r = pd.DataFrame(rows)
    ticks = int(r.n.sum())
    say()
    say("  Ticks on which the voicing gate had reset the output to 0 (scoring window):")
    say("  %-9s %-11s %7s %13s %11s %13s %12s"
        % ("channel", "task", "trials", "with reset", "reset ticks", "last s only", "force = 0"))
    for ch in CHANNELS:
        for task in TASKS:
            s = r[(r.channel == ch) & (r.task == task)]
            say("  %-9s %-11s %7d %13d %11d %13d %12d"
                % (ch, task, len(s), int((s.n_bad > 0).sum()), int(s.n_bad.sum()),
                   int(s.bad_last_s_only.sum()), int((s.n_zero > 0).sum())))
    say("  all trials: %d of %d have at least one reset tick; %d of %d window ticks (%.2f %%) are reset"
        % (int((r.n_bad > 0).sum()), len(r), int(r.n_bad.sum()), ticks, 100.0 * r.n_bad.sum() / ticks))
    say("  ('force = 0': trials with any zero-force tick, resets and clips at the lower bound together)")
    say("  trials with a reset tick before 5000 ms: %d (pitch %d, intensity %d); resets only after 5000 ms"
        " (whistle ended before the trial): %d"
        % (int(((r.n_bad > 0) & ~r.bad_last_s_only).sum()),
           int(((r.n_bad > 0) & ~r.bad_last_s_only & (r.channel == "pitch")).sum()),
           int(((r.n_bad > 0) & ~r.bad_last_s_only & (r.channel == "intensity")).sum()),
           int(r.bad_last_s_only.sum())))
    half = r[r.n_bad > r.n / 2.0]
    say("  trials in which more than half of the window ticks are reset: %d" % len(half))
    for _, x in r.sort_values(["longest", "n_bad"], ascending=False).head(3).iterrows():
        say("    %s %s %s trial %d: %d of %d ticks reset (%.2f s by tick times); longest run %d ticks"
            " = %.2f s at the nominal %.1f ms tick, %.2f s by tick times"
            % (x.pid, x.channel, x.task, x.trial, x.n_bad, x.n, x.bad_ms / 1000.0, x.longest,
               x.longest * NOMINAL_TICK_S, 1000 * NOMINAL_TICK_S, x.longest_ms / 1000.0))
    return r


# ---------------------------------------------------------------------------
# 2. Session log
# ---------------------------------------------------------------------------
def session_log(folder, label):
    s = read(folder, "session-log_data.csv")
    cal = read(folder, "calibration-log_data.csv")
    ol = read(folder, "openloop-log_data.csv")
    say("  %s: %d sessions; devices: %s" % (label, len(s), "; ".join(sorted(set(s.Device)))))
    for name, d in (("session log", s), ("calibration log", cal), ("open-loop log", ol)):
        say("    mu in the %-15s: %s" % (name, ", ".join("%.4f" % v for v in sorted(set(d.SampleRateMultiplier)))))
    say("    room level dB(A): typical M = %.2f, SD = %.2f (range of typical %g..%g);"
        " lowest minimum %g, highest maximum %g"
        % (s.RoomDbaTyp.mean(), s.RoomDbaTyp.std(ddof=1), s.RoomDbaTyp.min(), s.RoomDbaTyp.max(),
           s.RoomDbaMin.min(), s.RoomDbaMax.max()))
    say("    minimum column M = %.2f, maximum column M = %.2f; ambient-check frames %d..%d"
        % (s.RoomDbaMin.mean(), s.RoomDbaMax.mean(), s.Frames.min(), s.Frames.max()))
    for c in ("BroadbandMeanDbFS", "BroadbandPeakDbFS", "BandpassedMeanDbFS", "BandpassedPeakDbFS"):
        say("    %-19s M = %7.2f, SD = %5.2f, range %6.1f..%6.1f dBFS"
            % (c, s[c].mean(), s[c].std(ddof=1), s[c].min(), s[c].max()))


# ---------------------------------------------------------------------------
# 3. Open-loop production noise
# ---------------------------------------------------------------------------
def openloop(tr):
    say()
    say("=" * 78)
    say("3. OPEN-LOOP PRODUCTION NOISE")
    say("=" * 78)
    ol = read(PT, "openloop-log_data.csv")
    raw = read(PT, "openloop-raw_data.csv")
    say("  segments: %d (%d participants x %d); valid frames per segment %d..%d, M = %.1f; frame rows in raw file: %d"
        % (len(ol), ol.Participant.nunique(), ol.groupby("Participant").size().max(),
           ol.Frames.min(), ol.Frames.max(), ol.Frames.mean(), len(raw)))
    chk = raw.groupby(["Participant", "Segment"]).size()
    assert (chk.reindex(pd.MultiIndex.from_frame(ol[["Participant", "Segment"]])).values == ol.Frames.values).all()
    pp = ol.groupby("Participant")[["PitchSdPoints", "AmpSdPoints", "PitchSdSemitones", "AmpSdDb"]].mean().loc[PIDS]
    for c, lab in (("PitchSdPoints", "pitch noise (points)"), ("AmpSdPoints", "amplitude noise (points)"),
                   ("PitchSdSemitones", "pitch SD (semitones)"), ("AmpSdDb", "amplitude SD (dB)")):
        say("  %-26s participant means: M = %6.3f, SD = %6.3f, range %6.3f..%6.3f"
            % (lab, pp[c].mean(), pp[c].std(ddof=1), pp[c].min(), pp[c].max()))
    r = ttest1(pp.AmpSdPoints - pp.PitchSdPoints)
    say("  ratio of means (amplitude / pitch): %.3f" % (pp.AmpSdPoints.mean() / pp.PitchSdPoints.mean()))
    say("  paired t: t(%d) = %.2f, p = %.3g, d_z = %.3f; amplitude > pitch in %d of %d participants"
        % (r["df"], r["t"], r["p"], r["dz"], int((pp.AmpSdPoints > pp.PitchSdPoints).sum()), len(pp)))
    say("  over the %d segments: median AmpSdPoints %.2f, median PitchSdPoints %.2f;"
        " median over participants of the median segment pitch SD %.4f semitones"
        % (len(ol), ol.AmpSdPoints.median(), ol.PitchSdPoints.median(),
           ol.groupby("Participant").PitchSdSemitones.median().loc[PIDS].median()))

    say()
    say("  Open-loop noise and task performance (Pearson r over the 24 participants):")
    m = {k: block_means(tr, PIDS, k) for k in ("TIB", "SD", "Success", "PearsonR")}
    for ch, noise in (("intensity", pp.AmpSdPoints), ("pitch", pp.PitchSdPoints)):
        for task, key in (("maintenance", "TIB"), ("maintenance", "Success"), ("maintenance", "SD"),
                          ("tracking", "TIB"), ("tracking", "PearsonR")):
            v = np.array([m[key][(p, ch, task)] for p in PIDS])
            rr, p = stats.pearsonr(noise.values, v)
            say("    %-9s %-11s %-8s ~ %-9s noise: r = %+.3f, p = %s"
                % (ch, task, key, ch, rr, fp(p)))

    say()
    say("  Tone amplitude and f0 in open loop, per participant (all frames of the three segments):")
    rs = pd.Series({p: float(np.corrcoef(np.log(g.FrequencyHz.values), 20 * np.log10(g.Amplitude.values))[0, 1])
                    for p, g in raw.groupby("Participant")}).loc[PIDS]
    v = np.sort(rs.values)
    say("    r(ln f0, dB): median %+.3f (12th and 13th of 24 sorted values: %+.3f, %+.3f), mean %+.3f"
        % (np.median(v), v[11], v[12], v.mean()))
    say("    positive in %d of %d participants; |r| > 0.3 in %d; range %+.3f..%+.3f"
        % (int((v > 0).sum()), len(v), int((np.abs(v) > 0.3).sum()), v.min(), v.max()))
    w = raw.copy()
    for c, f in (("st", lambda d: 12 * np.log2(d.FrequencyHz)), ("db", lambda d: 20 * np.log10(d.Amplitude))):
        w[c] = f(w)
        w[c] = w[c] - w.groupby(["Participant", "Segment"])[c].transform("mean")
    rw = np.sort([float(np.corrcoef(g.st, g.db)[0, 1]) for _, g in w.groupby("Participant")])
    say("    with segment means removed (within-segment association only): median %+.3f, positive in %d of %d"
        % (np.median(rw), int((rw > 0).sum()), len(rw)))
    return pp


# ---------------------------------------------------------------------------
# 4. Order
# ---------------------------------------------------------------------------
def order_check(tr):
    say()
    say("=" * 78)
    say("4. ORDER: channel difference (pitch - intensity) by sequence")
    say("=" * 78)
    say("  sequence A (pitch maintenance first): %s..%s, n = %d; sequence B (intensity tracking first): %s..%s, n = %d"
        % (SEQ_A[0], SEQ_A[-1], len(SEQ_A), SEQ_B[0], SEQ_B[-1], len(SEQ_B)))
    say("  %-11s %-9s %9s %9s %9s %7s %7s"
        % ("task", "measure", "A", "B", "t", "p", "d"))
    for task, keys in (("maintenance", ("TIB", "Success", "RMSE")), ("tracking", ("TIB", "PearsonR", "RMSE"))):
        for key in keys:
            m = block_means(tr, PIDS, key)
            diff = {p: m[(p, "pitch", task)] - m[(p, "intensity", task)] for p in PIDS}
            r = ttest2([diff[p] for p in SEQ_A], [diff[p] for p in SEQ_B])
            say("  %-11s %-9s %+9.3f %+9.3f %+9.2f %7s %+7.2f   (t(%d), p = %.4f)"
                % (task, key, r["ma"], r["mb"], r["t"], fp(r["p"]), r["d"], r["df"], r["p"]))


# ---------------------------------------------------------------------------
# 5. Fatigue
# ---------------------------------------------------------------------------
def fatigue_check(tr):
    say()
    say("=" * 78)
    say("5. FATIGUE: first half vs second half of each block (participant means)")
    say("=" * 78)
    say("  %-11s %-9s %-9s %9s %9s %8s %7s %8s" % ("task", "channel", "measure", "first", "second",
                                                  "t(23)", "p", "p exact"))
    rows = []
    for task, keys in (("maintenance", ("TIB", "Success", "RMSE")),
                       ("tracking", ("TIB", "PearsonR", "zr", "RMSE"))):
        half = NTR[task] // 2
        for ch in CHANNELS:
            d = tr[(tr.task == task) & (tr.channel == ch)].copy()
            d["zr"] = np.arctanh(d.PearsonR) if task == "tracking" else np.nan
            for key in keys:
                a = d[d.Trial <= half].groupby("pid")[key].mean().loc[PIDS]
                b = d[d.Trial > half].groupby("pid")[key].mean().loc[PIDS]
                r = ttest1(a.values - b.values)
                s = 100.0 if key in ("TIB", "Success") else 1.0
                higher_better = key != "RMSE"
                improved = (b.mean() > a.mean()) == higher_better
                rows.append(dict(task=task, channel=ch, key=key, p=r["p"], improved=improved))
                say("  %-11s %-9s %-9s %9.3f %9.3f %+8.2f %7s %8.4f  %s"
                    % (task, ch, key, s * a.mean(), s * b.mean(), r["t"], fp(r["p"]), r["p"],
                       "second half better" if improved else "second half worse"))
    r = pd.DataFrame(rows)
    sig = r[r.p < 0.05]
    say("  significant half differences: %d, of which the second half was worse: %d"
        % (len(sig), int((~sig.improved).sum())))
    prim = r[(r.channel == "intensity") & (((r.task == "maintenance") & r.key.isin(["TIB", "Success"])) |
                                           ((r.task == "tracking") & r.key.isin(["TIB", "PearsonR"])))]
    say("  intensity, primary measures (maintenance TIB, Success; tracking TIB, r): smallest p = %.4f" % prim.p.min())


# ---------------------------------------------------------------------------
# 6. Recency in the subjective measures
# ---------------------------------------------------------------------------
def recency_check():
    say()
    say("=" * 78)
    say("6. RECENCY IN THE SUBJECTIVE MEASURES")
    say("=" * 78)
    xl = os.path.join(PT, "subjective-measures.xlsx")
    tlx = pd.read_excel(xl, sheet_name="NASA-TLX raw", header=2)
    sus = pd.read_excel(xl, sheet_name="SUS", header=2)
    pref = pd.read_excel(xl, sheet_name="Preference", header=2)
    code = {"P%02d" % i: "pa%d" % i for i in range(1, 25)}

    tlx = tlx[tlx.iloc[:, 0].isin(code)].copy()
    sub = tlx.iloc[:, 2:8].to_numpy(dtype=float, copy=True)  # MD PD TD PE EF FR
    sub[:, 3] = 100.0 - sub[:, 3]                          # Performance as load
    tlx["raw"] = sub.mean(axis=1)
    assert np.allclose(tlx.raw.values, tlx.iloc[:, 9].astype(float).values), "raw TLX column"
    tlx["pid"] = tlx.iloc[:, 0].map(code)
    tlx["ch"] = tlx.iloc[:, 1].str.split("-").str[1].str.lower()
    raw_ch = tlx.groupby(["pid", "ch"]).raw.mean()

    sus = sus[sus.iloc[:, 0].isin(code)].copy()
    q = sus.iloc[:, 2:12].astype(float).values
    score = 2.5 * ((q[:, 0::2] - 1).sum(axis=1) + (5 - q[:, 1::2]).sum(axis=1))
    assert np.allclose(score, sus.iloc[:, 12].astype(float).values), "SUS score column"
    sus["score"] = score
    sus["pid"] = sus.iloc[:, 0].map(code)
    sus["ch"] = sus.iloc[:, 1].str.lower()
    s = sus.set_index(["pid", "ch"]).score

    pref = pref[pref.iloc[:, 0].isin(code)]
    pr = dict(zip(pref.iloc[:, 0].map(code), pref.iloc[:, 1].str.lower()))
    for lab, grp, last in (("A", SEQ_A, "intensity"), ("B", SEQ_B, "pitch")):
        k = sum(pr[p] == "pitch" for p in grp)
        say("  sequence %s (final block %-9s): pitch preferred by %d of %d" % (lab, last, k, len(grp)))
    k = sum(pr[p] == "pitch" for p in SEQ_A)
    say("  two-sided binomial test, participants whose final block was intensity: p = %.6f"
        % stats.binomtest(k, len(SEQ_A), 0.5).pvalue)
    for lab, f in (("raw TLX, pitch - intensity", lambda p: raw_ch[(p, "pitch")] - raw_ch[(p, "intensity")]),
                   ("SUS, pitch - intensity", lambda p: s[(p, "pitch")] - s[(p, "intensity")]),
                   ("SUS pitch (A: after tracking, B: after maintenance)", lambda p: s[(p, "pitch")]),
                   ("SUS intensity (A: after tracking, B: after maintenance)", lambda p: s[(p, "intensity")])):
        r = ttest2([f(p) for p in SEQ_A], [f(p) for p in SEQ_B])
        say("  %-55s A %+7.2f  B %+7.2f  t(%d) = %+.2f, p = %s (%.4f)"
            % (lab, r["ma"], r["mb"], r["df"], r["t"], fp(r["p"]), r["p"]))


# ---------------------------------------------------------------------------
# 7. Pilot
# ---------------------------------------------------------------------------
def pilot():
    say()
    say("=" * 78)
    say("7. PILOT MEASUREMENTS (3 participants)")
    say("=" * 78)
    tr = load_trials(PILOT, PILOT_IDS)
    per = tr.groupby("pid").size()
    say("  trials per participant: %s" % ", ".join("%s: %d" % (p, per[p]) for p in PILOT_IDS))
    ol = read(PILOT, "openloop-log_data.csv")
    ol["Participant"] = ol.Participant.astype(str)
    say("  open-loop segments: %d; valid frames %d..%d" % (len(ol), ol.Frames.min(), ol.Frames.max()))
    say("  over the segments: pitch SD M = %.3f semitones, amplitude SD M = %.3f dB"
        % (ol.PitchSdSemitones.mean(), ol.AmpSdDb.mean()))
    pp = ol.groupby("Participant")[["PitchSdPoints", "AmpSdPoints"]].mean().loc[PILOT_IDS]
    noise = pp.AmpSdPoints.mean()
    say("  force points, participant means: pitch %.3f +/- %.3f, amplitude %.3f +/- %.3f (M +/- SD); ratio %.2f"
        % (pp.PitchSdPoints.mean(), pp.PitchSdPoints.std(ddof=1), noise, pp.AmpSdPoints.std(ddof=1),
           noise / pp.PitchSdPoints.mean()))
    say("  tolerance: band 15 points / amplitude noise = %.3f; 1.40 x noise = %.2f points"
        % (BAND / noise, 1.40 * noise))

    raw = read(PILOT, "openloop-raw_data.csv")
    ratios = {"pitch": [], "amplitude": []}
    for _, g in raw.groupby(["Participant", "Segment"], sort=True):
        g = g.sort_values("SampleIndex")
        for lab, x in (("pitch", 12 * np.log2(g.FrequencyHz.values)), ("amplitude", 20 * np.log10(g.Amplitude.values))):
            sm = pd.Series(x).rolling(5).mean().dropna().values          # 5 frames = 250 ms
            ratios[lab].append(np.std(sm) / np.std(x))
    say("  250 ms (5-frame) moving average: SD after / SD before, mean over segments: pitch %.3f, amplitude %.3f"
        % (np.mean(ratios["pitch"]), np.mean(ratios["amplitude"])))

    cal = read(PILOT, "calibration-log_data.csv")
    cal["Participant"] = cal.Participant.astype(str)
    b = bounds(cal)
    span = pd.Series({p: b[p]["max_midi"] - b[p]["min_midi"] for p in PILOT_IDS})
    rng = pd.Series({p: 20 * math.log10(b[p]["max_amp"] / b[p]["min_amp"]) for p in PILOT_IDS})
    say("  calibrated pitch span: %.2f +/- %.2f semitones (%s)"
        % (span.mean(), span.std(ddof=1), ", ".join("%.2f" % v for v in span)))
    say("  calibrated intensity range: %.2f +/- %.2f dB (%s)"
        % (rng.mean(), rng.std(ddof=1), ", ".join("%.2f" % v for v in rng)))
    widest = rng.idxmax()
    say("  widest intensity range: participant %s. Pitch - intensity by participant:" % widest)
    for task, key in (("maintenance", "TIB"), ("maintenance", "Success"), ("maintenance", "RMSE"),
                      ("tracking", "TIB"), ("tracking", "PearsonR"), ("tracking", "RMSE")):
        m = block_means(tr, PILOT_IDS, key)
        adv = {p: m[(p, "pitch", task)] - m[(p, "intensity", task)] for p in PILOT_IDS}
        if key == "RMSE":
            adv = {p: -v for p, v in adv.items()}         # advantage = lower pitch error
        smallest = min(adv, key=adv.get)
        say("    %-11s %-8s advantage: %s; smallest: participant %s"
            % (task, key, ", ".join("%s %+.3f" % (p, adv[p]) for p in PILOT_IDS), smallest))

    say("  Acquisition, maintenance: start of the first in-band tick interval, from trial onset")
    for ch in CHANNELS:
        start, tick = [], []
        for pid in PILOT_IDS:
            raw = trial_file(PILOT, pid, ch, "maintenance", raw=True)
            for _, g in raw.groupby("Trial", sort=True):
                g = g.sort_values("TimeMs")
                t = g.TimeMs.values.astype(float)
                inb = np.abs(g.Force.values - g.Target.values) <= BAND
                i = int(np.argmax(inb))
                assert inb[i], "a pilot maintenance trial never entered the band"
                start.append((t[i - 1] if i > 0 else 0.0) / 1000.0)
                tick.append(t[i] / 1000.0)
        say("    %-9s mean %.3f s over %d trials (time of the first in-band tick itself: %.3f s)"
            % (ch, np.mean(start), len(start), np.mean(tick)))
    say()
    session_log(PILOT, "pilot session log")


# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    tr = load_trials(PT, PIDS)
    bnd = bounds(read(PT, "calibration-log_data.csv"))
    data_overview(tr, bnd)
    say()
    say("=" * 78)
    say("2. SESSION LOG")
    say("=" * 78)
    session_log(PT, "main study")
    openloop(tr)
    order_check(tr)
    fatigue_check(tr)
    recency_check()
    pilot()


if __name__ == "__main__":
    main()
