"""Low and high targets in the intensity maintenance task (Appendix F-D and Table 11 of the paper).

Reproduces the numbers reported for the extreme targets of the intensity
maintenance task: how often the tone amplitude fell outside the calibrated
range and was clipped, what the clipping before smoothing did to the scored
result, where the gate dropouts fall, what limits the failed trials at target
20, the within-trial variability of the amplitude reading at each target, and
the time course of the undershoot at target 80.

Reads only the public data package:
    <DATA_DIR>/participant tests/pa<N>_intensity_maintenance_raw_data.csv  (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_intensity_maintenance_data.csv      self-check
    <DATA_DIR>/participant tests/calibration-log_data.csv                  intensity bounds
and imports, from <DATA_DIR>/scripts, dsp_chain.py (median of three and One
Euro filter of the chain) and score_trials.py (scoring definitions).

Usage:
    python target_extremes.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/, next to dsp_chain.py). Nothing is written; the
results go to standard output, and every run prints the same text. The run
takes less than a minute.

Definitions
  * Scoring window: TimeMs >= 1000 (every trial ends at 6000 ms). Band:
    |Force - target| <= 15. TIB, Success, SD and Bias as in score_trials.py.
  * U, the force before clipping:
        U = 100 (20 log10 A - dB_min) / (dB_max - dB_min),
    with A the logged Amplitude and dB_min, dB_max the participant's bounds
    after the last intensity calibration row (as in frequency_dependence.py).
    The chain clips U to 0..100 and then applies the median of three and the
    One Euro filter.
  * Dropout: Amplitude = 0. The chain has reset after six invalid frames
    (300 ms); FrequencyHz and Force are 0 on these ticks and U is undefined.
    Valid ticks are the ticks with Amplitude > 0.
  * Participant values are means over the participant's 5 trials at a
    target. Tests are over the 24 participants: one-sample t, two-sided, with
    a 95% confidence interval. Pooled values are taken over the 120 trials of
    a target, or over their ticks.
  * Slopes are ordinary least-squares slopes against time in seconds.
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
    sys.exit("usage: python target_extremes.py [DATA_DIR]")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DATA, "scripts"))
import dsp_chain as D  # noqa: E402
import score_trials as ST  # noqa: E402

# io_path adds the Windows extended-length prefix, so that a deeply nested
# package folder (paths over 260 characters) can still be read.
PT = ST.io_path(os.path.join(DATA, "participant tests"))

PIDS = ["pa%d" % i for i in range(1, 25)]
TARGETS = (20, 40, 60, 80)
WINDOW_START_MS = ST.WINDOW_START_MS
BAND = ST.BAND
HOLD_MS = ST.HOLD_MS
LAST_SECOND_MS = 5000               # start of the final second of the trial
FRAME_MS = 1000.0 * D.FRAME / D.FS  # the chain runs once per 50 ms audio frame
PHASE_STEP_MS = 0.5                 # grid of the fitted frame-clock phase
MODELS = ("M1", "M2")
PER_TRIAL_COLS = ("TIB", "Success", "SD", "Bias", "LongestStreak")


# ---------------------------------------------------------------------------
# Formatting and statistics
# ---------------------------------------------------------------------------
def fp(p):
    """p value as printed."""
    return "%.4f" % p if p >= 0.0001 else "%.1e" % p


def pct(k, n):
    return 100.0 * k / n


def ols_slope(x, y):
    """Least-squares slope and intercept of y on x."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    xm, ym = x.mean(), y.mean()
    b = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    return b, ym - b * xm


def one_sample(x):
    """Mean, SD, 95% CI and one-sample t test against 0 over participants."""
    x = np.asarray(x, np.float64)
    n = x.size
    m, sd = x.mean(), x.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    res = stats.ttest_1samp(x, 0.0)
    return dict(n=n, mean=m, sd=sd, lo=m - half, hi=m + half, t=res.statistic, p=res.pvalue)


def test_text(r, nd=2):
    f = "%+." + str(nd) + "f"
    return ((f + "  95%% CI [" + f + ", " + f + "]  t(%d) = %+.2f  p = %s")
            % (r["mean"], r["lo"], r["hi"], r["n"] - 1, r["t"], fp(r["p"])))


def participant_means(trials, values):
    """Mean over each participant's trials of one target, in PIDS order."""
    s = pd.Series(values, index=[tr["pid"] for tr in trials])
    return s.groupby(level=0).mean().reindex(PIDS).to_numpy()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_bounds():
    """(dB_min, dB_max) per participant, from the last intensity calibration row."""
    c = pd.read_csv(os.path.join(PT, "calibration-log_data.csv"))
    c = c[c.Type.isin(["MinIntensity", "MaxIntensity"])]
    last = c.groupby("Participant", sort=False).tail(1).set_index("Participant")
    bounds = {p: (20.0 * math.log10(float(last.loc[p, "MinAmplitude"])),
                  20.0 * math.log10(float(last.loc[p, "MaxAmplitude"]))) for p in PIDS}
    return bounds, len(c), int(c.Accepted.sum())


def load_trials(bounds):
    """One record per trial, in participant and trial order.

    Values are parsed as Python parses them (round_trip), which is how
    score_trials.py reads the same files."""
    trials = []
    for pid in PIDS:
        raw = pd.read_csv(os.path.join(PT, "%s_intensity_maintenance_raw_data.csv" % pid),
                          float_precision="round_trip")
        raw = raw.sort_values(["Trial", "SampleIndex"])
        db_min, db_max = bounds[pid]
        for trial, g in raw.groupby("Trial", sort=True):
            amp = g.Amplitude.to_numpy(np.float64)
            with np.errstate(invalid="ignore"):
                db = 20.0 * np.log10(np.where(amp > 0, amp, np.nan))
            assert g.Target.nunique() == 1
            t = g.TimeMs.to_numpy(np.int64)
            trials.append(dict(
                pid=pid, trial=int(trial), target=int(g.Target.iloc[0]), t=t,
                force=g.Force.to_numpy(np.float64), amp=amp, f0=g.FrequencyHz.to_numpy(np.float64),
                db_rel=db - db_min,                             # dB above the calibrated floor
                u=100.0 * (db - db_min) / (db_max - db_min),    # force before clipping
                win=t >= WINDOW_START_MS, valid=amp > 0))
    return trials


def by_target(trials, target):
    return [tr for tr in trials if tr["target"] == target]


# ---------------------------------------------------------------------------
# Scoring (definitions of score_trials.py)
# ---------------------------------------------------------------------------
def band_scores(t_ms, inb):
    """TIB and longest in-band streak (ms) over the scoring window.

    As in score_trials.score_maintenance: each tick stands for the real time
    since the tick before it, and the first tick of the window for the time
    since the window opened. inb is a per-tick in-band flag, so that the same
    rules can score a trial in which some ticks are counted as in band."""
    prev = WINDOW_START_MS
    in_band = consec = longest = 0.0
    for t, b in zip(t_ms.tolist(), inb.tolist()):
        if t < WINDOW_START_MS:
            continue
        dt = t - prev
        prev = t
        if b:
            in_band += dt
            consec += dt
            if consec > longest:
                longest = consec
        else:
            consec = 0.0
    return in_band / max(1.0, prev - WINDOW_START_MS), longest


def score_trial(t_ms, force, target):
    """TIB, Success, SD, Bias and LongestStreak, computed as score_trials does
    (population SD, sums taken left to right)."""
    tib, longest = band_scores(t_ms, np.abs(force - target) <= BAND)
    f = force[t_ms >= WINDOW_START_MS].tolist()
    assert len(f) >= ST.MIN_SAMPLES
    n = len(f)
    mean = sum(f) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in f) / n)
    return dict(TIB=tib, Success=1 if longest >= HOLD_MS else 0, SD=sd, Bias=mean - target,
                LongestStreak=longest / 1000.0)


def check_scoring(trials, scores):
    """Compare the scorer with score_trials.score_maintenance and with the
    per-trial files of the package. Returns (trials that differ from
    score_trials, cells that differ from the files, cells compared)."""
    differ = cells_differ = cells = 0
    for pid in PIDS:
        ref, _ = ST.score_maintenance(os.path.join(PT, "%s_intensity_maintenance_raw_data.csv" % pid))
        ref = {r["Trial"]: r for r in ref}
        pub = pd.read_csv(os.path.join(PT, "%s_intensity_maintenance_data.csv" % pid),
                          dtype=str).set_index("Trial")
        for tr, sc in zip(trials, scores):
            if tr["pid"] != pid:
                continue
            r = ref[tr["trial"]]
            differ += int(any(sc[k] != r[k] for k in PER_TRIAL_COLS))
            for k in PER_TRIAL_COLS:
                cells += 1
                cells_differ += int(ST.fmt(sc[k]) != pub.loc[str(tr["trial"]), k])
    return differ, cells_differ, cells


# ---------------------------------------------------------------------------
# Reconstruction of the chain output from the logged amplitude
# ---------------------------------------------------------------------------
def pre_smoothing(u):
    """Input to the median of three: column 0 as implemented (U clipped to
    0..100), column 1 without the clip at 0 (U clipped at 100 only)."""
    return np.array([min(max(u, 0.0), 100.0), min(u, 100.0)])


def fit_frame_counts(t, new, valid_pair):
    """Frames completed in each tick interval under a 50 ms frame clock.

    The tick log (about 62.7 ms) subsamples the 50 ms frame stream, so some
    intervals hold two frames. The clock phase is chosen on a 0.5 ms grid so
    that the counts agree best with the observed pattern: a tick whose logged
    value changed should follow at least one frame, a tick whose value did not
    change should follow none (the first phase with the fewest disagreements
    is used)."""
    best = None
    for phi in np.arange(0.0, FRAME_MS, PHASE_STEP_MS):
        n = np.floor((t - phi) / FRAME_MS)
        c = np.r_[0, np.diff(n)].astype(int)
        bad = (((~new) & (c > 0)) | (new & (c == 0))) & valid_pair
        s = int(bad.sum())
        if best is None or s < best[0]:
            best = (s, c)
    return best[1]


def reconstruct(tr, model):
    """Chain output per tick for both pre-smoothing variants (n_ticks x 2).

    U -> clip -> median of three -> One Euro (dsp_chain, dt = 50 ms). A tick
    whose logged (Amplitude, FrequencyHz) did not change holds the output. A
    dropout tick resets the median buffer and the filter to 0. The state at
    the first tick is taken from the logged Force.
      M1: a changed tick advances the chain by one frame.
      M2: a changed tick advances the chain by the number of frames the fitted
          frame clock places in the interval; the missing frames get values
          interpolated linearly in U (that is, in dB) between the two logged
          values. The first tick after a dropout advances it by one frame."""
    u = tr["u"]
    n = u.size
    reset = np.isnan(u)
    key = np.stack([tr["amp"], tr["f0"]], axis=1)
    new = np.r_[True, np.any(key[1:] != key[:-1], axis=1)] & ~reset
    valid_pair = np.r_[False, ~reset[1:] & ~reset[:-1]]
    if model == "M2":
        frames = fit_frame_counts(tr["t"].astype(np.float64), new, valid_pair)
    else:
        frames = np.ones(n, int)
    every = np.ones(2, bool)
    med = D._Median3((2,))
    euro = D._OneEuro((2,), **D.ONE_EURO)
    out = np.zeros((n, 2))
    if reset[0]:
        med.reset_to(0.0, every)
        euro.reset_to(0.0, every)
        cur = np.zeros(2)
    else:
        med.buf[:] = pre_smoothing(u[0])[:, None]
        med.count[:] = 3
        cur = np.full(2, tr["force"][0])
        euro.reset_to(0.0, every)
        euro.x[:] = cur
    out[0] = cur
    for i in range(1, n):
        if reset[i]:
            med.reset_to(0.0, every)
            euro.reset_to(0.0, every)
            cur = np.zeros(2)
        elif new[i]:
            steps = max(1, frames[i]) if model == "M2" else 1
            if reset[i - 1]:
                steps = 1
            vals = [u[i - 1] + (u[i] - u[i - 1]) * k / steps for k in range(1, steps)] + [u[i]]
            for v in vals:
                m = med.step(pre_smoothing(v), every)
                cur = np.array(euro.step(m, D.FILTER_DT, every), dtype=np.float64)
        out[i] = cur
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_checks(trials, scores, n_cal_rows, n_cal_accepted):
    print("0. Checks")
    differ, cells_differ, cells = check_scoring(trials, scores)
    print("  scorer vs score_trials.score_maintenance: %d of %d trials differ" % (differ, len(trials)))
    print("  scorer vs the per-trial files: %d of %d cells differ (%s)"
          % (cells_differ, cells, ", ".join(PER_TRIAL_COLS)))
    amp0 = np.concatenate([tr["amp"] == 0 for tr in trials])
    f00 = np.concatenate([tr["f0"] == 0 for tr in trials])
    force = np.concatenate([tr["force"] for tr in trials])
    print("  ticks with Amplitude = 0: %d; of these with FrequencyHz != 0: %d; ticks with FrequencyHz = 0 "
          "and Amplitude != 0: %d; largest |Force| on them: %.3f"
          % (amp0.sum(), (amp0 & ~f00).sum(), (f00 & ~amp0).sum(), np.abs(force[amp0]).max()))
    ends = sorted({int(tr["t"][-1]) for tr in trials})
    nwin = [int(tr["win"].sum()) for tr in trials]
    print("  last tick of every trial at %s ms; ticks in the scoring window per trial: %d-%d"
          % ("/".join(str(e) for e in ends), min(nwin), max(nwin)))
    print("  intensity calibration rows: %d, accepted: %d" % (n_cal_rows, n_cal_accepted))


def section_reconstruction(trials, recon):
    print("\n1. Reconstruction of the logged Force from U (window ticks)")
    print("   U -> clip 0..100 -> median of three -> One Euro, compared with the logged Force")
    print("   %-5s %-7s %6s %10s %8s %22s" % ("model", "target", "ticks", "Pearson r", "MAE", "same band call (%)"))
    for model in MODELS:
        for target in ("all",) + TARGETS:
            sel = [(tr, rc) for tr, rc in zip(trials, recon[model])
                   if target == "all" or tr["target"] == target]
            logged = np.concatenate([tr["force"][tr["win"]] for tr, _ in sel])
            rec = np.concatenate([rc[tr["win"], 0] for tr, rc in sel])
            tg = np.concatenate([np.full(tr["win"].sum(), tr["target"]) for tr, _ in sel])
            same = (np.abs(logged - tg) <= BAND) == (np.abs(rec - tg) <= BAND)
            print("   %-5s %-7s %6d %10.4f %8.3f %22.2f"
                  % (model, target, logged.size, np.corrcoef(logged, rec)[0, 1],
                     np.abs(rec - logged).mean(), 100.0 * same.mean()))


def section_clipping(trials):
    print("\n2. U outside the calibrated range (window ticks; % of all window ticks)")
    print("   %-6s %6s %18s %12s %18s %12s" % ("target", "ticks", "U < 0: ticks", "trials", "U > 100: ticks",
                                            "trials"))
    for target in TARGETS:
        tt = by_target(trials, target)
        n = sum(int(tr["win"].sum()) for tr in tt)
        lo = [int((tr["win"] & tr["valid"] & (tr["u"] < 0)).sum()) for tr in tt]
        hi = [int((tr["win"] & tr["valid"] & (tr["u"] > 100)).sum()) for tr in tt]
        print("   %-6d %6d %9d (%5.2f%%) %8d/%d %9d (%5.2f%%) %8d/%d"
              % (target, n, sum(lo), pct(sum(lo), n), sum(k > 0 for k in lo), len(tt),
                 sum(hi), pct(sum(hi), n), sum(k > 0 for k in hi), len(tt)))
    t20 = by_target(trials, 20)
    below = np.concatenate([-tr["db_rel"][tr["win"] & tr["valid"] & (tr["u"] < 0)] for tr in t20])
    print("  target 20, ticks with U < 0 (n = %d): median %.2f dB below the calibrated floor"
          % (below.size, np.median(below)))


def section_counterfactual(trials, scores, recon):
    print("\n3. Success with and without the clip at 0 before smoothing")
    print("   no clip at 0: logged Force + (chain output with U clipped at 100 only - chain output as")
    print("   implemented), both reconstructed with M2, rounded to 0.001 point as in the log, and scored")
    print("   with the definitions of score_trials.py")
    cf = []
    for tr, rc in zip(trials, recon["M2"]):
        force = np.round(tr["force"] + (rc[:, 1] - rc[:, 0]), 3)
        cf.append(score_trial(tr["t"], force, tr["target"]))
    print("   %-6s %8s %14s %7s %5s" % ("target", "logged", "no clip at 0", "gained", "lost"))
    for target in TARGETS:
        idx = [i for i, tr in enumerate(trials) if tr["target"] == target]
        a = np.array([scores[i]["Success"] for i in idx])
        b = np.array([cf[i]["Success"] for i in idx])
        print("   %-6d %4d/%d %10d/%d %7d %5d"
              % (target, a.sum(), a.size, b.sum(), b.size, ((b == 1) & (a == 0)).sum(),
                 ((b == 0) & (a == 1)).sum()))
    idx = [i for i, tr in enumerate(trials) if tr["target"] == 20]
    t20 = [trials[i] for i in idx]
    sd_log = participant_means(t20, [scores[i]["SD"] for i in idx]).mean()
    sd_cf = participant_means(t20, [cf[i]["SD"] for i in idx]).mean()
    print("  target 20, SD of Force (mean of participant means, as in Table 1): logged %.2f, no clip at 0 %.2f"
          % (sd_log, sd_cf))


def section_dropouts(trials, scores):
    print("\n4. Dropouts (Amplitude = 0), window ticks")
    print("   %-6s %6s %18s   %s" % ("target", "ticks", "dropout ticks", "of these in the last 1000 ms"))
    for target in TARGETS:
        tt = by_target(trials, target)
        n = sum(int(tr["win"].sum()) for tr in tt)
        d = np.concatenate([tr["t"][tr["win"] & ~tr["valid"]] for tr in tt])
        late = int((d > LAST_SECOND_MS).sum())
        print("   %-6d %6d %9d (%5.2f%%)   %3d/%3d (%5.1f%%)"
              % (target, n, d.size, pct(d.size, n), late, d.size, pct(late, d.size)))
    idx = [i for i, tr in enumerate(trials) if tr["target"] == 20]
    has = np.array([bool((trials[i]["win"] & ~trials[i]["valid"]).any()) for i in idx])
    succ = np.array([scores[i]["Success"] for i in idx])
    print("  target 20, success of trials with a dropout in the window %d/%d (%.1f%%), without %d/%d (%.1f%%)"
          % (succ[has].sum(), has.sum(), pct(succ[has].sum(), has.sum()),
             succ[~has].sum(), (~has).sum(), pct(succ[~has].sum(), (~has).sum())))


def section_failed_20(trials, scores):
    print("\n5. Failed trials at target 20")
    target = 20
    failed = [tr for tr, sc in zip(trials, scores) if tr["target"] == target and sc["Success"] == 0]
    below = above = 0
    classes = dict(below=0, above=0, either=0, both=0, neither=0)
    for tr in failed:
        f = tr["force"]
        inb = np.abs(f - target) <= BAND
        lo = f < target - BAND
        hi = f > target + BAND
        below += int((lo & tr["win"]).sum())
        above += int((hi & tr["win"]).sum())
        passes = [band_scores(tr["t"], m)[1] >= HOLD_MS for m in (inb | lo, inb | hi, inb | lo | hi)]
        if passes[0] and passes[1]:
            classes["either"] += 1
        elif passes[0]:
            classes["below"] += 1
        elif passes[1]:
            classes["above"] += 1
        elif passes[2]:
            classes["both"] += 1
        else:
            classes["neither"] += 1
    n_out = below + above
    print("  failed trials: %d of %d" % (len(failed), len(by_target(trials, target))))
    print("  out-of-band window ticks in the failed trials: %d; below the band %d (%.1f%%), above %d (%.1f%%)"
          % (n_out, below, pct(below, n_out), above, pct(above, n_out)))
    print("  failed trials that pass when out-of-band ticks are counted as in band:")
    print("    with the below-band ticks, but not with the above-band ticks: %d" % classes["below"])
    print("    with the above-band ticks, but not with the below-band ticks: %d" % classes["above"])
    print("    with either side alone:                                     %d" % classes["either"])
    print("    only with both sides together:                              %d" % classes["both"])
    print("    not even with both:                                         %d" % classes["neither"])


def section_variability(trials):
    print("\n6. Within-trial variability of the tone amplitude (valid window ticks)")
    sd_db, sd_f0 = {}, {}
    print("   %-6s %32s %34s" % ("target", "SD of dB reading (trial mean)", "SD of f0, semitones (trial mean)"))
    for target in TARGETS:
        tt = by_target(trials, target)
        a, b = [], []
        for tr in tt:
            m = tr["win"] & tr["valid"]
            a.append(np.std(tr["db_rel"][m]))
            f0 = tr["f0"][m]
            b.append(np.std(12.0 * np.log2(f0 / np.median(f0))))
        sd_db[target] = participant_means(tt, a)
        sd_f0[target] = participant_means(tt, b)
        print("   %-6d %32.2f %34.3f" % (target, np.mean(a), np.mean(b)))
    print("  participants with a larger dB SD at 20 than at the other target (participant means, Wilcoxon"
          " signed-rank):")
    for other in (40, 60, 80):
        w = stats.wilcoxon(sd_db[20], sd_db[other])
        print("    20 vs %d: %d of %d, p = %s"
              % (other, int((sd_db[20] > sd_db[other]).sum()), len(PIDS), fp(w.pvalue)))


def trial_time_course(tr):
    """Per-trial quantities of the time-course analysis (bias = Force - target)."""
    t, f, target = tr["t"], tr["force"], tr["target"]
    w = tr["win"]
    ts = t / 1000.0
    m15 = w & (t < LAST_SECOND_MS)
    m56 = w & (t >= LAST_SECOND_MS)
    ok = tr["valid"]
    e = f - target
    r = dict(bias=e[w].mean(), level=e[m15].mean(), last=e[m56].mean(),
             bin12=e[(t >= 1000) & (t < 2000)].mean())
    r["change"] = r["last"] - r["level"]                  # mean 5-6 s minus mean 1-5 s
    b, a = ols_slope(ts[m15], f[m15])
    r["slope15"] = b
    r["at_1s"] = a + b * 1.0 - target                     # fitted bias at 1 s, 1-5 s fit
    r["trend_part"] = b * (ts[m15].mean() - 1.0)          # level = at_1s + trend_part
    r["last_part"] = m56[w].mean() * r["change"]          # bias = level + last_part
    r["last_part_valid"] = e[w & ok].mean() - e[m15 & ok].mean()
    u = tr["u"]
    r["u_ge_target_15"] = np.mean(u[m15 & ok] >= target)
    r["clip100"] = np.mean(np.minimum(u[w & ok], 100.0) - u[w & ok])
    return r


def section_time_course(trials):
    print("\n7. Time course of the bias (Force - target), participant means, t over 24 participants")
    tc = [trial_time_course(tr) for tr in trials]
    rows = (("level: mean bias 1-5 s", "level", 2),
            ("bias 1-2 s", "bin12", 2),
            ("slope 1-5 s, points/s", "slope15", 3),
            ("last-second change: mean 5-6 s - mean 1-5 s", "change", 2))
    for target in TARGETS:
        idx = [i for i, tr in enumerate(trials) if tr["target"] == target]
        tt = [trials[i] for i in idx]
        print("  target %d" % target)
        for label, key, nd in rows:
            r = one_sample(participant_means(tt, [tc[i][key] for i in idx]))
            print("    %-46s %s" % (label, test_text(r, nd)))
    idx = [i for i, tr in enumerate(trials) if tr["target"] == 80]
    tt = [trials[i] for i in idx]
    pm = {k: participant_means(tt, [tc[i][k] for i in idx])
          for k in ("bias", "at_1s", "trend_part", "last_part", "level", "last_part_valid")}
    bias = pm["bias"].mean()
    print("  target 80, Bias = fitted bias at 1 s (1-5 s fit) + trend part within 1-5 s + last-second part:")
    print("    %+.2f = %+.2f + (%+.2f) + (%+.2f)"
          % (bias, pm["at_1s"].mean(), pm["trend_part"].mean(), pm["last_part"].mean()))
    for label, key in (("fitted bias at 1 s", "at_1s"), ("trend part", "trend_part"),
                       ("last-second part", "last_part"),
                       ("last-second part, valid ticks only", "last_part_valid")):
        print("    %-46s %s" % (label, test_text(one_sample(pm[key]), 2)))
    print("    share of Bias: level (mean bias 1-5 s) %.1f%%, last-second part %.1f%%"
          % (100.0 * pm["level"].mean() / bias, 100.0 * pm["last_part"].mean() / bias))
    return tc


def section_target80(trials, scores, tc):
    print("\n8. Target 80: failed and successful trials, reach and the clip at 100")
    idx = [i for i, tr in enumerate(trials) if tr["target"] == 80]
    for label, s in (("failed", 0), ("successful", 1)):
        sel = [i for i in idx if scores[i]["Success"] == s]
        print("  %-10s trials: %3d; bias 1-2 s %+.2f; slope 1-5 s %+.2f points/s (means over trials)"
              % (label, len(sel), np.mean([tc[i]["bin12"] for i in sel]),
                 np.mean([tc[i]["slope15"] for i in sel])))
    tt = [trials[i] for i in idx]
    share = participant_means(tt, [tc[i]["u_ge_target_15"] for i in idx])
    print("  share of valid 1-5 s ticks with U >= 80 (participant means of trial shares): "
          "M %.1f%%, SD %.1f%%, range %.1f-%.1f%%"
          % (100 * share.mean(), 100 * share.std(ddof=1), 100 * share.min(), 100 * share.max()))
    r = one_sample(participant_means(tt, [tc[i]["clip100"] for i in idx]))
    print("  effect of the clip at 100 alone on the bias, mean of min(U, 100) - U over valid window ticks:")
    print("    %s  (SD %.3f)" % (test_text(r, 3), r["sd"]))


def section_trial_number(trials, scores):
    print("\n9. Target 80: Bias against trial number in the block (1-20)")
    slopes = []
    for pid in PIDS:
        sel = [(tr["trial"], sc["Bias"]) for tr, sc in zip(trials, scores)
               if tr["pid"] == pid and tr["target"] == 80]
        slopes.append(ols_slope([s[0] for s in sel], [s[1] for s in sel])[0])
    r = one_sample(slopes)
    print("  per-participant OLS slope of Bias on Trial (5 trials each), points per trial:")
    print("    %s" % test_text(r, 3))


def main():
    # Plain line feeds, so that saved output is the same file on every platform.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    bounds, n_cal_rows, n_cal_accepted = load_bounds()
    trials = load_trials(bounds)
    scores = [score_trial(tr["t"], tr["force"], tr["target"]) for tr in trials]
    n_win = sum(int(tr["win"].sum()) for tr in trials)
    print("Low and high targets in the intensity maintenance task")
    print("%d participants, %d trials (%s per target), %d ticks in the scoring window"
          % (len(PIDS), len(trials), "/".join(str(len(by_target(trials, t))) for t in TARGETS), n_win))
    print("")
    section_checks(trials, scores, n_cal_rows, n_cal_accepted)
    recon = {m: [reconstruct(tr, m) for tr in trials] for m in MODELS}
    section_reconstruction(trials, recon)
    section_clipping(trials)
    section_counterfactual(trials, scores, recon)
    section_dropouts(trials, scores)
    section_failed_20(trials, scores)
    section_variability(trials)
    tc = section_time_course(trials)
    section_target80(trials, scores, tc)
    section_trial_number(trials, scores)


if __name__ == "__main__":
    main()
