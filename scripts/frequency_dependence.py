"""Frequency dependence of the intensity estimate (Appendix A-G, its table and Fig. 9).

For a tone of constant amplitude the intensity reading still depends on the
tone frequency: through the scalloping loss of the Hann-windowed Goertzel bins
(20 Hz apart), through the slope of the band-pass filter inside the whistle
range, and through the three-frame median frequency, which trails the pitch
when the pitch changes quickly. This script measures the three effects with
the signal chain in dsp_chain.py, driven by synthetic tones, replays the
recorded pitch tracks of the study as tones of constant amplitude, compares
the prediction with the recorded open-loop amplitudes, and estimates the
effect on the scored results and on the calibrated bounds.

Reads only the public data package:
    <DATA_DIR>/participant tests/calibration-log_data.csv   calibrated bounds
    <DATA_DIR>/participant tests/openloop-log_data.csv      open-loop segments
    <DATA_DIR>/participant tests/openloop-raw_data.csv      open-loop frames
    <DATA_DIR>/participant tests/pa<N>_intensity_<task>_raw_data.csv   (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_raw_data.csv   scored again
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv       self-check

Usage:
    python frequency_dependence.py [DATA_DIR] [OUT_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/, next to dsp_chain.py). OUT_DIR defaults to
frequency_dependence_output/ next to this script; it receives the CSV tables
and results.txt, which lists every number reported in the appendix. The run
takes several minutes. Random pitch walks and the calibration model use a set
random seed, so every run gives the same numbers. With another seed the medians
of the random walks move by a few hundredths (force SD about +/-0.03 points).

Scoring follows score_trials.py: scoring window 1000 ms to trial end, band
+/-15 points, success = 3 s in band without interruption, time measures
weighted by the real interval between ticks, population SD and RMSE. The
script checks this by scoring the recorded force again and comparing the result
with the per-trial files of the package.
"""
import glob
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsp_chain as D  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

CALIBRATION_FILE = "calibration-log_data.csv"
OPENLOOP_FILE = "openloop-log_data.csv"
OPENLOOP_FRAMES_FILE = "openloop-raw_data.csv"

PIDS = ["pa%d" % i for i in range(1, 25)]
TASKS = ("maintenance", "tracking")
RANDOM_SEED = 12345

# Scoring definitions of score_trials.py.
WINDOW_START_MS = 1000
HOLD_MS = 3000
BAND = 15.0
MIN_SAMPLES = 20

# Synthetic tests.
PREROLL = 20                     # frames (1 s) before the analysed part, so every state is settled
SWEEP_SETTLE, SWEEP_MEASURE = 30, 10
FM_CARRIERS = (("centre", 1200.0), ("quarter", 1205.0), ("edge", 1210.0))
FM_DEPTHS = (0.05, 0.15, 0.3)    # peak deviation, semitones
FM_RATES = (0.5, 1.0, 2.0, 5.0)  # Hz
FM_PHASES = 8                    # the 50 ms frame clock samples a 5 Hz modulation at only 4 phases
FM_FRAMES = 600                  # 30 s
WALK_SEGMENTS, WALK_FRAMES, WALK_RATE = 40, 120, 200.0
TASK_FRAMES = 120                # 6 s trial
SCORE_FRAME = 20                 # first replay frame inside the scoring window
MC_DRAWS = 200

TEXT = io.StringIO()
NUMBERS = []


def say(s=""):
    print(s)
    TEXT.write(s + "\n")
    sys.stdout.flush()


def progress(s):
    """Console-only progress line (not part of results.txt)."""
    print(s)
    sys.stdout.flush()


def report(key, value, fmt, text):
    """Record one reported quantity: key, value, and the value as printed."""
    printed = fmt % value
    NUMBERS.append(dict(key=key, value=value, printed=printed, description=text))
    say("  %-44s %s   %s" % (key, printed, text))


def psd(x):
    """Population standard deviation."""
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2))) if x.size else np.nan


def pct(x, q):
    return float(np.percentile(np.asarray(x, np.float64), q))


# ---------------------------------------------------------------------------
# Package data
# ---------------------------------------------------------------------------
def load_bounds():
    """Intensity bounds in effect after the last intensity calibration attempt."""
    c = pd.read_csv(os.path.join(PT, CALIBRATION_FILE))
    c = c[c.Type.isin(["MinIntensity", "MaxIntensity"])]
    last = c.groupby("Participant", sort=False).tail(1).set_index("Participant")
    b = last[["MinAmplitude", "MaxAmplitude", "IntensityRangeDb"]].astype(float).copy()
    b["R"] = 20.0 * np.log10(b.MaxAmplitude / b.MinAmplitude)
    return b.loc[PIDS]


def load_raw(pid, channel, task):
    d = pd.read_csv(os.path.join(PT, "%s_%s_%s_raw_data.csv" % (pid, channel, task)))
    d = d.sort_values(["Trial", "SampleIndex"]).reset_index(drop=True)
    d["F"] = d["Force"] if task == "maintenance" else d["Actual"]
    return d


DATA = PT = OUTDIR = None
BOUNDS = None
R_MED = R_MIN = R_MAX = PTS_PER_DB = A_CENTRE = A_SWEEP = None
MID_BOUNDS = None


def configure(data_dir, out_dir):
    """Set the input and output folders and the quantities derived from the bounds."""
    global DATA, PT, OUTDIR, BOUNDS, R_MED, R_MIN, R_MAX, PTS_PER_DB, A_CENTRE, A_SWEEP, MID_BOUNDS
    DATA = os.path.abspath(data_dir)
    PT = os.path.join(DATA, "participant tests")
    OUTDIR = os.path.abspath(out_dir)
    os.makedirs(OUTDIR, exist_ok=True)
    BOUNDS = load_bounds()
    A_CENTRE = float(np.sqrt(BOUNDS.MinAmplitude.median() * BOUNDS.MaxAmplitude.median()))
    R_MED, R_MIN, R_MAX = float(BOUNDS.R.median()), float(BOUNDS.R.min()), float(BOUNDS.R.max())
    PTS_PER_DB = 100.0 / R_MED
    # Force mapping of the sweeps and FM tests: the median range, centred on A_CENTRE.
    MID_BOUNDS = np.array([[A_CENTRE * 10 ** (-R_MED / 40.0), A_CENTRE * 10 ** (R_MED / 40.0)]])
    # Tone amplitude whose best-case reading (bin centre, band-pass maximum on the
    # bin grid) equals A_CENTRE, i.e. force 50 at the median range.
    h_grid_max = float(D.bandpass_response_db(np.arange(700.0, 4500.01, D.BIN_HZ)).max())
    A_SWEEP = A_CENTRE / D.COHERENT_GAIN / 10 ** (h_grid_max / 20.0)


def own_bounds(pids):
    b = BOUNDS.loc[list(pids)]
    return np.stack([b.MinAmplitude.values, b.MaxAmplitude.values], axis=1)[:, None, :]


def amp_for_level(pid, points, f_ref):
    """Peak amplitude whose bin-centred reading at f_ref maps to `points` with the
    participant's bounds. It only keeps the replayed force off the 0/100 limits."""
    r = BOUNDS.loc[pid]
    level = r.MinAmplitude * 10 ** (points / 100.0 * r.R / 20.0)
    return level / D.COHERENT_GAIN / 10 ** (D.bandpass_response_db(f_ref)[0] / 20.0)


def run_batches(specs, n_frames, preroll, variants, batch=32):
    """Synthesise and run a list of traces of equal length.

    specs: dicts with 'track' (number or f(t)), 'amp', 'bounds' (M, 2) and
    optionally 'gate' and 'ideal'. Returns the chain outputs for the analysed
    frames (pre-roll removed), stacked over traces."""
    out = {}
    for s in range(0, len(specs), batch):
        part = specs[s:s + batch]
        xs, tf = zip(*[D.synthesize(p["track"], p["amp"], n_frames, preroll=preroll, gate=p.get("gate"))
                       for p in part])
        ideal = np.array([p.get("ideal", 0.0) for p in part])
        o = D.run_chain(np.stack(xs), np.stack([p["bounds"] for p in part]), true_f0=np.stack(tf),
                        ideal_tone=ideal, variants=variants)
        o["f_true"] = np.stack(tf)
        for k, v in o.items():
            out.setdefault(k, []).append(v[:, preroll:])
    return {k: np.concatenate(v) for k, v in out.items()}


def db(x):
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.asarray(x, np.float64))


# ---------------------------------------------------------------------------
# 1. Constant-amplitude sweeps
# ---------------------------------------------------------------------------
def sweep(f_list, label):
    specs = [dict(track=f, amp=A_SWEEP, bounds=MID_BOUNDS[0][None, :]) for f in f_list]
    o = run_batches(specs, SWEEP_SETTLE + SWEEP_MEASURE, 0, ("task", "true", "true_unfiltered"), batch=200)
    m = slice(SWEEP_SETTLE, SWEEP_SETTLE + SWEEP_MEASURE)
    t = pd.DataFrame(dict(f=f_list, valid_frames=o["valid"].sum(axis=1)))
    for v in ("task", "true", "true_unfiltered"):
        t[v + "_amp"] = o[v + "_amp"][:, m].mean(axis=1)
        t[v + "_force"] = o[v + "_force"][:, m, 0].mean(axis=1)
    a = db(o["task_amp"][:, m])
    t["task_frame_p2p_db"] = a.max(axis=1) - a.min(axis=1)
    t["raw_f0_error_hz"] = o["f0_raw"][:, -1] - t.f
    t["median_f0_error_hz"] = o["task_probe"][:, -1] - t.f
    say("  %s: %d tones" % (label, len(t)))
    progress("    (%.0f s)" % (time.time() - T0))
    return t


def sweeps():
    say("\n1. Constant-amplitude sweeps")
    wide = sweep(np.round(700.0 + 0.5 * np.arange(7601), 6), "700-4500 Hz, 0.5 Hz steps")
    fine = sweep(np.round(1000.0 + 0.1 * np.arange(4001), 6), "1000-1400 Hz, 0.1 Hz steps")
    ib = int(wide.task_amp.idxmax())
    best, f_best = float(wide.task_amp[ib]), float(wide.f[ib])
    for t in (wide, fine):
        t["att_db"] = db(t.task_amp / best)
        t["scalloping_db"] = db(t.true_unfiltered_amp / (D.COHERENT_GAIN * A_SWEEP))
        t["bandpass_db"] = D.bandpass_response_db(t.f.values) - D.bandpass_response_db(f_best)[0]
        t["force_change_points"] = t.task_force - float(wide.task_force[ib])
    wide.to_csv(os.path.join(OUTDIR, "sweep_wide.csv"), index=False)
    fine.to_csv(os.path.join(OUTDIR, "sweep_fine.csv"), index=False)

    hb = D.bandpass_response_db(np.arange(700.0, 4500.01, 0.5))
    f05 = np.arange(700.0, 4500.01, 0.5)
    f0_ticks = np.concatenate([pd.read_csv(fn, usecols=["FrequencyHz"]).FrequencyHz.values
                               for fn in sorted(glob.glob(os.path.join(PT, "pa*_intensity_*_raw_data.csv")))])
    f0_ticks = f0_ticks[f0_ticks > 0]
    both = pd.concat([wide, fine])
    say("  every frame valid: %s" % bool((both.valid_frames == SWEEP_SETTLE + SWEEP_MEASURE).all()))
    report("sweep_best_frequency_hz", f_best, "%.1f", "sweep: frequency of the largest reading")
    report("bandpass_peak_hz", float(f05[hb.argmax()]), "%.1f", "band-pass response: frequency of its maximum")
    report("scalloping_span_db", float(fine.scalloping_db.max() - fine.scalloping_db.min()), "%.2f",
           "scalloping only (no band-pass, true frequency), span 1.0-1.4 kHz")
    report("scalloping_span_wide_db", float(wide.scalloping_db.max() - wide.scalloping_db.min()), "%.2f",
           "scalloping only, span 0.7-4.5 kHz")
    report("scalloping_span_points", float(fine.true_unfiltered_force.max() - fine.true_unfiltered_force.min()),
           "%.1f", "scalloping only, span in force points (median range)")
    report("bandpass_rise_1000_1400_db", float(np.diff(D.bandpass_response_db([1000.0, 1400.0]))[0]), "%.2f",
           "band-pass response at 1.4 kHz minus at 1.0 kHz")
    report("share_f0_1000_1400_pct", 100.0 * float(np.mean((f0_ticks >= 1000) & (f0_ticks <= 1400))), "%.0f",
           "percent of the recorded intensity-trial pitch values within 1.0-1.4 kHz")
    report("span_1000_1400_db", float(fine.att_db.max() - fine.att_db.min()), "%.2f",
           "full chain, span 1.0-1.4 kHz")
    report("span_1000_1400_points", float(fine.task_force.max() - fine.task_force.min()), "%.1f",
           "full chain, span 1.0-1.4 kHz in force points (median range)")
    report("span_700_4500_db", float(wide.att_db.max() - wide.att_db.min()), "%.2f", "full chain, span 0.7-4.5 kHz")
    report("span_700_4500_points", float(wide.task_force.max() - wide.task_force.min()), "%.1f", "full chain, span 0.7-4.5 kHz in force points (median range)")
    report("frame_to_frame_p2p_max_db", float(both.task_frame_p2p_db.max()), "%.1e",
           "steady tones: largest change between the measured frames")
    report("median_vs_true_frequency_max_db", float(np.abs(db(both.task_amp / both.true_amp)).max()), "%.1e",
           "steady tones: largest difference, probe at the median vs the true frequency")
    report("yin_error_max_hz", float(both.raw_f0_error_hz.abs().max()), "%.2f",
           "steady tones: largest YIN error (below 20 Hz, so the nearest bin is probed)")
    report("range_ratio_narrowest", R_MED / R_MIN, "%.2f", "median range / narrowest range (factor on the points)")
    report("range_ratio_widest", R_MED / R_MAX, "%.2f", "median range / widest range (factor on the points)")
    report("range_median_db", R_MED, "%.1f", "median calibrated intensity range")
    report("range_min_db", R_MIN, "%.1f", "narrowest calibrated intensity range")
    report("range_max_db", R_MAX, "%.1f", "widest calibrated intensity range")
    return best, f_best


def linearity():
    say("\n   Linearity (steady tones, peak amplitude 0.0005 x 2^j, j = 0..10)")
    freqs = (700.5, 1200.0, 1205.0, 1210.0, 1793.5, 4490.0)
    amps = 0.0005 * 2.0 ** np.arange(11)
    specs = [dict(track=f, amp=a, bounds=MID_BOUNDS[0][None, :]) for f in freqs for a in amps]
    o = run_batches(specs, SWEEP_SETTLE + SWEEP_MEASURE, 0, ("task",), batch=66)
    r = o["task_amp"][:, SWEEP_SETTLE:].mean(axis=1).reshape(len(freqs), len(amps))
    steps = np.diff(db(r), axis=1)
    pd.DataFrame(steps, index=freqs, columns=["step_%d" % j for j in range(1, len(amps))]).to_csv(
        os.path.join(OUTDIR, "linearity_steps_db.csv"))
    report("doubling_step_db", float(steps.mean()), "%.2f", "mean change of the reading for a doubling of the amplitude")
    report("doubling_step_max_dev_db", float(np.abs(steps - 20 * np.log10(2)).max()), "%.1e",
           "largest deviation from 20 log10(2)")


# ---------------------------------------------------------------------------
# 2. Frequency-modulated tones
# ---------------------------------------------------------------------------
def fm_track(fc, depth, rate, phase):
    return lambda t: fc * 2.0 ** (depth / 12.0 * np.sin(2.0 * np.pi * rate * t + phase))


def fm_tests():
    say("\n2. Frequency-modulated tones (constant amplitude, median range)")
    specs, meta = [], []
    for tag, fc in FM_CARRIERS:
        for ph in range(FM_PHASES):
            phase = 2.0 * np.pi * ph / FM_PHASES
            conds = [(0.0, 0.0)] + [(d, r) for d in FM_DEPTHS for r in FM_RATES]
            for d, r in conds:
                specs.append(dict(track=fm_track(fc, d, r, phase), amp=A_SWEEP, bounds=MID_BOUNDS[0][None, :]))
                meta.append(dict(carrier=tag, fc=fc, depth=d, rate=r, phase=ph))
    o = run_batches(specs, FM_FRAMES, PREROLL, ("task", "true"), batch=12)
    rows = []
    for i, m in enumerate(meta):
        v = o["valid"][i]
        r = dict(m, valid_frames=int(v.sum()))
        for var in ("task", "true"):
            r[var + "_amp_sd_db"] = psd(db(o[var + "_amp"][i, v]))
            r[var + "_force_sd"] = psd(o[var + "_force"][i, v, 0])
        err = np.abs(o["task_probe"][i, v] - o["f_true"][i, v])
        r["median_err_gt20_pct"] = 100.0 * float(np.mean(err > 20.0))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUTDIR, "fm_by_phase.csv"), index=False)
    key = ["carrier", "fc", "depth", "rate"]
    t["d_force_sd"] = (t.task_force_sd - t.true_force_sd).abs()
    t["d_amp_sd_db"] = (t.task_amp_sd_db - t.true_amp_sd_db).abs()
    g = t.groupby(key, sort=False).agg(
        task_amp_sd_db=("task_amp_sd_db", "mean"), true_amp_sd_db=("true_amp_sd_db", "mean"),
        task_force_sd=("task_force_sd", "mean"), true_force_sd=("true_force_sd", "mean"),
        median_err_gt20_pct=("median_err_gt20_pct", "mean"), d_force_sd_max=("d_force_sd", "max"),
        d_amp_sd_db_max=("d_amp_sd_db", "max")).reset_index()
    tt = np.linspace(0.0, 1.0, 200001)
    g["max_rate_hz_per_s"] = [float(np.abs(np.gradient(fm_track(r.fc, r.depth, r.rate, 0.0)(tt), tt)).max())
                              for r in g.itertuples()]
    g.to_csv(os.path.join(OUTDIR, "fm_conditions.csv"), index=False)
    mod = g[g.depth > 0]
    fast = mod[(mod.depth == 0.3) & (mod.rate == 5.0)]
    rest = mod[~((mod.depth == 0.3) & (mod.rate == 5.0))]
    report("fm_conditions", len(mod), "%d", "number of modulation conditions")
    report("fm_amp_sd_max_db", float(mod.task_amp_sd_db.max()), "%.2f",
           "largest SD of the amplitude after the energy window")
    report("fm_force_sd_max_points", float(mod.task_force_sd.max()), "%.1f",
           "largest SD of the force after the full chain (median range)")
    report("fm_median_vs_true_other_max_db", float(max(rest.d_amp_sd_db_max.max(), 0.0)), "%.1e",
           "all conditions except +/-0.3 st at 5 Hz: amplitude SD, median vs true frequency")
    report("fm_median_vs_true_other_max_points", float(rest.d_force_sd_max.max()), "%.1e",
           "the same for the force SD")
    report("fm_fast_max_rate_hz_s", float(fast.max_rate_hz_per_s.max()), "%.0f",
           "+/-0.3 st at 5 Hz: largest rate of pitch change")
    report("fm_fast_median_err_gt20_pct", float(fast.median_err_gt20_pct.mean()), "%.0f",
           "+/-0.3 st at 5 Hz: percent of frames with the median more than 20 Hz off")
    report("fm_fast_force_sd_change_max", float(fast.d_force_sd_max.max()), "%.2f",
           "+/-0.3 st at 5 Hz: largest change of the force SD, median vs true frequency")
    say("  force SD by rate (mean over carriers and depths): " +
        ", ".join("%g Hz %.2f" % (r, mod[mod.rate == r].task_force_sd.mean()) for r in FM_RATES))
    return g


def probe_error_loss():
    """Worst-case loss of the three-probe maximum for a probe error of e Hz."""
    f_true = 1200.0 + np.linspace(-10.0, 10.0, 4001)
    out = {}
    for e in (25.0, 30.0):
        worst = 0.0
        for sgn in (1.0, -1.0):
            worst = min(worst, float(D.scalloping_db(f_true, f_true + sgn * e).min()))
        out[e] = worst
    report("probe_error_25hz_worst_db", -out[25.0], "%.1f", "worst loss of the reading for a 25 Hz probe error")
    report("probe_error_30hz_worst_db", -out[30.0], "%.1f", "worst loss of the reading for a 30 Hz probe error")
    report("frame_step_limit_hz_s", D.PROBE_STEP_HZ / (D.FRAME / D.FS), "%.0f",
           "rate of a 20 Hz change from one frame to the next")


# ---------------------------------------------------------------------------
# 3. Random pitch walks scaled to each participant
# ---------------------------------------------------------------------------
def random_walks(ol_seg):
    say("\n3. Random pitch walks (40 x 6 s per participant, own bounds)")
    rng = np.random.default_rng(RANDOM_SEED)
    per = ol_seg.groupby("Participant", sort=False).agg(pitch_sd=("PitchSdSemitones", "median"),
                                                        f0=("MedianF0Hz", "median"),
                                                        meas_sd_db=("AmpSdDb", "median")).loc[PIDS]
    n = int(round(WALK_FRAMES * D.FRAME / D.FS * WALK_RATE)) + 1
    t = np.arange(n) / WALK_RATE
    specs, meta = [], []
    for pid, r in per.iterrows():
        A = amp_for_level(pid, 50.0, r.f0)
        for k in range(WALK_SEGMENTS):
            s = np.cumsum(rng.standard_normal(n))
            s -= s.mean()
            s *= r.pitch_sd / s.std()          # population SD over the segment equals the target exactly
            specs.append(dict(track=D.interp_track(t, r.f0 * 2.0 ** (s / 12.0)), amp=A, bounds=own_bounds([pid])[0]))
            meta.append(dict(pid=pid, segment=k))
    o = run_batches(specs, WALK_FRAMES, PREROLL, ("task", "raw"), batch=48)
    rows = []
    for i, m in enumerate(meta):
        v = o["valid"][i]
        rows.append(dict(m, valid_frames=int(v.sum()), task_amp_sd_db=psd(db(o["task_amp"][i, v])),
                         raw_amp_sd_db=psd(db(o["raw_amp"][i, v])), task_force_sd=psd(o["task_force"][i, v, 0])))
    seg = pd.DataFrame(rows)
    seg.to_csv(os.path.join(OUTDIR, "random_walk_segments.csv"), index=False)
    p = seg.groupby("pid", sort=False)[["task_amp_sd_db", "raw_amp_sd_db", "task_force_sd"]].mean().join(per)
    # The open-loop amplitude was measured at the current YIN estimate, so the share uses that probe.
    p["share_var"] = (p.raw_amp_sd_db / p.meas_sd_db) ** 2
    p.to_csv(os.path.join(OUTDIR, "random_walk_participants.csv"))
    report("walk_pitch_sd_median_st", float(per.pitch_sd.median()), "%.2f",
           "median over participants of the open-loop pitch SD (semitones)")
    report("walk_amp_sd_median_db", float(p.task_amp_sd_db.median()), "%.2f", "random walks: amplitude SD after the energy window (median over participants)")
    report("walk_force_sd_median_points", float(p.task_force_sd.median()), "%.2f", "random walks: force SD after the full chain (median over participants)")
    report("walk_share_median_pct", 100.0 * float(p.share_var.median()), "%.1f",
           "random walks: percent of the measured open-loop amplitude variance (median)")
    return p


# ---------------------------------------------------------------------------
# 4. Replays of the recorded pitch tracks
# ---------------------------------------------------------------------------
def openloop_replay(ol_seg, ol_raw):
    say("\n4a. Replay of the 72 open-loop pitch tracks (constant amplitude, own bounds)")
    specs, meta = [], []
    for (pid, seg), d in ol_raw.groupby(["Participant", "Segment"], sort=False):
        d = d.sort_values("SampleIndex")
        f = d.FrequencyHz.values.astype(np.float64)
        t = d.TimeMs.values / 1000.0 + 0.5 * D.FRAME / D.FS      # frame centre; TimeMs is the frame start
        specs.append(dict(track=D.interp_track(t, f), amp=amp_for_level(pid, 50.0, float(np.median(f))),
                          bounds=own_bounds([pid])[0]))
        meta.append(dict(pid=pid, segment=int(seg), n=len(f)))
    n_max = max(m["n"] for m in meta)
    o = run_batches(specs, n_max, PREROLL, ("raw", "task", "true"), batch=36)
    lg = ol_seg.set_index(["Participant", "Segment"])
    rows = []
    for i, m in enumerate(meta):
        v = o["valid"][i, :m["n"]]
        r = dict(m, valid_frames=int(v.sum()))
        for var in ("raw", "task", "true"):
            r[var + "_amp_sd_db"] = psd(db(o[var + "_amp"][i, :m["n"]][v]))
        r["meas_sd_db"] = float(lg.loc[(m["pid"], m["segment"]), "AmpSdDb"])
        r["meas_sd_pts"] = float(lg.loc[(m["pid"], m["segment"]), "AmpSdPoints"])
        rows.append(r)
    t = pd.DataFrame(rows)
    t["R"] = BOUNDS.R.reindex(t.pid).values
    t["raw_sd_pts"] = t.raw_amp_sd_db * 100.0 / t.R
    t["share_var"] = (t.raw_amp_sd_db / t.meas_sd_db) ** 2
    t.to_csv(os.path.join(OUTDIR, "openloop_replay.csv"), index=False)
    report("openloop_segments", len(t), "%d", "open-loop segments replayed")
    report("openloop_err_sd_db", float(t.raw_amp_sd_db.median()), "%.2f", "open loop: error SD (median over segments)")
    report("openloop_err_sd_points", float(t.raw_sd_pts.median()), "%.2f", "open loop: error SD in force points (median)")
    report("openloop_share_pct", 100.0 * float(t.share_var.median()), "%.1f",
           "open loop: percent of the within-segment variance (median)")
    report("openloop_measured_sd_points", float(t.meas_sd_pts.median()), "%.1f",
           "open loop: measured SD in force points (median)")
    return t


def task_specs(best_gain):
    """One replay per intensity trial: the recorded median-f0 track at constant
    amplitude. A tick reads the newest finished frame, whose centre is on
    average 50 ms before the tick. Ticks with f0 = 0 (no whistle for >= 300 ms,
    state reset) become silent frames, so the replayed chain goes through the
    same resets and median warm-up as in the trial."""
    specs, meta = [], []
    for task in TASKS:
        for pid in PIDS:
            d = load_raw(pid, "intensity", task)
            for tr, g in d.groupby("Trial", sort=True):
                g = g.sort_values("TimeMs")
                f = g.FrequencyHz.values.astype(np.float64)
                ok = f > 0
                if ok.sum() < MIN_SAMPLES:
                    continue
                t = g.TimeMs.values / 1000.0 - 0.05
                level = float(g.Target.iloc[0]) if task == "maintenance" else 50.0
                A = amp_for_level(pid, level, float(np.median(f[ok])))
                specs.append(dict(track=D.interp_track(t[ok], f[ok]), gate=D.nearest_gate(t, ok.astype(int)), amp=A,
                                  bounds=own_bounds([pid])[0], ideal=A * best_gain))
                meta.append(dict(task=task, pid=pid, trial=int(tr),
                                 level=float(g.Target.iloc[0]) if task == "maintenance" else float(g.SpeedPercent.iloc[0])))
    return specs, meta


def tick_frame(time_ms):
    """Replay frame read by a tick at time_ms (frame centre about 50 ms earlier)."""
    return np.clip(np.round((np.asarray(time_ms, np.float64) - 75.0) / 50.0).astype(int), 0, TASK_FRAMES - 1)


def running_median_dev(x, half=10):
    s = pd.Series(x)
    return (s - s.rolling(2 * half + 1, center=True, min_periods=1).median()).values


def task_replay(best_gain, per_trial):
    say("\n4b. Replay of the recorded pitch tracks of all intensity trials")
    specs, meta = task_specs(best_gain)
    o = run_batches(specs, TASK_FRAMES, PREROLL, ("task", "true", "ideal"), batch=40)
    art = o["task_force"][:, :, 0] - o["ideal_force"][:, :, 0]
    art_true = o["true_force"][:, :, 0] - o["ideal_force"][:, :, 0]
    rows = []
    for i, m in enumerate(meta):
        sc = slice(SCORE_FRAME, TASK_FRAMES)
        a, at = art[i, sc], art_true[i, sc]
        both = o["valid"][i, sc] & (o["task_amp"][i, sc] > 0) & (o["ideal_amp"][i, sc] > 0)
        pt = per_trial[(m["pid"], "intensity", m["task"])].loc[m["trial"]]
        r = dict(m, err_sd_pts=psd(a), err_true_sd_pts=psd(at),
                 err_sd_db=psd(db(o["task_amp"][i, sc][both] / o["ideal_amp"][i, sc][both])),
                 err_maxabs_pts=float(np.abs(a - a.mean()).max()),
                 err_true_maxabs_pts=float(np.abs(at - at.mean()).max()),
                 meas_sd=float(pt.SD) if m["task"] == "maintenance" else np.nan, meas_rmse=float(pt.RMSE))
        rows.append(r)
    t = pd.DataFrame(rows)
    # Share of the measured variability: within-trial variance in maintenance, mean
    # squared error in tracking (the target moves, so the SD would include the target).
    t["share"] = np.where(t.task == "maintenance", (t.err_sd_pts / t.meas_sd) ** 2, (t.err_sd_pts / t.meas_rmse) ** 2)
    mt = t[t.task == "maintenance"]
    tk = t[t.task == "tracking"]
    t.to_csv(os.path.join(OUTDIR, "task_replay.csv"), index=False)
    report("replayed_trials", len(t), "%d", "intensity trials replayed")
    report("maint_err_sd_db", float(mt.err_sd_db.median()), "%.2f", "maintenance: error SD (median over trials)")
    report("maint_err_sd_points", float(mt.err_sd_pts.median()), "%.2f", "maintenance: error SD in force points (median)")
    report("maint_share_pct", 100.0 * float(mt.share.median()), "%.1f",
           "maintenance: percent of the within-trial variance (median)")
    report("maint_measured_sd_points", float(mt.meas_sd.median()), "%.1f", "maintenance: measured SD in force points (median)")
    report("track_err_sd_db", float(tk.err_sd_db.median()), "%.2f", "tracking: error SD (median over trials)")
    report("track_err_rms_points", float(tk.err_sd_pts.median()), "%.2f", "tracking: RMS error in force points (median)")
    report("track_share_pct", 100.0 * float(tk.share.median()), "%.1f", "tracking: percent of the mean squared error (median)")
    report("maint_trials_err_gt5", int((mt.err_maxabs_pts > 5).sum()), "%d",
           "maintenance trials with an error > 5 points in the scoring window")
    report("maint_trials_err_gt5_true_f0", int((mt.err_true_maxabs_pts > 5).sum()), "%d",
           "the same with the probe at the true frequency")
    report("maint_trials", len(mt), "%d", "maintenance trials")
    report("track_trials", len(tk), "%d", "tracking trials")

    # Largest dip and the pitch glide in which it occurred.
    j = int(t.err_maxabs_pts.values.argmax())
    sc = np.arange(SCORE_FRAME, TASK_FRAMES)
    a = art[j, sc]
    k = int(sc[np.abs(a - a.mean()).argmax()])
    ft = o["f_true"][j]
    glide = (ft[k] - ft[k - 6]) / (6 * D.FRAME / D.FS)
    report("largest_dip_points", float(t.err_maxabs_pts.max()), "%.0f", "largest error in force points (all intensity trials)")
    report("largest_dip_glide_hz_s", float(glide), "%.0f",
           "rate of pitch change over the 300 ms before the frame of the largest error")
    say("  largest dip: %s %s trial %d, frame %d, true f0 %.1f Hz, probe %.1f Hz" %
        (meta[j]["task"], meta[j]["pid"], meta[j]["trial"], k, ft[k], o["task_probe"][j, k]))

    # Ticks with a predicted dip of more than 6 dB, relative to the local level of the prediction.
    pred_db = np.full(o["task_amp"].shape, np.nan)
    ok = o["valid"] & (o["task_amp"] > 0) & (o["ideal_amp"] > 0)
    pred_db[ok] = db(o["task_amp"][ok] / o["ideal_amp"][ok])
    ticks = []
    for i, m in enumerate(meta):
        g = load_raw_cached(m["pid"], m["task"])
        g = g[g.Trial == m["trial"]].sort_values("TimeMs")
        g = g[(g.FrequencyHz > 0) & (g.Amplitude > 0)]
        p = pred_db[i, tick_frame(g.TimeMs.values)]
        has = np.isfinite(p)
        ticks.append(pd.DataFrame(dict(time_ms=g.TimeMs.values[has], dip_db=running_median_dev(p[has]))))
    tk_all = pd.concat(ticks, ignore_index=True)
    deep = tk_all[tk_all.dip_db < -6.0]
    report("deep_dip_ticks", len(deep), "%d", "ticks with a predicted dip larger than 6 dB")
    report("deep_dip_first_1500ms_pct", 100.0 * float((deep.time_ms < 1500).mean()), "%.0f",
           "percent of them in the first 1.5 s of a trial")
    return o, meta, t


_RAW_CACHE = {}


def load_raw_cached(pid, task, channel="intensity"):
    key = (pid, channel, task)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = load_raw(pid, channel, task)
    return _RAW_CACHE[key]


# ---------------------------------------------------------------------------
# 5. The prediction in the recorded open-loop amplitudes
# ---------------------------------------------------------------------------
def energy_gain_db(g_db):
    """Gain seen after the three-frame energy window, within one segment."""
    p = pd.Series(10 ** (np.asarray(g_db) / 10.0)).rolling(D.ENERGY_FRAMES, min_periods=1).mean()
    return 10.0 * np.log10(p.values)


def openloop_check(ol_seg, ol_raw):
    say("\n5. The predicted error in the recorded open-loop amplitudes")
    r = ol_raw.sort_values(["Participant", "Segment", "SampleIndex"]).reset_index(drop=True).copy()
    f = r.FrequencyHz.values.astype(np.float64)
    r["y"] = db(r.Amplitude.values)
    r["g"] = D.estimator_gain_db(f)
    grp = r.groupby(["Participant", "Segment"], sort=False)
    r["g_eff"] = np.concatenate([energy_gain_db(d.g.values) for _, d in grp])
    for c in ("y", "g_eff"):
        r[c + "_c"] = r[c] - grp[c].transform("mean")
    fit = sm.OLS(r.y_c.values, sm.add_constant(r.g_eff_c.values)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(r.Participant)[0]})
    report("openloop_slope", float(fit.params[1]), "%.2f", "open loop: slope of the measured amplitude on the predicted error")
    report("openloop_slope_se", float(fit.bse[1]), "%.2f", "its SE, clustered by participant")
    report("openloop_r2", float(fit.rsquared), "%.3f", "R^2 (within-segment variance explained)")
    # Frames within 2 Hz of a bin centre vs within 2 Hz of a bin edge (each about a fifth of the frames).
    off = np.abs(f / D.BIN_HZ - np.round(f / D.BIN_HZ))
    b = pd.cut(off, np.linspace(0.0, 0.5, 6), include_lowest=True)
    prof = r.groupby(b, observed=True).agg(frames=("y_c", "size"), measured_db=("y_c", "mean"),
                                           predicted_db=("g_eff_c", "mean"))
    prof.to_csv(os.path.join(OUTDIR, "openloop_offset_profile.csv"))
    report("openloop_centre_minus_edge_measured_db", float(prof.measured_db.iloc[0] - prof.measured_db.iloc[-1]), "%.2f",
           "mean level near a bin centre minus near a bin edge, measured")
    report("openloop_centre_minus_edge_predicted_db", float(prof.predicted_db.iloc[0] - prof.predicted_db.iloc[-1]),
           "%.2f", "the same, predicted")
    say("  frames in the two outer fifths: %d and %d of %d" % (prof.frames.iloc[0], prof.frames.iloc[-1], len(r)))

    # Open-loop comparison with the predicted gain removed from every frame.
    r["y_comp"] = r.y - r.g_eff
    seg = r.groupby(["Participant", "Segment"], sort=False).y_comp.agg(lambda v: float(np.std(v))).rename("sd_comp")
    lg = ol_seg.set_index(["Participant", "Segment"]).join(seg)
    lg["pts_comp"] = lg.sd_comp * 100.0 / lg.IntensityRangeDb
    pp = lg.groupby(level=0, sort=False)[["AmpSdPoints", "pts_comp", "PitchSdPoints"]].mean().loc[PIDS]
    res = {}
    for col in ("AmpSdPoints", "pts_comp"):
        diff = pp[col] - pp.PitchSdPoints
        res[col] = (pp[col].mean(), pp[col].mean() / pp.PitchSdPoints.mean(), diff.mean() / diff.std(ddof=1),
                    stats.ttest_rel(pp[col], pp.PitchSdPoints).pvalue)
    report("openloop_intensity_noise_points", res["AmpSdPoints"][0], "%.2f", "open-loop intensity noise (force points, mean over participants)")
    report("openloop_intensity_noise_comp_points", res["pts_comp"][0], "%.2f", "the same with the predicted gain removed")
    report("openloop_ratio", res["AmpSdPoints"][1], "%.2f", "ratio of intensity to pitch noise")
    report("openloop_ratio_comp", res["pts_comp"][1], "%.2f", "the same with the predicted gain removed")
    report("openloop_dz", res["AmpSdPoints"][2], "%.2f", "d_z of the channel difference")
    report("openloop_dz_comp", res["pts_comp"][2], "%.2f", "the same with the predicted gain removed")
    return r


# ---------------------------------------------------------------------------
# 6. Scoring (definitions of score_trials.py), vectorised over trials
# ---------------------------------------------------------------------------
class Block:
    """All trials of one channel and task, as padded (trial x tick) arrays."""

    def __init__(self, channel, task):
        self.channel, self.task = channel, task
        parts = []
        for pid in PIDS:
            d = load_raw_cached(pid, task, channel).copy()
            d["pid"] = pid
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        d["key"] = d.pid + "|" + d.Trial.astype(str)
        keys = pd.unique(d.key)
        ti = pd.Series(np.arange(len(keys)), index=keys)
        d["ti"] = ti[d.key].values
        d["pos"] = d.groupby("ti").cumcount()
        self.d = d
        n, J = len(keys), int(d.pos.max()) + 1
        self.n, self.J = n, J
        self.mask = np.zeros((n, J), bool)
        self.mask[d.ti, d.pos] = True
        first = d.groupby("ti").first()
        self.pid = first.pid.values
        self.trial = first.Trial.values
        self.level = (first.Target if task == "maintenance" else first.SpeedPercent).values.astype(np.float64)

    def grid(self, col, fill=0.0):
        a = np.full((self.n, self.J), fill, np.float64)
        a[self.d.ti, self.d.pos] = self.d[col].values
        return a


def score(block, force):
    """Per-trial TIB, Success, SD, RMSE, Bias (maintenance) or RMSE, PearsonR,
    TIB (tracking) of the force array (trials x ticks)."""
    t = block.grid("TimeMs")
    win = block.mask & (t >= WINDOW_START_MS)
    tg = block.grid("Target")
    prev_win = np.zeros_like(win)
    prev_win[:, 1:] = win[:, :-1]
    prev_t = np.zeros_like(t)
    prev_t[:, 1:] = t[:, :-1]
    dt = np.where(win, t - np.where(prev_win, prev_t, WINDOW_START_MS), 0.0)
    inb = win & (np.abs(force - tg) <= BAND)
    last_t = np.where(win, t, -np.inf).max(axis=1)
    span = np.maximum(1.0, last_t - WINDOW_START_MS)
    tib = (dt * inb).sum(axis=1) / span
    n = win.sum(axis=1)
    nn = np.maximum(n, 1)
    out = dict(TIB=tib)
    if block.task == "maintenance":
        consec = np.zeros(block.n)
        longest = np.zeros(block.n)
        for j in range(block.J):
            consec = np.where(inb[:, j], consec + dt[:, j], np.where(win[:, j], 0.0, consec))
            longest = np.maximum(longest, consec)
        mean = (force * win).sum(axis=1) / nn
        out["Success"] = (longest >= HOLD_MS).astype(np.float64)
        out["SD"] = np.sqrt((((force - mean[:, None]) ** 2) * win).sum(axis=1) / nn)
        out["RMSE"] = np.sqrt((((force - tg) ** 2) * win).sum(axis=1) / nn)
        out["Bias"] = mean - block.level
    else:
        ma = (force * win).sum(axis=1) / nn
        mt = (tg * win).sum(axis=1) / nn
        da, dtg = (force - ma[:, None]) * win, (tg - mt[:, None]) * win
        out["RMSE"] = np.sqrt((((force - tg) ** 2) * win).sum(axis=1) / nn)
        out["PearsonR"] = (da * dtg).sum(axis=1) / np.sqrt((da ** 2).sum(axis=1) * (dtg ** 2).sum(axis=1))
        out["z"] = np.arctanh(out["PearsonR"])
    res = pd.DataFrame(out)
    res.loc[n < MIN_SAMPLES, :] = np.nan
    res["pid"] = block.pid
    res["trial"] = block.trial
    res["level"] = block.level
    return res


PRIMARY = {"maintenance": ["TIB", "Success"], "tracking": ["z", "TIB"]}
SECONDARY = {"maintenance": ["SD", "RMSE"], "tracking": ["RMSE"]}


def participant_means(res, metrics):
    return res.groupby("pid", sort=False)[metrics].mean().loc[PIDS]


def channel_test(pitch_means, intensity_means, metric):
    """Pitch minus intensity over participant means: d_z and paired-t p."""
    a, b = pitch_means[metric], intensity_means[metric]
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1)), float(stats.ttest_rel(a, b).pvalue)


# ---------------------------------------------------------------------------
# 7. Effect on the scored results
# ---------------------------------------------------------------------------
def remove_replayed_error(o, meta, blocks, pitch_means):
    """First analysis: subtract the replayed error from the recorded force of every
    intensity trial and score it again (no behavioural response assumed)."""
    say("\n7a. Replayed error removed from the recorded force")
    art = o["task_force"][:, :, 0] - o["ideal_force"][:, :, 0]
    index = {(m["task"], m["pid"], m["trial"]): i for i, m in enumerate(meta)}
    out = {}
    for task in TASKS:
        blk = blocks[("intensity", task)]
        F = blk.grid("F")
        t = blk.grid("TimeMs")
        err = np.zeros_like(F)
        for r in range(blk.n):
            i = index.get((task, blk.pid[r], int(blk.trial[r])))
            if i is None:
                continue
            e = art[i]
            err[r] = e[tick_frame(t[r])] - e[SCORE_FRAME:].mean()
        base = score(blk, F)
        rem = score(blk, F - err)
        dbl = score(blk, F + err)
        out[task] = (base, rem, dbl)
    base, rem, dbl = out["maintenance"]
    d_tib = 100.0 * (rem.TIB - base.TIB)
    report("remove_tib_change_pp", float(d_tib.mean()), "%+.2f", "maintenance TIB change, error removed (mean, percentage points)")
    report("remove_tib_p5_pp", pct(d_tib, 5), "%+.1f", "5th percentile of trials")
    report("remove_tib_p95_pp", pct(d_tib, 95), "%+.1f", "95th percentile of trials")
    report("remove_success_change_pp", 100.0 * float((rem.Success - base.Success).mean()), "%+.1f",
           "maintenance success change, error removed (percentage points)")
    report("double_tib_change_pp", 100.0 * float((dbl.TIB - base.TIB).mean()), "%+.1f", "error doubled: TIB change (percentage points)")
    report("double_success_change_pp", 100.0 * float((dbl.Success - base.Success).mean()), "%+.1f",
           "error doubled: success change (percentage points)")
    for m, lab in (("TIB", "tib"), ("Success", "success")):
        dz0, p0 = channel_test(pitch_means["maintenance"], participant_means(base, [m]), m)
        dz1, p1 = channel_test(pitch_means["maintenance"], participant_means(rem, [m]), m)
        report("remove_dz_%s_recorded" % lab, dz0, "%.2f", "channel d_z, recorded force")
        report("remove_dz_%s_removed" % lab, dz1, "%.2f", "channel d_z, error removed")
        report("remove_p_%s_removed" % lab, p1, "%.1e", "its p")
    base, rem, _ = out["tracking"]
    report("remove_track_rmse_change", float((rem.RMSE - base.RMSE).mean()), "%+.2f",
           "tracking RMSE change, error removed (mean, force points)")


def tick_tables(blocks):
    """Per-tick predicted gain at the recorded median f0, duplicate ticks (the same
    frame read twice) and the gain after the energy window."""
    for blk in blocks.values():
        if blk.channel != "intensity":
            continue
        d = blk.d
        v = d.Amplitude.values > 0
        f = d.FrequencyHz.values.astype(np.float64)
        s_db = np.zeros(len(d))
        b_db = np.zeros(len(d))
        s_db[v] = D.scalloping_db(f[v])
        b_db[v] = D.bandpass_response_db(f[v])
        d["S_dB"], d["B_dB"] = s_db, b_db
        d["g_dB"] = s_db + b_db
        same = np.r_[False, (d.Amplitude.values[1:] == d.Amplitude.values[:-1]) &
                     (d.FrequencyHz.values[1:] == d.FrequencyHz.values[:-1]) &
                     (d.ti.values[1:] == d.ti.values[:-1])]
        d["dup"] = same & v
        # energy mean of the gain over this tick and the earlier distinct ticks of the
        # same whistle within 100 ms (frames k-2..k lie within about 100 ms of frame k)
        ge = np.full(len(d), np.nan)
        t, g, ti, dup, amp = d.TimeMs.values, d.g_dB.values, d.ti.values, d.dup.values, d.Amplitude.values
        for i in np.flatnonzero(v):
            acc = [10 ** (g[i] / 10.0)]
            j = i - 1
            while j >= 0 and ti[j] == ti[i] and t[i] - t[j] <= 100 and amp[j] > 0:
                if not dup[j]:
                    acc.append(10 ** (g[j] / 10.0))
                j -= 1
            ge[i] = 10.0 * np.log10(np.mean(acc))
        d["geff_dB"] = ge


def trial_mean(blk, col):
    d = blk.d
    w = (d.Amplitude > 0) & ~d.dup & (d.TimeMs >= WINDOW_START_MS)
    m = d[w].groupby("ti")[col].mean()
    return d.ti.map(m).fillna(0.0).values


def participant_tick_mean(blocks, col):
    vals = {}
    for pid in PIDS:
        x = [blocks[("intensity", t)].d for t in TASKS]
        x = np.concatenate([dd[col].values[(dd.pid == pid).values & (dd.Amplitude.values > 0)] for dd in x])
        vals[pid] = float(x.mean())
    return vals


def port(blk, comp_db, db_min, db_max, dt_mode):
    """Force mapping, median of three and One Euro filter applied tick by tick to the
    recorded amplitude plus comp_db (dB). A tick with amplitude 0 is the reset
    state; a duplicate tick holds the previous output; a trial that starts inside
    a whistle starts from its own first value. dt_mode 'tick' steps the One Euro
    filter with the real interval between ticks, 'frame' with the 50 ms of the
    application; the recorded ticks are a sample of the frames, so both are
    approximations."""
    A = blk.grid("Amplitude")
    T = blk.grid("TimeMs")
    DUP = blk.grid("dup").astype(bool)
    C = np.zeros_like(A)
    C[blk.d.ti, blk.d.pos] = comp_db
    n = blk.n
    started = np.zeros(n, bool)
    tprev = np.zeros(n)
    mb = np.zeros(n)
    mc = np.zeros(n)
    xp = np.zeros(n)
    dxp = np.zeros(n)
    prev = np.zeros(n)
    out = np.zeros_like(A)
    tau_d = 1.0 / (2.0 * np.pi * D.ONE_EURO["d_cutoff"])
    for j in range(blk.J):
        m = blk.mask[:, j]
        a = A[:, j]
        is_reset = m & (a <= 0)
        is_dup = m & ~is_reset & DUP[:, j] & started
        is_new = m & ~is_reset & ~is_dup
        fr = np.clip(100.0 * (db(np.maximum(a, 1e-12)) + C[:, j] - db_min) / (db_max - db_min), 0.0, 100.0)
        first = is_new & ~started
        step = is_new & started
        med = np.median(np.stack([mb, mc, fr]), axis=0)
        dt = np.full(n, D.FILTER_DT) if dt_mode == "frame" else np.maximum((T[:, j] - tprev) / 1000.0, 1e-3)
        dx = (med - xp) / dt
        a_d = 1.0 / (1.0 + tau_d / dt)
        dxh = a_d * dx + (1.0 - a_d) * dxp
        cut = D.ONE_EURO["min_cutoff"] + D.ONE_EURO["beta"] * np.abs(dxh)
        al = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * cut)) / dt)
        xh = al * med + (1.0 - al) * xp
        new_mb = np.where(step, mc, np.where(first, fr, 0.0))
        new_mc = np.where(step, fr, np.where(first, fr, 0.0))
        mb = np.where(is_reset | is_new, new_mb, mb)
        mc = np.where(is_reset | is_new, new_mc, mc)
        xp = np.where(is_reset, 0.0, np.where(first, fr, np.where(step, xh, xp)))
        dxp = np.where(is_reset | first, 0.0, np.where(step, dxh, dxp))
        o = np.where(is_reset, 0.0, np.where(is_dup, prev, np.where(first, fr, np.where(step, xh, prev))))
        started |= is_reset | is_new
        tprev = np.where(m, T[:, j], tprev)
        prev = np.where(m, o, prev)
        out[:, j] = o
    return out


def bound_db(blk, em=None, ex=None):
    b = BOUNDS.loc[blk.pid]
    lo = db(b.MinAmplitude.values)
    hi = db(b.MaxAmplitude.values)
    if em is not None:
        lo = lo - pd.Series(em).loc[blk.pid].values
        hi = hi - pd.Series(ex).loc[blk.pid].values
    return lo, hi


def gain_compensation(blocks, pitch_means):
    """Second analysis: divide every recorded amplitude by the predicted gain at its
    recorded frequency and map it to force again. The variants differ in the
    reference level of the gain (trial mean 'W' or participant mean 'A'), in the
    gain used (per tick 'g', after the energy window 'geff', scalloping only 'S')
    and in the One Euro time step; each is compared with the uncompensated port."""
    say("\n7b. Recorded amplitudes divided by the predicted gain (port of the force chain)")
    gbar = participant_tick_mean(blocks, "g_dB")
    sbar = participant_tick_mean(blocks, "S_dB")
    rows = []
    for task in TASKS:
        blk = blocks[("intensity", task)]
        d = blk.d
        pbar_g = d.pid.map(gbar).values
        pbar_s = d.pid.map(sbar).values
        comp = {
            "none": np.zeros(len(d)),
            "W_g": -(d.g_dB.values - trial_mean(blk, "g_dB")),
            "W_geff": -(np.nan_to_num(d.geff_dB.values) - trial_mean(blk, "geff_dB")),
            "W_S": -(d.S_dB.values - trial_mean(blk, "S_dB")),
            "A_g": -(d.g_dB.values - pbar_g),
            "A_S": -(d.S_dB.values - pbar_s),
        }
        lo, hi = bound_db(blk)
        metrics = PRIMARY[task] + SECONDARY[task]
        for dt_mode in ("tick", "frame"):
            for name, c in comp.items():
                res = score(blk, port(blk, c, lo, hi, dt_mode))
                im = participant_means(res, metrics)
                for mtr in metrics:
                    dz, p = channel_test(pitch_means[task], im, mtr)
                    rows.append(dict(task=task, dt=dt_mode, variant=name, metric=mtr, intensity_mean=float(im[mtr].mean()),
                                     dz=dz, p=p))
    t = pd.DataFrame(rows)
    base = t[t.variant == "none"].set_index(["task", "dt", "metric"])
    v = t[t.variant != "none"].copy()
    v["dz_change"] = v.dz.values - base.loc[list(zip(v.task, v.dt, v.metric)), "dz"].values
    v["mean_change"] = v.intensity_mean.values - base.loc[list(zip(v.task, v.dt, v.metric)), "intensity_mean"].values
    pd.concat([t[t.variant == "none"], v]).to_csv(os.path.join(OUTDIR, "gain_compensation.csv"), index=False)
    prim = v[[m in PRIMARY[k] for k, m in zip(v.task, v.metric)]]
    sec = v[[m in SECONDARY[k] for k, m in zip(v.task, v.metric)]]
    suc = v[(v.task == "maintenance") & (v.metric == "Success")]
    report("comp_success_change_min_pp", 100.0 * float(suc.mean_change.min()), "%+.1f",
           "intensity success change, lowest over the variants (percentage points)")
    report("comp_success_change_max_pp", 100.0 * float(suc.mean_change.max()), "%+.1f",
           "intensity success change, highest over the variants (percentage points)")
    report("comp_dz_change_primary_max", float(prim.dz_change.abs().max()), "%.2f",
           "largest d_z change, four primary measures")
    report("comp_dz_change_sd_rmse_max", float(sec.dz_change.abs().max()), "%.3f",
           "largest d_z change, SD and RMSE")
    report("comp_p_max", float(pd.concat([prim, sec]).p.max()), "%.1e", "largest p of these comparisons")
    return t


def calibration_model(blocks, ol_raw, pitch_means):
    say("\n8. Calibration bounds")
    # Scalloping part of a bound: open-loop segments stand in for calibration
    # recordings (the pitch during calibration is not in the package). A bound is
    # the linear mean over three recordings, each the linear mean over its frames.
    r = ol_raw.copy()
    r["S_lin"] = 10 ** (D.scalloping_db(r.FrequencyHz.values.astype(np.float64)) / 20.0)
    seg = r.groupby(["Participant", "Segment"], sort=False).S_lin.mean()
    s3 = seg.groupby(level=0, sort=False).mean().loc[PIDS]
    s3 = pd.Series(db(s3.values), index=s3.index)
    task_s = pd.Series(participant_tick_mean(blocks, "S_dB"))
    eps3 = s3 - task_s.loc[PIDS]
    report("steady_pitch_bound_error_db", -float(D.scalloping_db(np.array([1210.0]))[0]), "%.2f",
           "steady pitch: largest bound error")
    report("steady_pitch_bound_error_points", -float(D.scalloping_db(np.array([1210.0]))[0]) * PTS_PER_DB, "%.1f",
           "the same in force points (median range)")
    report("bound_scalloping_sd_db", float(eps3.std(ddof=1)), "%.2f",
           "bound over three recordings, scalloping part, SD over participants")
    report("bound_scalloping_sd_points", float(eps3.std(ddof=1)) * PTS_PER_DB, "%.1f", "the same in force points (median range)")

    # Band-pass part: louder whistles tended to be higher, so the pitch at the two
    # bounds is extrapolated from the force-pitch relation of the task ticks.
    rows = []
    for pid in PIDS:
        d = pd.concat([blocks[("intensity", t)].d for t in TASKS])
        d = d[(d.pid == pid) & (d.Amplitude > 0) & ~d.dup]
        semi = 12.0 * np.log2(d.FrequencyHz / d.FrequencyHz.median())
        b = np.polyfit(d.F, semi, 1)
        fmed = d.FrequencyHz.median()
        f_lo, f_hi = fmed * 2 ** (np.polyval(b, 0.0) / 12.0), fmed * 2 ** (np.polyval(b, 100.0) / 12.0)
        bmean = d.B_dB.mean()
        rows.append(dict(pid=pid, semitones_per_100_points=100.0 * b[0], f_at_0=f_lo, f_at_100=f_hi,
                         eps_min=float(D.bandpass_response_db(f_lo)[0] - bmean),
                         eps_max=float(D.bandpass_response_db(f_hi)[0] - bmean)))
    pe = pd.DataFrame(rows).set_index("pid")
    pe["eps3"] = eps3
    pe["range_widening_db"] = pe.eps_max - pe.eps_min
    pe["range_widening_pct"] = 100.0 * pe.range_widening_db / BOUNDS.IntensityRangeDb.loc[PIDS]
    pe.to_csv(os.path.join(OUTDIR, "calibration_bounds.csv"))
    report("bandpass_range_widening_db", float(pe.range_widening_db.median()), "%.2f",
           "band-pass slope: widening of the calibrated range (median)")
    report("bandpass_range_widening_pct", float(pe.range_widening_pct.median()), "%.1f", "the same in percent of the range (median)")

    # Monte Carlo: both bound errors added to the gain compensation (participant-mean
    # reference, per-tick gain, real tick interval). A participant's bounds are shared
    # by the two tasks, so each draw applies the same errors to both.
    rng = np.random.default_rng(RANDOM_SEED + 1)
    pool = (eps3 - eps3.mean()).values
    gbar = participant_tick_mean(blocks, "g_dB")
    rows = []
    for k in range(MC_DRAWS):
        em = {p: rng.choice(pool) + pe.loc[p, "eps_min"] for p in PIDS}
        ex = {p: rng.choice(pool) + pe.loc[p, "eps_max"] for p in PIDS}
        for task in TASKS:
            blk = blocks[("intensity", task)]
            comp = -(blk.d.g_dB.values - blk.d.pid.map(gbar).values)
            lo, hi = bound_db(blk, em, ex)
            res = score(blk, port(blk, comp, lo, hi, "tick"))
            metrics = PRIMARY[task] + SECONDARY[task]
            im = participant_means(res, metrics)
            for mtr in metrics:
                dz, p = channel_test(pitch_means[task], im, mtr)
                rows.append(dict(draw=k, task=task, metric=mtr, dz=dz, p=p))
    mc = pd.DataFrame(rows)
    mc.to_csv(os.path.join(OUTDIR, "calibration_montecarlo.csv"), index=False)
    report("mc_draws", mc.draw.nunique(), "%d", "random draws")
    report("mc_p_max", float(mc.p.max()), "%.1e", "largest p over draws and comparisons")


# ---------------------------------------------------------------------------
def main(argv):
    here_out = os.path.join(HERE, "frequency_dependence_output")
    configure(argv[1] if len(argv) > 1 else os.path.dirname(HERE), argv[2] if len(argv) > 2 else here_out)
    say("Frequency dependence of the intensity estimate")
    say("participants: %d; median range %.2f dB (%.3f points per dB); sweep amplitude %.6f" %
        (len(BOUNDS), R_MED, PTS_PER_DB, A_SWEEP))
    ol_seg = pd.read_csv(os.path.join(PT, OPENLOOP_FILE))
    ol_raw = pd.read_csv(os.path.join(PT, OPENLOOP_FRAMES_FILE))

    say("\n0. Scoring check and pitch-channel scores")
    blocks = {(c, t): Block(c, t) for c in ("pitch", "intensity") for t in TASKS}
    per_trial, pitch_means, worst = {}, {}, 0.0
    for (c, t), blk in blocks.items():
        res = score(blk, blk.grid("F"))
        for pid in PIDS:
            ref = pd.read_csv(os.path.join(PT, "%s_%s_%s_data.csv" % (pid, c, t))).set_index("Trial")
            per_trial[(pid, c, t)] = ref
            mine = res[res.pid == pid].set_index("trial")
            for col in (["TIB", "Success", "SD", "RMSE"] if t == "maintenance" else ["TIB", "RMSE", "PearsonR"]):
                worst = max(worst, float(np.nanmax(np.abs(mine[col].values - ref.loc[mine.index, col].values))))
        if c == "pitch":
            pitch_means[t] = participant_means(res, PRIMARY[t] + SECONDARY[t])
    say("  recorded force scored again vs the per-trial files of the package: max |diff| %.1e" % worst)

    best, f_best = sweeps()
    linearity()
    fm_tests()
    probe_error_loss()
    random_walks(ol_seg)
    openloop_replay(ol_seg, ol_raw)
    o, meta, _ = task_replay(best / A_SWEEP, per_trial)
    openloop_check(ol_seg, ol_raw)
    tick_tables(blocks)
    remove_replayed_error(o, meta, blocks, pitch_means)
    gain_compensation(blocks, pitch_means)
    calibration_model(blocks, ol_raw, pitch_means)

    pd.DataFrame(NUMBERS).to_csv(os.path.join(OUTDIR, "numbers.csv"), index=False)
    with io.open(os.path.join(OUTDIR, "results.txt"), "w", encoding="ascii", newline="\n") as f:
        f.write(TEXT.getvalue())


T0 = time.time()
if __name__ == "__main__":
    main(sys.argv)
