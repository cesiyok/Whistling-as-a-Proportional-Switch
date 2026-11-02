"""Subjective measures and calibrated range (Sections V-A, V-D, V-E, V-G, Appendix A-H and the recency rows of Table 13 of the paper).

Reproduces the numbers the paper reports for
  * the calibration: calibrated pitch span and intensity range, force-space
    sensitivity, accepted and rejected calibration attempts, frames per
    calibration recording, and the acceptance rules of Appendix A-H;
  * the subjective measures: NASA-TLX (raw score and the six subscales), SUS
    and the forced-choice preference (Table 5), the 2 x 2 ANOVA on raw TLX,
    and the check of the subjective measures against the block sequence
    (Section V-D, recency);
  * the between-participant analysis of calibrated intensity range and of
    open-loop production noise (Section V-G).

Reads only the public data package:
    <DATA_DIR>/participant tests/subjective-measures.xlsx          opened read-only
    <DATA_DIR>/participant tests/calibration-log_data.csv
    <DATA_DIR>/participant tests/openloop-log_data.csv
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv   (N = 1..24)
    <DATA_DIR>/participant tests/per-participant-summary_data.csv  self-check
and imports, from <DATA_DIR>/scripts, score_trials.py (path helper only).

Usage:
    python subjective_and_range.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/, next to score_trials.py). Nothing is written; the
results go to standard output, and every run prints the same text. The run
takes a few seconds.

Definitions
  * Participants P01..P24 of the workbook are pa1..pa24 of the CSV files.
  * NASA-TLX: the Performance subscale is administered as 0 = unsuccessful,
    100 = very successful and enters the raw score as load, 100 - x. Raw TLX
    is the unweighted mean of the six load terms. A participant's value for a
    channel is the mean of that channel's two blocks (maintenance, tracking).
  * SUS: odd items contribute x - 1, even items 5 - x; the sum times 2.5.
  * Wilcoxon signed-rank test, pitch vs intensity over the 24 participants.
    Zero differences are dropped (Wilcoxon's method); n is the number of
    non-zero differences and W the smaller of the two signed-rank sums, tied
    |differences| taking their mid-rank. p is exact: the two-sided
    probability of the signed-rank sum over all 2^n sign patterns of the n
    non-zero differences, given their ranks. Z is the normal score with the
    variance corrected for tied ranks (t = size of each group of ties),
        Z = (W - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24 - sum(t^3 - t)/48),
    so Z <= 0, and r = |Z| / sqrt(N) with N = 24 (all participants).
  * Channel x Task ANOVA on raw TLX (2 x 2, within participants): every
    effect has one degree of freedom, so F(1, 23) is the square of the
    one-sample t of its contrast over participants, and
    partial eta^2 = F / (F + 23).
  * Forced choice: exact two-sided binomial test against 0.5, Clopper-Pearson
    95% interval for the proportion choosing pitch.
  * Block sequences (Appendix A, Counterbalancing and Block Order):
    A = P-M, I-M, P-T, I-T and B = I-T, P-T, I-M, P-M. As documented in
    COLUMNS.md, participants 1-12 followed A
    and participants 13-24 followed B (SEQUENCE_A and SEQUENCE_B below).
    The SUS of a channel is taken after its last block, which is a tracking
    block in A and a maintenance block in B for both channels; the last
    block of the session is intensity in A and pitch in B. Groups are
    compared with Student's two-sample t (pooled variance, 22 df).
  * Calibrated bounds: the bounds in effect after the participant's last row
    of the calibration log (each row carries the bounds after that attempt).
    Pitch span R = MaxMidi - MinMidi in semitones and intensity range
    R = 20 log10(MaxAmplitude / MinAmplitude) in dB, both as logged.
    Force-space sensitivity s = 100 / R, in points per semitone or per dB.
    The group value is s at the mean range, 100 / mean(R); the mean of the
    participants' own values of s is printed beside it.
  * Participant performance: the mean of the trial values in the
    participant's per-trial file of that block (empty cells skipped). The
    per-trial files are used rather than the summary file because the summary
    file rounds the means to four decimals. TIB and Success are in percent.
    Channel difference = pitch - intensity. Tracking r is the plain mean of
    the trial correlations; the association with range is also given for r
    averaged in Fisher z (z = arctanh r).
  * Open-loop amplitude noise: the mean of AmpSdPoints over the participant's
    three open-loop segments (force points).
  * Associations: Pearson r over the 24 participants with its two-sided p;
    slopes are ordinary least-squares slopes of the outcome on the predictor,
    with their standard errors.
"""
import math
import os
import re
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1:2] in (["-h"], ["--help"]):
    print(__doc__)
    sys.exit(0)
if len(sys.argv) > 2:
    sys.exit("usage: python subjective_and_range.py [DATA_DIR]")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DATA, "scripts"))
import score_trials as ST  # noqa: E402

# io_path adds the Windows extended-length prefix, so that a deeply nested
# package folder (paths over 260 characters) can still be read.
PT = ST.io_path(os.path.join(DATA, "participant tests"))
XLSX = os.path.join(PT, "subjective-measures.xlsx")

N = 24
PIDS = ["pa%d" % i for i in range(1, N + 1)]
CODES = ["P%02d" % i for i in range(1, N + 1)]    # workbook codes, same numbering
CHANNELS = ("pitch", "intensity")
TASKS = ("maintenance", "tracking")
N_TRIALS = {"maintenance": 20, "tracking": 15}
SEQUENCE_A = CODES[:12]      # P-M, I-M, P-T, I-T: last block intensity
SEQUENCE_B = CODES[12:]      # I-T, P-T, I-M, P-M: last block pitch
SUBSCALES = (("MD", "Mental demand"), ("PD", "Physical demand"), ("TD", "Temporal demand"),
             ("PE_load", "Performance (as load)"), ("EF", "Effort"), ("FR", "Frustration"))
PITCH_MIN_SPAN = 2.0         # semitones: acceptance rule for a pitch bound
AMP_MIN_RATIO = 1.2          # acceptance rule A_max >= 1.2 A_min
MIN_FRAMES = 60              # 3 s of sustained production at 50 ms per DSP frame
NOMINAL_FRAMES = 120         # a 6 s recording at 50 ms per DSP frame


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fp(p):
    """p value: three decimals, or two significant digits below .001."""
    return "%.1e" % p if p < 1e-3 else "%.3f" % p


def msd(x, w=6):
    return "%*.2f +- %5.2f" % (w, np.mean(x), np.std(x, ddof=1))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def sheet_rows(wb, name, columns):
    """Participant rows of one workbook sheet as a DataFrame.

    The header row is the row whose first cell is 'Participant'. A column is
    found by the first line of its header cell (several header cells carry
    an explanation on further lines). Only rows whose first cell is a code
    P01..P24 are kept, which leaves out title, blank and count rows."""
    rows = list(wb[name].iter_rows(values_only=True))
    h = next(i for i, r in enumerate(rows) if r and r[0] == "Participant")
    first_line = [str(c).split("\n")[0].strip() if c is not None else "" for c in rows[h]]
    idx = {}
    for key, head in columns:
        hits = [j for j, c in enumerate(first_line) if c == head]
        assert len(hits) == 1, (name, head, hits)
        idx[key] = hits[0]
    recs = []
    for r in rows[h + 1:]:
        if r and isinstance(r[0], str) and re.fullmatch(r"P\d\d", r[0].strip()):
            recs.append({k: (r[j].strip() if isinstance(r[j], str) else r[j]) for k, j in idx.items()})
    return pd.DataFrame(recs), rows


def load_workbook():
    """TLX, SUS and preference tables, and the workbook's own summary cells.

    read_only opens the file for reading only; the workbook is never saved."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    try:
        tlx, _ = sheet_rows(wb, "NASA-TLX raw", [
            ("P", "Participant"), ("Block", "Block"), ("MD", "MD"), ("PD", "PD"), ("TD", "TD"),
            ("PE_rep", "PE as reported"), ("EF", "EF"), ("FR", "FR"),
            ("PE_book", "PE as load"), ("Raw_book", "Raw TLX")])
        sus, _ = sheet_rows(wb, "SUS", [("P", "Participant"), ("Channel", "Channel")]
                            + [("Q%d" % i, "Q%d" % i) for i in range(1, 11)]
                            + [("Score_book", "SUS score")])
        pref, pref_all = sheet_rows(wb, "Preference", [("P", "Participant"),
                                                        ("Choice", "Preferred technique")])
        summary = list(wb["Summary"].iter_rows(values_only=True))
    finally:
        wb.close()
    counts = {str(r[0]).strip(): r[1] for r in pref_all
              if r and isinstance(r[0], str) and r[0].strip() in ("Pitch", "Intensity")}
    book = {str(r[0]).strip(): r[1] for r in summary if r and isinstance(r[0], str)
            and isinstance(r[1], (int, float))}
    return tlx, sus, pref, counts, book


def read_csv(name):
    return pd.read_csv(os.path.join(PT, name), float_precision="round_trip")


def participant_means():
    """{(channel, task, column): array over PIDS} of the per-trial means, and the
    number of trials per block."""
    cols = {"maintenance": ("TIB", "Success", "SD"), "tracking": ("TIB", "PearsonR")}
    out, n_trials = {}, {}
    for ch in CHANNELS:
        for task in TASKS:
            vals = {c: [] for c in cols[task]}
            zr = []
            for pid in PIDS:
                d = read_csv("%s_%s_%s_data.csv" % (pid, ch, task))
                n_trials[(pid, ch, task)] = len(d)
                for c in cols[task]:
                    vals[c].append(d[c].dropna().mean())
                if task == "tracking":
                    zr.append(np.arctanh(d["PearsonR"].dropna().to_numpy()).mean())
            for c in cols[task]:
                out[(ch, task, c)] = np.array(vals[c])
            if task == "tracking":
                out[(ch, task, "zr")] = np.array(zr)
    return out, n_trials


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilcoxon(a, b):
    """Signed-rank test as defined in the module docstring."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    n = len(nz)
    rk = stats.rankdata(np.abs(nz))
    w = min(rk[nz > 0].sum(), rk[nz < 0].sum())
    mu = n * (n + 1) / 4.0
    _, tie = np.unique(rk, return_counts=True)
    var_t = n * (n + 1) * (2 * n + 1) / 24.0 - (tie ** 3 - tie).sum() / 48.0
    z = (w - mu) / math.sqrt(var_t)
    # exact null distribution of the signed-rank sum: doubled ranks are integers
    r2 = np.rint(2 * rk).astype(int)
    dist = np.zeros(r2.sum() + 1)
    dist[0] = 1.0
    for k in r2:
        nxt = dist.copy()
        nxt[k:] += dist[:dist.size - k]
        dist = nxt
    dist /= dist.sum()
    s2 = np.arange(dist.size)
    p = min(1.0, dist[np.abs(s2 - 2 * mu) >= abs(2 * w - 2 * mu) - 1e-9].sum())
    return dict(n=n, W=w, Z=z, p=p, r=abs(z) / math.sqrt(len(d)))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    t = stats.ttest_1samp(d, 0.0)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h, t.statistic, t.pvalue, d.mean() / d.std(ddof=1)


def two_groups(a, b):
    t = stats.ttest_ind(a, b)             # Student, pooled variance
    return np.mean(a), np.mean(b), t.statistic, t.pvalue, len(a) + len(b) - 2


def assoc(x, y):
    r, p = stats.pearsonr(x, y)
    lr = stats.linregress(x, y)
    return r, p, lr.slope, lr.stderr, lr.intercept


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_calibration():
    print("1. Calibration (calibration-log_data.csv)")
    c = read_csv("calibration-log_data.csv")
    types = ("MinPitch", "MaxPitch", "MinIntensity", "MaxIntensity")
    print("   %d calibration rows (attempts), %d accepted, %d rejected"
          % (len(c), int(c.Accepted.sum()), int((c.Accepted == 0).sum())))
    for t in types:
        s = c[c.Type == t]
        print("     %-13s %2d rows: %2d accepted, %d rejected"
              % (t, len(s), int(s.Accepted.sum()), int((s.Accepted == 0).sum())))
    per = c.groupby(["Participant", "Type"], sort=False).size()
    extra = per[per > 1]
    print("   participants with more than one row for a bound: %s"
          % (", ".join("%s %s (%d rows)" % (p, t, k) for (p, t), k in extra.items()) or "none"))
    print("   capture devices: %d distinct (%s); sampling-rate multiplier mu: %s"
          % (c.Device.nunique(), "; ".join(sorted(c.Device.unique())),
             ", ".join("%.4f" % v for v in sorted(c.SampleRateMultiplier.unique()))))

    fr = c[["Frames1", "Frames2", "Frames3"]].to_numpy(float).ravel()
    print("   frames per calibration recording (3 per row, %d recordings): M = %.1f, SD = %.1f, "
          "min %d, max %d" % (len(fr), fr.mean(), fr.std(ddof=1), fr.min(), fr.max()))
    print("     recordings below %d frames (3 s): %d; above %d frames (6 s at 50 ms): %d; "
          "missing frame counts: %d"
          % (MIN_FRAMES, int((fr < MIN_FRAMES).sum()), NOMINAL_FRAMES, int((fr > NOMINAL_FRAMES).sum()),
             int(np.isnan(fr).sum())))
    mv = c[["Trial1", "Trial2", "Trial3"]].mean(axis=1)
    print("   bound = mean of the three recordings: max |MeasuredValue - mean(Trial1..3)| = %.1e"
          % (mv - c.MeasuredValue).abs().max())
    sp_all = c.PitchSpanSemitones.dropna()
    rg_all = c.IntensityRangeDb.dropna()
    print("   smallest pitch span in any row %.3f semitones (rule: >= %.0f); smallest intensity range "
          "in any row %.2f dB = ratio %.2f (rule: ratio >= %.1f, i.e. >= %.2f dB)"
          % (sp_all.min(), PITCH_MIN_SPAN, rg_all.min(), 10 ** (rg_all.min() / 20), AMP_MIN_RATIO,
             20 * math.log10(AMP_MIN_RATIO)))

    last = c.groupby("Participant", sort=False).tail(1).set_index("Participant").loc[PIDS]
    span = last.PitchSpanSemitones.to_numpy(float)
    rng = last.IntensityRangeDb.to_numpy(float)
    d_span = np.abs((last.MaxMidi - last.MinMidi).to_numpy(float) - span).max()
    d_rng = np.abs(20 * np.log10(last.MaxAmplitude.to_numpy(float) / last.MinAmplitude.to_numpy(float))
                   - rng).max()
    print("   final bounds (after each participant's last row), N = %d:" % len(PIDS))
    print("     pitch span      %s semitones  (range %.2f-%.2f)" % (msd(span), span.min(), span.max()))
    print("     intensity range %s dB         (range %.2f-%.2f)" % (msd(rng), rng.min(), rng.max()))
    print("     check against the logged bounds: max |span - (MaxMidi - MinMidi)| = %.3f; "
          "max |range - 20 log10(Max/Min)| = %.4f dB (rounding of the logged bounds)" % (d_span, d_rng))
    print("     narrowest intensity range %s (%.2f dB), widest %s (%.2f dB)"
          % (PIDS[int(rng.argmin())], rng.min(), PIDS[int(rng.argmax())], rng.max()))
    for lab, R, unit in (("pitch", span, "semitone"), ("intensity", rng, "dB")):
        s = 100.0 / R
        print("     sensitivity s, %-9s: 100/mean(R) = %.2f points per %s; mean of 100/R = %.2f; "
              "participants %.2f-%.2f" % (lab, 100.0 / R.mean(), unit, s.mean(), s.min(), s.max()))
    return dict(zip(PIDS, rng))


def section_subjective(tlx, sus, pref, counts, book):
    print("2. Subjective measures (subjective-measures.xlsx), N = %d" % N)
    # ---- NASA-TLX
    t = tlx.copy()
    for k in ("MD", "PD", "TD", "PE_rep", "EF", "FR", "PE_book", "Raw_book"):
        t[k] = t[k].astype(float)
    t["PE_load"] = 100.0 - t["PE_rep"]
    t["Raw"] = t[[k for k, _ in SUBSCALES]].mean(axis=1)
    t["Channel"] = t.Block.str.split("-").str[1].str.lower()
    t["Task"] = t.Block.str.split("-").str[0]
    assert len(t) == 4 * N and set(t.P) == set(CODES)
    print("   workbook checks: %d TLX rows; 'PE as load' = 100 - reported: %s; 'Raw TLX' = mean of "
          "the six load terms: max diff %.1e" % (len(t), bool((t.PE_book == t.PE_load).all()),
                                                  (t.Raw_book - t.Raw).abs().max()))
    cell = t.groupby(["P", "Channel"])[["Raw", "PE_rep"] + [k for k, _ in SUBSCALES]].mean()
    cell_task = t.set_index(["P", "Channel", "Task"])["Raw"]

    def chan(key, ch):
        return cell.xs(ch, level="Channel").loc[CODES, key].to_numpy()

    print()
    print("   Table 5: M +- SD by channel (TLX: mean of the channel's two blocks); Wilcoxon signed-rank")
    print("   %-24s %15s %15s %7s %9s %5s %4s  %s"
          % ("measure", "pitch", "intensity", "Z", "p exact", "r", "n", "pitch lower load"))

    def line(label, a, b, lower):
        w = wilcoxon(a, b)
        print("   %-24s %s %s %7.2f %9s %5.2f %4d  %s"
              % (label, msd(a), msd(b), w["Z"], fp(w["p"]), w["r"], w["n"], lower))

    for key, label in (("Raw", "Raw TLX (0-100)"),) + SUBSCALES:
        a, b = chan(key, "pitch"), chan(key, "intensity")
        line(label if key == "Raw" else "  " + label, a, b, "%d/%d" % (int((a < b).sum()), N))
    a, b = chan("PE_rep", "pitch"), chan("PE_rep", "intensity")
    print("   %-24s %s %s   (administered direction, higher = more successful)"
          % ("  Performance (reported)", msd(a), msd(b)))

    # ---- SUS
    s = sus.copy()
    q = s[["Q%d" % i for i in range(1, 11)]].to_numpy(float)
    s["SUS"] = 2.5 * ((q[:, 0::2] - 1).sum(axis=1) + (5 - q[:, 1::2]).sum(axis=1))
    s["Channel"] = s.Channel.str.lower()
    sus_ch = {ch: s[s.Channel == ch].set_index("P").loc[CODES, "SUS"].to_numpy() for ch in CHANNELS}
    a, b = sus_ch["pitch"], sus_ch["intensity"]
    line("SUS (0-100)", a, b, "pitch higher %d/%d" % (int((a > b).sum()), N))

    # ---- preference
    choice = pref.set_index("P").loc[CODES, "Choice"].str.lower()
    k = int((choice == "pitch").sum())
    bt = stats.binomtest(k, N, 0.5, alternative="two-sided")
    ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    print("   %-24s pitch %d/%d, intensity %d/%d; exact binomial p = %s; Clopper-Pearson 95%% CI "
          "[%.3f, %.3f]" % ("Preferred channel", k, N, N - k, N, fp(bt.pvalue), ci.low, ci.high))
    print("   workbook checks: %d SUS rows, 'SUS score' = item scoring: max diff %.1e; preference counts "
          "sheet pitch %s, intensity %s"
          % (len(s), np.abs(s.Score_book.astype(float) - s.SUS).max(), counts.get("Pitch"),
             counts.get("Intensity")))
    bm = {blk: t[t.Block == blk].Raw.mean() for blk in ("Maint-Pitch", "Maint-Intensity",
                                                         "Track-Pitch", "Track-Intensity")}
    diffs = [abs(book[blk] - v) for blk, v in bm.items() if blk in book]
    diffs += [abs(book[ch.capitalize()] - sus_ch[ch].mean()) for ch in CHANNELS if ch.capitalize() in book]
    print("   workbook 'Summary' sheet: %d group means compared, max diff %.1e" % (len(diffs), max(diffs)))

    # ---- subscale gaps, ceilings
    print()
    gaps = sorted(((chan(k, "intensity").mean() - chan(k, "pitch").mean(), lab, k) for k, lab in SUBSCALES),
                  reverse=True)
    print("   subscale gaps (intensity - pitch), largest first:")
    for g, lab, k in gaps:
        print("     %-22s %5.2f  (pitch %.2f vs intensity %.2f)"
              % (lab, g, chan(k, "pitch").mean(), chan(k, "intensity").mean()))

    # ---- 2 x 2 ANOVA
    y = {(ch, tk): cell_task.xs((ch, tk), level=("Channel", "Task")).loc[CODES].to_numpy()
         for ch in CHANNELS for tk in ("Maint", "Track")}
    contrasts = (
        ("Channel", (y["pitch", "Maint"] + y["pitch", "Track"] - y["intensity", "Maint"]
                     - y["intensity", "Track"]) / 2),
        ("Task", (y["pitch", "Maint"] + y["intensity", "Maint"] - y["pitch", "Track"]
                  - y["intensity", "Track"]) / 2),
        ("Channel x Task", (y["pitch", "Maint"] - y["pitch", "Track"]) - (y["intensity", "Maint"]
                                                                          - y["intensity", "Track"])))
    print()
    print("   raw TLX, 2 (Channel) x 2 (Task) repeated-measures ANOVA:")
    for lab, cvec in contrasts:
        tt = stats.ttest_1samp(cvec, 0.0)
        F = tt.statistic ** 2
        print("     %-15s F(1,%d) = %6.2f  p = %s  partial eta^2 = %.3f"
              % (lab, N - 1, F, fp(tt.pvalue), F / (F + N - 1)))
    print("     block means: %s" % ", ".join("%s %.2f" % (blk, v) for blk, v in bm.items()))

    # ---- SUS paired difference
    dm, lo, hi, tv, pv, dz = paired(a, b)
    print()
    print("   SUS pitch - intensity: %+.2f  95%% CI [%+.2f, %+.2f]  t(%d) = %.2f  p = %s  d_z = %.2f"
          % (dm, lo, hi, N - 1, tv, fp(pv), dz))
    print("   pitch ceilings: raw TLX %.2f of 100, SUS %.2f of 100"
          % (chan("Raw", "pitch").mean(), sus_ch["pitch"].mean()))
    return chan, sus_ch, choice


def section_order(chan, sus_ch, choice):
    print("3. Subjective measures and block sequence (A: participants 1-12, B: 13-24)")
    ia = [CODES.index(p) for p in SEQUENCE_A]
    ib = [CODES.index(p) for p in SEQUENCE_B]
    for lab, grp, last in (("A", SEQUENCE_A, "intensity"), ("B", SEQUENCE_B, "pitch")):
        k = int((choice.loc[grp] == "pitch").sum())
        print("   sequence %s (last block %-9s): pitch preferred by %d/%d; two-sided binomial p = %s"
              % (lab, last, k, len(grp), fp(stats.binomtest(k, len(grp), 0.5).pvalue)))
    rows = (("raw TLX pitch - intensity", chan("Raw", "pitch") - chan("Raw", "intensity")),
            ("SUS pitch - intensity", sus_ch["pitch"] - sus_ch["intensity"]),
            ("SUS pitch", sus_ch["pitch"]),
            ("SUS intensity", sus_ch["intensity"]))
    print("   %-26s %8s %8s   Student t (A vs B)" % ("", "A", "B"))
    for lab, v in rows:
        ma, mb, tv, pv, df = two_groups(v[ia], v[ib])
        f = "%+8.2f" if "-" in lab else "%8.2f"
        print("   %-26s %s %s   t(%d) = %+.2f  p = %s" % (lab, f % ma, f % mb, df, tv, fp(pv)))
    print("   (SUS follows a tracking block in A and a maintenance block in B, for both channels)")


def section_range(rng_by_pid):
    print("4. Calibrated intensity range and open-loop noise vs performance, N = %d" % N)
    pm, n_trials = participant_means()
    bad = {k: v for k, v in n_trials.items() if v != N_TRIALS[k[2]]}
    print("   per-trial files: %d blocks (%d participants x 2 channels x 2 tasks), %d trials; "
          "blocks with a missing trial: %d" % (len(n_trials), N, sum(n_trials.values()), len(bad)))
    summ = read_csv("per-participant-summary_data.csv")
    dev = 0.0
    for (ch, task, col), v in pm.items():
        if col == "zr":
            continue
        s = summ[(summ.Channel == ch) & (summ.Task == task)].set_index("Participant").loc[PIDS, col]
        dev = max(dev, np.abs(s.to_numpy(float) - v).max())
    print("   per-trial means vs per-participant-summary_data.csv: max |diff| = %.1e (4-decimal rounding)"
          % dev)
    ol = read_csv("openloop-log_data.csv")
    seg = ol.groupby("Participant").size().loc[PIDS]
    amp_noise = ol.groupby("Participant").AmpSdPoints.mean().loc[PIDS].to_numpy()
    rng = np.array([rng_by_pid[p] for p in PIDS])
    ol_rng = ol.groupby("Participant").IntensityRangeDb.first().loc[PIDS].to_numpy(float)
    print("   open-loop segments per participant: %s; open-loop log range = calibration range: %s"
          % ("/".join(str(v) for v in sorted(seg.unique())), bool(np.allclose(ol_rng, rng))))
    print("   observed intensity range %.2f-%.2f dB" % (rng.min(), rng.max()))

    def show(lab, x, y, unit):
        r, p, b, se, a = assoc(x, y)
        print("   %-40s r = %+.3f  p = %s  R^2 = %.3f  slope %+.3f %s (SE %.3f)"
              % (lab, r, fp(p), r * r, b, unit, se))
        return a, b

    P = pm
    pct = 100.0
    print("   maintenance, predictor = calibrated intensity range (dB):")
    a, b = show("success, pitch - intensity (%)", rng,
                pct * (P["pitch", "maintenance", "Success"] - P["intensity", "maintenance", "Success"]),
                "pp/dB")
    print("     fitted difference = %.2f %+.3f x range; zero at %.1f dB (observed maximum %.2f dB)"
          % (a, b, -a / b, rng.max()))
    show("TIB, pitch - intensity (%)", rng,
         pct * (P["pitch", "maintenance", "TIB"] - P["intensity", "maintenance", "TIB"]), "pp/dB")
    show("intensity within-trial SD (points)", rng, P["intensity", "maintenance", "SD"], "points/dB")
    show("intensity TIB (%)", rng, pct * P["intensity", "maintenance", "TIB"], "pp/dB")
    show("intensity success (%)", rng, pct * P["intensity", "maintenance", "Success"], "pp/dB")
    print("   maintenance, predictor = open-loop amplitude noise (points):")
    show("intensity TIB (%)", amp_noise, pct * P["intensity", "maintenance", "TIB"], "pp/point")
    show("intensity within-trial SD (points)", amp_noise, P["intensity", "maintenance", "SD"],
         "points/point")
    print("   tracking, predictor = calibrated intensity range (dB):")
    show("TIB, pitch - intensity (%)", rng,
         pct * (P["pitch", "tracking", "TIB"] - P["intensity", "tracking", "TIB"]), "pp/dB")
    show("intensity TIB (%)", rng, pct * P["intensity", "tracking", "TIB"], "pp/dB")
    show("intensity r (plain mean of trial r)", rng, P["intensity", "tracking", "PearsonR"], "per dB")
    zr = P["intensity", "tracking", "zr"]
    r1, p1 = stats.pearsonr(rng, zr)
    r2, p2 = stats.pearsonr(rng, np.tanh(zr))
    print("     r averaged in Fisher z: r = %+.3f  p = %s (z scale); r = %+.3f  p = %s (back-transformed)"
          % (r1, fp(p1), r2, fp(p2)))


def main():
    # Plain line feeds, so that saved output is the same file on every platform.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    print("Subjective measures and calibrated range")
    print("")
    rng_by_pid = section_calibration()
    print("")
    tlx, sus, pref, counts, book = load_workbook()
    chan, sus_ch, choice = section_subjective(tlx, sus, pref, counts, book)
    print("")
    section_order(chan, sus_ch, choice)
    print("")
    section_range(rng_by_pid)


if __name__ == "__main__":
    main()
