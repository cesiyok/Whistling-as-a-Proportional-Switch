"""Result figures of the paper (Figs. 5-8): maintenance error by target,
representative maintenance traces, tracking fidelity by rate, and NASA-TLX.

Reads only the public data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv       (N = 1..24)
    <DATA_DIR>/participant tests/pa<N>_<channel>_maintenance_raw_data.csv
    <DATA_DIR>/participant tests/subjective-measures.xlsx

Usage:
    python result_figures.py [DATA_DIR] [OUT_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). OUT_DIR defaults to result_figures_output/ next
to this script; it receives fig3_maint_by_target.png, fig4_traces.png,
fig5_track_by_rate.png and fig6_tlx.png (the file names used by the
manuscript source). Nothing is written into DATA_DIR unless OUT_DIR points
there. The plotted values are printed to standard output, and every run
prints the same text.

Definitions
  * Every point is the mean over the 24 participants of the participant's
    mean in that cell (5 trials per target, 3 per rate); error bars are 95 %
    confidence intervals of that mean (t distribution, 23 df).
  * Maintenance: RMSE and SD by target. Tracking: PearsonR (zero lag),
    BestLagR and TIB (in percent) by target rate.
  * Representative traces: at target 60, for each channel, the trial whose SD
    is closest to the median SD of the 120 trials of that channel and target.
    With an even number of trials the median lies between two trials; ties are
    broken by participant number, then trial number. The legend gives the
    trial's SD and TIB from the per-trial file.
  * NASA-TLX: per participant and channel, each subscale is averaged over the
    two blocks of the channel; Performance is plotted as load (100 - x). Raw
    TLX is the mean of the six subscales, averaged over the two blocks.
"""
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1:2] in (["-h"], ["--help"]):
    print(__doc__)
    sys.exit(0)
if len(sys.argv) > 3:
    sys.exit("usage: python result_figures.py [DATA_DIR] [OUT_DIR]")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "result_figures_output")
PT = os.path.join(DATA, "participant tests")
if os.path.normcase(OUT).startswith(os.path.normcase(os.path.join(DATA, "participant tests"))) or \
        os.path.normcase(OUT).startswith(os.path.normcase(os.path.join(DATA, "pilot tests"))):
    sys.exit("refusing to write figures into a data folder: choose another OUT_DIR")

PIDS = ["pa%d" % i for i in range(1, 25)]
CHANNELS = ("pitch", "intensity")
LABEL = {"pitch": "Pitch", "intensity": "Intensity"}
COLOR = {"pitch": "#1F4E79", "intensity": "#E31A1C"}
MARKER = {"pitch": "o", "intensity": "s"}
TARGETS = (20, 40, 60, 80)
RATES = (20, 35, 50, 65, 80)
TRACE_TARGET = 60
BAND = 15.0
DPI = 500
CW = 3.45                      # IEEE column width (inches)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1.1,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
GRID = dict(color="0.89", lw=0.5)
ERR = dict(capsize=2, elinewidth=0.8, capthick=0.8)


def say(s=""):
    print(s)


def load(ch, task):
    parts = []
    for pid in PIDS:
        d = pd.read_csv(os.path.join(PT, "%s_%s_%s_data.csv" % (pid, ch, task)), float_precision="round_trip")
        d["pid"] = pid
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cells(d, key, level_col, levels):
    """Participant x level matrix of cell means (24 x len(levels))."""
    m = d.groupby(["pid", level_col])[key].mean().unstack()
    m = m.reindex(index=PIDS, columns=list(levels))
    assert not m.isna().any().any(), "empty cell"
    return m.values


def mean_ci(y):
    y = np.asarray(y, float)
    n = y.shape[0]
    return y.mean(axis=0), y.std(axis=0, ddof=1) / math.sqrt(n) * stats.t.ppf(0.975, n - 1)


def report(title, x, rows):
    say(title)
    say("  %-26s %s" % ("", " ".join("%12s" % v for v in x)))
    for lab, m, h in rows:
        say("  %-26s %s" % (lab, " ".join("%6.3f+-%5.3f" % (a, b) for a, b in zip(m, h))))


def errorline(ax, x, m, h, ch, label, dashed=False):
    kw = dict(marker=MARKER[ch], color=COLOR[ch], ms=3.4, label=label, **ERR)
    if dashed:
        kw.update(ls=(0, (4, 2)), alpha=0.75, mfc="white", ms=3.2)
    ax.errorbar(x, m, yerr=h, **kw)


# ---------------------------------------------------------------------------
def fig_maintenance(maint):
    fig, ax = plt.subplots(1, 2, figsize=(CW, 1.75), sharex=True)
    x = np.array(TARGETS, float)
    rows = []
    for k, (key, ylab) in enumerate((("RMSE", "RMSE (force points)"), ("SD", "Within-trial SD (points)"))):
        for ch in CHANNELS:
            m, h = mean_ci(cells(maint[ch], key, "Target", TARGETS))
            errorline(ax[k], x, m, h, ch, LABEL[ch])
            rows.append(("%s %s" % (ch, key), m, h))
        ax[k].set_xlabel("Target force level (points)")
        ax[k].set_ylabel(ylab)
        ax[k].set_xticks(x)
        ax[k].set_ylim(bottom=0)
        ax[k].grid(True, **GRID)
        ax[k].set_axisbelow(True)
    ax[0].legend(frameon=False, loc="upper left", handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig3_maint_by_target.png"), dpi=DPI)
    plt.close(fig)
    report("fig3_maint_by_target: mean +- 95% CI half-width, by target", TARGETS, rows)


def pick_trace(d):
    """Trial at TRACE_TARGET whose SD is closest to the median SD of the channel."""
    c = d[d.Target == TRACE_TARGET].copy()
    c["num"] = c.pid.str[2:].astype(int)
    c = c.sort_values(["num", "Trial"])
    med = float(np.median(c.SD.values))
    dist = (c.SD - med).abs()
    best = c.loc[dist == dist.min()]
    return best.iloc[0], med, best


def fig_traces(maint):
    fig, ax = plt.subplots(figsize=(CW, 2.0))
    say("fig4_traces: trials at target %d whose SD is closest to the channel median" % TRACE_TARGET)
    for ch in CHANNELS:
        r, med, tied = pick_trace(maint[ch])
        raw = pd.read_csv(os.path.join(PT, "%s_%s_maintenance_raw_data.csv" % (r.pid, ch)))
        g = raw[raw.Trial == r.Trial].sort_values("TimeMs")
        t, f = g.TimeMs.values / 1000.0, g.Force.values
        onset = t[np.argmax(f > 0)] if (f > 0).any() else float("nan")
        ax.plot(t, f, color=COLOR[ch], lw=1.1, label="%s (SD %.1f, TIB %.2f)" % (LABEL[ch], r.SD, r.TIB))
        say("  %-9s median SD %.5f over %d trials; closest: %s" % (
            ch, med, int((maint[ch].Target == TRACE_TARGET).sum()),
            "; ".join("%s trial %d SD %.4f" % (x.pid, x.Trial, x.SD) for _, x in tied.iterrows())))
        say("  %-9s plotted: %s trial %d, SD %.4f, TIB %.4f, Bias %+.3f, first non-zero force at %.3f s"
            % (ch, r.pid, r.Trial, r.SD, r.TIB, r.Bias, onset))
    ax.axhspan(TRACE_TARGET - BAND, TRACE_TARGET + BAND, color="0.55", alpha=0.18, lw=0)
    ax.axhline(TRACE_TARGET, color="0.45", lw=0.8, ls=(0, (4, 2)))
    ax.axvspan(0, 1.0, color="0.5", alpha=0.13, lw=0)
    ax.text(0.5, 96, "excluded\nfrom scoring", ha="center", va="top", fontsize=6, color="0.4")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time within trial (s)")
    ax.set_ylabel("Force (points)")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig4_traces.png"), dpi=DPI)
    plt.close(fig)


def fig_tracking(track):
    fig, ax = plt.subplots(1, 2, figsize=(CW, 1.75))
    xs = np.array(RATES) / 100.0
    rows = []
    for ch in CHANNELS:
        m, h = mean_ci(cells(track[ch], "PearsonR", "SpeedPercent", RATES))
        errorline(ax[0], xs, m, h, ch, LABEL[ch] + " $r$")
        rows.append(("%s r" % ch, m, h))
        m, h = mean_ci(cells(track[ch], "BestLagR", "SpeedPercent", RATES))
        errorline(ax[0], xs, m, h, ch, LABEL[ch] + " best-lag $r$", dashed=True)
        rows.append(("%s best-lag r" % ch, m, h))
        m, h = mean_ci(100.0 * cells(track[ch], "TIB", "SpeedPercent", RATES))
        errorline(ax[1], xs, m, h, ch, LABEL[ch])
        rows.append(("%s TIB (%%)" % ch, m, h))
    ax[0].set_ylabel("Correlation with target")
    ax[0].set_ylim(0.55, 1.0)
    ax[0].legend(frameon=False, loc="lower left", fontsize=5.6, handletextpad=0.35, labelspacing=0.25)
    ax[1].set_ylabel("Time in band (%)")
    ax[1].set_ylim(50, 100)
    ax[1].legend(frameon=False, loc="lower left", handletextpad=0.4)
    for a in ax:
        a.set_xlabel("Target rate (Hz)")
        a.set_xticks(xs)
        a.set_xticklabels(["%.2f" % v for v in xs])
        a.grid(True, **GRID)
        a.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig5_track_by_rate.png"), dpi=DPI)
    plt.close(fig)
    report("fig5_track_by_rate: mean +- 95% CI half-width, by rate (Hz x 100)", RATES, rows)


def fig_tlx():
    xl = os.path.join(PT, "subjective-measures.xlsx")
    t = pd.read_excel(xl, sheet_name="NASA-TLX raw", header=2)
    code = {"P%02d" % i: "pa%d" % i for i in range(1, 25)}
    t = t[t.iloc[:, 0].isin(code)].copy()
    sub = t.iloc[:, 2:8].to_numpy(dtype=float, copy=True)     # MD PD TD PE EF FR
    sub[:, 3] = 100.0 - sub[:, 3]                              # Performance as load
    names = ["MD", "PD", "TD", "PE", "EF", "FR"]
    for j, n in enumerate(names):
        t[n] = sub[:, j]
    t["Raw"] = sub.mean(axis=1)
    assert np.allclose(t.Raw.values, t.iloc[:, 9].astype(float).values), "raw TLX column"
    t["pid"] = t.iloc[:, 0].map(code)
    t["ch"] = t.iloc[:, 1].str.split("-").str[1].str.lower()
    per = t.groupby(["pid", "ch"])[names + ["Raw"]].mean()
    labels = ["Mental", "Physical", "Temporal", "Perform.*", "Effort", "Frustr.", "Raw TLX"]
    fig, ax = plt.subplots(figsize=(CW, 2.0))
    xi = np.arange(len(labels))
    w = 0.38
    rows = []
    for off, ch in zip((-w / 2, w / 2), CHANNELS):
        y = np.array([[per.loc[(p, ch), n] for n in names + ["Raw"]] for p in PIDS])
        m, h = mean_ci(y)
        ax.bar(xi + off, m, w, yerr=h, color=COLOR[ch], label=LABEL[ch], capsize=1.8,
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
        rows.append((ch, m, h))
    ax.axvline(len(labels) - 1.5, color="0.6", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Load rating (0\u2013100)")
    ax.set_ylim(0, 45)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig6_tlx.png"), dpi=DPI)
    plt.close(fig)
    report("fig6_tlx: mean +- 95% CI half-width (Performance as load)", names + ["Raw"], rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    os.makedirs(OUT, exist_ok=True)
    maint = {ch: load(ch, "maintenance") for ch in CHANNELS}
    track = {ch: load(ch, "tracking") for ch in CHANNELS}
    fig_maintenance(maint)
    say()
    fig_traces(maint)
    say()
    fig_tracking(track)
    say()
    fig_tlx()
    say()
    say("written: fig3_maint_by_target.png, fig4_traces.png, fig5_track_by_rate.png, fig6_tlx.png")


if __name__ == "__main__":
    main()
