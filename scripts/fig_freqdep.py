"""Fig. 9: frequency dependence of the intensity estimate.

(a) Output of the full intensity chain for pure tones of constant amplitude
    between 1.0 and 1.4 kHz, relative to its maximum, with the band-pass
    response; right axis in force points for the median calibrated range.
(b) Recorded pitch tracks replayed as tones of constant amplitude: SD of the
    resulting error against the SD measured in the same maintenance trial or
    open-loop segment, with lines for an error of 100 %, 10 % and 1 % of the
    measured variance.

Reads the tables written by frequency_dependence.py:
    sweep_fine.csv, task_replay.csv, openloop_replay.csv, numbers.csv

Usage:
    python fig_freqdep.py [OUT_DIR]

OUT_DIR is the output folder of frequency_dependence.py (default:
frequency_dependence_output/ next to this script). The figure is written there
as fig_freqdep.pdf, with a PNG preview.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frequency_dependence_output"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
})
INTENS = "#E31A1C"          # intensity colour used in the other figures

numbers = pd.read_csv(os.path.join(OUT, "numbers.csv")).set_index("key").value
PTS = 100.0 / float(numbers["range_median_db"])     # force points per dB at the median range

fine = pd.read_csv(os.path.join(OUT, "sweep_fine.csv"))
task = pd.read_csv(os.path.join(OUT, "task_replay.csv"))
ol = pd.read_csv(os.path.join(OUT, "openloop_replay.csv"))
m = task[task.task == "maintenance"]

W = 242.67 / 72.27          # IEEE column width in inches
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.0), gridspec_kw={"height_ratios": [1, 1.05]})

# (a) sweep
a.plot(fine.f, fine.att_db, color=INTENS, lw=0.9, label="Constant-amplitude tone")
a.plot(fine.f, fine.bandpass_db, color="k", lw=0.9, ls="--", label="Band-pass response alone")
a.set_xlim(1000, 1400)
a.set_ylim(-2.4, 0.15)
a.set_xlabel("Tone frequency (Hz)")
a.set_ylabel("Output re maximum (dB)")
a.grid(True, color="0.88", lw=0.5)
a.legend(loc="lower right", frameon=False, handlelength=2.2)
ra = a.twinx()
lo, hi = a.get_ylim()
ra.set_ylim(lo * PTS, hi * PTS)
ra.set_ylabel("Change in force (points)")
a.text(0.015, 0.96, "(a)", transform=a.transAxes, va="top", fontweight="bold")

# (b) replayed pitch tracks
lim = (0.3, 40)
xs = np.array(lim)
for share, ls, lab in [(1.0, "-", "100 %"), (0.1, "--", "10 %"), (0.01, ":", "1 %")]:
    b.plot(xs, xs * np.sqrt(share), color="0.45", lw=0.7, ls=ls)
    x_lab = 37 if share < 1 else 0.8
    b.text(x_lab, x_lab * np.sqrt(share) * 1.18, lab, color="0.35", fontsize=6.5, ha="right", va="bottom",
           rotation=45, rotation_mode="anchor")
b.scatter(m.meas_sd, m.err_sd_pts, s=6, color=INTENS, alpha=0.55, lw=0, label="Maintenance trials (%d)" % len(m))
b.scatter(ol.meas_sd_pts, ol.raw_sd_pts, s=9, facecolor="none", edgecolor="k", lw=0.6, marker="s",
          label="Open-loop segments (%d)" % len(ol))
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(*lim)
b.set_ylim(*lim)
ticks = [0.5, 1, 2, 5, 10, 20]
b.set_xticks(ticks)
b.set_xticklabels(["%g" % v for v in ticks])
b.set_yticks(ticks)
b.set_yticklabels(["%g" % v for v in ticks])
b.minorticks_off()
b.set_xlabel("Measured SD (force points)")
b.set_ylabel("Estimator error SD (force points)")
b.grid(True, color="0.88", lw=0.5)
b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handletextpad=0.3, borderaxespad=0.3)
b.text(0.97, 0.04, "(b)", transform=b.transAxes, ha="right", va="bottom", fontweight="bold")

fig.tight_layout(h_pad=0.8)
fig.savefig(os.path.join(OUT, "fig_freqdep.pdf"))
fig.savefig(os.path.join(OUT, "fig_freqdep_preview.png"), dpi=300)
print("written: %s" % os.path.join(OUT, "fig_freqdep.pdf"))
