"""Separate analysis of the scored trials closest to a break (Appendix F-C, Table 10).

Reads the per-trial files of the data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv     (N = 1..24)

Usage:
    python near_break_analysis.py [DATA_DIR] [OUT_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). OUT_DIR defaults to near_break_output/ next to
this script; it receives the result tables and results.txt.

Definitions (as in the manuscript text):
  * 'first k' = trials 1..k of a block, 'last k' = trials n-k+1..n (n = 20 maintenance, 15 tracking),
    'rest' = the remaining trials of the same block.
  * Difficulty removal: from each trial value subtract that participant's mean for the same block and
    the same target (maintenance) or rate (tracking).
  * Per participant: mean(first k) - mean(rest) of the difficulty-removed values (same for last k);
    paired t over the 24 participants, 95 % CI, d_z = mean/SD of the differences.
  * Holm across the 8 first-k tests (2 channels x 2 tasks x 2 primary outcomes), separately across
    the 8 last-k tests.
  * Primary outcomes: maintenance TIB and Success; tracking Fisher z(r) and TIB.
  * Channel difference: pitch minus intensity of participant means, all trials vs. trials with
    k < Trial <= n-k.
"""
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
PT = os.path.join(DATA, "participant tests")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "near_break_output")
os.makedirs(OUTDIR, exist_ok=True)

PIDS = ["pa%d" % i for i in range(1, 25)]
CHANNELS = ("pitch", "intensity")
TASKS = ("maintenance", "tracking")
NTR = {"maintenance": 20, "tracking": 15}
DIFF = {"maintenance": "Target", "tracking": "SpeedPercent"}
PRIMARY = {"maintenance": ["TIB", "Success"], "tracking": ["zr", "TIB"]}
SCALE = {"TIB": 100.0, "Success": 100.0, "zr": 1.0, "RMSE": 1.0, "PearsonR": 1.0}

LOG = io.StringIO()


def say(s=""):
    print(s)
    LOG.write(s + "\n")


# --------------------------------------------------------------------------- data
def load():
    frames = []
    for pid in PIDS:
        for ch in CHANNELS:
            for task in TASKS:
                f = os.path.join(PT, "%s_%s_%s_data.csv" % (pid, ch, task))
                d = pd.read_csv(f)
                d["pid"], d["channel"], d["task"] = pid, ch, task
                d["diff"] = d[DIFF[task]].astype(int)
                if task == "tracking":
                    d["zr"] = np.arctanh(d["PearsonR"].astype(float))
                frames.append(d)
    return pd.concat(frames, ignore_index=True)


def check(df):
    say("Loaded %d trial rows from %s" % (len(df), PT))
    for task in TASKS:
        d = df[df.task == task]
        n = NTR[task]
        g = d.groupby(["pid", "channel"])
        assert (g.size() == n).all(), "block size"
        assert (g.Trial.apply(lambda t: sorted(t) == list(range(1, n + 1)))).all(), "trial numbers"
        per = d.groupby(["pid", "channel", "diff"]).size()
        say("  %-11s rows=%d, blocks=%d, trials/block=%d, trials per target/rate cell: %s"
            % (task, len(d), g.ngroups, n, sorted(per.unique())))
        for o in PRIMARY[task] + ["RMSE"]:
            na = int(d[o].isna().sum())
            say("    %-8s missing=%d" % (o, na))
            assert na == 0
    t = df[df.task == "tracking"].PearsonR
    say("  tracking PearsonR range %.4f .. %.4f (no clipping needed)" % (t.min(), t.max()))
    for task in TASKS:
        d = df[df.task == task]
        e = d[d.Trial <= 3].groupby(["pid", "channel"])["diff"].mean()
        say("  difficulty mix, %-11s first-3 mean %s = %.2f (all trials %.2f)"
            % (task, DIFF[task], e.mean(), d["diff"].mean()))


# --------------------------------------------------------------------------- helpers
def ttest1(x):
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    return dict(n=n, diff=m, lo=m - h, hi=m + h, sd=sd, t=t, df=n - 1, p=p, dz=m / sd)


def holm(p):
    """Own Holm step-down; cross-checked against statsmodels below."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    ref = multipletests(p, method="holm")[1]
    assert np.allclose(adj, ref), "Holm mismatch"
    return adj


def fp(p):
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def remove_difficulty(d, col):
    return d[col] - d.groupby(["pid", "channel", "task", "diff"])[col].transform("mean")


# --------------------------------------------------------------------------- 1. first/last k
def near_break(df, k, adjust=True):
    rows = []
    for task in TASKS:
        n = NTR[task]
        for ch in CHANNELS:
            d = df[(df.task == task) & (df.channel == ch)].copy()
            for o in PRIMARY[task]:
                d["y"] = remove_difficulty(d, o) if adjust else d[o]
                for pos, sel in (("first", d.Trial <= k), ("last", d.Trial > n - k)):
                    a = d[sel].groupby("pid")["y"].mean()
                    b = d[~sel].groupby("pid")["y"].mean()
                    r = ttest1((a - b).reindex(PIDS))
                    # identical paired-t via scipy as a cross-check
                    tt = stats.ttest_rel(a.reindex(PIDS), b.reindex(PIDS))
                    assert np.isclose(tt.pvalue, r["p"]) and np.isclose(tt.statistic, r["t"])
                    r.update(k=k, pos=pos, task=task, channel=ch, outcome=o, adjusted=adjust,
                             mean_k=a.mean(), mean_rest=b.mean())
                    rows.append(r)
    R = pd.DataFrame(rows)
    R["p_holm"] = np.nan
    for pos in ("first", "last"):
        idx = R.index[R.pos == pos]
        R.loc[idx, "p_holm"] = holm(R.loc[idx, "p"].values)
    return R


def show_near(R, title):
    say()
    say(title)
    say("%-6s %-11s %-9s %-8s %9s %22s %7s %7s %7s %6s"
        % ("pos", "task", "channel", "outcome", "diff", "95% CI", "t(23)", "p", "pHolm", "d_z"))
    for _, r in R.iterrows():
        s = SCALE[r.outcome]
        say("%-6s %-11s %-9s %-8s %+9.3f [%+9.3f,%+9.3f] %+7.2f %7s %7s %+6.2f"
            % (r.pos, r.task, r.channel, r.outcome, s * r["diff"], s * r.lo, s * r.hi, r.t,
               fp(r.p), fp(r.p_holm), r.dz))


# --------------------------------------------------------------------------- 2. mixed models
def lmm(formula, data, terms):
    data = data.reset_index(drop=True)
    md = smf.mixedlm(formula, data, groups=data["pid"])
    for meth in ("lbfgs", "bfgs", "powell", "nm"):
        try:
            r = md.fit(reml=True, method=meth)
        except Exception:  # noqa
            continue
        if not r.converged or not all(np.isfinite(r.bse[t]) for t in terms):
            continue
        return {t: dict(coef=r.params[t], se=r.bse[t], p=r.pvalues[t]) for t in terms}, meth
    return None, None


def mixed(df, k):
    rows = []
    for task in TASKS:
        n = NTR[task]
        for ch in CHANNELS:
            d = df[(df.task == task) & (df.channel == ch)].copy()
            d["first"] = (d.Trial <= k).astype(int)
            d["last"] = (d.Trial > n - k).astype(int)
            d["pos_c"] = d.Trial - (n + 1) / 2.0
            for o in PRIMARY[task]:
                d["y"] = d[o].astype(float)
                # M: y ~ first + last + linear trial position + target/rate, random intercept
                res, meth = lmm("y ~ first + last + pos_c + C(diff)", d, ["first", "last", "pos_c"])
                # trend-only model (no first/last indicators), for comparison with 'Practice.'
                res0, _ = lmm("y ~ pos_c + C(diff)", d, ["pos_c"])
                row = dict(k=k, task=task, channel=ch, outcome=o, method=meth,
                           trend_only_slope=res0["pos_c"]["coef"], trend_only_p=res0["pos_c"]["p"])
                for t in ("first", "last", "pos_c"):
                    row[t + "_coef"] = res[t]["coef"]
                    row[t + "_lo"] = res[t]["coef"] - 1.96 * res[t]["se"]
                    row[t + "_hi"] = res[t]["coef"] + 1.96 * res[t]["se"]
                    row[t + "_p"] = res[t]["p"]
                if o == "Success":
                    try:
                        g = smf.gee("y ~ first + last + pos_c + C(diff)", "pid", d,
                                    family=sm.families.Binomial(),
                                    cov_struct=sm.cov_struct.Exchangeable()).fit()
                        for t in ("first", "last", "pos_c"):
                            row["gee_%s_OR" % t] = np.exp(g.params[t])
                            row["gee_%s_p" % t] = g.pvalues[t]
                    except Exception as e:  # noqa  (pitch success: near-complete separation)
                        row["gee_err"] = str(e)[:80]
                rows.append(row)
    return pd.DataFrame(rows)


def show_mixed(M):
    say()
    say("Mixed model per channel x task x outcome: y ~ first_k + last_k + trial position (linear) +")
    say("C(target|rate), random intercept per participant (REML). Coefficients in outcome units x scale.")
    say("%-11s %-9s %-8s %28s %28s %22s %20s"
        % ("task", "channel", "outcome", "first_k [95% CI] p", "last_k [95% CI] p",
           "position slope p", "trend-only slope p"))
    for _, r in M.iterrows():
        s = SCALE[r.outcome]
        say("%-11s %-9s %-8s %+7.3f [%+7.3f,%+7.3f] %5s %+7.3f [%+7.3f,%+7.3f] %5s %+9.4f %6s %+9.4f %6s"
            % (r.task, r.channel, r.outcome, s * r.first_coef, s * r.first_lo, s * r.first_hi,
               fp(r.first_p), s * r.last_coef, s * r.last_lo, s * r.last_hi, fp(r.last_p),
               s * r.pos_c_coef, fp(r.pos_c_p), s * r.trend_only_slope, fp(r.trend_only_p)))
        if r.outcome == "Success" and "gee_first_OR" in r and r.gee_first_OR == r.gee_first_OR:
            say("%31s GEE logit OR: first %.2f p=%s, last %.2f p=%s, position %.3f p=%s"
                % ("", r.gee_first_OR, fp(r.gee_first_p), r.gee_last_OR, fp(r.gee_last_p),
                   r.gee_pos_c_OR, fp(r.gee_pos_c_p)))


# --------------------------------------------------------------------------- 3. channel difference
def channel_diff(df, k):
    rows = []
    for task in TASKS:
        n = NTR[task]
        for o in PRIMARY[task]:
            for subset, keep in (("all", df.Trial >= 1),
                                 ("without first/last %d" % k, (df.Trial > k) & (df.Trial <= n - k))):
                d = df[(df.task == task) & keep]
                # (a) plain participant mean of the kept trials
                g = d.groupby(["pid", "channel"])[o].mean().unstack().reindex(PIDS)
                r = ttest1(g["pitch"] - g["intensity"])
                # (b) difficulty-balanced: mean of target/rate cell means (cells left empty are skipped)
                c = d.groupby(["pid", "channel", "diff"])[o].mean().groupby(["pid", "channel"]).mean()
                c = c.unstack().reindex(PIDS)
                rb = ttest1(c["pitch"] - c["intensity"])
                ncell = d.groupby(["pid", "channel"])["diff"].nunique()
                r.update(k=k, task=task, outcome=o, subset=subset,
                         pitch=g["pitch"].mean(), intensity=g["intensity"].mean(),
                         bal_diff=rb["diff"], bal_dz=rb["dz"], bal_p=rb["p"],
                         min_cells=int(ncell.min()), n_trials_kept=len(d))
                rows.append(r)
    return pd.DataFrame(rows)


def show_channel(C):
    say()
    say("Channel difference (pitch - intensity), participant means; (bal) = mean of target/rate cell means")
    say("%-3s %-11s %-8s %-22s %8s %8s %9s %22s %7s %7s %6s | %9s %6s %7s %s"
        % ("k", "task", "outcome", "subset", "pitch", "intens", "diff", "95% CI", "t(23)", "p", "d_z",
           "bal diff", "bal dz", "bal p", "min cells"))
    for _, r in C.iterrows():
        s = SCALE[r.outcome]
        say("%-3d %-11s %-8s %-22s %8.3f %8.3f %+9.3f [%+9.3f,%+9.3f] %+7.2f %7s %6.2f | %+9.3f %6.2f %7s %d"
            % (r.k, r.task, r.outcome, r.subset, s * r.pitch, s * r.intensity, s * r["diff"],
               s * r.lo, s * r.hi, r.t, fp(r.p), r.dz, s * r.bal_diff, r.bal_dz, fp(r.bal_p),
               r.min_cells))


# --------------------------------------------------------------------------- 4. half-block (Fatigue text)
def half_block(df):
    rows = []
    for task, outs in (("maintenance", ["TIB", "Success", "RMSE"]),
                       ("tracking", ["TIB", "PearsonR", "zr", "RMSE"])):
        half = NTR[task] // 2          # maintenance 1-10 vs 11-20; tracking 1-7 vs 8-15
        for ch in CHANNELS:
            d = df[(df.task == task) & (df.channel == ch)]
            for o in outs:
                a = d[d.Trial <= half].groupby("pid")[o].mean().reindex(PIDS)
                b = d[d.Trial > half].groupby("pid")[o].mean().reindex(PIDS)
                r = ttest1(a - b)
                r.update(task=task, channel=ch, outcome=o, first=a.mean(), second=b.mean())
                rows.append(r)
    H = pd.DataFrame(rows)
    say()
    say("Half-block check (Fatigue paragraph): first half minus second half, participant means")
    for _, r in H.iterrows():
        s = SCALE.get(r.outcome, 1.0)
        say("  %-11s %-9s %-8s %8.3f -> %8.3f  diff %+7.3f  t(23)=%+6.2f  p=%.4f"
            % (r.task, r.channel, r.outcome, s * r["first"], s * r.second, s * r["diff"], r.t, r.p))
    return H


# --------------------------------------------------------------------------- 5. LaTeX + numbers
def sgn(x, nd):
    s = ("%+." + str(nd) + "f") % x
    return "0" if float(s) == 0 and nd == 0 else s


def cell(m, lo, hi, nd):
    return "$%s$ $[%s,%s]$" % (sgn(m, nd), sgn(lo, nd), sgn(hi, nd))


def pl(p):
    return "$<.001$" if p < 0.001 else ("$1.000$" if p >= 0.9995 else "$%s$" % ("%.3f" % p).lstrip("0"))


LABEL = {"TIB": "TIB", "Success": "Success", "zr": "$r$ (z)"}
NDIG = {"TIB": 1, "Success": 1, "zr": 2}


def latex_rows(R3, C3):
    out = []
    for task in TASKS:
        for ch in CHANNELS:
            out.append(r"\multicolumn{5}{@{}l}{\emph{%s %s}}\\" % (ch.capitalize(), task))
            for o in PRIMARY[task]:
                s, nd = SCALE[o], NDIG[o]
                f = R3[(R3.pos == "first") & (R3.task == task) & (R3.channel == ch) & (R3.outcome == o)].iloc[0]
                l = R3[(R3.pos == "last") & (R3.task == task) & (R3.channel == ch) & (R3.outcome == o)].iloc[0]
                out.append("%s & %s & %s & %s & %s\\\\" % (
                    LABEL[o], cell(s * f["diff"], s * f.lo, s * f.hi, nd), pl(f.p_holm),
                    cell(s * l["diff"], s * l.lo, s * l.hi, nd), pl(l.p_holm)))
    out.append("--- channel block ---")
    for task in TASKS:
        out.append(r"\multicolumn{5}{@{}l}{\emph{Pitch $-$ intensity, %s}}\\" % task)
        for o in PRIMARY[task]:
            s, nd = SCALE[o], NDIG[o]
            a = C3[(C3.task == task) & (C3.outcome == o) & (C3.subset == "all")].iloc[0]
            b = C3[(C3.task == task) & (C3.outcome == o) & (C3.subset != "all")].iloc[0]
            out.append("%s & %s & $%.2f$ & %s & $%.2f$\\\\" % (
                LABEL[o], cell(s * a["diff"], s * a.lo, s * a.hi, nd), a.dz,
                cell(s * b["diff"], s * b.lo, s * b.hi, nd), b.dz))
    return "\n".join(out)


def numbers_dump(Rk, M, C, H):
    lines = []
    for _, r in Rk.iterrows():
        s = SCALE[r.outcome]
        lines.append("near-break k=%d %-5s %-11s %-9s %-7s diff=%+.4f CI=[%+.4f,%+.4f] t=%.3f p=%.5f pHolm=%.5f dz=%+.3f"
                     % (r.k, r.pos, r.task, r.channel, r.outcome, s * r["diff"], s * r.lo, s * r.hi,
                        r.t, r.p, r.p_holm, r.dz))
    for _, r in M.iterrows():
        s = SCALE[r.outcome]
        lines.append("mixed k=%d %-11s %-9s %-7s first=%+.4f p=%.5f  last=%+.4f p=%.5f  slope=%+.5f p=%.5f"
                     % (r.k, r.task, r.channel, r.outcome, s * r.first_coef, r.first_p,
                        s * r.last_coef, r.last_p, s * r.pos_c_coef, r.pos_c_p))
    for _, r in C.iterrows():
        s = SCALE[r.outcome]
        lines.append("channel k=%d %-11s %-7s %-22s diff=%+.4f CI=[%+.4f,%+.4f] t=%.3f p=%.3g dz=%.4f"
                     % (r.k, r.task, r.outcome, r.subset, s * r["diff"], s * r.lo, s * r.hi, r.t,
                        r.p, r.dz))
    for _, r in H.iterrows():
        s = SCALE.get(r.outcome, 1.0)
        lines.append("half %-11s %-9s %-8s %.3f -> %.3f p=%.5f"
                     % (r.task, r.channel, r.outcome, s * r["first"], s * r.second, r.p))
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    df = load()
    check(df)
    allR = []
    adjR = {}
    for k in (3, 5):
        R = near_break(df, k, adjust=True)
        adjR[k] = R
        show_near(R, "Near-break test, k = %d, difficulty removed (TIB/Success in points, z in z units)" % k)
        allR.append(R)
        Rr = near_break(df, k, adjust=False)
        show_near(Rr, "Near-break test, k = %d, raw (difficulty NOT removed), for reference" % k)
        allR.append(Rr)
    pd.concat(allR).to_csv(os.path.join(OUTDIR, "near_break_tests.csv"), index=False,
                           float_format="%.6g")
    M = pd.concat([mixed(df, 3), mixed(df, 5)], ignore_index=True)
    for k in (3, 5):
        say()
        say("---- mixed model, k = %d ----" % k)
        show_mixed(M[M.k == k])
    M.to_csv(os.path.join(OUTDIR, "mixed_models.csv"), index=False, float_format="%.6g")
    C = pd.concat([channel_diff(df, 3), channel_diff(df, 5)], ignore_index=True)
    show_channel(C)
    C.to_csv(os.path.join(OUTDIR, "channel_difference.csv"), index=False, float_format="%.6g")
    H = half_block(df)
    H.to_csv(os.path.join(OUTDIR, "half_block.csv"), index=False, float_format="%.6g")
    tex = latex_rows(adjR[3], C[C.k == 3])
    with io.open(os.path.join(OUTDIR, "table_rows.tex"), "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    say()
    say("LaTeX rows for tab:breaks (k = 3):")
    say(tex)
    nums = numbers_dump(pd.concat([adjR[3], adjR[5]]), M, C, H)
    with io.open(os.path.join(OUTDIR, "numbers.txt"), "w", encoding="utf-8") as f:
        f.write(nums + "\n")
    say()
    say(nums)
    with io.open(os.path.join(OUTDIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write(LOG.getvalue())
