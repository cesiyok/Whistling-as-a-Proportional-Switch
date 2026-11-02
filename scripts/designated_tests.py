"""Designated outcomes of H1-H4 and the decision rule across them (Section IV-H, Appendix F-G, Table 14).

Uses the functions of maintenance_tracking.py, mixed_models.py and
subjective_and_range.py (same folder), which read the data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv   (N = 1..24)
    <DATA_DIR>/participant tests/subjective-measures.xlsx

Usage:
    python designated_tests.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). Results are printed to stdout.

One test per designated outcome, nine in all:
  H1 (maintenance, channel effect)
     time-in-band: Channel main effect of the 2 x 4 repeated-measures ANOVA
     success: channel term of the trial-level mixed-effects logistic model (Wald z)
  H2 (tracking, channel effect)
     correlation (Fisher z) and time-in-band: Channel main effects of the 2 x 5 ANOVA
  H3 (Channel x Target interaction)
     time-in-band: interaction of the 2 x 4 ANOVA, Greenhouse-Geisser p
     success: likelihood-ratio test of the categorical interaction at the supremum
              of the likelihood (trial level; pitch succeeded on every trial at 80)
  H4 (subjective)
     raw TLX and SUS: Wilcoxon signed-rank test, exact p
     forced-choice preference: exact two-sided binomial test against 0.5
The Holm step-down correction is applied over all nine p values (alpha = .05).
Decision rule: a hypothesis is supported when every one of its designated
outcomes is significant after this correction, partly supported when some are,
and not supported when none is. H3 also carries a shape prediction (the largest
intensity disadvantage at both extreme targets), which is judged descriptively
in Section V-B and does not enter this rule.
"""
import io
import contextlib
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maintenance_tracking as MT   # noqa: E402
import mixed_models as MM           # noqa: E402
import subjective_and_range as SR   # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")

ALPHA = 0.05


def anova_tests():
    maint = MT.load_trials("maintenance")
    track = MT.load_trials("tracking")
    tib_m = MT.rm_anova(MT.cells(maint, "TIB", "Target", MT.TARGETS))
    r_t = MT.rm_anova(MT.cells(track, "PearsonR", "SpeedPercent", MT.RATES, z=True))
    tib_t = MT.rm_anova(MT.cells(track, "TIB", "SpeedPercent", MT.RATES))

    def f(r, key, pkey="p"):
        e = r[key]
        txt = "F(%d, %d) = %.2f" % (e["df1"], e["df2"], e["F"])
        if pkey == "p_gg":
            txt += ", eps = %.2f" % e["eps"]
        return txt, e[pkey]
    return {"h1_tib": f(tib_m, "ch"), "h2_r": f(r_t, "ch"), "h2_tib": f(tib_t, "ch"),
            "h3_tib": f(tib_m, "int", "p_gg")}


def success_tests():
    m = MM.load("maintenance")
    y, g = m["Success"].values.astype(float), m["pid"].values
    X, names = MM.design(m, ["pitch", "T40", "T60", "T80"])
    res = MM.fit_ri(X, y, g)
    k = names.index("pitch")
    b, se = res["theta"][k], np.sqrt(res["cov"][k, k])
    z = b / se
    sep = (m["pitch"] == 1) & (m["Target"] == 80)
    sub = m.loc[~sep]
    Xi, _ = MM.design(sub, ["pitch", "T40", "T60", "T80", "pitch:T40", "pitch:T60"])
    ri = MM.fit_ri(Xi, sub["Success"].values.astype(float), sub["pid"].values)
    chi2, p = MM.lrt(res["nll"], ri["nll"], 3)
    return {"h1_success": ("OR = %.1f, z = %.2f" % (np.exp(b), z), 2 * stats.norm.sf(abs(z))),
            "h3_success": ("chi2(3) = %.2f" % chi2, p)}


def subjective_tests():
    tlx, sus, pref, _, _ = SR.load_workbook()
    t = tlx.copy()
    for key in ("MD", "PD", "TD", "PE_rep", "EF", "FR"):
        t[key] = t[key].astype(float)
    t["PE_load"] = 100.0 - t["PE_rep"]
    t["Raw"] = t[[key for key, _ in SR.SUBSCALES]].mean(axis=1)
    t["Channel"] = t.Block.str.split("-").str[1].str.lower()
    raw = t.groupby(["P", "Channel"])["Raw"].mean()
    s = sus.copy()
    q = s[["Q%d" % i for i in range(1, 11)]].to_numpy(float)
    s["SUS"] = 2.5 * ((q[:, 0::2] - 1).sum(axis=1) + (5 - q[:, 1::2]).sum(axis=1))
    s["Channel"] = s.Channel.str.lower()
    out = {}
    for key, a, b in (("h4_tlx", raw.xs("pitch", level="Channel").loc[SR.CODES].to_numpy(),
                       raw.xs("intensity", level="Channel").loc[SR.CODES].to_numpy()),
                      ("h4_sus", s[s.Channel == "pitch"].set_index("P").loc[SR.CODES, "SUS"].to_numpy(),
                       s[s.Channel == "intensity"].set_index("P").loc[SR.CODES, "SUS"].to_numpy())):
        w = SR.wilcoxon(a, b)
        out[key] = ("Z = %.2f (n = %d)" % (w["Z"], w["n"]), w["p"])
    choice = pref.set_index("P").loc[SR.CODES, "Choice"].str.lower()
    k = int((choice == "pitch").sum())
    out["h4_pref"] = ("%d of %d chose pitch" % (k, len(choice)),
                      stats.binomtest(k, len(choice), 0.5, alternative="two-sided").pvalue)
    return out


ROWS = (("H1", "h1_tib", "maintenance time-in-band", "channel, RM ANOVA"),
        ("H1", "h1_success", "maintenance success", "channel, mixed-effects logistic"),
        ("H2", "h2_r", "tracking correlation (Fisher z)", "channel, RM ANOVA"),
        ("H2", "h2_tib", "tracking time-in-band", "channel, RM ANOVA"),
        ("H3", "h3_tib", "maintenance time-in-band", "Channel x Target, RM ANOVA (GG)"),
        ("H3", "h3_success", "maintenance success", "Channel x Target, trial-level LRT"),
        ("H4", "h4_tlx", "raw TLX", "Wilcoxon signed-rank, exact"),
        ("H4", "h4_sus", "SUS", "Wilcoxon signed-rank, exact"),
        ("H4", "h4_pref", "forced-choice preference", "exact binomial"))


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        res = {}
        res.update(anova_tests())
        res.update(success_tests())
        res.update(subjective_tests())
    p = np.array([res[key][1] for _, key, _, _ in ROWS])
    p_holm = MT.holm(p)
    print("Designated outcomes of H1-H4, Holm correction over all nine tests (alpha = %.2f)" % ALPHA)
    print("%-3s %-34s %-34s %-28s %10s %10s  %s" % ("H", "outcome", "test", "statistic", "p", "p Holm", "sig."))
    for (h, key, lab, test), pv, ph in zip(ROWS, p, p_holm):
        print("%-3s %-34s %-34s %-28s %10.2g %10.2g  %s"
              % (h, lab, test, res[key][0], pv, ph, "yes" if ph < ALPHA else "no"))
    print()
    print("Decision rule: supported if all designated outcomes are significant after correction")
    for h in ("H1", "H2", "H3", "H4"):
        sig = [ph < ALPHA for (hh, _, _, _), ph in zip(ROWS, p_holm) if hh == h]
        verdict = "supported" if all(sig) else ("partly supported" if any(sig) else "not supported")
        print("   %s: %d of %d designated outcomes significant -> %s" % (h, sum(sig), len(sig), verdict))


if __name__ == "__main__":
    main()
