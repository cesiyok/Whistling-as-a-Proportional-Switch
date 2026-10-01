"""Trial-level models for maintenance success and for practice effects (Appendix F-A, F-E, Table 12).

Reads the per-trial files of the data package:
    <DATA_DIR>/participant tests/pa<N>_<channel>_<task>_data.csv     (N = 1..24)

Usage:
    python mixed_models.py [DATA_DIR]

DATA_DIR defaults to the parent of the folder holding this script (the script
belongs in DATA_DIR/scripts/). All results are printed to stdout.

Models
  1. Maintenance success, mixed-effects logistic regression (primary model):
         logit P(success) = b0 + b1*pitch + g40*T40 + g60*T60 + g80*T80 + u_i,
         u_i ~ N(0, sigma^2), one random intercept per participant i.
     Reference levels: intensity channel, target 20. Fitted by maximum likelihood;
     the marginal likelihood of each participant is computed by adaptive
     Gauss-Hermite quadrature (25 nodes, centred on the mode of u_i). Standard
     errors come from the numerically differentiated Hessian of the negative
     log-likelihood; 95 % CIs are Wald intervals on the log-odds scale,
     exponentiated. ICC = sigma^2 / (sigma^2 + pi^2/3).
  2. Channel x Target interaction models, fitted separately from model 1:
     categorical (pitch x T40, T60, T80) and continuous (target centred at 50,
     divided by 30). Tested with likelihood-ratio tests against the matching
     additive model. Pitch succeeded on every trial at target 80, so the ML
     estimate of the pitch x T80 term is infinite; the likelihood then attains
     its supremum when those trials are predicted with probability 1, which is
     the same as fitting the interaction model without them.
  3. Robustness: random slope for channel (Laplace approximation), GEE with an
     exchangeable working correlation, conditional logit with participant strata,
     and target entered as a continuous predictor.
  4. Practice effects: for each outcome a linear mixed model
         y = Channel * Trial + Difficulty + (1 | participant) + (1 | participant:channel),
     Trial centred within the task, Difficulty = target (maintenance) or rate
     (tracking) as a factor, pitch as reference channel, fitted by REML
     (statsmodels MixedLM). Robustness: an added random slope for Trial per
     participant and channel. Success uses model 1 with Channel * Trial added.
"""
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
PT = os.path.join(DATA, "participant tests")
PIDS = list(range(1, 25))
CHANNELS = ("pitch", "intensity")
NODES = 25

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")


def load(task):
    parts = []
    for pid in PIDS:
        for ch in CHANNELS:
            fn = os.path.join(PT, "pa%d_%s_%s_data.csv" % (pid, ch, task))
            d = pd.read_csv(fn, float_precision="round_trip")
            d["pid"] = pid
            d["ch"] = ch
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["pitch"] = (d["ch"] == "pitch").astype(float)
    return d


# ----------------------------------------------------------------------------
# Random-intercept logistic regression by maximum likelihood
# ----------------------------------------------------------------------------
def quadrature(nq):
    if nq == 1:
        return np.array([0.0]), np.array([np.sqrt(2.0 * np.pi)])
    return np.polynomial.hermite_e.hermegauss(nq)  # weight exp(-x^2/2)


def ri_negloglik(X, y, groups, nq):
    """Negative marginal log-likelihood; theta = (beta, log sigma)."""
    xq, wq = quadrature(nq)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    def f(theta):
        beta, sig = theta[:-1], np.exp(theta[-1])
        eta = X @ beta
        total = 0.0
        for ix in idx:
            e, yy = eta[ix], y[ix]
            u = 0.0
            for _ in range(100):
                p = special.expit(e + u)
                step = (np.sum(yy - p) - u / sig ** 2) / (-np.sum(p * (1 - p)) - 1 / sig ** 2)
                u -= step
                if abs(step) < 1e-12:
                    break
            p = special.expit(e + u)
            s = 1.0 / np.sqrt(np.sum(p * (1 - p)) + 1.0 / sig ** 2)
            uu = u + s * xq
            lin = e[:, None] + uu[None, :]
            ll = (yy[:, None] * lin - np.logaddexp(0.0, lin)).sum(0)
            lp = ll - uu ** 2 / (2 * sig ** 2) - 0.5 * np.log(2 * np.pi * sig ** 2) + xq ** 2 / 2
            total += special.logsumexp(lp, b=wq * s)
        return -total
    return f


def num_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    E = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                                 - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H


def num_grad(f, x, h=1e-6):
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


def fit_ri(X, y, groups, nq=NODES):
    f = ri_negloglik(X, y, groups, nq)
    r = optimize.minimize(f, np.zeros(X.shape[1] + 1), method="BFGS",
                          options={"gtol": 1e-8, "maxiter": 5000})
    x = r.x
    H = num_hessian(f, x)
    return {"theta": x, "nll": f(x), "cov": np.linalg.inv(H), "grad": np.abs(num_grad(f, x)).max(),
            "mineig": np.linalg.eigvalsh(H).min()}


def design(d, terms):
    cols, names = [np.ones(len(d))], ["(Intercept)"]
    tc = ((d["Target"] - 50.0) / 30.0).values
    for t in terms:
        if t == "pitch":
            v = d["pitch"].values
        elif t in ("T40", "T60", "T80"):
            v = (d["Target"] == int(t[1:])).astype(float).values
        elif t.startswith("pitch:T"):
            v = d["pitch"].values * (d["Target"] == int(t[7:])).astype(float).values
        elif t == "tc":
            v = tc
        elif t == "pitch:tc":
            v = d["pitch"].values * tc
        elif t == "trial":
            v = d["trial_c"].values
        elif t == "pitch:trial":
            v = d["pitch"].values * d["trial_c"].values
        cols.append(v)
        names.append(t)
    return np.column_stack(cols), names


def coef_line(res, names, term, label):
    k = names.index(term)
    b, se = res["theta"][k], np.sqrt(res["cov"][k, k])
    z = b / se
    return ("%-44s b = %+.3f  SE %.3f  OR = %.1f  95%% CI [%.1f, %.1f]  z = %.2f  p = %.2g"
            % (label, b, se, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), z,
               2 * stats.norm.sf(abs(z))))


def lrt(nll_small, nll_big, df):
    chi2 = 2.0 * (nll_small - nll_big)
    return chi2, stats.chi2.sf(chi2, df)


# ----------------------------------------------------------------------------
# Random intercept + random slope for channel, Laplace approximation
# ----------------------------------------------------------------------------
def rs_negloglik(X, y, z, groups):
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    def f(theta):
        beta = theta[:-3]
        s0, s1, rho = np.exp(theta[-3]), np.exp(theta[-2]), np.tanh(theta[-1])
        S = np.array([[s0 * s0, rho * s0 * s1], [rho * s0 * s1, s1 * s1]])
        Si = np.linalg.inv(S)
        logdetS = np.linalg.slogdet(S)[1]
        eta = X @ beta
        total = 0.0
        for ix in idx:
            e, yy = eta[ix], y[ix]
            Z = np.column_stack([np.ones(len(ix)), z[ix]])
            u = np.zeros(2)
            for _ in range(100):
                p = special.expit(e + Z @ u)
                g = Z.T @ (yy - p) - Si @ u
                Hn = Z.T @ (Z * (p * (1 - p))[:, None]) + Si
                step = np.linalg.solve(Hn, g)
                u += step
                if np.abs(step).max() < 1e-12:
                    break
            lin = e + Z @ u
            p = special.expit(lin)
            Hn = Z.T @ (Z * (p * (1 - p))[:, None]) + Si
            ll = np.sum(yy * lin - np.logaddexp(0.0, lin))
            total += ll - 0.5 * u @ Si @ u - 0.5 * logdetS - 0.5 * np.linalg.slogdet(Hn)[1]
        return -total
    return f


# ----------------------------------------------------------------------------
def section_success(m):
    print("1. Maintenance success: mixed-effects logistic regression (ML, adaptive Gauss-Hermite)")
    y, g = m["Success"].values.astype(float), m["pid"].values
    tab = m.pivot_table(index="ch", columns="Target", values="Success", aggfunc="sum")
    print("   successes (of 120 per cell):")
    for ch in CHANNELS:
        print("     %-9s %s" % (ch, "  ".join("%3d" % tab.loc[ch, t] for t in (20, 40, 60, 80))))
    print("   trials %d, participants %d; reference levels: intensity, target 20" % (len(m), len(PIDS)))
    X, names = design(m, ["pitch", "T40", "T60", "T80"])
    res = fit_ri(X, y, g)
    for t, lab in zip(names, ["intercept", "pitch (channel)", "target 40", "target 60", "target 80"]):
        k = names.index(t)
        b, se = res["theta"][k], np.sqrt(res["cov"][k, k])
        print("   %-18s b = %+.3f  SE %.3f  z = %+.2f  p = %.2g" % (lab, b, se, b / se, 2 * stats.norm.sf(abs(b / se))))
    print("   " + coef_line(res, names, "pitch", "channel odds ratio (pitch vs intensity)"))
    ls, se_ls = res["theta"][-1], np.sqrt(res["cov"][-1, -1])
    s2 = np.exp(2 * ls)
    print("   random-intercept variance sigma^2 = %.2f (95%% CI %.2f to %.2f, from log sigma %.3f SE %.3f)"
          % (s2, np.exp(2 * (ls - 1.96 * se_ls)), np.exp(2 * (ls + 1.96 * se_ls)), ls, se_ls))
    print("   ICC = %.3f" % (s2 / (s2 + np.pi ** 2 / 3)))
    print("   log-likelihood %.3f; max |gradient| %.1e; smallest Hessian eigenvalue %.2f"
          % (-res["nll"], res["grad"], res["mineig"]))
    for nq in (1, 5, 11):
        r2 = fit_ri(X, y, g, nq)
        k = names.index("pitch")
        print("   nodes %2d: OR = %.2f  SE(b) %.4f  sigma^2 %.3f  log-lik %.3f"
              % (nq, np.exp(r2["theta"][k]), np.sqrt(r2["cov"][k, k]), np.exp(2 * r2["theta"][-1]), -r2["nll"]))
    add_nll = res["nll"]

    print()
    print("2. Channel x Target interaction (separate models, likelihood-ratio tests)")
    Xc, nc = design(m, ["pitch", "tc"])
    rc = fit_ri(Xc, y, g)
    Xci, nci = design(m, ["pitch", "tc", "pitch:tc"])
    rci = fit_ri(Xci, y, g)
    k = nci.index("pitch:tc")
    b, se = rci["theta"][k], np.sqrt(rci["cov"][k, k])
    chi2, p = lrt(rc["nll"], rci["nll"], 1)
    print("   continuous target: pitch x target b = %+.2f  SE %.2f  Wald p = %.2f;  LRT chi2(1) = %.2f  p = %.2f"
          % (b, se, 2 * stats.norm.sf(abs(b / se)), chi2, p))
    sep = (m["pitch"] == 1) & (m["Target"] == 80)
    print("   pitch at target 80: %d of %d trials successful (separation)" % (int(m.loc[sep, "Success"].sum()), int(sep.sum())))
    sub = m.loc[~sep]
    Xi, ni = design(sub, ["pitch", "T40", "T60", "T80", "pitch:T40", "pitch:T60"])
    ri = fit_ri(Xi, sub["Success"].values.astype(float), sub["pid"].values)
    chi2, p = lrt(add_nll, ri["nll"], 3)
    print("   categorical target: LRT chi2(3) = %.2f  p = %.2f (interaction likelihood at its supremum)" % (chi2, p))
    print("   max |gradient| %.1e; smallest Hessian eigenvalue %.2f" % (ri["grad"], ri["mineig"]))

    print()
    print("3. Robustness of the channel odds ratio")
    print("   " + coef_line(rc, nc, "pitch", "target continuous (random intercept, ML)"))
    Xs, ns = design(m, ["pitch", "T40", "T60", "T80"])
    f = rs_negloglik(Xs, y, m["pitch"].values, g)
    th0 = np.concatenate([res["theta"][:-1], [res["theta"][-1], np.log(0.5), 0.0]])
    r = optimize.minimize(f, th0, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
    H = num_hessian(f, r.x)
    eig = np.linalg.eigvalsh(H)
    cov = np.linalg.pinv(H) if eig.min() <= 0 else np.linalg.inv(H)
    b, se = r.x[1], np.sqrt(cov[1, 1])
    print("   random slope for channel (Laplace): OR = %.1f  95%% CI [%.1f, %.1f]; SD intercept %.2f, SD slope %.2f, "
          "correlation %.2f; LRT vs random intercept chi2(2) = %.2f  p = %.2f; smallest Hessian eigenvalue %.3g"
          % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), np.exp(r.x[-3]), np.exp(r.x[-2]),
             np.tanh(r.x[-1]), 2 * (fit_ri(X, y, g, 1)["nll"] - f(r.x)),
             stats.chi2.sf(max(0.0, 2 * (fit_ri(X, y, g, 1)["nll"] - f(r.x))), 2), eig.min()))
    gee = smf.gee("Success ~ pitch + C(Target)", "pid", data=m, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = gee.params["pitch"], gee.bse["pitch"]
    print("   GEE, exchangeable (robust SE): OR = %.1f  95%% CI [%.1f, %.1f]  z = %.2f  p = %.2g"
          % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), b / se, gee.pvalues["pitch"]))
    Xcl = pd.get_dummies(m["Target"].astype(int), prefix="T", drop_first=True).astype(float)
    Xcl.insert(0, "pitch", m["pitch"].values)
    cl = ConditionalLogit(m["Success"].values, Xcl, groups=m["pid"].values).fit(disp=False)
    b, se = cl.params["pitch"], cl.bse["pitch"]
    print("   conditional logit (participant strata): OR = %.1f  95%% CI [%.1f, %.1f]  z = %.2f  p = %.2g"
          % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), b / se, 2 * stats.norm.sf(abs(b / se))))


def mixedlm(d, y, slope):
    vc = {"blk": "0 + C(ch)"}
    if slope:
        vc["bslope"] = "0 + C(ch):trial_c"
    md = smf.mixedlm('%s ~ C(ch, Treatment("pitch")) * trial_c + C(diff)' % y, d, groups="pid",
                     re_formula="1", vc_formula=vc)
    fit = None
    for method in ("lbfgs", "bfgs", "powell", "nm"):
        fit = md.fit(reml=True, method=method, maxiter=5000)
        if fit.converged:
            break
    return fit


def section_practice(m, t):
    print("4. Practice effects: Channel x Trial (points per trial; pitch is the reference channel)")
    m = m.copy()
    t = t.copy()
    m["TIBp"] = 100 * m["TIB"]
    t["TIBp"] = 100 * t["TIB"]
    t["rz"] = np.arctanh(t["PearsonR"].clip(-0.999999, 0.999999))
    m["diff"] = m["Target"].astype(int).astype(str)
    t["diff"] = t["SpeedPercent"].astype(int).astype(str)
    m["trial_c"] = m["Trial"] - m["Trial"].mean()
    t["trial_c"] = t["Trial"] - t["Trial"].mean()
    for label, d, y in (("maintenance TIB (%)", m, "TIBp"), ("maintenance RMSE", m, "RMSE"),
                        ("tracking r (Fisher z)", t, "rz"), ("tracking TIB (%)", t, "TIBp"),
                        ("tracking RMSE", t, "RMSE")):
        for slope in (False, True):
            r = mixedlm(d, y, slope)
            k = [n for n in r.params.index if n.endswith(":trial_c")][0]
            cov = r.cov_params()
            bp, sp = r.params["trial_c"], r.bse["trial_c"]
            bi = bp + r.params[k]
            si = np.sqrt(cov.loc["trial_c", "trial_c"] + cov.loc[k, k] + 2 * cov.loc["trial_c", k])
            bd, sd = r.params[k], r.bse[k]
            print("   %-22s %-13s pitch %+.3f (p = %.3f)  intensity %+.3f (p = %.3f)  "
                  "intensity - pitch %+.3f  95%% CI [%+.3f, %+.3f]  p = %.3f  converged %s"
                  % (label, "random slope" if slope else "intercepts", bp, 2 * stats.norm.sf(abs(bp / sp)),
                     bi, 2 * stats.norm.sf(abs(bi / si)), bd, bd - 1.96 * sd, bd + 1.96 * sd,
                     2 * stats.norm.sf(abs(bd / sd)), r.converged))
    # success: random-intercept logistic model with Channel * Trial
    y, g = m["Success"].values.astype(float), m["pid"].values
    X, names = design(m, ["pitch", "T40", "T60", "T80", "trial", "pitch:trial"])
    res = fit_ri(X, y, g)
    ki, kp = names.index("trial"), names.index("pitch:trial")
    bi, si = res["theta"][ki], np.sqrt(res["cov"][ki, ki])
    bp = bi + res["theta"][kp]
    sp = np.sqrt(res["cov"][ki, ki] + res["cov"][kp, kp] + 2 * res["cov"][ki, kp])
    bd, sd = res["theta"][kp], np.sqrt(res["cov"][kp, kp])
    print("   %-22s %-13s pitch %+.3f (p = %.3f)  intensity %+.3f (p = %.3f)  "
          "intensity - pitch %+.3f  95%% CI [%+.3f, %+.3f]  p = %.3f  (log-odds per trial)"
          % ("maintenance success", "intercepts", bp, 2 * stats.norm.sf(abs(bp / sp)), bi,
             2 * stats.norm.sf(abs(bi / si)), -bd, -bd - 1.96 * sd, -bd + 1.96 * sd, 2 * stats.norm.sf(abs(bd / sd))))


if __name__ == "__main__":
    maint = load("maintenance")
    track = load("tracking")
    section_success(maint)
    print()
    section_practice(maint, track)
