"""
theory_validation.py
====================
Numerical validation of every quantitative claim in the *Main Results* of the
revised manuscript.  Produces publication-quality PDF figures and a JSON file of
machine-checked numbers (slopes, slacks, rates, success probabilities).

Validated statements
---------------------
  Lemma 1  (coefficient decay)        :  |w_j(alpha)| ~ C j^{-(1+alpha)}
  Lemma 2  (tempered-gradient align.) :  <grad f, g> >= S(||grad f||^2 - L h M ||grad f||)
  Theorem 1 (local rate)              :  limsup ||grad f(x_t)|| <= L h M   (linear approach)
  Theorem 2 (global convergence)      :  P(f(b_t)-f* > d) <= (1 - eps q_d)^t -> 0
  Effect of (alpha, lambda)           :  exploration/exploitation trade-off

Run:  PYTHONUTF8=1 python theory_validation.py
"""

from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import get_benchmarks, make_quadratic
from tfabc import (TFABC, ABC, PSO, GA, gl_weights, tempered_weights,
                   tempered_fractional_gradient, jmax_for)

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130, "savefig.bbox": "tight", "lines.linewidth": 1.8,
})
RESULTS = {}


# --------------------------------------------------------------------- #
def fig_coefficient_decay():
    """Lemma 1: magnitudes of the fractional weights decay like j^{-(1+alpha)}."""
    Jmax = 4000
    j = np.arange(1, Jmax + 1)
    alphas = [0.3, 0.5, 0.7, 0.9]
    fitted = {}
    plt.figure(figsize=(6.4, 4.6))
    for a in alphas:
        w = np.abs(gl_weights(a, Jmax))[1:]
        # least-squares slope on log-log over the asymptotic tail
        tail = j > 200
        slope, intercept = np.polyfit(np.log(j[tail]), np.log(w[tail]), 1)
        fitted[a] = slope
        plt.loglog(j, w, label="alpha=%.1f (slope %.3f)" % (a, slope))
    # reference line  j^{-(1+alpha)} for alpha=0.7
    plt.loglog(j, j ** (-(1 + 0.7)) * 0.3, "k--", lw=1.0,
               label="ref  j^-(1+0.7)")
    plt.xlabel("memory index  j")
    plt.ylabel("|w_j(alpha)|")
    plt.title("Algebraic decay of the tempered GL weights")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "coeff_decay.pdf"))
    plt.close()
    RESULTS["lemma1_fitted_slopes"] = {("alpha_%.1f" % a): float(s)
                                       for a, s in fitted.items()}
    RESULTS["lemma1_predicted_slopes"] = {("alpha_%.1f" % a): -(1 + a)
                                          for a in alphas}


# --------------------------------------------------------------------- #
def fig_alignment():
    """Lemma 2: verify  <grad f, g(x)> >= S(||grad f||^2 - L h M ||grad f||)
    on a strongly convex quadratic at many random points and step sizes."""
    d = 8
    f, grad, Q, x_star, mu, L = make_quadratic(d, cond=40.0, seed=3)
    rng = np.random.default_rng(7)
    alpha, lam = 0.7, 0.3
    J = jmax_for(lam)
    _, _, S, M = tempered_weights(alpha, lam, J)

    hs = [0.0, 0.01, 0.05, 0.1, 0.2]
    npts = 400
    min_slack = np.inf
    plt.figure(figsize=(6.4, 4.6))
    for h in hs:
        lhs, rhs = [], []
        for _ in range(npts):
            x = x_star + rng.uniform(-3, 3, d)
            g = tempered_fractional_gradient(grad, x, alpha, lam, h, J)
            gn = grad(x)
            ng = np.linalg.norm(gn)
            lhs.append(float(gn @ g))
            rhs.append(float(S * (ng ** 2 - L * h * M * ng)))
        lhs, rhs = np.array(lhs), np.array(rhs)
        slack = (lhs - rhs).min()
        min_slack = min(min_slack, slack)
        plt.scatter(rhs, lhs, s=8, alpha=0.4, label="h=%.2f" % h)
    lim = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lim, lim, "k--", lw=1.0, label="y = x (bound is tight here)")
    plt.xlabel("lower bound  S(||g||^2 - L h M ||g||)")
    plt.ylabel("inner product  <grad f, g>")
    plt.title("Gradient-alignment inequality (points on/above y=x)")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "alignment_validation.pdf"))
    plt.close()
    RESULTS["lemma2"] = dict(S=S, M=M, mu=mu, L=L,
                             min_slack=float(min_slack),
                             holds=bool(min_slack >= -1e-8))


def fig_sc_alignment():
    """Lemma 3.6 (the FIX for the old Theorem 3.7): verify the strong-convexity
    alignment used in the Robbins--Siegmund stochastic-stability proof,

        <x - x*, g(x)>  >=  S*mu*||x-x*||^2  -  S*rho*||x-x*||,   rho = L h M,

    where g is the tempered fractional-memory gradient.  This bound is between
    (x - x*) and g and is NOT the gradient-alignment lemma; the earlier
    manuscript wrongly derived it from the latter.  We confirm it numerically at
    many random points / step sizes on a strongly convex quadratic."""
    d = 8
    f, grad, Q, x_star, mu, L = make_quadratic(d, cond=40.0, seed=5)
    rng = np.random.default_rng(13)
    alpha, lam = 0.7, 0.3
    J = jmax_for(lam)
    _, _, S, M = tempered_weights(alpha, lam, J)
    hs = [0.0, 0.02, 0.05, 0.1]
    npts = 400
    min_slack = np.inf
    plt.figure(figsize=(6.4, 4.6))
    for h in hs:
        rho = L * h * M
        lhs, rhs = [], []
        for _ in range(npts):
            x = x_star + rng.uniform(-3, 3, d)
            g = tempered_fractional_gradient(grad, x, alpha, lam, h, J)
            dv = x - x_star
            nv = np.linalg.norm(dv)
            lhs.append(float(dv @ g))
            rhs.append(float(S * mu * nv ** 2 - S * rho * nv))
        lhs, rhs = np.array(lhs), np.array(rhs)
        min_slack = min(min_slack, (lhs - rhs).min())
        plt.scatter(rhs, lhs, s=8, alpha=0.4, label="h=%.2f" % h)
    lim = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lim, lim, "k--", lw=1.0, label="y = x")
    plt.xlabel("lower bound  S*mu*||x-x*||^2 - S*rho*||x-x*||")
    plt.ylabel("inner product  <x-x*, g(x)>")
    plt.title("Strong-convexity alignment (points on/above y=x)")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "sc_alignment_validation.pdf"))
    plt.close()
    RESULTS["lemma_sc"] = dict(S=S, M=M, mu=mu, L=L,
                               min_slack=float(min_slack),
                               holds=bool(min_slack >= -1e-8))


# --------------------------------------------------------------------- #
def fig_local_rate():
    """Theorem 1: deterministic tempered-fractional descent converges linearly
    to an O(L h M) stationarity neighbourhood; radius scales with h."""
    d = 8
    f, grad, Q, x_star, mu, L = make_quadratic(d, cond=40.0, seed=3)
    alpha, lam = 0.7, 0.3
    J = jmax_for(lam)
    _, _, S, M = tempered_weights(alpha, lam, J)
    eta = 1.0 / (L * S)          # safe step
    T = 400
    hs = [0.0, 0.02, 0.05, 0.1]
    rng = np.random.default_rng(11)

    plt.figure(figsize=(6.4, 4.6))
    limits = {}
    for h in hs:
        x = x_star + rng.uniform(-3, 3, d)
        gn = np.empty(T)
        for t in range(T):
            g = tempered_fractional_gradient(grad, x, alpha, lam, h, J)
            x = x - eta * g
            gn[t] = np.linalg.norm(grad(x))
        plt.semilogy(gn, label="h=%.2f  (pred. floor %.2e)" % (h, L * h * M))
        if h > 0:
            plt.axhline(L * h * M, color="gray", ls=":", lw=0.8)
        limits[h] = float(np.median(gn[-20:]))
    plt.xlabel("iteration  t")
    plt.ylabel("||grad f(x_t)||")
    plt.title("Linear approach to the O(Delta) neighbourhood")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "local_rate.pdf"))
    plt.close()
    RESULTS["theorem1_residual_vs_h"] = {("h_%.2f" % h): v
                                         for h, v in limits.items()}
    RESULTS["theorem1_predicted_floor"] = {("h_%.2f" % h): float(L * h * M)
                                           for h in hs if h > 0}


# --------------------------------------------------------------------- #
def fig_global_convergence():
    """Theorem 2: success probability -> 1 and the geometric tail bound."""
    dim = 10
    bench = get_benchmarks(dim)
    names = ["Rastrigin", "Ackley", "Griewank", "Schwefel"]
    Tmax, nrun = 150, 20
    # delta defines the global-basin target set S_delta of Theorem 4.4
    delta = {"Rastrigin": 5.0, "Ackley": 2.0, "Griewank": 0.5, "Schwefel": 250.0}

    plt.figure(figsize=(6.4, 4.6))
    succ_summary = {}
    for nm in names:
        b = bench[nm]; lo, hi = b.box()
        gaps = np.empty((nrun, Tmax))
        for r in range(nrun):
            res = TFABC(b, lo, hi, SN=40, seed=1000 + r).optimize(Tmax)
            gaps[r] = res.history - b.f_star
        succ = (gaps <= delta[nm]).mean(axis=0)        # P(reached delta-set by t)
        plt.plot(np.arange(1, Tmax + 1), succ, label=nm)
        succ_summary[nm] = float(succ[-1])
    plt.xlabel("iteration  t")
    plt.ylabel("P( f(b_t) - f*  <=  delta )")
    plt.title("Empirical global-basin success probability")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "global_convergence.pdf"))
    plt.close()
    RESULTS["theorem2_final_success"] = succ_summary

    # geometric-tail check on Rastrigin: fit log P(fail) ~ t log(1-eps q)
    b = bench["Rastrigin"]; lo, hi = b.box()
    gaps = np.empty((120, Tmax))
    for r in range(120):
        res = TFABC(b, lo, hi, SN=40, seed=5000 + r).optimize(Tmax)
        gaps[r] = res.history - b.f_star
    pfail = (gaps > delta["Rastrigin"]).mean(axis=0)
    tt = np.arange(1, Tmax + 1)
    msk = (pfail > 1e-3) & (pfail < 1.0)
    if msk.sum() > 5:
        slope, _ = np.polyfit(tt[msk], np.log(pfail[msk]), 1)
        RESULTS["theorem2_geometric_decay_rate"] = float(slope)  # < 0 expected
    plt.figure(figsize=(6.4, 4.6))
    plt.semilogy(tt, np.maximum(pfail, 1e-3), label="empirical P(fail)")
    if msk.sum() > 5:
        plt.semilogy(tt, np.exp(slope * tt + np.polyfit(tt[msk],
                     np.log(pfail[msk]), 1)[1]), "k--",
                     label="geometric fit  exp(%.3f t)" % slope)
    plt.xlabel("iteration  t"); plt.ylabel("P( f(b_t)-f* > delta )")
    plt.title("Geometric decay of the failure probability")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "geometric_tail.pdf"))
    plt.close()


# --------------------------------------------------------------------- #
def fig_param_effect():
    """Effect of fractional order alpha and tempering lambda on convergence."""
    b = get_benchmarks(10)["Rastrigin"]; lo, hi = b.box()
    Tmax, nrun = 200, 20
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for a in [0.3, 0.5, 0.7, 0.9]:
        H = np.array([TFABC(b, lo, hi, alpha=a, seed=200 + r).optimize(Tmax).history
                      for r in range(nrun)])
        ax[0].semilogy(np.median(H, 0) + 1e-12, label="alpha=%.1f" % a)
    ax[0].set_title("Effect of fractional order alpha (lambda=0.3)")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("median best cost")
    ax[0].legend(fontsize=8)

    for lm in [0.1, 0.3, 0.6, 1.0]:
        H = np.array([TFABC(b, lo, hi, lam=lm, seed=300 + r).optimize(Tmax).history
                      for r in range(nrun)])
        ax[1].semilogy(np.median(H, 0) + 1e-12, label="lambda=%.1f" % lm)
    ax[1].set_title("Effect of tempering lambda (alpha=0.7)")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("median best cost")
    ax[1].legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "rate_vs_params.pdf"))
    plt.close()


# --------------------------------------------------------------------- #
def fig_benchmark_comparison():
    """TF-ABC vs ABC / PSO / GA on the multimodal suite: convergence + quality."""
    dim = 10
    bench = get_benchmarks(dim)
    names = ["Rastrigin", "Ackley", "Griewank", "Schwefel"]
    algos = {"TF-ABC": TFABC, "ABC": ABC, "PSO": PSO, "GA": GA}
    Tmax, nrun = 200, 25
    finals = {a: {} for a in algos}

    # convergence on Rastrigin (median + IQR band)
    plt.figure(figsize=(6.4, 4.6))
    b = bench["Rastrigin"]; lo, hi = b.box()
    for a, cls in algos.items():
        H = np.array([cls(b, lo, hi, seed=10 * k).optimize(Tmax).history
                      for k in range(nrun)])
        med = np.median(H, 0)
        q1, q3 = np.percentile(H, [25, 75], axis=0)
        x = np.arange(1, Tmax + 1)
        line, = plt.semilogy(x, med + 1e-12, label=a)
        plt.fill_between(x, q1 + 1e-12, q3 + 1e-12, alpha=0.15,
                         color=line.get_color())
    plt.xlabel("iteration"); plt.ylabel("best cost (median, IQR band)")
    plt.title("Benchmark convergence (Rastrigin-10)")
    plt.legend(fontsize=9)
    plt.savefig(os.path.join(FIGDIR, "benchmark_convergence.pdf"))
    plt.close()

    # final-quality box plots across the suite
    fig, axes = plt.subplots(1, len(names), figsize=(13, 3.8))
    for ax, nm in zip(axes, names):
        b = bench[nm]; lo, hi = b.box()
        data = []
        for a, cls in algos.items():
            vals = [cls(b, lo, hi, seed=7 * k + 1).optimize(Tmax).f
                    for k in range(nrun)]
            data.append(vals)
            finals[a][nm] = dict(mean=float(np.mean(vals)),
                                 std=float(np.std(vals)),
                                 best=float(np.min(vals)))
        ax.boxplot(data, tick_labels=list(algos.keys()), showfliers=False)
        ax.set_yscale("log"); ax.set_title(nm)
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Final solution quality over %d runs" % nrun)
    plt.savefig(os.path.join(FIGDIR, "benchmark_quality.pdf"))
    plt.close()
    RESULTS["benchmark_finals"] = finals


# --------------------------------------------------------------------- #
def main():
    import traceback
    stages = [("Lemma 1", fig_coefficient_decay),
              ("Lemma 2", fig_alignment),
              ("Lemma 3.6 (SC alignment)", fig_sc_alignment),
              ("Theorem 1", fig_local_rate),
              ("Theorem 2", fig_global_convergence),
              ("param effect", fig_param_effect),
              ("benchmark comparison", fig_benchmark_comparison)]
    for name, fn in stages:
        try:
            fn(); print("[ok] %s" % name, flush=True)
        except Exception:
            print("[FAIL] %s" % name, flush=True); traceback.print_exc()
        # write incrementally so a later failure never loses earlier numbers
        with open("theory_results.json", "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, indent=2)
    print("[done] wrote theory_results.json")


if __name__ == "__main__":
    main()
