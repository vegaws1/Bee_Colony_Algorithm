"""
run_gfi.py
==========
Tunes the FOPID voltage controller with TF-ABC (and the ABC/PSO/GA baselines),
runs every grid-forming-inverter scenario, and regenerates the ten figures used
by the manuscript plus a JSON of all table numbers.

Outputs (in figures/):
  convergence_curves.pdf  solution_quality.pdf
  voltage_step_response.pdf  current_step_response.pdf
  impedance_variation.pdf
  nonlinear_voltage.pdf  nonlinear_current.pdf
  current_estimation.pdf  voltage_estimation.pdf
  fopid_bode.pdf
and gfi_results.json

Run:  PYTHONUTF8=1 python run_gfi.py
"""

from __future__ import annotations
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

import gfi_model as G
from gfi_model import (PARAMS, fopid_tf, plant_ss, plant_tf, tf_mul, tf_add,
                       tf_feedback, tf_sens, cost)
from tfabc import TFABC, ABC, PSO, GA

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight",
                     "lines.linewidth": 1.8})

# search box  theta = [Kp, Ki, Kd, mu, lambda]
# theta = [Kp, Ki, Kd, mu (integral order), nu (derivative order)]
# Orders restricted to a physically sensible band for power electronics
# (mu, nu in [0.7,1.2]); Kd kept modest to avoid aggressive HF behaviour.
LO = np.array([0.05, 1.0, 1e-4, 0.70, 0.70])
HI = np.array([6.0, 120.0, 0.03, 1.20, 1.20])
NCOST = 2                      # Oustaloup order during tuning (fast, well-conditioned)
NFINAL = 4                    # Oustaloup order for final figures
UMAX = 1.5                    # per-unit inverter current limit
RES = {}


def cost_vec(X, N=NCOST):
    return np.array([cost(row, N=N) for row in np.atleast_2d(X)])


def cost_pid_vec(X, N=NCOST):
    out = []
    for row in np.atleast_2d(X):
        th = row.copy(); th[3] = 1.0; th[4] = 1.0      # force integer orders
        out.append(cost(th, N=N))
    return np.array(out)


# ======================================================================= #
#  Discrete closed-loop state simulator (volts) for the load-step figures #
# ======================================================================= #
def closed_loop_states(theta, Lg=None, T=0.6, Ts=2e-5, Vref=1.0,
                       load_times=(0.2, 0.4), Iload=0.3, N=NFINAL):
    """Closed-loop response (per unit) computed with `lsim` on the stable
    closed-loop transfer functions, which is numerically robust for the
    high-order Oustaloup FOPID realisation.  A reference step (Vref) is applied
    at t=0 and a load-current disturbance Iload acts as an output disturbance
    during `load_times`; the voltage is v = T*ref + S*d and the commanded
    current reference is i_ref = C S (ref - d)."""
    Gp = plant_tf(PARAMS, Lg)
    C = fopid_tf(theta, N=N)
    L = tf_mul(C, Gp)
    if not G._is_stable(L)[0]:
        n = int(T / Ts); t = np.arange(n) * Ts
        return dict(t=t, v=np.full(n, np.nan), i_inv=np.full(n, np.nan),
                    u=np.full(n, np.nan))
    Tcl = signal.TransferFunction(*tf_feedback(L))
    Scl = signal.TransferFunction(*tf_sens(L))
    CS = signal.TransferFunction(*tf_mul(C, tf_sens(L)))   # C/(1+CG)
    n = int(T / Ts)
    t = np.arange(n) * Ts
    ref = np.full(n, Vref)
    d = np.where((t >= load_times[0]) & (t < load_times[1]), Iload, 0.0)
    _, vr, _ = signal.lsim(Tcl, U=ref, T=t)
    _, vd, _ = signal.lsim(Scl, U=d, T=t)
    _, ur, _ = signal.lsim(CS, U=ref, T=t)
    _, ud, _ = signal.lsim(CS, U=d, T=t)
    v = vr + vd
    iref = ur - ud                                         # commanded current ref
    return dict(t=t, v=v, i_inv=iref, u=iref)


def load_step_metrics(t, v, Vref, win):
    """Max deviation and recovery (settling) over a disturbance window."""
    m = (t >= win[0]) & (t < win[1])
    seg = v[m]
    dev = np.max(np.abs(seg - Vref))
    tol = 0.02 * Vref
    idx = np.where(np.abs(seg - Vref) > tol)[0]
    rec = (t[m][idx[-1]] - win[0]) if idx.size else 0.0
    return float(dev), float(rec)


# ======================================================================= #
#  1. Optimiser tuning: convergence + quality                             #
# ======================================================================= #
def tune_all():
    algos = {"TF-ABC": TFABC, "ABC": ABC, "PSO": PSO, "GA": GA}
    Tmax, SN = 50, 20
    hist, best = {}, {}
    print("  tuning headline run ...")
    for name, cls in algos.items():
        t0 = time.time()
        r = cls(cost_vec, LO, HI, SN=SN, seed=1).optimize(Tmax)
        hist[name] = r.history
        best[name] = (r.x, r.f)
        print("    %-7s best=%.4f  (%.1fs)" % (name, r.f, time.time() - t0))

    # convergence figure
    plt.figure(figsize=(6.4, 4.6))
    for name in algos:
        plt.plot(np.arange(1, Tmax + 1), hist[name], label=name)
    plt.xlabel("iteration"); plt.ylabel("best cost  J(theta)")
    plt.title("FOPID tuning convergence")
    plt.legend(); plt.savefig(os.path.join(FIGDIR, "convergence_curves.pdf"))
    plt.close()

    # quality distribution over many independent runs with COMMON random seeds
    # (paired across optimisers -> valid Friedman / Wilcoxon signed-rank tests)
    from scipy import stats as sstats
    nrun, Tq, SNq = 30, 30, 15
    finals = {name: [] for name in algos}
    print("  tuning quality runs (%d each) ..." % nrun)
    for k in range(nrun):
        for name, cls in algos.items():
            finals[name].append(
                cls(cost_vec, LO, HI, SN=SNq, seed=100 + k).optimize(Tq).f)
    finals = {k: np.asarray(v, float) for k, v in finals.items()}

    plt.figure(figsize=(6.4, 4.6))
    plt.boxplot([finals[a] for a in algos], tick_labels=list(algos.keys()),
                showfliers=False)
    plt.ylabel("final cost  J(theta*)")
    plt.title("Solution-quality distribution (%d runs)" % nrun)
    plt.savefig(os.path.join(FIGDIR, "solution_quality.pdf"))
    plt.close()

    def q(a, pc):
        return float(np.percentile(a, pc))
    RES["tuning_best_cost"] = {k: float(v[1]) for k, v in best.items()}
    RES["tuning_best_theta"] = {k: [float(z) for z in v[0]]
                                for k, v in best.items()}
    RES["tuning_quality"] = {
        k: dict(mean=float(v.mean()), std=float(v.std()),
                median=float(np.median(v)), iqr=q(v, 75) - q(v, 25),
                best=float(v.min()), worst=float(v.max()))
        for k, v in finals.items()}
    # nonparametric tests, paired by common seed
    try:
        friedman_p = float(sstats.friedmanchisquare(
            *[finals[a] for a in algos]).pvalue)
    except Exception:
        friedman_p = float("nan")
    wilco = {}
    for a in algos:
        if a == "TF-ABC":
            continue
        try:
            wilco[a] = float(sstats.wilcoxon(finals["TF-ABC"], finals[a]).pvalue)
        except Exception:
            wilco[a] = float("nan")
    RES["tuning_stats"] = dict(nrun=nrun, friedman_p=friedman_p,
                               wilcoxon_vs_TFABC=wilco)
    # rank the optimisers by mean final cost (lower is better)
    order = sorted(algos, key=lambda a: RES["tuning_quality"][a]["mean"])
    RES["tuning_rank_by_mean"] = order
    print("    rank by mean cost:", " < ".join(order))
    return best["TF-ABC"][0]


# ======================================================================= #
#  2. PID baseline                                                         #
# ======================================================================= #
def tune_pid():
    r = TFABC(cost_pid_vec, LO, HI, SN=20, seed=2).optimize(50)
    th = r.x.copy(); th[3] = 1.0; th[4] = 1.0
    RES["pid_theta"] = [float(z) for z in th]
    RES["pid_cost"] = float(r.f)
    return th


# ======================================================================= #
#  3. Load-step response figures                                          #
# ======================================================================= #
def step_response_figs(theta_fopid, theta_pid):
    """Load-disturbance response WITH a physical current limit (|i_ref|<=UMAX)
    and back-calculation anti-windup, so the plotted voltage and current are
    physically realistic (no overflow): see gfi_model.simulate_saturated."""
    Vref = 1.0
    Tend, Tw = 0.30, (0.10, 0.20)
    a = G.simulate_saturated(theta_fopid, T=Tend, Ts=1e-5, load_times=Tw,
                             Iload=0.4, umax=UMAX, Kaw=20.0, N=3)
    b = G.simulate_saturated(theta_pid, T=Tend, Ts=1e-5, load_times=Tw,
                             Iload=0.4, umax=UMAX, Kaw=20.0, N=3)

    # voltage
    plt.figure(figsize=(7, 4.4))
    plt.plot(a["t"], a["v"], label="TF-ABC FOPID")
    plt.plot(b["t"], b["v"], "--", label="PID")
    plt.axhline(Vref, color="k", lw=0.8, ls=":")
    for xx in Tw:
        plt.axvline(xx, color="0.7", lw=0.6, ls=":")
    plt.xlabel("time (s)"); plt.ylabel("capacitor voltage  v_c (pu)")
    plt.title("Voltage response to a 0.4 pu load step (applied 0.1 s, removed 0.2 s)")
    plt.legend(); plt.savefig(os.path.join(FIGDIR, "voltage_step_response.pdf"))
    plt.close()

    # current: raw demand vs saturated reference (+limits), and integral state
    fig, ax = plt.subplots(2, 1, figsize=(7, 5.4), sharex=True)
    ax[0].plot(a["t"], a["i_raw"], color="0.6", lw=1.0,
               label="raw demand $i_{raw}$ (pre-limit)")
    ax[0].plot(a["t"], a["i_ref"], color="C0",
               label="saturated reference $i_{ref}$")
    ax[0].axhline(UMAX, color="r", lw=0.8, ls=":")
    ax[0].axhline(-UMAX, color="r", lw=0.8, ls=":", label="$\\pm%.1f$ pu limit" % UMAX)
    ax[0].set_ylim(-2.2 * UMAX, 2.2 * UMAX)
    ax[0].set_ylabel("current (pu)"); ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title("Current limiting and back-calculation anti-windup (FOPID)")
    ax[1].plot(a["t"], a["i_int"], color="C2", label="integral term $K_i\\,I$")
    for xx in Tw:
        ax[0].axvline(xx, color="0.8", lw=0.6, ls=":")
        ax[1].axvline(xx, color="0.8", lw=0.6, ls=":")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("integral term (pu)")
    ax[1].legend(fontsize=8)
    plt.savefig(os.path.join(FIGDIR, "current_step_response.pdf"))
    plt.close()

    # reference-tracking ISE comparison on the nominal plant (small-signal)
    Gp = plant_tf(PARAMS)
    def step_ise(theta):
        L = tf_mul(fopid_tf(theta, N=NFINAL), Gp)
        if not G._is_stable(L)[0]:
            return np.nan
        tt = np.linspace(0, 0.06, 2000)
        _, y = signal.step(signal.TransferFunction(*tf_feedback(L)), T=tt)
        return float(np.sum((1 - y) ** 2) * (tt[1] - tt[0]))
    ise_f, ise_p = step_ise(theta_fopid), step_ise(theta_pid)
    dev_f, rec_f = load_step_metrics(a["t"], a["v"], Vref, Tw)
    dev_p, rec_p = load_step_metrics(b["t"], b["v"], Vref, Tw)
    RES["step_response"] = dict(
        fopid_max_dev=dev_f, fopid_settle=rec_f,
        pid_max_dev=dev_p, pid_settle=rec_p,
        fopid_iref_peak=float(np.nanmax(np.abs(a["i_ref"]))),
        fopid_iraw_peak=float(np.nanmax(np.abs(a["i_raw"]))), umax=UMAX,
        fopid_ise=float(ise_f), pid_ise=float(ise_p),
        ise_improvement_pct=(100 * (ise_p - ise_f) / ise_p
                             if (ise_p and np.isfinite(ise_p) and
                                 np.isfinite(ise_f)) else float("nan")))


# ======================================================================= #
#  4. Impedance-variation robustness                                      #
# ======================================================================= #
def impedance_fig(theta_fopid, theta_pid):
    """Robustness to grid-impedance variation: tune once on nominal L_g, then
    evaluate the closed-loop step ISE as L_g (hence the LCL resonance) varies.
    Unstable closed loops are recorded as NaN and annotated."""
    Lgs = np.linspace(0.5e-3, 3e-3, 14)
    ise_f, ise_p = [], []
    for Lg in Lgs:
        Gp = plant_tf(PARAMS, Lg)
        for theta, store in ((theta_fopid, ise_f), (theta_pid, ise_p)):
            L = tf_mul(fopid_tf(theta, N=NFINAL), Gp)
            if not G._is_stable(L)[0]:
                store.append(np.nan); continue
            t = np.linspace(0, 0.06, 2000)
            try:
                _, y = signal.step(signal.TransferFunction(*tf_feedback(L)), T=t)
                store.append(float(np.sum((1 - y) ** 2) * (t[1] - t[0]))
                             if np.all(np.isfinite(y)) else np.nan)
            except Exception:
                store.append(np.nan)
    ise_f, ise_p = np.array(ise_f), np.array(ise_p)
    plt.figure(figsize=(6.8, 4.4))
    plt.semilogy(Lgs * 1e3, ise_f, "o-", label="TF-ABC FOPID")
    plt.semilogy(Lgs * 1e3, ise_p, "s--", label="PID")
    # mark PID instability
    bad = np.isnan(ise_p)
    if bad.any():
        for xg in Lgs[bad] * 1e3:
            plt.axvline(xg, color="r", lw=0.6, ls=":", alpha=0.5)
        plt.plot([], [], "r:", label="PID unstable")
    plt.xlabel("grid inductance  L_g (mH)")
    plt.ylabel("closed-loop step ISE")
    plt.title("Robustness to grid-impedance variation")
    plt.legend(); plt.savefig(os.path.join(FIGDIR, "impedance_variation.pdf"))
    plt.close()
    npid_unstable = int(np.sum(np.isnan(ise_p)))
    nfo_unstable = int(np.sum(np.isnan(ise_f)))
    RES["impedance"] = dict(
        Lg_mH=[float(x) for x in Lgs * 1e3],
        ise_fopid=[None if np.isnan(x) else float(x) for x in ise_f],
        ise_pid=[None if np.isnan(x) else float(x) for x in ise_p],
        pid_unstable_count=npid_unstable, fopid_unstable_count=nfo_unstable,
        n_points=len(Lgs))


# ======================================================================= #
#  4b. Loop-level robustness margins across the L_g sweep                  #
# ======================================================================= #
def margins_table(theta_fopid, theta_pid):
    """Frequency-domain robustness across the grid-impedance sweep: sensitivity
    peaks, gain/phase margins, and closed-loop pole real part."""
    rows = []
    for Lg in (0.5, 1.0, 2.0, 3.0):
        m = G.loop_margins(theta_fopid, Lg=Lg * 1e-3, N=NFINAL)
        rows.append(dict(ctrl="FOPID", Lg_mH=Lg, **m))
    mp = G.loop_margins(theta_pid, Lg=1.0e-3, N=NFINAL)
    rows.append(dict(ctrl="PID", Lg_mH=1.0, **mp))
    RES["margins"] = rows
    for r in rows:
        print("    %-5s Lg=%.1f mH  Sinf=%.2f Tinf=%.2f GM=%.1fdB PM=%.0fdeg poleRe=%.3g"
              % (r["ctrl"], r["Lg_mH"], r["sinf"], r["tinf"], r["gm_db"],
                 r["pm_deg"], r["max_pole_re"]))


# ======================================================================= #
#  5. Nonlinear-load THD                                                  #
# ======================================================================= #
def nonlinear_figs(theta_fopid, theta_pid):
    rf = G.thd_under_nonlinear_load(theta_fopid, N=NFINAL)
    rp = G.thd_under_nonlinear_load(theta_pid, N=NFINAL)
    if rf and rp:
        t, vf, dist, thdf, inf = rf
        _, vp, _, thdp, inp = rp
        plt.figure(figsize=(7, 4.2))
        plt.plot(t * 1e3, vf,
                 label="TF-ABC FOPID (THD=%.2f%%, $V_{rms}$=%.3f)" % (thdf, inf["vrms"]))
        plt.plot(t * 1e3, vp, "--",
                 label="PID (THD=%.2f%%, $V_{rms}$=%.3f)" % (thdp, inp["vrms"]))
        plt.xlabel("time (ms)"); plt.ylabel("output voltage (pu)")
        plt.title("Voltage waveform under nonlinear load")
        plt.legend(); plt.savefig(os.path.join(FIGDIR, "nonlinear_voltage.pdf"))
        plt.close()

        plt.figure(figsize=(7, 4.2))
        plt.plot(t * 1e3, dist, color="C3", label="nonlinear load current")
        plt.xlabel("time (ms)"); plt.ylabel("load current (pu)")
        plt.title("Harmonic (rectifier) load current")
        plt.legend(); plt.savefig(os.path.join(FIGDIR, "nonlinear_current.pdf"))
        plt.close()
        RES["thd"] = dict(
            fopid=float(thdf), pid=float(thdp),
            fopid_vrms=inf["vrms"], pid_vrms=inp["vrms"],
            fopid_v1=inf["v1"], pid_v1=inp["v1"],
            fopid_ess=inf["ess"], pid_ess=inp["ess"])


# ======================================================================= #
#  6. Observer estimation figures                                         #
# ======================================================================= #
def observer_figs():
    r = G.run_observer(theta_obs=(0.7, 0.5), noise_std=0.02, seed=3)
    t = r["t"]
    plt.figure(figsize=(7, 4.2))
    plt.plot(t, r["x"][:, 2], label="true  i_g")
    plt.plot(t, r["xh"][:, 2], "--", label="estimated  i_g")
    plt.xlabel("time (s)"); plt.ylabel("grid-side current (pu)")
    plt.title("Tempered fractional observer: grid-current estimation")
    plt.legend(); plt.savefig(os.path.join(FIGDIR, "current_estimation.pdf"))
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(t, r["x"][:, 1], label="true  v_c")
    plt.plot(t, r["xh"][:, 1], "--", label="estimated  v_c")
    plt.plot(t, r["y_meas"][:, 1], color="0.7", lw=0.5, alpha=0.5, label="noisy meas.")
    plt.xlabel("time (s)"); plt.ylabel("capacitor voltage (pu)")
    plt.title("Tempered fractional observer: voltage estimation")
    plt.legend(); plt.savefig(os.path.join(FIGDIR, "voltage_estimation.pdf"))
    plt.close()
    RES["observer"] = dict(mae_iinv=r["mae_iinv"], mae_vc=r["mae_vc"],
                           mae_ig=r["mae_ig"])


# ======================================================================= #
#  7. FOPID Bode plot                                                     #
# ======================================================================= #
def bode_fig(theta_fopid, theta_pid):
    w = np.logspace(0, 5, 500)
    Hf = G.fopid_freqresp(theta_fopid, w, N=NFINAL)
    Hp = G.fopid_freqresp(theta_pid, w, N=NFINAL)
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 6), sharex=True)
    ax[0].semilogx(w, 20 * np.log10(np.abs(Hf)), label="TF-ABC FOPID")
    ax[0].semilogx(w, 20 * np.log10(np.abs(Hp)), "--", label="PID")
    ax[0].set_ylabel("magnitude (dB)"); ax[0].legend()
    ax[0].set_title("FOPID controller Bode plot")
    ax[1].semilogx(w, np.angle(Hf, deg=True), label="TF-ABC FOPID")
    ax[1].semilogx(w, np.angle(Hp, deg=True), "--", label="PID")
    ax[1].set_ylabel("phase (deg)"); ax[1].set_xlabel("frequency (rad/s)")
    plt.savefig(os.path.join(FIGDIR, "fopid_bode.pdf"))
    plt.close()


# ======================================================================= #
def _save():
    with open("gfi_results.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2)


def main():
    import traceback
    t0 = time.time()
    print("[1/7] tuning optimisers ...", flush=True)
    theta_fopid = tune_all(); _save()
    print("[2/7] tuning PID baseline ...", flush=True)
    theta_pid = tune_pid(); _save()
    print("    FOPID theta* =", [round(float(x), 4) for x in theta_fopid], flush=True)
    print("    PID   theta* =", [round(float(x), 4) for x in theta_pid], flush=True)
    steps = [("step response", step_response_figs),
             ("impedance variation", impedance_fig),
             ("loop margins", margins_table),
             ("nonlinear/THD", nonlinear_figs),
             ("observer", lambda a, b: observer_figs()),
             ("bode", bode_fig)]
    for i, (name, fn) in enumerate(steps, start=3):
        print("[%d/7] %s ..." % (i, name), flush=True)
        try:
            fn(theta_fopid, theta_pid)
        except Exception:
            print("[FAIL] %s" % name, flush=True); traceback.print_exc()
        _save()
    print("[done] gfi_results.json written  (%.1fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
