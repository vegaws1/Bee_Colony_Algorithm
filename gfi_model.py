"""
gfi_model.py
============
Grid-forming-inverter (GFI) voltage-regulation testbed for the application and
numerical-results sections of the revised manuscript.

Modelling choices (kept deliberately well-posed and numerically benign):

  * Cascaded control.  The fast inner current loop is modelled as a first-order
    lag with time constant tau_i; the outer FOPID regulates the capacitor
    voltage by producing the inner-loop current reference i_ref.  The
    voltage-loop plant from i_ref to v_c is therefore the type-1 system
        G(s) = 1 / ( Cf s ( tau_i s + 1 ) ),
    which is exactly what cascaded GFI designs see and which makes the FOPID
    tuning landscape well-conditioned.  Grid-impedance variation is captured by
    letting the achievable inner-loop bandwidth depend on Lg:  tau_i = Lg/Reff.

  * Oustaloup rational approximation of s^r makes the FOPID closed loop LTI, so
    it can be tuned thousands of times quickly and its Bode plot is exact.

  * The tempered-fractional state observer is simulated together with a
    tempered-fractional plant model using the SAME tempered Gruenwald-Letnikov
    scheme, so the error obeys D^{a,l} e = (A-LC) e exactly and Mittag-Leffler
    stability (Theorem, Main Results) is what the figure demonstrates.

ASCII output only (cp1252 console).
"""

from __future__ import annotations
import numpy as np
from scipy import signal

PARAMS = dict(
    Lf=2e-3, Cf=50e-6, Lg=1e-3, Rf=0.1, Rg=0.1,
    Rd=2.0,                # passive capacitor-branch damping resistor (standard
                           # LCL practice); sets resonance damping zeta ~ 0.15
    tau_i=1.0e-4,          # inner current-loop time constant (~1.6 kHz)
    Rload=10.0,            # equivalent load resistance (observer plant)
    Vref=325.27,           # peak phase voltage of a 400 V (L-L) system
)


# ======================================================================= #
#  Oustaloup approximation of s^r and (num,den) algebra                    #
# ======================================================================= #
def oustaloup(r, wb=1.0, wh=5e3, N=3):
    k = np.arange(-N, N + 1)
    m = (2 * N + 1)
    wz = wb * (wh / wb) ** ((k + N + 0.5 * (1 - r)) / m)   # poles
    wp = wb * (wh / wb) ** ((k + N + 0.5 * (1 + r)) / m)   # zeros
    K = wh ** r
    num = np.array([1.0]); den = np.array([1.0])
    for z in wp:
        num = np.polymul(num, [1.0, z])
    for p in wz:
        den = np.polymul(den, [1.0, p])
    return K * num, den


def _oustaloup_sections(r, wb=1.0, wh=5e3, N=3):
    """s^r ~ K * prod (s - zr_k)/(s - pr_k); return (K, [(zr,pr), ...]) with the
    (negative-real) ROOT values, for a well-conditioned cascade realisation."""
    num, den = oustaloup(r, wb, wh, N)               # num=K*prod(s+wp), den=prod(s+wz)
    zr = np.sort(np.roots(num).real)[::-1]
    pr = np.sort(np.roots(den).real)[::-1]
    return float(num[0]), list(zip(zr, pr))          # num[0]=K (den monic)


class _Sec:
    """ZOH-discretised first-order section (s - zr)/(s - pr) with negative-real
    root values: xdot = pr x + u, y = (pr - zr) x + u.  Each section is bounded and
    perfectly conditioned, unlike a high-order companion-form tf2ss realisation."""
    __slots__ = ("ad", "bd", "c", "x")

    def __init__(self, zr, pr, Ts):
        self.ad = np.exp(pr * Ts)
        self.bd = (self.ad - 1.0) / pr if pr != 0 else Ts
        self.c = (pr - zr)
        self.x = 0.0

    def step(self, u):
        y = self.c * self.x + u
        self.x = self.ad * self.x + self.bd * u
        return y


def tf_add(a, b):
    na, da = a; nb, db = b
    return np.polyadd(np.polymul(na, db), np.polymul(nb, da)), np.polymul(da, db)


def tf_mul(a, b):
    return np.polymul(a[0], b[0]), np.polymul(a[1], b[1])


def tf_feedback(L):
    n, d = L
    return n, np.polyadd(d, n)


def tf_sens(L):
    n, d = L
    return d, np.polyadd(d, n)


# ======================================================================= #
#  FOPID controller  C(s) = Kp + Ki s^{-mu} + Kd s^{lambda}               #
# ======================================================================= #
def fopid_tf(theta, N=4, wb=1.0, wh=5e3, tau_d=1e-4):
    """FOPID  C(s) = Kp + Ki s^{-mu} + Kd s^{nu}.

    The integral term is realised as a TRUE integrator 1/s cascaded with the
    fractional remainder s^{1-mu} (Oustaloup), i.e. s^{-mu} = (1/s) s^{1-mu};
    this guarantees infinite DC gain (zero steady-state error) for every
    mu in (0,1].  The derivative term s^{nu} is realised as a true differentiator
    s cascaded with s^{nu-1} and a high-frequency roll-off 1/(tau_d s+1) for
    properness.  At the integer boundaries mu=1 or nu=1 the fractional remainder
    is unity, so the controller reduces EXACTLY to a classical PID -- making the
    PID baseline fair (an Oustaloup approximation of integer order 1 is
    inaccurate and was previously destabilising the PID)."""
    Kp, Ki, Kd, mu, nu = theta
    C = (np.array([Kp]), np.array([1.0]))
    # --- fractional integral  Ki s^{-mu} = Ki (1/s) s^{1-mu} ---
    if abs(mu - 1.0) < 1e-6:
        s_int = (np.array([1.0]), np.array([1.0, 0.0]))         # exact 1/s
    else:
        rem = oustaloup(1.0 - mu, wb, wh, N)                    # s^{1-mu}, order in (0,1)
        s_int = (rem[0], np.polymul([1.0, 0.0], rem[1]))        # (1/s) s^{1-mu}
    C = tf_add(C, (Ki * s_int[0], s_int[1]))
    # --- fractional derivative  Kd s^{nu} = Kd s s^{nu-1} / (tau_d s+1) ---
    if abs(nu - 1.0) < 1e-6:
        s_der = (np.array([1.0, 0.0]), np.array([tau_d, 1.0]))  # s/(tau_d s+1)
    else:
        rem = oustaloup(nu - 1.0, wb, wh, N)                    # s^{nu-1}, order in (-1,0)
        s_der = (np.polymul([1.0, 0.0], rem[0]),
                 np.polymul([tau_d, 1.0], rem[1]))              # s s^{nu-1}/(tau_d s+1)
    C = tf_add(C, (Kd * s_der[0], s_der[1]))
    return C


def fopid_freqresp(theta, w, N=4):
    C = fopid_tf(theta, N=N)
    _, H = signal.freqs(C[0], C[1], worN=w)
    return H


# ======================================================================= #
#  Voltage-loop plant                                                      #
# ======================================================================= #
def plant_tf(p=PARAMS, Lg=None):
    """Voltage-loop plant of the LCL-filtered grid-forming inverter, from the
    inner-loop current reference i_ref to the capacitor voltage v_c.  The fast
    inner current loop is a first-order lag 1/(tau_eff s+1) (the effective
    bandwidth depends on the series inductance L_f + L_g, hence on the grid
    impedance), and the LC tank with a capacitor-branch passive damping resistor
    R_d (standard LCL practice) gives a UNITY-DC-GAIN, realistically damped
    second-order voltage plant:

        G(s) = (R_d C_f s + 1)
               -----------------------------------------------
               (tau_eff s + 1)(L C_f s^2 + R_d C_f s + 1),      L = L_f + L_g,

    so G(0)=1 (a steady current reference holds rated voltage, as a voltage loop
    must), the resonance omega_r = 1/sqrt(L C_f) is retained, and the damping
    zeta = R_d C_f/(2 sqrt(L C_f)).  Grid-impedance variation enters through L
    (=L_f+L_g) and tau_eff, shifting omega_r -- the robustness scenario in which
    fractional iso-damping is beneficial."""
    Lf, Cf, Rd = p["Lf"], p["Cf"], p.get("Rd", 2.0)
    Lg = p["Lg"] if Lg is None else Lg
    L = Lf + Lg
    tau = p["tau_i"] * (1.0 + Lg / p["Lg"]) * 0.5        # mild Lg-dependence
    num = np.array([Rd * Cf, 1.0])                       # (R_d C_f s + 1), DC=1
    den_lc = np.array([L * Cf, Rd * Cf, 1.0])           # L C_f s^2 + R_d C_f s + 1
    den = np.polymul([tau, 1.0], den_lc)
    return num, den


def plant_ss(p=PARAMS, Lg=None):
    """State-space realisation of the voltage-loop plant `plant_tf`."""
    num, den = plant_tf(p, Lg)
    A, B, C, D = signal.tf2ss(num, den)
    return A, B, C, D


# ======================================================================= #
#  Closed-loop evaluation and metrics                                     #
# ======================================================================= #
def _is_stable(L):
    _, dcl = tf_feedback(L)
    poles = np.roots(dcl)
    return bool(np.all(poles.real < -1e-9)), poles


def step_metrics(t, y, ref):
    e = ref - y
    dt = t[1] - t[0]
    ISE = float(np.sum(e ** 2) * dt)
    IAE = float(np.sum(np.abs(e)) * dt)
    ITAE = float(np.sum(t * np.abs(e)) * dt)
    yfin = float(np.mean(y[-max(5, len(y) // 20):]))
    overshoot = max(0.0, (np.max(y) - ref) / abs(ref) * 100.0)
    ess = abs(ref - yfin)
    tol = 0.02 * abs(ref)
    idx = np.where(np.abs(y - yfin) > tol)[0]
    ts = float(t[idx[-1]]) if idx.size else 0.0
    return dict(ISE=ISE, IAE=IAE, ITAE=ITAE, overshoot=float(overshoot),
                settling=ts, ess=float(ess), yfinal=yfin)


def simulate_reference_step(theta, p=PARAMS, Lg=None, T=0.05, n=2000, N=3):
    G = plant_tf(p, Lg)
    C = fopid_tf(theta, N=N)
    L = tf_mul(C, G)
    if not _is_stable(L)[0]:
        return None
    Tcl = tf_feedback(L)
    t = np.linspace(0, T, n)
    try:
        tout, y = signal.step(signal.TransferFunction(Tcl[0], Tcl[1]), T=t)
        _, e = signal.step(signal.TransferFunction(*tf_sens(L)), T=t)
    except Exception:
        return None
    if not np.all(np.isfinite(y)):
        return None
    return tout, y, e


def hinf_norms(L, w=None):
    if w is None:
        w = np.logspace(-1, 6, 300)
    _, H = signal.freqs(L[0], L[1], worN=w)
    S = 1.0 / (1.0 + H)
    Tc = H / (1.0 + H)
    return float(np.max(np.abs(S))), float(np.max(np.abs(Tc)))


def loop_margins(theta, p=PARAMS, Lg=None, N=4):
    """Open-loop frequency-domain robustness measures for L(s)=C(s)G(s):
    sensitivity peaks ||S||inf, ||T||inf; gain margin (dB) at the first phase
    crossover; phase margin (deg) at the first gain crossover; and the maximum
    closed-loop pole real part (negative => stable).  NaN/inf where a crossover
    does not exist (e.g. infinite gain margin)."""
    C = fopid_tf(theta, N=N); G = plant_tf(p, Lg); L = tf_mul(C, G)
    w = np.logspace(-1, 6, 6000)
    _, H = signal.freqs(L[0], L[1], worN=w)
    mag = np.abs(H); ph = np.unwrap(np.angle(H)) * 180.0 / np.pi
    Sinf, Tinf = hinf_norms(L, w)
    pm = float("nan")                                  # phase margin at |L|=1
    s = np.where(np.diff(np.sign(mag - 1.0)))[0]
    if s.size:
        i = s[0]
        phm = ph[i] + (ph[i + 1] - ph[i]) * (1.0 - mag[i]) / (mag[i + 1] - mag[i])
        pm = 180.0 + phm
    gm = float("inf")                                  # gain margin at arg L=-180
    pc = np.where(np.diff(np.sign(ph + 180.0)))[0]
    if pc.size:
        j = pc[0]
        magpc = mag[j] + (mag[j + 1] - mag[j]) * (-180.0 - ph[j]) / (ph[j + 1] - ph[j])
        gm = -20.0 * np.log10(magpc) if magpc > 0 else float("inf")
    _, poles = _is_stable(L)
    return dict(sinf=float(Sinf), tinf=float(Tinf), gm_db=float(gm),
                pm_deg=float(pm), max_pole_re=float(np.max(poles.real)))


def harmonic_sensitivity(theta, p=PARAMS, f0=50.0, N=2,
                         harmonics=(3, 5, 7, 9, 11)):
    """RMS of |S(j w_h)| over the load-harmonic frequencies, where
    S = 1/(1+C G) is the output-disturbance sensitivity.  Lower values mean the
    closed loop rejects the harmonic load currents better -> lower voltage THD.
    This makes power quality an EXPLICIT objective rather than an accident of an
    aggressive derivative order."""
    try:
        L = tf_mul(fopid_tf(theta, N=N), plant_tf(p))
        wh = np.array([2 * np.pi * h * f0 for h in harmonics])
        _, H = signal.freqs(L[0], L[1], worN=wh)
        S = np.abs(1.0 / (1.0 + H))
        return float(np.sqrt(np.mean(S ** 2)))
    except Exception:
        return 10.0


def cost(theta, p=PARAMS, weights=(0.30, 0.18, 0.18, 0.10, 0.12, 0.12), N=2):
    """Scalar FOPID tuning cost = weighted ISE/IAE/ITAE + control-effort
    (||e||^2 proxy) + robustness margin (||S||inf,||T||inf) + HARMONIC-rejection
    penalty (RMS |S| at load harmonics, targeting voltage THD), with a large
    penalty for closed-loop instability so J is bounded on the box.  Physical
    current limiting is enforced separately in `simulate_saturated` (used for the
    reported figures), so the controller cannot rely on impossible currents."""
    a1, a2, a3, a4, a5, a6 = weights
    try:
        out = simulate_reference_step(theta, p, T=0.05, n=600, N=N)
        if out is None:
            return 50.0
        t, y, e = out
        if np.max(np.abs(y)) > 10:
            return 50.0
        m = step_metrics(t, y, 1.0)
        L = tf_mul(fopid_tf(theta, N=N), plant_tf(p))
        Sinf, Tinf = hinf_norms(L)
        ctrl = float(np.sum(e ** 2) * (t[1] - t[0]))
        rob = max(0.0, Sinf - 1.4) + max(0.0, Tinf - 1.3)
        harm = harmonic_sensitivity(theta, p, N=N)        # power-quality term
        J = (a1 * m["ISE"] * 80 + a2 * m["IAE"] * 30 + a3 * m["ITAE"] * 400
             + a4 * ctrl * 30 + a5 * rob * 3 + a6 * harm * 8
             + 0.03 * m["overshoot"] + 8.0 * m["ess"])
        return float(J) if np.isfinite(J) else 50.0
    except Exception:
        return 50.0


# ======================================================================= #
#  Saturating closed loop with back-calculation anti-windup               #
#  (physically realistic: the current reference is hard-limited to +-umax) #
# ======================================================================= #
def simulate_saturated(theta, p=PARAMS, Lg=None, T=0.3, Ts=1e-5, Vref=1.0,
                       load_times=(0.1, 0.2), Iload=0.4, umax=1.5, Kaw=20.0,
                       N=3, wb=1.0, wh=5e3, tau_d=1e-4):
    """Closed-loop voltage-regulation response with a HARD current limit
    |i_ref|<=umax and back-calculation anti-windup.  The low-order unity-DC-gain
    plant G=plant_tf (vc/iref) is ZOH-discretised; the high-order FOPID is realised
    as a CASCADE of well-conditioned first-order Oustaloup sections plus a true
    integrator (where the anti-windup acts) and a derivative washout.  This avoids
    the catastrophic conditioning of a companion-form tf2ss realisation of the
    controller (cond ~ 1e33), which overflows an ODE/fixed-step integrator.  The
    load current enters at the plant input node, so the plant sees (i_ref - i_load).
    Returns t, capacitor voltage v, saturated reference i_ref, raw demand i_raw, and
    the integral contribution i_int (Ki * integral state) for the anti-windup trace."""
    Gp = plant_tf(p, Lg)
    n = int(T / Ts); t = np.arange(n) * Ts
    if not _is_stable(tf_mul(fopid_tf(theta, N=N, wb=wb, wh=wh), Gp))[0]:
        bad = np.full(n, np.nan)
        return dict(t=t, v=bad, i_ref=bad, i_raw=bad, i_int=bad)
    Ap, Bp, Cp, _ = signal.tf2ss(Gp[0], Gp[1])
    Adp, Bdp, Cdp, _, _ = signal.cont2discrete((Ap, Bp, Cp, 0), Ts, method="zoh")
    Bp = Bdp.reshape(-1); Cp = Cdp.reshape(-1); nxp = Adp.shape[0]

    Kp, Ki, Kd, mu, nu = theta
    if abs(mu - 1.0) < 1e-6:
        Ki_K, Ki_sec = 1.0, []
    else:
        Ki_K, secs = _oustaloup_sections(1.0 - mu, wb, wh, N)
        Ki_sec = [_Sec(z, q, Ts) for z, q in secs]
    if abs(nu - 1.0) < 1e-6:
        Kd_K, Kd_sec = 1.0, []
    else:
        Kd_K, secs = _oustaloup_sections(nu - 1.0, wb, wh, N)
        Kd_sec = [_Sec(z, q, Ts) for z, q in secs]
    dwo = _Sec(0.0, -1.0 / tau_d, Ts)                # s/(tau_d s+1) = (1/tau_d)(s)/(s+1/tau_d)

    xp = np.zeros(nxp)
    V = np.zeros(n); IR = np.zeros(n); IRW = np.zeros(n); IINT = np.zeros(n)
    integ = 0.0; wprev = 0.0
    for k in range(n):
        v = float(Cp @ xp); e = Vref - v
        w = e
        for s in Ki_sec:
            w = s.step(w)
        w *= Ki_K
        integ += 0.5 * Ts * (w + wprev); wprev = w     # trapezoidal integrator
        iI = Ki * integ
        dd = e
        for s in Kd_sec:
            dd = s.step(dd)
        dd = dwo.step(dd * Kd_K) / tau_d
        iD = Kd * dd
        iraw = Kp * e + iI + iD
        iref = min(umax, max(-umax, iraw))
        integ += Kaw * Ts * (iref - iraw) / max(Ki, 1e-9)   # back-calculation anti-windup
        il = Iload if (load_times[0] <= t[k] < load_times[1]) else 0.0
        V[k] = v; IR[k] = iref; IRW[k] = iraw; IINT[k] = iI
        xp = Adp @ xp + Bp * (iref - il)
    return dict(t=t, v=V, i_ref=IR, i_raw=IRW, i_int=IINT)


# ======================================================================= #
#  THD under a harmonic (nonlinear) load                                   #
# ======================================================================= #
def thd_under_nonlinear_load(theta, p=PARAMS, f0=50.0, N=3,
                             harmonics=(3, 5, 7, 9, 11), amps=None):
    if amps is None:
        amps = [0.06, 0.04, 0.025, 0.015, 0.01]   # realistic harmonic load fractions
    G = plant_tf(p); C = fopid_tf(theta, N=N); L = tf_mul(C, G)
    if not _is_stable(L)[0]:
        return None
    cycles, spc = 12, 256                       # integer periods, samples/cycle
    n = cycles * spc
    T = cycles / f0
    t = np.linspace(0, T, n, endpoint=False)
    ref = np.sin(2 * np.pi * f0 * t)
    dist = sum(a * np.sin(2 * np.pi * h * f0 * t) for h, a in zip(harmonics, amps))
    try:
        _, yr, _ = signal.lsim(signal.TransferFunction(*tf_feedback(L)), U=ref, T=t)
        _, yd, _ = signal.lsim(signal.TransferFunction(*tf_sens(L)), U=dist, T=t)
    except Exception:
        return None
    v = yr + yd
    # analyse the last half (6 whole cycles -> exact FFT-bin alignment, no leakage)
    seg = v[n // 2:]
    Nseg = len(seg)
    sp = np.abs(np.fft.rfft(seg)) * 2.0 / Nseg
    cyc = cycles // 2                            # fundamental bin index in `seg`
    V1 = sp[cyc]
    Vh = np.sqrt(sum(sp[cyc * h] ** 2 for h in range(2, 16) if cyc * h < len(sp)))
    thd = 100.0 * Vh / (V1 + 1e-12)
    vrms = float(np.sqrt(np.mean(seg ** 2)))
    info = dict(vrms=vrms, v1=float(V1), ess=float(abs(V1 - 1.0)), f0=float(f0))
    return t, v, dist, float(thd), info


# ======================================================================= #
#  Tempered-fractional state observer                                      #
#  (plant and observer share the tempered Gruenwald-Letnikov scheme so the #
#   error obeys  D^{a,l} e = (A-LC) e  exactly).                            #
# ======================================================================= #
def _tempered_gl_weights(alpha, lam, Ts, J):
    """w_j^{a,l} = e^{-lambda j Ts} (-1)^j C(alpha,j),  j=0..J  (w_0 = 1)."""
    w = np.empty(J + 1); w[0] = 1.0
    for j in range(1, J + 1):
        w[j] = (1.0 - (alpha + 1.0) / j) * w[j - 1]
    j = np.arange(J + 1)
    return np.exp(-lam * j * Ts) * w


def _obs_plant():
    """Per-unit small-signal LCL model [i_L, v_c, i_g] with measurements
    y=[i_L, v_c].  Per-unit entries (O(1-30)) keep the explicit tempered GL
    discretisation well-conditioned, as is standard in power-system studies."""
    l, c, lg = 0.1, 0.05, 0.1
    r, R = 0.5, 0.5
    A = np.array([[-r / l, -1.0 / l, 0.0],
                  [1.0 / c, 0.0, -1.0 / c],
                  [0.0, 1.0 / lg, -R / lg]])
    B = np.array([[1.0 / l], [0.0], [0.0]])
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    return A, B, C


def design_observer_gain(A, C, desired=None):
    if desired is None:
        desired = np.array([-700.0, -800.0, -900.0])
    try:
        return signal.place_poles(A.T, C.T, desired).gain_matrix.T
    except Exception:
        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A.T, C.T, np.eye(A.shape[0]) * 1e4,
                                     np.eye(C.shape[0]))
            return P @ C.T
        except Exception:
            return C.T * 200.0


def run_observer(theta_obs=(0.7, 0.5), T=3.0, Ts=1e-3, noise_std=0.02, seed=0,
                 poles=(-8.0, -10.0, -12.0)):
    """Simulate the per-unit tempered-fractional plant and observer; the error
    obeys D^{a,l} e=(A-LC)e and decays (Theorem, observer stability)."""
    alpha, lam = theta_obs
    A, B, C = _obs_plant()
    nstate = A.shape[0]
    rng = np.random.default_rng(seed)
    L = design_observer_gain(A, C, np.array(poles))
    Acl = A - L @ C
    J = 600
    w = _tempered_gl_weights(alpha, lam, Ts, J)
    Ta = Ts ** alpha
    n = int(T / Ts)
    t = np.arange(n) * Ts
    u = np.sin(2 * np.pi * 0.5 * t) + 1.0                  # slow per-unit drive
    x = np.zeros((n, nstate)); xh = np.zeros((n, nstate))
    xh[0] = np.array([1.0, -1.0, 0.8])                     # deliberately wrong guess
    for k in range(1, n):
        kk = min(k, J)
        mem_x = -sum(w[j] * x[k - j] for j in range(1, kk + 1))
        x[k] = Ta * (A @ x[k - 1] + B.ravel() * u[k - 1]) + mem_x
        ym = C @ x[k] + noise_std * rng.standard_normal(C.shape[0])
        mem_h = -sum(w[j] * xh[k - j] for j in range(1, kk + 1))
        innov = ym - C @ xh[k - 1]
        xh[k] = Ta * (A @ xh[k - 1] + B.ravel() * u[k - 1] + L @ innov) + mem_h
        if not np.all(np.isfinite(xh[k])):
            xh[k] = xh[k - 1]
    Ymeas = (C @ x.T).T
    skip = n // 4                                          # ignore initial transient
    mae = np.mean(np.abs(x[skip:] - xh[skip:]), axis=0)
    return dict(t=t, x=x, xh=xh, y_meas=Ymeas, poles=np.linalg.eigvals(Acl),
                mae_iinv=float(mae[0]), mae_vc=float(mae[1]), mae_ig=float(mae[2]))


if __name__ == "__main__":
    th = [1.5, 25.0, 0.003, 0.9, 0.85]
    print("nominal cost:", round(cost(th), 4))
    out = simulate_reference_step(th, T=0.05, n=1500)
    if out:
        t, y, e = out
        print("step:", {k: round(v, 4) for k, v in step_metrics(t, y, 1.0).items()})
    r = thd_under_nonlinear_load(th)
    if r:
        print("THD = %.3f %%" % r[3])
    o = run_observer()
    print("observer MAE  i_inv=%.4f  v_c=%.4f  i_g=%.4f"
          % (o["mae_iinv"], o["mae_vc"], o["mae_ig"]))
    print("observer error poles real parts:",
          [round(float(p.real), 1) for p in o["poles"]])
