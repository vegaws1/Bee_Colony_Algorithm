"""
tfabc.py
========
Corrected implementation of the Tempered Fractional Artificial Bee Colony
(TF-ABC) optimiser, together with the tempered-fractional-gradient utilities
that the *Main Results* of the revised manuscript reason about, and three fair
baselines (ABC, PSO, GA) for the comparison figures.

The implementation fixes the issues identified in the evaluation of the original
manuscript:

  (1) The search space is the COMPACT box  B = prod_k [lo_k, hi_k]  (every move
      is projected back onto B).  This is the domain of Theorem 2 and makes the
      gradient automatically bounded (no strong-convexity / bounded-gradient
      contradiction on R^d).

  (2) ELITISM: the global best `gbest` never deteriorates.  Together with the
      uniform scout re-initialisation (a per-iteration sampling floor) this is
      exactly the Solis-Wets condition behind the corrected global-convergence
      theorem.

  (3) Correct minimisation signs:
        * fitness  = 1/(1+J)   (lower cost  ->  higher fitness),
        * attraction is TOWARDS the best / neighbour:  (x_best - x_i).

  (4) The "tempered fractional memory" uses the normalised, non-negative weights
        c_j = |w_j(alpha)| e^{-lambda j},   p_j = c_j / sum_j c_j,
      i.e. the velocity memory is a genuine convex combination of past moves -
      the object the alignment lemma (Lemma 2) is proved for.

ASCII-only printing (cp1252 console).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ======================================================================= #
#  Tempered fractional weights  (Lemma 1 / Lemma 2 of the Main Results)    #
# ======================================================================= #
def gl_weights(alpha: float, J: int) -> np.ndarray:
    """Gruenwald-Letnikov coefficients  w_j^{(alpha)} = (-1)^j C(alpha, j),
    j = 0..J, via the stable recursion  w_0 = 1,  w_j = (1-(alpha+1)/j) w_{j-1}.
    """
    w = np.empty(J + 1)
    w[0] = 1.0
    for j in range(1, J + 1):
        w[j] = (1.0 - (alpha + 1.0) / j) * w[j - 1]
    return w


def tempered_weights(alpha: float, lam: float, J: int):
    """Return (c, p, S, M) where
        c_j = |w_j^{(alpha)}| e^{-lambda j},     j = 0..J
        S   = sum_j c_j           (normalising constant of Lemma 2)
        p   = c / S               (probability weights, sum to 1)
        M   = sum_j j p_j         (mean memory depth, finite)
    """
    w = gl_weights(alpha, J)
    j = np.arange(J + 1)
    c = np.abs(w) * np.exp(-lam * j)
    S = float(c.sum())
    p = c / S
    M = float((j * p).sum())
    return c, p, S, M


def jmax_for(lam: float, eps: float = 1e-5) -> int:
    """Memory-truncation depth.  The discarded EXPONENTIAL mass at index j is
    e^{-lambda j}; requiring e^{-lambda J} <= eps gives

        J_max = ceil( ln(1/eps) / lambda ).

    For eps = 1e-5 this is ceil(11.5129/lambda) (NOT 5/lambda; the latter gives
    only e^{-5} ~ 6.7e-3).  This is the corrected Proposition 3.7 bound."""
    return int(np.ceil(np.log(1.0 / eps) / max(lam, 1e-12)))


def tempered_history_gradient(grad_hist, alpha, lam, J=None):
    """Tempered fractional-memory gradient SURROGATE along the optimisation
    trajectory (the causal object the reframed Main Results analyse):

        g_t = sum_{j=0}^{min(t,J)} c_j * grad(theta_{t-j}),

    with c_j = |w_j(alpha)| e^{-lambda j}.  Because it uses only PAST iterates
    theta_{t-j}, every evaluation point lies in the compact search box -- this
    removes the leave-the-domain / choice-of-direction / projection problems of
    a spatial  x - j h u  definition.  `grad_hist` is a list of past gradients,
    NEWEST LAST:  grad_hist[-1]=grad(theta_t), grad_hist[-1-j]=grad(theta_{t-j})."""
    if J is None:
        J = jmax_for(lam)
    c, _, _, _ = tempered_weights(alpha, lam, J)
    L = len(grad_hist)
    K = min(J + 1, L)
    g = np.zeros_like(np.asarray(grad_hist[-1], float))
    for j in range(K):
        g += c[j] * np.asarray(grad_hist[-1 - j], float)
    return g


def tempered_fractional_gradient(grad, x, alpha, lam, h, J=None, direction=None):
    """Spatial-shift tempered gradient g(x)=sum_j c_j grad(x - j h u).  Retained
    ONLY as a numerical probe of the alignment inequality at an interior point;
    the manuscript's analysis itself uses `tempered_history_gradient`."""
    if J is None:
        J = jmax_for(lam)
    x = np.asarray(x, float)
    if direction is None:
        u = np.ones_like(x); u = u / np.linalg.norm(u)
    else:
        u = np.asarray(direction, float)
        u = u / (np.linalg.norm(u) + 1e-15)
    c, _, _, _ = tempered_weights(alpha, lam, J)
    g = np.zeros_like(x, dtype=float)
    for j in range(J + 1):
        g += c[j] * grad(x - j * h * u)
    return g


# ======================================================================= #
#  Result container                                                       #
# ======================================================================= #
@dataclass
class OptResult:
    x: np.ndarray
    f: float
    history: np.ndarray          # best-so-far cost per iteration (elitist)
    nfev: int
    name: str = ""
    extra: dict = field(default_factory=dict)


# ======================================================================= #
#  TF-ABC optimiser                                                       #
# ======================================================================= #
class TFABC:
    """Tempered Fractional Artificial Bee Colony optimiser (minimisation)."""

    def __init__(self, func, lo, hi, SN=40, alpha=0.7, lam=0.3,
                 eta0=0.6, beta=0.6, limit=None, c1=1.5, c2=1.0,
                 noise0=0.3, mom0=0.35, scout_floor=0.01, seed=0):
        self.func = func
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.d = self.lo.size
        self.SN = SN
        self.alpha = alpha
        self.lam = lam
        self.eta0 = eta0
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.noise0 = noise0
        self.mom0 = mom0                         # tempered-momentum strength
        self.scout_floor = scout_floor          # per-iter uniform-sampling floor
        self.limit = limit if limit is not None else max(10, SN * self.d // 2)
        self.rng = np.random.default_rng(seed)
        self.J = jmax_for(lam)
        _, self.p, self.S, self.M = tempered_weights(alpha, lam, self.J)

    # -- helpers ------------------------------------------------------- #
    def _clip(self, X):
        return np.clip(X, self.lo, self.hi)

    def _rand_pop(self, n):
        return self.lo + (self.hi - self.lo) * self.rng.random((n, self.d))

    def _memory_term(self, hist):
        """Tempered fractional memory: convex combination of the last moves,
        sum_j p_j * Delta x^{t-j}.  `hist` is a deque-like list (newest last)."""
        if not hist:
            return 0.0
        L = len(hist)
        K = min(self.J + 1, L)
        # newest move gets weight p_0, etc.
        acc = np.zeros((self.SN, self.d))
        for j in range(K):
            acc += self.p[j] * hist[-1 - j]
        return acc

    # -- main loop ----------------------------------------------------- #
    def optimize(self, Tmax=200):
        rng = self.rng
        d, SN = self.d, self.SN
        X = self._rand_pop(SN)
        f = self.func(X)
        nfev = SN
        trials = np.zeros(SN, int)
        moves = []                                  # accepted Delta x history
        gi = int(np.argmin(f))
        gbest, gval = X[gi].copy(), float(f[gi])
        hist = np.empty(Tmax)

        ar = np.arange(SN)
        for t in range(Tmax):
            frac = t / max(Tmax - 1, 1)
            mom = self.mom0 * (1.0 - frac)                   # tempered momentum decays

            # ---------- Employed bee phase: ABC single-dimension search
            #            enhanced with the tempered fractional momentum -------- #
            mem = self._memory_term(moves)
            K = (ar + rng.integers(1, SN, SN)) % SN          # neighbour != i
            jdim = rng.integers(0, d, SN)
            phi = rng.uniform(-1.0, 1.0, SN)
            Xnew = X.copy()
            Xnew[ar, jdim] += phi * (X[ar, jdim] - X[K, jdim])
            Xnew = self._clip(Xnew + mom * mem)              # fractional momentum
            fnew = self.func(Xnew)
            nfev += SN
            improved = fnew < f
            dX = np.where(improved[:, None], Xnew - X, 0.0)
            X = np.where(improved[:, None], Xnew, X)
            f = np.where(improved, fnew, f)
            trials = np.where(improved, 0, trials + 1)
            moves.append(dX)
            if len(moves) > self.J + 1:
                moves.pop(0)

            # ---------- Onlooker bee phase: roulette + single-dimension search,
            #            with gentle elite guidance for exploitation -------- #
            fitness = 1.0 / (1.0 + (f - f.min()))           # minimisation-correct
            prob = fitness / fitness.sum()
            sel = rng.choice(SN, size=SN, p=prob)
            K2 = (sel + rng.integers(1, SN, SN)) % SN
            jd = rng.integers(0, d, SN)
            phi2 = rng.uniform(-1.0, 1.0, SN)
            cand = X[sel].copy()
            cand[ar, jd] += phi2 * (X[sel, jd] - X[K2, jd])
            cand = self._clip(cand + 0.08 * rng.random((SN, 1)) * (gbest - X[sel]))
            fcand = self.func(cand)
            nfev += SN
            better = fcand < f[sel]
            idx = sel[better]
            X[idx] = cand[better]
            f[idx] = fcand[better]
            trials[idx] = 0

            # ---------- Scout bee phase (abandonment + sampling floor) ---------- #
            scouts = np.where(trials > self.limit)[0]
            extra = np.where(rng.random(SN) < self.scout_floor)[0]  # Solis-Wets floor
            scouts = np.unique(np.concatenate([scouts, extra]).astype(int))
            if scouts.size:
                X[scouts] = self._rand_pop(scouts.size)
                f[scouts] = self.func(X[scouts])
                nfev += scouts.size
                trials[scouts] = 0
                for Mv in moves:                 # reset per-individual memory:
                    Mv[scouts] = 0.0             # a scouted bee starts a new trajectory

            # ---------- Elitism ---------- #
            gi = int(np.argmin(f))
            if f[gi] < gval:
                gbest, gval = X[gi].copy(), float(f[gi])
            hist[t] = gval

        return OptResult(gbest, gval, hist, nfev, "TF-ABC",
                         extra=dict(S=self.S, M=self.M, J=self.J))


# ======================================================================= #
#  Baselines:  standard ABC, PSO, GA                                       #
# ======================================================================= #
class ABC:
    def __init__(self, func, lo, hi, SN=40, limit=None, seed=0):
        self.func, self.lo, self.hi = func, np.asarray(lo, float), np.asarray(hi, float)
        self.d = self.lo.size
        self.SN = SN
        self.limit = limit if limit is not None else max(10, SN * self.d // 2)
        self.rng = np.random.default_rng(seed)

    def _clip(self, X):
        return np.clip(X, self.lo, self.hi)

    def optimize(self, Tmax=200):
        rng = self.rng
        X = self.lo + (self.hi - self.lo) * rng.random((self.SN, self.d))
        f = self.func(X)
        nfev = self.SN
        trials = np.zeros(self.SN, int)
        gi = int(np.argmin(f)); gbest, gval = X[gi].copy(), float(f[gi])
        hist = np.empty(Tmax)
        for t in range(Tmax):
            # employed
            K = rng.integers(0, self.SN, self.SN)
            jdim = rng.integers(0, self.d, self.SN)
            phi = rng.uniform(-1, 1, self.SN)
            Xnew = X.copy()
            Xnew[np.arange(self.SN), jdim] += phi * (X[np.arange(self.SN), jdim]
                                                     - X[K, jdim])
            Xnew = self._clip(Xnew)
            fnew = self.func(Xnew); nfev += self.SN
            imp = fnew < f
            X[imp], f[imp], trials[imp] = Xnew[imp], fnew[imp], 0
            trials[~imp] += 1
            # onlooker
            fit = 1.0 / (1.0 + (f - f.min())); prob = fit / fit.sum()
            sel = rng.choice(self.SN, self.SN, p=prob)
            K2 = rng.integers(0, self.SN, self.SN)
            jd = rng.integers(0, self.d, self.SN)
            phi2 = rng.uniform(-1, 1, self.SN)
            cand = X[sel].copy()
            cand[np.arange(self.SN), jd] += phi2 * (X[sel, jd] - X[K2, jd])
            cand = self._clip(cand)
            fc = self.func(cand); nfev += self.SN
            bet = fc < f[sel]; idx = sel[bet]
            X[idx], f[idx], trials[idx] = cand[bet], fc[bet], 0
            # scout
            sc = np.where(trials > self.limit)[0]
            if sc.size:
                X[sc] = self.lo + (self.hi - self.lo) * rng.random((sc.size, self.d))
                f[sc] = self.func(X[sc]); nfev += sc.size; trials[sc] = 0
            gi = int(np.argmin(f))
            if f[gi] < gval: gbest, gval = X[gi].copy(), float(f[gi])
            hist[t] = gval
        return OptResult(gbest, gval, hist, nfev, "ABC")


class PSO:
    def __init__(self, func, lo, hi, SN=40, w=0.72, c1=1.49, c2=1.49, seed=0):
        self.func, self.lo, self.hi = func, np.asarray(lo, float), np.asarray(hi, float)
        self.d = self.lo.size; self.SN = SN
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = np.random.default_rng(seed)

    def optimize(self, Tmax=200):
        rng = self.rng
        span = self.hi - self.lo
        X = self.lo + span * rng.random((self.SN, self.d))
        V = 0.1 * span * rng.standard_normal((self.SN, self.d))
        f = self.func(X); nfev = self.SN
        P, Pf = X.copy(), f.copy()
        gi = int(np.argmin(f)); g, gval = X[gi].copy(), float(f[gi])
        hist = np.empty(Tmax)
        for t in range(Tmax):
            r1 = rng.random((self.SN, self.d)); r2 = rng.random((self.SN, self.d))
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (g - X)
            X = np.clip(X + V, self.lo, self.hi)
            f = self.func(X); nfev += self.SN
            imp = f < Pf; P[imp], Pf[imp] = X[imp], f[imp]
            gi = int(np.argmin(Pf))
            if Pf[gi] < gval: g, gval = P[gi].copy(), float(Pf[gi])
            hist[t] = gval
        return OptResult(g, gval, hist, nfev, "PSO")


class GA:
    def __init__(self, func, lo, hi, SN=40, pc=0.9, pm=0.1, eta=15.0, seed=0):
        self.func, self.lo, self.hi = func, np.asarray(lo, float), np.asarray(hi, float)
        self.d = self.lo.size; self.SN = SN
        self.pc, self.pm, self.eta = pc, pm, eta
        self.rng = np.random.default_rng(seed)

    def _tournament(self, f):
        a = self.rng.integers(0, self.SN, self.SN)
        b = self.rng.integers(0, self.SN, self.SN)
        return np.where(f[a] < f[b], a, b)

    def optimize(self, Tmax=200):
        rng = self.rng
        span = self.hi - self.lo
        X = self.lo + span * rng.random((self.SN, self.d))
        f = self.func(X); nfev = self.SN
        gi = int(np.argmin(f)); g, gval = X[gi].copy(), float(f[gi])
        hist = np.empty(Tmax)
        for t in range(Tmax):
            par = X[self._tournament(f)]
            rng.shuffle(par)
            child = par.copy()
            for i in range(0, self.SN - 1, 2):
                if rng.random() < self.pc:                  # SBX crossover
                    u = rng.random(self.d)
                    bq = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta + 1)),
                                  (1 / (2 * (1 - u))) ** (1 / (self.eta + 1)))
                    p1, p2 = par[i], par[i + 1]
                    child[i] = 0.5 * ((1 + bq) * p1 + (1 - bq) * p2)
                    child[i + 1] = 0.5 * ((1 - bq) * p1 + (1 + bq) * p2)
            mask = rng.random((self.SN, self.d)) < self.pm  # Gaussian mutation
            child += mask * (0.1 * span * rng.standard_normal((self.SN, self.d)))
            child = np.clip(child, self.lo, self.hi)
            fc = self.func(child); nfev += self.SN
            # elitist replacement
            allX = np.vstack([X, child]); allf = np.concatenate([f, fc])
            order = np.argsort(allf)[:self.SN]
            X, f = allX[order], allf[order]
            if f[0] < gval: g, gval = X[0].copy(), float(f[0])
            hist[t] = gval
        return OptResult(g, gval, hist, nfev, "GA")


OPTIMIZERS = {"TF-ABC": TFABC, "ABC": ABC, "PSO": PSO, "GA": GA}


if __name__ == "__main__":
    from benchmarks import get_benchmarks
    b = get_benchmarks(10)["Rastrigin"]
    lo, hi = b.box()
    r = TFABC(b, lo, hi, seed=1).optimize(200)
    print("TF-ABC on Rastrigin-10: best = %.4e  (nfev=%d, S=%.3f, M=%.3f)"
          % (r.f, r.nfev, r.extra["S"], r.extra["M"]))
