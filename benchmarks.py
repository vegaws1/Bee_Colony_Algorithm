"""
benchmarks.py
=============
Standard global-optimisation test functions used to *validate the theory* of the
revised manuscript (global convergence of the elitist TF-ABC optimiser and the
local convergence rate of the tempered-fractional refinement).

Every function is wrapped in a small dataclass that records:
    * the search box  [lo, hi]^d        (the COMPACT domain used in Theorem 2),
    * the global minimiser  x*  and value  f*  (known in closed form),
    * whether the landscape is multimodal (needed to make the global-search
      result non-trivial).

All functions are written for vectorised evaluation:  f(X) with X of shape
(n, d) returns an array of n objective values.

ASCII-only output by design (this Windows console is cp1252).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Benchmark:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    lo: float
    hi: float
    dim: int
    x_star: np.ndarray
    f_star: float
    multimodal: bool

    def box(self):
        """Return (lower, upper) bound vectors of length dim."""
        return (np.full(self.dim, self.lo), np.full(self.dim, self.hi))

    def __call__(self, X):
        return self.f(np.atleast_2d(X))


# --------------------------------------------------------------------------- #
#  Convex / unimodal reference (used for the LOCAL rate experiments)          #
# --------------------------------------------------------------------------- #
def sphere(X):
    return np.sum(X ** 2, axis=1)


def make_quadratic(d, cond=30.0, seed=0):
    """A strongly convex quadratic f(x)=0.5 (x-x*)^T Q (x-x*) with prescribed
    condition number `cond`.  mu = lambda_min(Q), L = lambda_max(Q).  Used to
    validate the descent lemma and the local linear-to-neighbourhood rate."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    U, _ = np.linalg.qr(A)
    eig = np.linspace(1.0, cond, d)            # mu = 1, L = cond
    Q = U @ np.diag(eig) @ U.T
    Q = 0.5 * (Q + Q.T)
    x_star = rng.uniform(-2, 2, size=d)
    mu, L = float(eig.min()), float(eig.max())

    def f(X):
        D = X - x_star
        return 0.5 * np.einsum('ni,ij,nj->n', D, Q, D)

    def grad(x):
        return Q @ (x - x_star)

    return f, grad, Q, x_star, mu, L


# --------------------------------------------------------------------------- #
#  Multimodal landscapes (used for the GLOBAL convergence experiments)        #
# --------------------------------------------------------------------------- #
def rastrigin(X):
    A = 10.0
    d = X.shape[1]
    return A * d + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)


def ackley(X):
    d = X.shape[1]
    s1 = np.sum(X ** 2, axis=1)
    s2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    return (-20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
            - np.exp(s2 / d) + 20.0 + np.e)


def griewank(X):
    d = X.shape[1]
    s = np.sum(X ** 2, axis=1) / 4000.0
    i = np.arange(1, d + 1)
    p = np.prod(np.cos(X / np.sqrt(i)), axis=1)
    return s - p + 1.0


def schwefel(X):
    d = X.shape[1]
    return 418.9828872724338 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
                  + (1.0 - X[:, :-1]) ** 2, axis=1)


def get_benchmarks(dim=10):
    """Return the standard multimodal suite at a given dimension."""
    d = dim
    bs = [
        Benchmark("Sphere", sphere, -5.12, 5.12, d, np.zeros(d), 0.0, False),
        Benchmark("Rastrigin", rastrigin, -5.12, 5.12, d, np.zeros(d), 0.0, True),
        Benchmark("Ackley", ackley, -32.768, 32.768, d, np.zeros(d), 0.0, True),
        Benchmark("Griewank", griewank, -600.0, 600.0, d, np.zeros(d), 0.0, True),
        Benchmark("Schwefel", schwefel, -500.0, 500.0, d,
                  np.full(d, 420.9687), 0.0, True),
        Benchmark("Rosenbrock", rosenbrock, -5.0, 10.0, d, np.ones(d), 0.0, False),
    ]
    return {b.name: b for b in bs}


if __name__ == "__main__":
    # quick self-check at the known optimum (ASCII output only)
    for name, b in get_benchmarks(5).items():
        val = float(b(b.x_star.reshape(1, -1))[0])
        print("%-12s f(x*) = %+.6e  (target %.1e)" % (name, val, b.f_star))
