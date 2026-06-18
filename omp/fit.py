"""Closed-form fits to the best-measured directions (deliverable 2).

We fit interpretable formulae to the best-measured weight ratios as functions
of the true parameters:

* **Spin** -- ratio ``c2/c1`` of the best-measured ``c1*chi1z + c2*chi2z``
  combination.  chi_eff predicts ``c2/c1 = q``; we fit a smooth generalisation.
* **Mass** -- ratio ``w2/w1`` of the best-measured ``w1*log m1 + w2*log m2``
  combination.  This interpolates between the chirp-mass gradient ratio
  ``r_chirp(q) = (3+2q)/(2+3q)`` (low total mass) and the total-mass gradient
  ratio ``r_total(q) = q`` (high total mass).  The blend ``s`` is a sigmoid in
  ``log10(Mtot)`` whose midpoint is a fitted transition mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


# --- reference (analytic) weight ratios ------------------------------------

def r_chieff(q):
    """chi_eff spin weight ratio c2/c1."""
    return np.asarray(q, dtype=float)


def r_chirp_logmass(q):
    """Chirp-mass gradient weight ratio w2/w1 in (log m1, log m2)."""
    q = np.asarray(q, dtype=float)
    return (3 + 2 * q) / (2 + 3 * q)


def r_total_logmass(q):
    """Total-mass gradient weight ratio w2/w1 in (log m1, log m2)."""
    return np.asarray(q, dtype=float)


# --- spin fit --------------------------------------------------------------

def _spin_model(q, a, b):
    """Generalised chi_eff: c2/c1 = q * (1 + a (1-q) + b (1-q)^2)."""
    return q * (1.0 + a * (1.0 - q) + b * (1.0 - q) ** 2)


@dataclass
class SpinFit:
    a: float
    b: float
    rms: float
    max_dev_from_chieff: float

    def predict(self, q):
        return _spin_model(np.asarray(q, float), self.a, self.b)

    def formula(self) -> str:
        return (f"c2/c1 = q * (1 + {self.a:+.4f}*(1-q) "
                f"{self.b:+.4f}*(1-q)^2)   [chi_eff: c2/c1 = q]")


def fit_spin(q, ratio) -> SpinFit:
    q = np.asarray(q, float)
    ratio = np.asarray(ratio, float)
    (a, b), _ = curve_fit(_spin_model, q, ratio, p0=[0.0, 0.0])
    pred = _spin_model(q, a, b)
    rms = float(np.sqrt(np.mean((pred - ratio) ** 2)))
    max_dev = float(np.max(np.abs(ratio - r_chieff(q))))
    return SpinFit(a=float(a), b=float(b), rms=rms,
                   max_dev_from_chieff=max_dev)


# --- mass fit --------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mass_model(X, logM0, k):
    """Chirp<->total interpolation. X = (q, Mtot)."""
    q, Mtot = X
    s = _sigmoid(k * (np.log10(Mtot) - logM0))
    return (1.0 - s) * r_chirp_logmass(q) + s * r_total_logmass(q)


@dataclass
class MassFit:
    logM0: float           # log10 transition total mass [Msun]
    k: float               # sharpness of the transition
    rms: float

    @property
    def transition_mass(self) -> float:
        return 10.0 ** self.logM0

    def predict(self, q, Mtot):
        return _mass_model((np.asarray(q, float), np.asarray(Mtot, float)),
                           self.logM0, self.k)

    def formula(self) -> str:
        return (f"w2/w1 = (1-s)*r_chirp(q) + s*r_total(q),  "
                f"s = sigmoid({self.k:.3f}*(log10 Mtot - {self.logM0:.3f})); "
                f"transition Mtot ~ {self.transition_mass:.0f} Msun")


def fit_mass(q, Mtot, ratio) -> MassFit:
    q = np.asarray(q, float)
    Mtot = np.asarray(Mtot, float)
    ratio = np.asarray(ratio, float)
    (logM0, k), _ = curve_fit(
        _mass_model, (q, Mtot), ratio,
        p0=[np.log10(np.median(Mtot)), 3.0],
        maxfev=20000,
    )
    pred = _mass_model((q, Mtot), logM0, k)
    rms = float(np.sqrt(np.mean((pred - ratio) ** 2)))
    return MassFit(logM0=float(logM0), k=float(k), rms=rms)


# --- mass fit, 2D (transition + overshoot bump) ----------------------------
#
# The (Mtot, q) grid (scripts/mass_grid2d.py) shows the surface is a monotonic,
# q-independent chirp->total transition *plus* an overshoot ridge localised in
# log10(Mtot) and growing with (1-q).  We model exactly that:
#
#   w2/w1 = (1-s) r_chirp(q) + s r_total(q)  +  A (1-q) exp[-((x-mu)/w)^2 / 2]
#
# with x = log10(Mtot), s = sigmoid(k (x - logM0)).  The first part is the
# transition (its midpoint logM0 is the ~q-independent transition mass); the
# second is the overshoot bump (amplitude A at q->0, centre mu, width w).

def _mass_model_2d(X, logM0, k, A, mu, w):
    q, Mtot = X
    x = np.log10(Mtot)
    s = _sigmoid(k * (x - logM0))
    base = (1.0 - s) * r_chirp_logmass(q) + s * r_total_logmass(q)
    bump = A * (1.0 - q) * np.exp(-0.5 * ((x - mu) / w) ** 2)
    return base + bump


@dataclass
class MassFit2D:
    logM0: float           # log10 transition total mass [Msun]
    k: float               # transition sharpness
    A: float               # overshoot bump amplitude (at q -> 0)
    mu: float              # log10 bump-centre total mass [Msun]
    w: float               # bump width in log10(Mtot)
    rms: float

    @property
    def transition_mass(self) -> float:
        return 10.0 ** self.logM0

    @property
    def bump_mass(self) -> float:
        return 10.0 ** self.mu

    def predict(self, q, Mtot):
        return _mass_model_2d((np.asarray(q, float), np.asarray(Mtot, float)),
                              self.logM0, self.k, self.A, self.mu, self.w)

    def formula(self) -> str:
        return (
            "w2/w1 = (1-s) r_chirp(q) + s r_total(q) + A (1-q) "
            f"exp[-((x-mu)/w)^2/2],  x=log10 Mtot, "
            f"s=sigmoid({self.k:.2f}(x-{self.logM0:.3f}));\n"
            f"    transition Mtot ~ {self.transition_mass:.0f} Msun, "
            f"bump: A={self.A:.3f} at Mtot ~ {self.bump_mass:.0f} Msun, "
            f"width(log10)={self.w:.3f}"
        )


def fit_mass_2d(q, Mtot, ratio) -> MassFit2D:
    q = np.asarray(q, float).ravel()
    Mtot = np.asarray(Mtot, float).ravel()
    ratio = np.asarray(ratio, float).ravel()
    p0 = [np.log10(225.0), 5.0, 0.6, np.log10(120.0), 0.15]
    bounds = ([1.5, 0.5, 0.0, 1.5, 0.03],
              [3.0, 50.0, 5.0, 3.0, 0.8])
    (logM0, k, A, mu, w), _ = curve_fit(
        _mass_model_2d, (q, Mtot), ratio, p0=p0, bounds=bounds, maxfev=40000,
    )
    pred = _mass_model_2d((q, Mtot), logM0, k, A, mu, w)
    rms = float(np.sqrt(np.mean((pred - ratio) ** 2)))
    return MassFit2D(logM0=float(logM0), k=float(k), A=float(A),
                     mu=float(mu), w=float(w), rms=rms)
