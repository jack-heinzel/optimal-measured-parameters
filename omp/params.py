"""Source-parameter bookkeeping and conversion to gwfast's event dictionary.

gwfast's aligned-spin waveforms use the parameter vector

    [Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z]

with ``Mc`` the *detector-frame* chirp mass (Msun), ``eta`` the symmetric mass
ratio, ``dL`` the luminosity distance (Gpc), sky angles ``theta, phi`` (rad),
inclination ``iota``, polarisation ``psi``, coalescence time ``tcoal`` (in
fractions of a day in gwfast's convention) and phase ``Phicoal``, and aligned
spin components ``chi1z, chi2z``.

This module converts from the physical parameters we reason about -- source-frame
primary mass ``m1``, mass ratio ``q = m2/m1 <= 1``, redshift ``z``, and aligned
spins -- into that dictionary, handling the ``(1+z)`` mass redshifting and the
redshift->luminosity-distance map via an astropy cosmology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical gwfast aligned-spin parameter order.
PAR_ORDER = [
    "Mc", "eta", "dL", "theta", "phi", "iota", "psi",
    "tcoal", "Phicoal", "chi1z", "chi2z",
]

# Intrinsic sub-spaces we study (indices into PAR_ORDER).
SPIN_PARS = ["chi1z", "chi2z"]
MASS_PARS = ["Mc", "eta"]


def _cosmo():
    from astropy.cosmology import Planck18
    return Planck18


# --- mass-coordinate conversions -------------------------------------------

def m1q_to_m1m2(m1, q):
    """(primary mass, q=m2/m1) -> (m1, m2)."""
    m1 = np.asarray(m1, dtype=float)
    q = np.asarray(q, dtype=float)
    return m1, q * m1


def m1m2_to_Mceta(m1, m2):
    """Component masses -> (chirp mass, symmetric mass ratio)."""
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    Mtot = m1 + m2
    eta = (m1 * m2) / Mtot**2
    Mc = Mtot * eta ** 0.6
    return Mc, eta


def redshift_to_dL_Gpc(z):
    """Redshift -> luminosity distance in Gpc (Planck18)."""
    z = np.asarray(z, dtype=float)
    dL_Mpc = _cosmo().luminosity_distance(z).value
    return dL_Mpc / 1e3


@dataclass
class SourceParams:
    """A (vectorised) set of true source parameters in physical coordinates.

    Masses are *source-frame*; ``z`` redshifts them to the detector frame and
    sets the luminosity distance.  Scalars are broadcast to length-1 arrays.
    """

    m1: np.ndarray            # source-frame primary mass [Msun]
    q: np.ndarray             # mass ratio m2/m1 in (0, 1]
    z: np.ndarray             # redshift
    chi1z: np.ndarray         # aligned spin of body 1
    chi2z: np.ndarray         # aligned spin of body 2
    iota: np.ndarray = field(default_factory=lambda: np.array([np.pi / 3]))
    theta: np.ndarray = field(default_factory=lambda: np.array([np.pi / 3]))
    phi: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    psi: np.ndarray = field(default_factory=lambda: np.array([0.2]))
    tcoal: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    Phicoal: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def __post_init__(self):
        arrs = {k: np.atleast_1d(np.asarray(getattr(self, k), dtype=float))
                for k in self.__dataclass_fields__}
        n = max(a.size for a in arrs.values())
        for k, a in arrs.items():
            if a.size == 1 and n > 1:
                a = np.full(n, a.item())
            elif a.size != n:
                raise ValueError(f"length mismatch for {k!r}: {a.size} vs {n}")
            setattr(self, k, a)

    def detector_component_masses(self):
        """Detector-frame component masses ``(m1_det, m2_det)`` [Msun].

        Needed for the (log m1, log m2) Jacobian, since gwfast differentiates
        w.r.t. detector-frame masses.
        """
        m1s, m2s = m1q_to_m1m2(self.m1, self.q)
        return m1s * (1.0 + self.z), m2s * (1.0 + self.z)

    def to_gwfast(self) -> dict:
        """Return the gwfast event dictionary (detector-frame Mc, dL in Gpc)."""
        m1s, m2s = m1q_to_m1m2(self.m1, self.q)
        # Redshift masses to the detector frame before forming Mc.
        m1_det = m1s * (1.0 + self.z)
        m2_det = m2s * (1.0 + self.z)
        Mc, eta = m1m2_to_Mceta(m1_det, m2_det)
        dL = redshift_to_dL_Gpc(self.z)
        return {
            "Mc": Mc, "eta": eta, "dL": dL,
            "theta": self.theta, "phi": self.phi, "iota": self.iota,
            "psi": self.psi, "tcoal": self.tcoal, "Phicoal": self.Phicoal,
            "chi1z": self.chi1z, "chi2z": self.chi2z,
        }


def chi_eff(chi1z, chi2z, q):
    """Effective aligned spin: (chi1z + q*chi2z)/(1+q).

    The reference 'best-measured spin' our eigen-analysis should generalise.
    """
    q = np.asarray(q, dtype=float)
    return (np.asarray(chi1z) + q * np.asarray(chi2z)) / (1.0 + q)
