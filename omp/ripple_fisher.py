"""Autodiff Fisher matrices from ripple (``ripplegw``) JAX waveforms.

gwfast's ``FisherMatr`` only consumes gwfast waveform objects, so to use
ripple's JAX waveforms -- which include the *precessing* models gwfast's build
lacks -- we build the Fisher directly from the waveform derivatives:

    F_ij = sum_d 4 Re int (d_i h)^* (d_j h) / S_n^d df,

with the derivatives obtained by ``jax.jacfwd`` (forward-mode AD, clean and
low-noise, exactly the property we need for smooth eigenvectors).  PSDs and the
detector list are reused from :mod:`omp.detectors`.

The class mirrors :class:`omp.fisher.FisherWorkspace`'s public surface
(``par_order`` + ``fisher(src, ...)`` returning ``(Npar, Npar, Nevents)``) so the
existing :mod:`omp.eigen` pipeline runs on it unchanged.

This first cut covers the *aligned-spin* generators (``IMRPhenomD``,
``IMRPhenomXAS``) for cross-checking against gwfast.  The mass slots are
``(log m1, log m2)`` directly: we parameterise the waveform call in
``(log m1, log m2)`` and let AD carry the chain rule through ``(Mc, eta)``, so no
explicit Jacobian is needed (cf. the manual one in :class:`FisherWorkspace`).

Detector projection uses fixed antenna factors ``(fplus, fcross)``: the
marginalised *intrinsic* eigen-directions (the c2/c1 spin and w2/w1 mass ratios)
are set by the waveform phase structure and are insensitive to the overall
antenna response, which is all the cross-check needs.
"""

from __future__ import annotations

import importlib

from . import compat  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from . import detectors as _det
from .params import SourceParams, redshift_to_dL_Gpc


# Aligned-spin hphc generators take
#   params = [Mc, eta, chi1z, chi2z, dL_Mpc, tc, phic, iota].
_WF_ALIGNED = ("IMRPhenomD", "IMRPhenomXAS")
# Precessing hphc generators take the full Cartesian spin vector at f_ref:
#   params = [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, tc, phiRef, incl].
_WF_PRECESSING = ("IMRPhenomPv2",)

# Physical parameter vectors this engine differentiates w.r.t.: intrinsic
# (log-mass + spins) plus the dominant nuisances we marginalise over.
RIPPLE_PAR_ORDER_ALIGNED = [
    "logm1", "logm2", "chi1z", "chi2z", "dL", "tcoal", "Phicoal", "iota",
]
RIPPLE_PAR_ORDER_PREC = [
    "logm1", "logm2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z",
    "dL", "tcoal", "Phicoal", "iota",
]


def _load_psd(path):
    """Load an ASD file -> (freqs, S_n = ASD**2) as jnp arrays."""
    data = np.loadtxt(path)
    f, asd = data[:, 0], data[:, 1]
    return jnp.asarray(f), jnp.asarray(asd ** 2)


class RippleFisherWorkspace:
    def __init__(self, network="L1_design", waveform="IMRPhenomD",
                 fmin=10.0, fmax=2048.0, npts=6000,
                 fplus=0.7, fcross=0.7):
        if waveform in _WF_ALIGNED:
            self.precessing = False
            self.par_order = list(RIPPLE_PAR_ORDER_ALIGNED)
        elif waveform in _WF_PRECESSING:
            self.precessing = True
            self.par_order = list(RIPPLE_PAR_ORDER_PREC)
        else:
            raise KeyError(f"unknown/unsupported waveform {waveform!r}; "
                           f"available: {_WF_ALIGNED + _WF_PRECESSING}")
        if network not in _det.NETWORKS:
            raise KeyError(f"unknown network {network!r}; "
                           f"available: {sorted(_det.NETWORKS)}")
        self.network = network
        self.waveform = waveform
        self.fmin = fmin
        self.fmax = fmax
        self.fplus = fplus
        self.fcross = fcross

        mod = importlib.import_module(f"ripplegw.waveforms.{waveform}")
        self._hphc = getattr(mod, f"gen_{waveform}_hphc")
        self.f = jnp.linspace(fmin, fmax, npts)
        self.f_ref = fmin
        self._psds = [_load_psd(p) for _, p in _det.NETWORKS[network]]

        # Compile the vmapped Fisher once.
        self._fisher_batch = jax.jit(jax.vmap(self._fisher_one))

    # --- waveform / inner product ------------------------------------------

    def _strain(self, phys):
        """Detector strain h(f) for one physical parameter vector."""
        logm1, logm2 = phys[0], phys[1]
        m1 = jnp.exp(logm1)
        m2 = jnp.exp(logm2)
        Mtot = m1 + m2
        Mc = (m1 * m2) ** 0.6 / Mtot ** 0.2
        eta = m1 * m2 / Mtot ** 2
        # Both layouts are [Mc, eta, <spins...>, dL, tc, phic, iota]: the engine
        # just swaps (logm1, logm2) -> (Mc, eta) and passes the rest through.
        # Aligned: spins = (chi1z, chi2z); precessing: (s1x,s1y,s1z,s2x,s2y,s2z).
        rip = jnp.concatenate([jnp.array([Mc, eta]), phys[2:]])
        hp, hc = self._hphc(self.f, rip, self.f_ref)
        return self.fplus * hp + self.fcross * hc

    def _fisher_one(self, phys):
        """Fisher matrix (Npar, Npar) for one event, summed over detectors."""
        # Forward-mode Jacobian of the complex strain w.r.t. real params.
        J = jax.jacfwd(self._strain)(phys)          # (Nf, Npar), complex
        npar = phys.shape[0]
        F = jnp.zeros((npar, npar))
        for fa, Sa in self._psds:
            S = jnp.interp(self.f, fa, Sa, left=jnp.inf, right=jnp.inf)
            w = jnp.where(jnp.isfinite(S) & (S > 0), 1.0 / S, 0.0)
            integ = (jnp.conj(J)[:, :, None] * J[:, None, :]
                     * w[:, None, None])
            F = F + 4.0 * jnp.real(jnp.trapezoid(integ, self.f, axis=0))
        return F

    # --- public API (mirrors FisherWorkspace) ------------------------------

    def _phys_array(self, src: SourceParams) -> jnp.ndarray:
        """Build the (Nevents, Npar) physical-parameter array from a source."""
        m1_det, m2_det = src.detector_component_masses()
        dL_Mpc = redshift_to_dL_Gpc(src.z) * 1.0e3
        if self.precessing:
            s1x, s1y, s1z, s2x, s2y, s2z = src.spins_cartesian()
            spins = [s1x, s1y, s1z, s2x, s2y, s2z]
        else:
            spins = [src.chi1z, src.chi2z]
        cols = [np.log(m1_det), np.log(m2_det), *spins,
                dL_Mpc, src.tcoal, src.Phicoal, src.iota]
        return jnp.asarray(np.stack(np.broadcast_arrays(*cols), axis=1))

    def fisher(self, src: SourceParams, batch_size=None) -> np.ndarray:
        """Return Fisher matrices with shape ``(Npar, Npar, Nevents)``.

        Parameter order is :attr:`par_order`; mass slots are ``(log m1, log m2)``
        in the detector frame, matching the gwfast ``logm1logm2`` convention.
        """
        phys = self._phys_array(src)
        nev = phys.shape[0]
        if batch_size is None or batch_size >= nev:
            F = np.asarray(self._fisher_batch(phys))          # (Nev, Np, Np)
        else:
            blocks = [np.asarray(self._fisher_batch(phys[s:s + batch_size]))
                      for s in range(0, nev, batch_size)]
            F = np.concatenate(blocks, axis=0)
        return np.transpose(F, (1, 2, 0))                     # (Np, Np, Nev)

    def index(self, name: str) -> int:
        return self.par_order.index(name)
