"""Thin wrapper around gwfast's Fisher-matrix machinery.

We do not re-implement Fisher matrices: ``GWSignal.FisherMatr`` /
``DetNet.FisherMatr`` already build them from JAX/analytic waveform derivatives
and the detector PSDs.  This module just selects the waveform + network and
returns the Fisher matrix with explicit parameter labels.
"""

from __future__ import annotations

from . import compat  # noqa: F401

import numpy as np

from . import detectors as _det
from .params import PAR_ORDER, SourceParams


_WF_FACTORIES = {
    "IMRPhenomD": lambda: _import_wf("IMRPhenomD"),
    "IMRPhenomHM": lambda: _import_wf("IMRPhenomHM"),
    "TaylorF2": lambda: _import_wf("TaylorF2_RestrictedPN"),
}


def _import_wf(name):
    import gwfast.waveforms as wf
    return getattr(wf, name)()


class FisherWorkspace:
    """Builds a detector network once and computes Fisher matrices on it.

    Parameters
    ----------
    network : str
        Key into :data:`omp.detectors.NETWORKS` (e.g. ``"L1_design"``).
    waveform : str
        One of the aligned-spin models in :data:`_WF_FACTORIES`.
    fmin, fmax : float
        Integration band (Hz).
    """

    def __init__(self, network="L1_design", waveform="IMRPhenomD",
                 fmin=10.0, fmax=2048.0, use_earth_motion=False, verbose=False,
                 mass_coords="Mceta"):
        if waveform not in _WF_FACTORIES:
            raise KeyError(f"unknown waveform {waveform!r}; "
                           f"available: {sorted(_WF_FACTORIES)}")
        if mass_coords not in ("Mceta", "m1m2", "logm1logm2"):
            raise ValueError(f"unknown mass_coords {mass_coords!r}")
        self.network = network
        self.waveform = waveform
        self.fmin = fmin
        self.fmax = fmax
        self.mass_coords = mass_coords
        self.par_order = self._par_order_for(mass_coords)
        self.net = _det.build_network(
            network, _WF_FACTORIES[waveform],
            fmin=fmin, fmax=fmax, use_earth_motion=use_earth_motion,
            verbose=verbose,
        )

    @staticmethod
    def _par_order_for(mass_coords):
        order = list(PAR_ORDER)
        if mass_coords == "m1m2":
            order[0:2] = ["m1", "m2"]
        elif mass_coords == "logm1logm2":
            order[0:2] = ["logm1", "logm2"]
        return order

    def snr(self, src: SourceParams, res=2000) -> np.ndarray:
        return np.asarray(self.net.SNR(src.to_gwfast(), res=res))

    def fisher(self, src: SourceParams, res=2000, df=None,
               batch_size=None) -> np.ndarray:
        """Return Fisher matrices with shape ``(Npar, Npar, Nevents)``.

        Parameter order is :attr:`par_order`.  For ``mass_coords='logm1logm2'``
        the mass slots are transformed from gwfast's detector-frame ``(m1, m2)``
        to ``(log m1, log m2)`` via the diagonal Jacobian ``F_log = D F D`` with
        ``D = diag(m1, m2, 1, ...)``.

        ``batch_size`` chunks the events to bound GPU memory (the JAX vmap over
        events allocates ~O(Nevents * Nfreq) at once).
        """
        ev = src.to_gwfast()
        use_m1m2 = self.mass_coords in ("m1m2", "logm1logm2")
        nev = len(np.atleast_1d(ev["dL"]))

        if batch_size is None or batch_size >= nev:
            F = np.asarray(self.net.FisherMatr(
                ev, res=res, df=df,
                use_m1m2=use_m1m2, use_chi1chi2=True, use_prec_ang=False,
            ))
        else:
            blocks = []
            for s in range(0, nev, batch_size):
                sl = slice(s, s + batch_size)
                ev_b = {k: np.atleast_1d(v)[sl] for k, v in ev.items()}
                blocks.append(np.asarray(self.net.FisherMatr(
                    ev_b, res=res, df=df,
                    use_m1m2=use_m1m2, use_chi1chi2=True, use_prec_ang=False,
                )))
            F = np.concatenate(blocks, axis=2)

        if self.mass_coords == "logm1logm2":
            m1_det, m2_det = src.detector_component_masses()
            npar = F.shape[0]
            jac = np.ones((npar, F.shape[2]))
            jac[0, :] = m1_det
            jac[1, :] = m2_det
            D = jac[:, None, :] * jac[None, :, :]   # outer product per event
            F = F * D
        return F

    def index(self, name: str) -> int:
        return self.par_order.index(name)

    def indices(self, names) -> list[int]:
        return [self.index(n) for n in names]
