#!/usr/bin/env python
"""Generate the eigenvector-field figures for notes/eigenvector_fields.md.

Produces, into notes/figs/:
  aligned_fields.png       -- chi_eff (spin) and chirp->total (mass) weight ratios
  prec_spectrum.png        -- spin measurability spectrum vs tilt (sloppy structure)
  prec_field.png           -- chi_eff & chi_p eigenvector components vs tilt
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import omp  # noqa: F401  (compat shim before gwfast)
from omp.params import SourceParams
from omp.fisher import FisherWorkspace
from omp.ripple_fisher import RippleFisherWorkspace
from omp.eigen import analyze_subspace, enforce_continuity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notes", "figs")
os.makedirs(OUT, exist_ok=True)

SPIN6 = ["s1x", "s1y", "s1z", "s2x", "s2y", "s2z"]
COMP_LABELS = [r"$s_{1x}$", r"$s_{1y}$", r"$s_{1z}$",
               r"$s_{2x}$", r"$s_{2y}$", r"$s_{2z}$"]


def aligned_fields():
    """Aligned chi_eff (spin) and chirp->total (mass) weight ratios."""
    gw = FisherWorkspace(waveform="IMRPhenomD", mass_coords="logm1logm2")

    # Spin: c2/c1 vs q at fixed mass.
    qg = np.linspace(0.25, 0.99, 16)
    src = SourceParams(m1=30.0, q=qg, z=0.1, chi1z=0.3, chi2z=0.1)
    F = gw.fisher(src, res=2000, batch_size=8)
    spin = analyze_subspace(F, gw.par_order, ["chi1z", "chi2z"],
                            sign_ref=[1.0, 1.0])
    spin = enforce_continuity(spin)
    c2c1 = np.array([r.best[1] / r.best[0] for r in spin])

    # Mass: w2/w1 vs Mtot at fixed q.
    Mt = np.geomspace(15.0, 400.0, 16)
    m1g = Mt / (1.0 + 0.6)
    src2 = SourceParams(m1=m1g, q=0.6, z=0.1, chi1z=0.0, chi2z=0.0)
    F2 = gw.fisher(src2, res=2000, batch_size=8)
    mass = analyze_subspace(F2, gw.par_order, ["logm1", "logm2"],
                            sign_ref=[1.0, 1.0])
    mass = enforce_continuity(mass)
    w2w1 = np.array([r.best[1] / r.best[0] for r in mass])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(qg, c2c1, "o-", label="best-measured (numerical)")
    a1.plot(qg, qg, "k--", label=r"$\chi_{\rm eff}$:  $c_2/c_1=q$")
    a1.set_xlabel("mass ratio $q$"); a1.set_ylabel(r"$c_2/c_1$")
    a1.set_title("Spin: best-measured direction (aligned)"); a1.legend()

    a2.plot(Mt, w2w1, "o-", label="best-measured (numerical)")
    a2.axhline((3 + 2 * 0.6) / (2 + 3 * 0.6), ls="--", color="C1",
               label=r"chirp $(3+2q)/(2+3q)$")
    a2.axhline(0.6, ls=":", color="C2", label=r"total  $q$")
    a2.set_xscale("log"); a2.set_xlabel(r"$M_{\rm tot}$ [$M_\odot$]")
    a2.set_ylabel(r"$w_2/w_1$ (log-mass)")
    a2.set_title("Mass: chirp $\\to$ total ($q=0.6$)"); a2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/aligned_fields.png", dpi=130)
    print("saved aligned_fields.png")


def _prec_grid(tilts, q=0.6, a1=0.6, a2=0.5, phi1=0.3, phi2=1.1):
    rp = RippleFisherWorkspace(waveform="IMRPhenomPv2", fmin=20.0, fmax=1024.0)
    src = SourceParams.from_spherical(
        m1=30.0, q=q, z=0.1, a1=a1, a2=a2,
        tilt1=tilts, tilt2=tilts, phi1=phi1, phi2=phi2)
    F = rp.fisher(src, batch_size=8)
    res = analyze_subspace(F, rp.par_order, SPIN6,
                           method="marginalize_fisher", sign_ref=[0, 0, 1, 0, 0, 1])
    return enforce_continuity(res)


def prec_spectrum():
    """Measurability (1/variance) of the 6 spin modes vs tilt -- sloppy spectrum."""
    tilts = np.deg2rad(np.linspace(5, 85, 18))
    res = _prec_grid(tilts)
    meas = np.array([1.0 / r.variances for r in res])  # (Ntilt, 6)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    names = ["mode 0 (chi_eff)", "mode 1 (chi_p-like)", "mode 2", "mode 3",
             "mode 4", "mode 5"]
    for j in range(6):
        ax.semilogy(np.rad2deg(tilts), np.clip(meas[:, j], 1e-3, None),
                    "o-", ms=3, label=names[j])
    ax.set_xlabel("spin tilt [deg]")
    ax.set_ylabel("measurability  $1/\\sigma^2_{\\rm whitened}$  (log)")
    ax.set_title("Spin sub-space eigen-spectrum (precessing, $q=0.6$)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/prec_spectrum.png", dpi=130)
    print("saved prec_spectrum.png")


def prec_field():
    """chi_eff (mode 0) and chi_p (mode 1) eigenvector components vs tilt."""
    tilts = np.deg2rad(np.linspace(5, 85, 18))
    res = _prec_grid(tilts)
    td = np.rad2deg(tilts)
    m0 = np.array([r.coeffs[:, 0] for r in res])  # (Ntilt, 6)
    m1 = np.array([r.coeffs[:, 1] for r in res])

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for j in range(6):
        a0.plot(td, m0[:, j], "-", label=COMP_LABELS[j])
        a1.plot(td, m1[:, j], "-", label=COMP_LABELS[j])
    a0.set_title("Best mode (generalised $\\chi_{\\rm eff}$)")
    a1.set_title("2nd mode (generalised $\\chi_p$)")
    for a in (a0, a1):
        a.set_xlabel("spin tilt [deg]"); a.axhline(0, color="k", lw=0.5)
    a0.set_ylabel("eigenvector component")
    a1.legend(ncol=2, fontsize=8, title="component")
    fig.tight_layout(); fig.savefig(f"{OUT}/prec_field.png", dpi=130)
    print("saved prec_field.png")


if __name__ == "__main__":
    aligned_fields()
    prec_spectrum()
    prec_field()
    print("all figures written to", OUT)
