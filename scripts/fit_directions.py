#!/usr/bin/env python
"""Fit closed-form formulae to the best-measured spin and mass directions.

Builds a 2D grid over (mass ratio q, primary mass m1) with a single vectorised
Fisher evaluation, extracts the best-measured spin and mass weight ratios, fits
interpretable models (see ``omp.fit``), and writes a summary + diagnostic plot.

Example
-------
    python scripts/fit_directions.py --network L1_design --nq 13 --nm 9
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omp.params import SourceParams                       # noqa: E402
from omp.fisher import FisherWorkspace                    # noqa: E402
from omp.eigen import analyze_subspace                    # noqa: E402
from omp import fit as omf                                # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", default="L1_design")
    p.add_argument("--waveform", default="IMRPhenomD")
    p.add_argument("--nq", type=int, default=13)
    p.add_argument("--qmin", type=float, default=0.2)
    p.add_argument("--qmax", type=float, default=1.0)
    p.add_argument("--nm", type=int, default=9)
    p.add_argument("--m1min", type=float, default=8.0)
    p.add_argument("--m1max", type=float, default=300.0)
    p.add_argument("--z", type=float, default=0.1)
    p.add_argument("--chi1z", type=float, default=0.2)
    p.add_argument("--chi2z", type=float, default=0.0)
    p.add_argument("--res", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16,
                   help="events per Fisher batch (bounds GPU memory)")
    p.add_argument("--out", default="data/fit.png")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    qg = np.linspace(args.qmin, args.qmax, args.nq)
    m1g = np.geomspace(args.m1min, args.m1max, args.nm)
    Q, M1 = np.meshgrid(qg, m1g, indexing="ij")
    q_flat, m1_flat = Q.ravel(), M1.ravel()

    src = SourceParams(m1=m1_flat, q=q_flat, z=args.z,
                       chi1z=args.chi1z, chi2z=args.chi2z)
    # Marginalised spin covariance is invariant to the mass coordinate choice,
    # so one Fisher in (log m1, log m2) serves both sub-spaces.
    ws = FisherWorkspace(network=args.network, waveform=args.waveform,
                         mass_coords="logm1logm2")
    F = ws.fisher(src, res=args.res, batch_size=args.batch)  # (Npar,Npar,Ngrid)

    spin_ratio = np.empty(q_flat.size)
    mass_ratio = np.empty(q_flat.size)
    for i in range(q_flat.size):
        sp = analyze_subspace(F[..., i:i+1], ws.par_order, ["chi1z", "chi2z"],
                              method="marginalize", whiten="subcov",
                              sign_ref=[1.0, q_flat[i]])[0].best
        ms = analyze_subspace(F[..., i:i+1], ws.par_order, ["logm1", "logm2"],
                              method="marginalize", whiten="subcov",
                              sign_ref=[1.0, 1.0])[0].best
        spin_ratio[i] = sp[1] / sp[0]
        mass_ratio[i] = ms[1] / ms[0]

    Mtot = m1_flat * (1.0 + q_flat)

    spin_fit = omf.fit_spin(q_flat, spin_ratio)
    mass_fit = omf.fit_mass(q_flat, Mtot, mass_ratio)

    print("=== SPIN (generalised chi_eff) ===")
    print(" ", spin_fit.formula())
    print(f"  fit RMS = {spin_fit.rms:.4e}, "
          f"max|data - chi_eff| = {spin_fit.max_dev_from_chieff:.4f}")
    print("=== MASS (chirp <-> total) ===")
    print(" ", mass_fit.formula())
    print(f"  fit RMS = {mass_fit.rms:.4e}")

    _plot(args.out, qg, m1g, q_flat, m1_flat, Mtot,
          spin_ratio, mass_ratio, spin_fit, mass_fit)


def _plot(out, qg, m1g, q_flat, m1_flat, Mtot,
          spin_ratio, mass_ratio, spin_fit, mass_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Spin: data points (light mass dependence) + fit + chi_eff
    ax1.scatter(q_flat, spin_ratio, s=12, alpha=0.5, label="data (all m1)")
    qq = np.linspace(qg.min(), qg.max(), 100)
    ax1.plot(qq, spin_fit.predict(qq), "r-", label="fit")
    ax1.plot(qq, omf.r_chieff(qq), "k--", label=r"$\chi_{\rm eff}$: $c_2/c_1=q$")
    ax1.set_xlabel("mass ratio $q$")
    ax1.set_ylabel(r"$c_2/c_1$")
    ax1.set_title("Spin: best-measured direction")
    ax1.legend()

    # Mass: ratio vs Mtot, coloured by q; fit curves at a few q
    sc = ax2.scatter(Mtot, mass_ratio, c=q_flat, s=14, cmap="viridis",
                     label="data")
    for q0 in [qg.min(), 0.5 * (qg.min() + qg.max()), qg.max()]:
        MM = np.geomspace(Mtot.min(), Mtot.max(), 100)
        ax2.plot(MM, mass_fit.predict(np.full_like(MM, q0), MM), "-",
                 lw=1.2, alpha=0.8)
    ax2.axvline(mass_fit.transition_mass, color="grey", ls=":",
                label=f"transition $\\sim${mass_fit.transition_mass:.0f} $M_\\odot$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"total mass $M_{\rm tot}$ [$M_\odot$]")
    ax2.set_ylabel(r"$w_2/w_1$ (log-mass)")
    ax2.set_title("Mass: chirp $\\to$ total")
    fig.colorbar(sc, ax=ax2, label="$q$")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
