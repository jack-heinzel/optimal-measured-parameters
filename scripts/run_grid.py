#!/usr/bin/env python
"""Compute best-measured intrinsic-parameter directions over a parameter grid.

For each grid point this evaluates the full Fisher matrix (gwfast) and extracts
the best-measured directions in the spin sub-space ``(chi1z, chi2z)`` -- the
generalisation of chi_eff -- and the mass sub-space ``(Mc, eta)`` -- the
generalisation of chirp mass.  Results are saved to an ``.npz`` for fitting/
plotting (see ``scripts/plot_grid.py``).

Example
-------
    python scripts/run_grid.py --network L1_design --nq 25 --out data/grid.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omp.params import SourceParams, chi_eff           # noqa: E402
from omp.fisher import FisherWorkspace                  # noqa: E402
from omp.eigen import analyze_subspace, enforce_continuity  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", default="L1_design")
    p.add_argument("--waveform", default="IMRPhenomD")
    p.add_argument("--fmin", type=float, default=10.0)
    p.add_argument("--fmax", type=float, default=2048.0)
    p.add_argument("--m1", type=float, default=35.0, help="source-frame m1 [Msun]")
    p.add_argument("--z", type=float, default=0.1)
    p.add_argument("--chi1z", type=float, default=0.2)
    p.add_argument("--chi2z", type=float, default=0.0)
    p.add_argument("--nq", type=int, default=25, help="grid points in q")
    p.add_argument("--qmin", type=float, default=0.2)
    p.add_argument("--qmax", type=float, default=1.0)
    p.add_argument("--method", default="marginalize",
                   choices=["marginalize", "whiten_then_marginalize", "condition"])
    p.add_argument("--whiten", default="subcov", choices=["subcov", "fisher", "none"])
    p.add_argument("--res", type=int, default=2000)
    p.add_argument("--out", default="data/grid.npz")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    whiten = None if args.whiten == "none" else args.whiten

    qs = np.linspace(args.qmin, args.qmax, args.nq)
    src = SourceParams(m1=args.m1, q=qs, z=args.z,
                       chi1z=args.chi1z, chi2z=args.chi2z)

    # Mass sub-space in (log m1, log m2) so the best-measured combination is an
    # interpretable power-law (chirp mass at low mass, total mass at high mass).
    ws = FisherWorkspace(network=args.network, waveform=args.waveform,
                         fmin=args.fmin, fmax=args.fmax,
                         mass_coords="logm1logm2")
    mass_pars = ["logm1", "logm2"]
    snr = ws.snr(src, res=args.res)
    F = ws.fisher(src, res=args.res)   # (Npar, Npar, Nq)

    # Spin sub-space -- chi_eff generalisation. Reference sign = chi_eff dir.
    spin = [analyze_subspace(F[..., i:i+1], ws.par_order, ["chi1z", "chi2z"],
                             method=args.method, whiten=whiten,
                             sign_ref=[1.0, qs[i]])[0]
            for i in range(len(qs))]
    spin = enforce_continuity(spin)

    # Mass sub-space -- chirp-mass generalisation in (log m1, log m2).
    # Reference sign = symmetric (1, 1) direction (total/chirp-mass-like).
    mass = [analyze_subspace(F[..., i:i+1], ws.par_order, mass_pars,
                             method=args.method, whiten=whiten,
                             sign_ref=[1.0, 1.0])[0]
            for i in range(len(qs))]
    mass = enforce_continuity(mass)

    spin_best = np.array([r.best for r in spin])          # (nq, 2)
    spin_var = np.array([r.variances for r in spin])       # (nq, 2)
    mass_best = np.array([r.best for r in mass])
    mass_var = np.array([r.variances for r in mass])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        q=qs, snr=snr,
        spin_best=spin_best, spin_var=spin_var,
        mass_best=mass_best, mass_var=mass_var,
        chi_eff_ratio=qs,                       # reference c2/c1 for chi_eff
        meta=dict(network=args.network, waveform=args.waveform,
                  m1=args.m1, z=args.z, chi1z=args.chi1z, chi2z=args.chi2z,
                  method=args.method, whiten=args.whiten),
    )
    print(f"saved {args.out}")
    print(f"  q in [{args.qmin}, {args.qmax}], {args.nq} pts, "
          f"SNR {snr.min():.1f}-{snr.max():.1f}")
    print(f"  spin best-measured c2/c1: "
          f"{spin_best[:,1]/spin_best[:,0]}")


if __name__ == "__main__":
    main()
