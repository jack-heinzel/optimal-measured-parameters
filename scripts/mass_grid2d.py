#!/usr/bin/env python
"""Map the best-measured (log m1, log m2) direction over the (Mtot, q) plane.

The best-measured mass combination interpolates between chirp mass
(weight ratio w2/w1 = (3+2q)/(2+3q)) at low total mass and total mass
(w2/w1 = q) at high total mass.  This grids the joint space to expose the
structure: the chirp plateau, the overshoot ridge, the transition, and the
q->1 near-degeneracy.

Writes notes/figs/mass_joint.png.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import omp  # noqa: F401
from omp.params import SourceParams
from omp.fisher import FisherWorkspace
from omp.eigen import analyze_subspace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notes", "figs")
os.makedirs(OUT, exist_ok=True)


def main(nq=22, nM=26, qmin=0.25, qmax=0.985, Mmin=15.0, Mmax=500.0,
         z=0.1, res=2000):
    qg = np.linspace(qmin, qmax, nq)
    Mg = np.geomspace(Mmin, Mmax, nM)
    Q, M = np.meshgrid(qg, Mg, indexing="ij")        # (nq, nM)
    q_flat, M_flat = Q.ravel(), M.ravel()
    m1_flat = M_flat / (1.0 + q_flat)

    src = SourceParams(m1=m1_flat, q=q_flat, z=z, chi1z=0.0, chi2z=0.0)
    ws = FisherWorkspace(waveform="IMRPhenomD", mass_coords="logm1logm2")
    F = ws.fisher(src, res=res, batch_size=26)

    ratio = np.empty(q_flat.size)
    for i in range(q_flat.size):
        b = analyze_subspace(F[..., i:i+1], ws.par_order, ["logm1", "logm2"],
                             method="marginalize", sign_ref=[1.0, 1.0])[0].best
        ratio[i] = b[1] / b[0]
    R = ratio.reshape(Q.shape)

    chirp = (3 + 2 * Q) / (2 + 3 * Q)                # q-only reference
    total = Q                                        # q-only reference
    # interpolation fraction: 0 at chirp, 1 at total (overshoot -> <0)
    frac = (R - chirp) / (total - chirp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ext = [np.log10(Mmin), np.log10(Mmax), qmin, qmax]

    im0 = axes[0].imshow(R, origin="lower", aspect="auto", extent=ext,
                         cmap="viridis")
    axes[0].set_title(r"best-measured $w_2/w_1$")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(R - chirp, origin="lower", aspect="auto", extent=ext,
                         cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[1].set_title(r"residual $w_2/w_1 - $ chirp$(q)$  (overshoot in red)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.clip(frac, -0.5, 1.5), origin="lower", aspect="auto",
                         extent=ext, cmap="magma")
    axes[2].set_title("chirp$\\to$total fraction  (0=chirp, 1=total)")
    fig.colorbar(im2, ax=axes[2])
    # contour of the half-transition
    axes[2].contour(np.log10(Mg)[None, :] + 0 * Q, Q, frac, levels=[0.5],
                    colors="cyan", linewidths=1.5)

    for ax in axes:
        ax.set_xlabel(r"$\log_{10} M_{\rm tot}$ [$M_\odot$]")
        ax.set_ylabel("mass ratio $q$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mass_joint.png", dpi=130)
    print("saved mass_joint.png")

    # quick text slice through a few q
    print("\n w2/w1 at selected (q, Mtot):")
    for qi in [0, nq // 2, nq - 1]:
        print(f"  q={qg[qi]:.2f}: chirp={ (3+2*qg[qi])/(2+3*qg[qi]):.3f} "
              f"total={qg[qi]:.3f} | ratio @Mtot="
              + ", ".join(f"{Mg[mi]:.0f}:{R[qi,mi]:.3f}"
                          for mi in [0, nM // 2, nM - 1]))


if __name__ == "__main__":
    main()
