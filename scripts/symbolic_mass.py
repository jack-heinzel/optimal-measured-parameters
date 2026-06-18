#!/usr/bin/env python
"""Symbolic regression (PySR) for the best-measured log-mass weight ratio.

Loads the cached (Mtot, q) grid (notes/figs/mass_joint.npz from
scripts/mass_grid2d.py) and searches for a closed form
    w2/w1 = f(q, x),   x = log10(Mtot)
from scratch -- no chirp/total priors fed in -- to see whether symbolic
regression independently recovers the transition + overshoot structure we fit by
hand in omp.fit.fit_mass_2d.

Prints the Pareto front (accuracy vs complexity) and writes
notes/figs/mass_pysr.png (data / SR model / residual).
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notes", "figs")
CACHE = os.path.join(OUT, "mass_joint.npz")


def main():
    if not os.path.exists(CACHE):
        sys.exit(f"missing {CACHE}; run scripts/mass_grid2d.py first")
    d = np.load(CACHE)
    Q, M, R = d["Q"], d["M"], d["R"]
    q = Q.ravel()
    x = np.log10(M).ravel()
    y = R.ravel()
    X = np.column_stack([q, x])

    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=120,
        populations=24,
        population_size=40,
        maxsize=28,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "tanh", "square"],
        elementwise_loss="loss(p, t) = (p - t)^2",
        model_selection="best",
        progress=False,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=0,
    )
    model.fit(X, y, variable_names=["q", "x"])

    print("\n=== PySR Pareto front (complexity / loss / equation) ===")
    eqs = model.equations_
    for _, row in eqs.iterrows():
        print(f"  c={int(row['complexity']):2d}  loss={row['loss']:.3e}  "
              f"{row['equation']}")

    best = model.sympy()
    pred = model.predict(X)
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    print("\nBest (model_selection='best'):")
    print("  ", best)
    print(f"  RMS = {rms:.4e}   (hand-fit 2D bump model ~0.030)")

    _plot(Q, M, R, pred.reshape(Q.shape), rms)


def _plot(Q, M, R, P, rms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [np.log10(M.min()), np.log10(M.max()), Q.min(), Q.max()]
    vmin, vmax = float(R.min()), float(R.max())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0].imshow(R, origin="lower", aspect="auto", extent=ext,
                     cmap="viridis", vmin=vmin, vmax=vmax)
    ax[0].set_title(r"data $w_2/w_1$"); fig.colorbar(a, ax=ax[0])
    b = ax[1].imshow(P, origin="lower", aspect="auto", extent=ext,
                     cmap="viridis", vmin=vmin, vmax=vmax)
    ax[1].set_title(f"PySR model (RMS={rms:.3f})"); fig.colorbar(b, ax=ax[1])
    c = ax[2].imshow(R - P, origin="lower", aspect="auto", extent=ext,
                     cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax[2].set_title("residual (data $-$ SR)"); fig.colorbar(c, ax=ax[2])
    for a_ in ax:
        a_.set_xlabel(r"$\log_{10} M_{\rm tot}$"); a_.set_ylabel("$q$")
    fig.tight_layout(); fig.savefig(f"{OUT}/mass_pysr.png", dpi=130)
    print("saved mass_pysr.png")


if __name__ == "__main__":
    main()
