#!/usr/bin/env python
"""Plot best-measured directions from a grid produced by ``run_grid.py``.

Shows (left) the best-measured spin combination's weight ratio c2/c1 versus the
chi_eff prediction (= q), and (right) the best-measured mass combination's
(Mc, eta) weights versus mass ratio.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", default="data/grid.npz")
    p.add_argument("--out", default="data/grid.png")
    args = p.parse_args(argv)

    d = np.load(args.grid, allow_pickle=True)
    q = d["q"]
    spin_best = d["spin_best"]
    mass_best = d["mass_best"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(q, q, "k--", label=r"$\chi_{\rm eff}$ prediction ($c_2/c_1=q$)")
    ax1.plot(q, spin_best[:, 1] / spin_best[:, 0], "o-",
             label="best-measured spin direction")
    ax1.set_xlabel("mass ratio $q$")
    ax1.set_ylabel(r"weight ratio $c_2/c_1$  ($c_1\chi_{1z}+c_2\chi_{2z}$)")
    ax1.set_title("Spin: generalisation of $\\chi_{\\rm eff}$")
    ax1.legend()

    # Best-measured mass combo: w1*log m1 + w2*log m2.  Reference gradient-weight
    # ratios in (log m1, log m2):  chirp mass -> (3+2q)/(2+3q);  total mass -> q.
    ax2.plot(q, mass_best[:, 1] / mass_best[:, 0], "o-",
             label=r"best-measured  $w_2/w_1$")
    ax2.plot(q, (3 + 2 * q) / (2 + 3 * q), "k--",
             label=r"chirp mass $(3{+}2q)/(2{+}3q)$")
    ax2.plot(q, q, "k:", label=r"total mass $=q$")
    ax2.set_xlabel("mass ratio $q$")
    ax2.set_ylabel(r"weight ratio $w_2/w_1$  ($w_1\log m_1+w_2\log m_2$)")
    ax2.set_title("Mass: generalisation of chirp mass")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
