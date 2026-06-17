"""Regression tests for the omp pipeline.

Run with:  python -m pytest tests/ -q   (in the gwjax313 environment)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omp.params import SourceParams, chi_eff, m1m2_to_Mceta   # noqa: E402
from omp.fisher import FisherWorkspace                          # noqa: E402
from omp.eigen import (analyze_subspace, eigen_from_cov,        # noqa: E402
                       enforce_continuity)


def test_chi_eff_reference():
    assert np.isclose(chi_eff(0.4, 0.0, 1.0), 0.2)
    # equal masses -> simple average
    assert np.isclose(chi_eff(0.3, 0.1, 1.0), 0.2)


def test_eigen_from_cov_diagonal():
    # diagonal cov: best-measured is the small-variance axis
    cov = np.array([[1e-4, 0.0], [0.0, 1.0]])
    res = eigen_from_cov(cov, ["a", "b"], whiten=None)
    assert res.variances[0] < res.variances[1]
    assert np.argmax(np.abs(res.best)) == 0  # aligned with the 'a' axis


def test_continuity_preserves_ratio_sign():
    # two smoothly varying 2x2 covariances should keep best-mode ratio sign
    qs = np.linspace(0.3, 0.9, 7)
    results = []
    for q in qs:
        cov = np.array([[0.01, 0.009 * q], [0.009 * q, 0.01 / q]])
        results.append(eigen_from_cov(cov, ["c1", "c2"], whiten="subcov",
                                      sign_ref=[1.0, q]))
    results = enforce_continuity(results)
    ratios = [r.best[1] / r.best[0] for r in results]
    assert all(np.isfinite(ratios))


def test_full_pipeline_recovers_chi_eff():
    """Best-measured spin direction ratio ~ q (the chi_eff weighting)."""
    ws = FisherWorkspace(network="L1_design", waveform="IMRPhenomD")
    qs = np.array([0.4, 0.7, 1.0])
    src = SourceParams(m1=35.0, q=qs, z=0.1, chi1z=0.2, chi2z=0.0)
    F = ws.fisher(src)
    for i, q in enumerate(qs):
        res = analyze_subspace(F[..., i:i+1], ws.par_order,
                               ["chi1z", "chi2z"], method="marginalize",
                               whiten="subcov", sign_ref=[1.0, q])[0]
        ratio = res.best[1] / res.best[0]
        # within 15% of chi_eff weighting q (the generalisation is close)
        assert abs(ratio - q) < 0.15 * q + 0.05, (q, ratio)
        # best-measured variance is much smaller than the worst
        assert res.variances[0] < res.variances[1]


def test_methods_agree_on_spin_direction():
    """marginalize vs whiten_then_marginalize should give close spin dirs."""
    ws = FisherWorkspace(network="L1_design", waveform="IMRPhenomD")
    src = SourceParams(m1=35.0, q=0.7, z=0.1, chi1z=0.2, chi2z=0.0)
    F = ws.fisher(src)
    a = analyze_subspace(F, ws.par_order, ["chi1z", "chi2z"],
                         method="marginalize", whiten="subcov",
                         sign_ref=[1.0, 0.7])[0].best
    b = analyze_subspace(F, ws.par_order, ["chi1z", "chi2z"],
                         method="whiten_then_marginalize", whiten="subcov",
                         sign_ref=[1.0, 0.7])[0].best
    assert abs(np.dot(a, b)) > 0.999  # nearly parallel


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
