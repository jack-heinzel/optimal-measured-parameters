"""Best-measured parameter directions from the Fisher matrix.

Given a full Fisher matrix over all parameters, we isolate an intrinsic
sub-space (spins or masses), reduce it to a 2x2 (or k x k) covariance, whiten
it so the parameters are dimensionless/comparable, and eigendecompose.  The
smallest-variance eigenvector is the *best-measured* combination -- the
generalisation of chi_eff (spins) or chirp mass (masses).

Inversion (marginalisation) and conditioning reuse gwfast's existing
``fisherTools`` (``CovMatr``, ``fixParams``) so the matrix conditioning is
handled the same way as in the rest of gwfast.

Three reductions are provided so we can compare numerical stability
(per CLAUDE.md, recipe (1) is preferred but must be checked):

* ``"marginalize"`` -- invert the full Fisher, take the sub-block of the
  covariance, then whiten.  (Recipe 1.)
* ``"whiten_then_marginalize"`` -- whiten the full Fisher by its diagonal,
  invert, then take the sub-block.  (Recipe 2.)
* ``"condition"`` -- fix (condition on) all other parameters and eigendecompose
  the Fisher sub-block directly (no marginalisation).  (Recipe 3.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gwfast.fisherTools as ft


# --------------------------------------------------------------------------
# Sub-space reductions (all return a k x k covariance per event)
# --------------------------------------------------------------------------

def _par_nums(par_order):
    return {name: i for i, name in enumerate(par_order)}


def marginalized_subcov(F, par_order, sub_pars, **cov_kwargs):
    """Invert full Fisher (gwfast ``CovMatr``), return sub-block covariance.

    Returns ``(subcov, eps)`` with ``subcov`` shape ``(k, k, Nev)`` and ``eps``
    the per-event inversion error from gwfast.
    """
    Cov, eps = ft.CovMatr(F, **cov_kwargs)
    idx = [par_order.index(p) for p in sub_pars]
    subcov = Cov[np.ix_(idx, idx)]
    return np.asarray(subcov), np.asarray(eps)


def whiten_then_marginalized_subcov(F, par_order, sub_pars, **cov_kwargs):
    """Whiten the full Fisher by its diagonal, invert, take sub-block.

    Whitening before inversion improves the conditioning of ``CovMatr`` when
    parameters span very different scales.  The returned covariance is mapped
    back to physical units.
    """
    F = np.asarray(F)
    npar, _, nev = F.shape
    d = np.sqrt(np.abs(np.diagonal(F, axis1=0, axis2=1).T))  # (npar, nev)
    D = d[:, None, :] * d[None, :, :]                        # outer per event
    Fw = F / D
    Covw, eps = ft.CovMatr(Fw, **cov_kwargs)
    Cov = Covw / D                                           # undo whitening
    idx = [par_order.index(p) for p in sub_pars]
    subcov = Cov[np.ix_(idx, idx)]
    return np.asarray(subcov), np.asarray(eps)


def conditioned_subfisher(F, par_order, sub_pars):
    """Condition on all other parameters (gwfast ``fixParams``); return the
    Fisher sub-block ``(k, k, Nev)``.  Its inverse is the conditional, *not*
    marginal, covariance."""
    par_nums = _par_nums(par_order)
    to_fix = [p for p in par_order if p not in sub_pars]
    Fsub, new_nums = ft.fixParams(F, par_nums, to_fix)
    idx = [new_nums[p] for p in sub_pars]
    Fsub = np.asarray(Fsub)[np.ix_(idx, idx)]
    return Fsub


# --------------------------------------------------------------------------
# Whitening + eigendecomposition
# --------------------------------------------------------------------------

@dataclass
class EigenResult:
    """Eigen-analysis of one sub-space at one parameter point.

    Attributes
    ----------
    sub_pars : list[str]
        Parameter names defining the sub-space (e.g. ``["chi1z", "chi2z"]``).
    variances : np.ndarray
        Eigen-variances in *whitened* space, ascending (best measured first).
    whitened_vecs : np.ndarray
        Columns are whitened-space eigenvectors ``b_j`` (orthonormal), ordered
        with ``variances``.
    coeffs : np.ndarray
        Columns are the corresponding *physical*-space coefficient vectors
        ``a_j = b_j / scales``, each normalised to unit Euclidean norm.  The
        first column is the best-measured combination -- the chi_eff /
        chirp-mass generalisation.
    scales : np.ndarray
        Whitening scales applied to each physical parameter.
    """

    sub_pars: list
    variances: np.ndarray
    whitened_vecs: np.ndarray
    coeffs: np.ndarray
    scales: np.ndarray

    @property
    def best(self) -> np.ndarray:
        """Physical-space coefficients of the best-measured combination."""
        return self.coeffs[:, 0]


def _whiten_scales(cov, how, fisher_diag=None):
    """Return whitening scales (length k) for a single k x k covariance."""
    if how is None:
        return np.ones(cov.shape[0])
    if how == "subcov":
        return np.sqrt(np.abs(np.diag(cov)))
    if how == "fisher":
        if fisher_diag is None:
            raise ValueError("whiten='fisher' requires fisher_diag")
        return 1.0 / np.sqrt(np.abs(fisher_diag))
    raise ValueError(f"unknown whiten mode {how!r}")


def eigen_from_cov(cov, sub_pars, whiten="subcov", fisher_diag=None,
                   sign_ref=None):
    """Eigen-analysis of a single k x k covariance matrix.

    Parameters
    ----------
    cov : (k, k) array
        Covariance of the sub-space parameters (marginal or conditional).
    whiten : {"subcov", "fisher", None}
        Whitening scheme.  ``"subcov"`` rescales by each parameter's marginal
        1-sigma (the diagonal of ``cov``); ``"fisher"`` by the conditional
        1-sigma ``1/sqrt(F_ii)`` (pass ``fisher_diag``); ``None`` disables it.
    sign_ref : (k,) array, optional
        Reference direction; the best-measured eigenvector's sign is flipped to
        have positive overlap with it (e.g. the chi_eff direction ``(1, q)``).
    """
    cov = np.asarray(cov, dtype=float)
    s = _whiten_scales(cov, whiten, fisher_diag)
    D_inv = np.diag(1.0 / s)
    cov_w = D_inv @ cov @ D_inv                      # whitened covariance
    cov_w = 0.5 * (cov_w + cov_w.T)                  # symmetrise
    vals, vecs = np.linalg.eigh(cov_w)               # ascending eigenvalues
    # ascending variance == best measured first (already ascending from eigh)
    coeffs = (vecs.T / s).T                           # a_j = b_j / s, columns
    coeffs = coeffs / np.linalg.norm(coeffs, axis=0, keepdims=True)
    if sign_ref is not None:
        sign_ref = np.asarray(sign_ref, dtype=float)
        if np.dot(coeffs[:, 0], sign_ref) < 0:
            coeffs[:, 0] *= -1
            vecs[:, 0] *= -1
    return EigenResult(
        sub_pars=list(sub_pars),
        variances=vals,
        whitened_vecs=vecs,
        coeffs=coeffs,
        scales=s,
    )


# --------------------------------------------------------------------------
# High-level driver
# --------------------------------------------------------------------------

_METHODS = {"marginalize", "whiten_then_marginalize", "condition"}


def analyze_subspace(F, par_order, sub_pars, *, method="marginalize",
                     whiten="subcov", sign_ref=None, **cov_kwargs):
    """Best-measured directions for a sub-space across all events in ``F``.

    ``F`` has shape ``(Npar, Npar, Nev)``.  Returns a list of
    :class:`EigenResult`, one per event.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {_METHODS}")
    F = np.asarray(F)
    nev = F.shape[2]

    if method == "condition":
        sub = conditioned_subfisher(F, par_order, sub_pars)  # Fisher blocks
        results = []
        for i in range(nev):
            cov_i = np.linalg.inv(sub[..., i])
            results.append(eigen_from_cov(cov_i, sub_pars, whiten=whiten,
                                          sign_ref=sign_ref))
        return results

    if method == "marginalize":
        subcov, _ = marginalized_subcov(F, par_order, sub_pars, **cov_kwargs)
    else:  # whiten_then_marginalize
        subcov, _ = whiten_then_marginalized_subcov(
            F, par_order, sub_pars, **cov_kwargs)

    results = []
    for i in range(nev):
        results.append(eigen_from_cov(subcov[..., i], sub_pars, whiten=whiten,
                                      sign_ref=sign_ref))
    return results


# --------------------------------------------------------------------------
# Smoothness / continuity across a parameter grid
# --------------------------------------------------------------------------

def enforce_continuity(results: list[EigenResult]) -> list[EigenResult]:
    """Reorder/sign-fix eigenvectors along a sequence for smoothness.

    Greedy: at each step, match each eigenvector to the previous step's
    eigenvectors by maximum absolute overlap (handling level crossings) and fix
    the sign for positive overlap.  Operates on the physical ``coeffs`` (and
    keeps ``variances``/``whitened_vecs`` consistent).
    """
    if not results:
        return results
    out = [results[0]]
    # Track modes using the *whitened* eigenvectors: they are orthonormal, so
    # the overlap matrix is close to a (signed) permutation and the greedy
    # assignment is unambiguous.  The physical (non-orthonormal) coeffs would
    # make best/worst modes overlap and mis-assign near level crossings.
    prev = results[0].whitened_vecs
    for res in results[1:]:
        W = res.whitened_vecs
        k = W.shape[1]
        overlap = prev.T @ W                       # (k_prev, k_cur)
        work = np.abs(overlap).copy()
        order = [-1] * k
        for _ in range(k):
            i, j = np.unravel_index(np.argmax(work), work.shape)
            order[i] = j
            work[i, :] = -np.inf
            work[:, j] = -np.inf
        perm = np.array(order)
        wv2 = res.whitened_vecs[:, perm].copy()
        C2 = res.coeffs[:, perm].copy()
        vars2 = res.variances[perm].copy()
        # sign fix against previous whitened vectors
        for j in range(k):
            if np.dot(prev[:, j], wv2[:, j]) < 0:
                wv2[:, j] *= -1
                C2[:, j] *= -1
        out.append(EigenResult(res.sub_pars, vars2, wv2, C2, res.scales))
        prev = wv2
    return out
