This is a repository for understanding the optimally measured intrinsic parameters of a gravitational wave source as a function of the true parameters.

## Goal

Compute the eigenspectrum of the Fisher information matrix at a parameter point \theta, and find the best-measured direction (eigenvector \xi(\theta) corresponding to the most-constrained combination of parameters). As we vary \theta, \xi(\theta) should vary as a smooth function.

We do this for a variety of reduced parameter sub-spaces. For example, the best-measured spin parameter (then second best, etc.) as a function of the true parameter \theta. Using Gaussian/Fisher theorems, we compute the eigenvector of the *marginalized* spin covariance matrix — the direction along which the spins are best constrained, marginalized over the other parameters (assuming Gaussianity as usual in the Fisher approximation).

The "best measured" spin parameter should be a generalization of
chi_eff = (a_1 * cos theta_1 + q * a_2 * cos theta_2) / (1 + q).
We do the analogous procedure for the mass parameters, generalizing chirp mass (which becomes total mass at high masses).

## Procedure and conventions

**Environment.** Use the `gwjax313` conda environment wherever possible (interpreter at `/home/jack.heinzel/.conda/envs/gwjax313/bin/python`). gwfast 1.1.2 was installed with `pip install --no-deps gwfast` (its pinned old astropy won't build on Python 3.13) plus `numdifftools`; it otherwise runs fine against the env's modern numpy/scipy/jax. The only incompatibility is `scipy.integrate.cumtrapz` (renamed), handled by the shim in `omp/compat.py` — **import `omp` (or `omp.compat`) before gwfast.** Runs on GPU automatically.

**Fisher tooling.** Use `gwfast` (JAX automatic differentiation) for waveform derivatives and PSD integration. Autodiff gives clean, low-noise derivatives, which matters for the eigenvectors to vary smoothly across the parameter grid. We reuse gwfast's existing methods throughout: `signal.GWSignal`/`network.DetNet` `FisherMatr` for the Fisher matrix, and `fisherTools.CovMatr` (inversion/marginalisation) and `fisherTools.fixParams` (conditioning).

**Spin model limitation (current).** This gwfast build only ships *aligned-spin* waveforms (TaylorF2, IMRPhenomD, IMRPhenomHM, NSBH); no precessing model (IMRPhenomXPHM absent, TEOBResumS not built). So spins enter only as aligned components `(chi1z, chi2z)` and the parameterisation is gwfast-native (see below), **not** `(a_i, cos tilt_i, phi_i)`. The aligned-spin case is exactly the chi_eff study: `chi_eff = (chi1z + q·chi2z)/(1+q)`. Precession / the chi_p generalisation is deferred until a precessing waveform is available.

**Detector network / PSD.** Keep the network and PSD *configurable* inputs rather than hardcoding one. Plan to produce results across several configurations (e.g. single detector, HLV at design, 3G ET+CE).

**Parameterization.** Intrinsic parameters:
- Spins: `a_i`, `cos(tilt_i)`, `phi_i` for i = 1, 2.
- Masses: source-frame primary mass `m_1` and mass ratio `q`.
- Redshift `z` as a parameter. Masses are source-frame; derivatives must account for the `(1+z)` scaling to detector frame and the distance/amplitude dependence on `z`.

**Whitening / normalization.** Fisher eigen-directions depend on parameter scaling (masses and spins carry different units). Whiten by the Fisher diagonal (rescale each parameter by its own 1-sigma) so the eigenproblem is dimensionless before extracting eigenvectors.

**Marginalize vs. whiten — to be determined experimentally.** The preferred recipe is (1): invert the full Fisher → take the spin (or mass) sub-covariance → whiten by that sub-covariance diagonal → the best-measured direction is the *smallest-eigenvalue* eigenvector of the whitened covariance. The open question is whether (1) is numerically stable; experiment with alternatives (whiten-then-marginalize, or working with the conditioned Fisher sub-block) and compare stability. Document what works.

**How many modes.** Track the top few eigen-directions per sub-space (e.g. the chi_eff-like best direction and a chi_p-like second-best for spins), not just the single best. Enforce sign/ordering continuity of eigenvectors across the parameter grid so they trace smooth functions of \theta.

## Deliverables

Both of:
1. Numerical eigenvectors + eigenvalues tabulated and plotted across the parameter grid.
2. Interpretable closed-form fitting formulae for the eigenvector components as functions of \theta, generalizing chi_eff and chirp mass. Fit these to the numerical grid.

## Code layout

- `omp/` — the package. Import it before gwfast (applies the scipy shim).
  - `compat.py` — scipy/numpy shims for gwfast on modern envs.
  - `detectors.py` — network presets (`L1_design`, `HLV_O4`, `HLV_design`, `ET`, `ET_CE`) built from gwfast's detector dict + shipped PSDs; `build_network()`.
  - `params.py` — `SourceParams` (source-frame m1, q, z, aligned spins, extrinsic angles) → gwfast event dict, with `(1+z)` mass redshifting and z→dL via Planck18; `chi_eff()` reference. gwfast parameter order: `[Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z]`.
  - `fisher.py` — `FisherWorkspace`: builds the network once, computes SNR and the full Fisher via gwfast.
  - `eigen.py` — sub-space reductions (`marginalize` [recipe 1, via `CovMatr`], `whiten_then_marginalize` [recipe 2], `condition` [recipe 3, via `fixParams`]), whitening + eigendecomposition (`eigen_from_cov`, `analyze_subspace`), and `enforce_continuity` (tracks modes across the grid by matching the *orthonormal whitened* eigenvectors, not the physical coeffs).
- `scripts/run_grid.py` — sweep a grid, save eigen-results to `.npz`. `scripts/plot_grid.py` — plot.
- `tests/test_pipeline.py` — regression tests (`python -m pytest tests/ -q`).

**Mass basis:** the mass sub-space uses `(log m1, log m2)` (`FisherWorkspace(mass_coords="logm1logm2")`, via gwfast's `use_m1m2=True` plus a diagonal log-Jacobian `F_log = diag(m)·F·diag(m)` on the detector-frame component masses). Reference gradient-weight ratios in this basis: chirp mass → `(3+2q)/(2+3q)`, total mass → `q`.

**Validation status (both sides confirmed):**
- *Spin* — best-measured `(chi1z, chi2z)` weight ratio tracks `q` (chi_eff): q=0.3→0.28, 0.5→0.46, 1.0→1.00.
- *Mass* — best-measured `(log m1, log m2)` weight ratio reproduces chirp mass at low mass and **transitions to total mass at high mass** (q=0.6: Mtot=16→1.11≈chirp 1.105; Mtot=480→0.63≈total 0.60), exactly the chirp-mass generalization.

## Closed-form fits (deliverable 2)

`omp/fit.py` + `scripts/fit_directions.py` (vectorised + batched Fisher over a `(q, m1)` grid).

- **Spin (done).** Best-measured spin direction is a smooth generalisation of chi_eff, slightly below it, with weak mass dependence (L1 design, IMRPhenomD):
  `c2/c1 = q*(1 - 0.337*(1-q) + 0.359*(1-q)^2)`, fit RMS ~0.018, max deviation from chi_eff ~0.10.
- **Mass (partial).** Confirms chirp→total transition near Mtot ~ 150-250 Msun, but the plain chirp↔total sigmoid fits poorly (RMS ~0.43) for two reasons: (a) a **non-monotonic overshoot** around Mtot ~ 100-130 Msun (ratio rises above the chirp value before dropping to total) that needs a richer model; (b) **ill-conditioning at equal mass (q=1)** where the two log-mass eigenvalues are near-degenerate and the "best" mode can flip to the unphysical antisymmetric direction (spurious `w2/w1 = -1`). Needs regularisation/conditioning at the degeneracy and a better functional form.

## Open methodological questions (to revisit)

- **Marginalize vs whiten order (recipes 1/2/3).** Implemented all three; recipe 1 preferred. Still need a systematic numerical-stability comparison across the grid, especially at high SNR / 3G networks where the Fisher is more ill-conditioned (cf. the q=1 mass degeneracy above).
- **Source-frame masses + z.** `params.py` redshifts masses to detector frame for the waveform, but the Fisher is currently taken w.r.t. gwfast-native `(Mc, eta)` (detector frame). Differentiating w.r.t. source-frame `(m1, q, z)` needs a Jacobian transform (gwfast provides `J_*` helpers); the mass-z degeneracy means z is essentially fixed by dL/cosmology.
