This is a repository for understanding the optimally measured intrinsic parameters of a gravitational wave source as a function of the true parameters.

## Goal

Compute the eigenspectrum of the Fisher information matrix at a parameter point \theta, and find the best-measured direction (eigenvector \xi(\theta) corresponding to the most-constrained combination of parameters). As we vary \theta, \xi(\theta) should vary as a smooth function.

We do this for a variety of reduced parameter sub-spaces. For example, the best-measured spin parameter (then second best, etc.) as a function of the true parameter \theta. Using Gaussian/Fisher theorems, we compute the eigenvector of the *marginalized* spin covariance matrix — the direction along which the spins are best constrained, marginalized over the other parameters (assuming Gaussianity as usual in the Fisher approximation).

The "best measured" spin parameter should be a generalization of
chi_eff = (a_1 * cos theta_1 + q * a_2 * cos theta_2) / (1 + q).
We do the analogous procedure for the mass parameters, generalizing chirp mass (which becomes total mass at high masses).

## Procedure and conventions

**Environment.** Use the `gwjax313` conda environment wherever possible (interpreter at `/home/jack.heinzel/.conda/envs/gwjax313/bin/python`). gwfast 1.1.2 was installed with `pip install --no-deps gwfast` (its pinned old astropy won't build on Python 3.13) plus `numdifftools`; it otherwise runs fine against the env's modern numpy/scipy/jax. The only incompatibility is `scipy.integrate.cumtrapz` (renamed), handled by the shim in `omp/compat.py` — **import `omp` (or `omp.compat`) before gwfast.** Runs on GPU automatically.

**Fisher tooling.** Two autodiff backends, both JAX:
- `gwfast` for the aligned-spin study — `signal.GWSignal`/`network.DetNet` `FisherMatr` for the Fisher, `fisherTools.CovMatr` (marginalisation) and `fisherTools.fixParams` (conditioning).
- `ripple` (PyPI `ripplegw`, import `ripplegw`) for **precessing** waveforms gwfast's build lacks. gwfast's `FisherMatr` can't consume ripple, so `omp/ripple_fisher.py` builds the Fisher directly: `F_ij = 4 Re ∫ (∂_i h)*(∂_j h)/S_n df` via `jax.jacfwd`. It mirrors `FisherWorkspace`'s interface so the `eigen` pipeline runs on it unchanged. `jimgw` is also installed (sampler/likelihood, unused so far). See [[ripple-jimgw-env]] memory: install jimgw with `jax==0.9.0 jaxlib==0.9.0` pinned or pip bumps jax to 0.10.2 and breaks the GPU CUDA build.

**Spin models.** gwfast ships only *aligned-spin* (TaylorF2, IMRPhenomD, IMRPhenomHM, NSBH). **Precession is now available via ripple** (`IMRPhenomPv2/XP/XPHM`, full 3D spins). `RippleFisherWorkspace(waveform="IMRPhenomPv2")` differentiates w.r.t. Cartesian spin components `(s1x,s1y,s1z,s2x,s2y,s2z)` at `f_ref`; chi_eff lives in the z-components, chi_p in the in-plane. `SourceParams.from_spherical(a_i, tilt_i, phi_i)` builds these from the `(a_i, cos tilt_i, phi_i)` parameterisation. **Never evaluate at exactly aligned (in-plane spin = 0)** or exactly `q=1`: ripple's precession angles / `sqrt(1-4eta)` are branch points there → nan Jacobian (use tiny tilt, q=0.999).

**Detector network / PSD.** Keep the network and PSD *configurable* inputs rather than hardcoding one. Plan to produce results across several configurations (e.g. single detector, HLV at design, 3G ET+CE).

**Parameterization.** Intrinsic parameters:
- Spins: `a_i`, `cos(tilt_i)`, `phi_i` for i = 1, 2.
- Masses: source-frame primary mass `m_1` and mass ratio `q`.
- Redshift `z` as a parameter. Masses are source-frame; derivatives must account for the `(1+z)` scaling to detector frame and the distance/amplitude dependence on `z`.

**Whitening / normalization.** Fisher eigen-directions depend on parameter scaling (masses and spins carry different units). Whiten by the Fisher diagonal (rescale each parameter by its own 1-sigma) so the eigenproblem is dimensionless before extracting eigenvectors.

**Marginalize vs. whiten — recipe choice.** Recipe (1) `marginalize`: invert the full Fisher → spin/mass sub-covariance → whiten by its diagonal → best-measured = *smallest-eigenvalue* eigenvector. Good for well-conditioned (aligned, 2D) sub-spaces. **For precession use `marginalize_fisher`** (`eigen.py`): the marginalised sub-Fisher is the Schur complement of the nuisance block (`Fbar = F_ss − F_sn F_nn⁻¹ F_ns`, never inverts the full matrix); eigendecompose it *directly* with Fisher-diagonal whitening (`1/√F_ii`), best-measured = *largest*-eigenvalue eigenvector. This is provably identical to recipe 1 on well-conditioned 2D cases but stable when the covariance is non-PD — which it is in the 6D spin space, where in-plane (chi_p) directions are measured ~1e3–1e6× worse than aligned (chi_eff) and inverting into a covariance gives spurious negative variances. `marginalize_schur` is the same Schur marginalisation but inverts to a covariance (use only when well-conditioned).

**How many modes.** Track the top few eigen-directions per sub-space (e.g. the chi_eff-like best direction and a chi_p-like second-best for spins), not just the single best. Enforce sign/ordering continuity of eigenvectors across the parameter grid so they trace smooth functions of \theta.

## Deliverables

Both of:
1. Numerical eigenvectors + eigenvalues tabulated and plotted across the parameter grid.
2. Interpretable closed-form fitting formulae for the eigenvector components as functions of \theta, generalizing chi_eff and chirp mass. Fit these to the numerical grid.

## Code layout

- `omp/` — the package. Import it before gwfast (applies the scipy shim).
  - `compat.py` — scipy/numpy shims for gwfast on modern envs.
  - `detectors.py` — network presets (`L1_design`, `HLV_O4`, `HLV_design`, `ET`, `ET_CE`) built from gwfast's detector dict + shipped PSDs; `build_network()`.
  - `params.py` — `SourceParams` (source-frame m1, q, z, spins, extrinsic angles) → gwfast event dict, with `(1+z)` mass redshifting and z→dL via Planck18; `chi_eff()` reference. Carries in-plane spins `s1x,s1y,s2x,s2y` (default 0 → aligned path unchanged); `spins_cartesian()` and `from_spherical(a_i, tilt_i, phi_i)` for precession. gwfast parameter order: `[Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z]`.
  - `fisher.py` — `FisherWorkspace`: builds the network once, computes SNR and the full Fisher via gwfast (aligned-spin).
  - `ripple_fisher.py` — `RippleFisherWorkspace`: autodiff Fisher from ripple waveforms (`jax.jacfwd` + PSD inner product), reusing `detectors.py` PSDs. Aligned (`IMRPhenomD`, `IMRPhenomXAS`) reproduces gwfast eigenvectors to 3 decimals; precessing (`IMRPhenomPv2`) gives the 6D spin block. Mass slots are `(log m1, log m2)` directly (AD through `(Mc, eta)`).
  - `eigen.py` — sub-space reductions (`marginalize` [recipe 1, via `CovMatr`], `marginalize_schur`/`marginalize_fisher` [Schur complement, covariance vs Fisher eigendecomp], `whiten_then_marginalize` [recipe 2], `condition` [recipe 3, via `fixParams`]), eigendecomposition (`eigen_from_cov` smallest-variance, `eigen_from_fisher` largest-eigenvalue, `analyze_subspace`), and `enforce_continuity` (tracks modes by matching *orthonormal whitened* eigenvectors).
- `scripts/run_grid.py` — sweep a grid, save eigen-results to `.npz`. `scripts/plot_grid.py` — plot.
- `tests/test_pipeline.py` — regression tests (`python -m pytest tests/ -q`).

**Mass basis:** the mass sub-space uses `(log m1, log m2)` (`FisherWorkspace(mass_coords="logm1logm2")`, via gwfast's `use_m1m2=True` plus a diagonal log-Jacobian `F_log = diag(m)·F·diag(m)` on the detector-frame component masses). Reference gradient-weight ratios in this basis: chirp mass → `(3+2q)/(2+3q)`, total mass → `q`.

**Validation status:**
- *Spin (aligned)* — best-measured `(chi1z, chi2z)` weight ratio tracks `q` (chi_eff): q=0.3→0.28, 0.5→0.46, 1.0→1.00. gwfast and ripple agree to 3 decimals.
- *Mass* — best-measured `(log m1, log m2)` weight ratio reproduces chirp mass at low mass and **transitions to total mass at high mass** (q=0.6: Mtot=16→1.11≈chirp 1.105; Mtot=480→0.63≈total 0.60), the chirp-mass generalization.
- *Spin (precessing, IMRPhenomPv2, `marginalize_fisher`)* — 6D spin field works: the **best-measured mode is generalized chi_eff** (z-dominated, `(s1z,s2z)` ratio ≈ q, e.g. q=0.6→0.55, in-plane fraction ~0.09); the **2nd mode is chi_p-like** (picks up the primary in-plane spin `s1x`). Remaining 4 in-plane directions are ~unmeasured (variance → ∞). Next: track modes 0,1 across a (q, tilt, m1) grid and fit a closed-form chi_p generalisation.

## Closed-form fits (deliverable 2)

`omp/fit.py` + `scripts/fit_directions.py` (vectorised + batched Fisher over a `(q, m1)` grid).

- **Spin (done).** Best-measured spin direction is a smooth generalisation of chi_eff, slightly below it, with weak mass dependence (L1 design, IMRPhenomD):
  `c2/c1 = q*(1 - 0.337*(1-q) + 0.359*(1-q)^2)`, fit RMS ~0.018, max deviation from chi_eff ~0.10.
- **Mass (2D fit done).** The `(Mtot, q)` map (`scripts/mass_grid2d.py`, cached `notes/figs/mass_joint.npz`) shows the surface separates into a **q-independent chirp→total transition** (sigmoid in log Mtot, midpoint Mtot ~ 216 Msun) **plus a separable overshoot bump** (amplitude ~0.94·(1−q), centred Mtot ~ 135 Msun, width 0.12 in log10). `omp.fit.fit_mass_2d` fits this form: **RMS ~0.030** vs ~0.12 for the bump-less sigmoid (4× better). Residual concentrates at the low-q overshoot tip (true bump is slightly asymmetric → skew-bump is the next refinement). The old 1D `fit_mass` (RMS ~0.43 on the (q,m1) grid) is superseded. q=1 still near-degenerate (chirp≈total there; a few points need gwfast's SVD fallback) but harmless since there is no structure to fit at q→1.

## Open methodological questions (to revisit)

- **Marginalize vs whiten order (recipes 1/2/3 + Schur/Fisher).** Recipe 1 for well-conditioned (aligned 2D); `marginalize_fisher` (Schur + Fisher-diagonal whitening, largest-eigenvalue) is the stable choice for ill-conditioned/sloppy sub-spaces (precessing 6D, and likely the q=1 mass degeneracy — worth retrying there). All recipes agree on well-conditioned cases. Still want a systematic stability sweep across grids / 3G networks.
- **Source-frame masses + z.** `params.py` redshifts masses to detector frame for the waveform, but the Fisher is currently taken w.r.t. gwfast-native `(Mc, eta)` (detector frame). Differentiating w.r.t. source-frame `(m1, q, z)` needs a Jacobian transform (gwfast provides `J_*` helpers); the mass-z degeneracy means z is essentially fixed by dL/cosmology.
