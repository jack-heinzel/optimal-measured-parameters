# optimal-measured-parameters

Best-measured intrinsic parameters of a gravitational-wave source as smooth
functions of the true parameters — generalisations of `chi_eff` (spins) and
chirp mass (masses), derived from the Fisher information matrix.

See `CLAUDE.md` for the full method, conventions, and status.

## Setup

Uses the `gwjax313` conda environment. gwfast is installed `--no-deps`
(`pip install --no-deps gwfast numdifftools`); `omp/compat.py` shims one removed
scipy symbol. **Import `omp` before gwfast.**

## Quick start

```python
from omp.params import SourceParams
from omp.fisher import FisherWorkspace
from omp.eigen import analyze_subspace

ws = FisherWorkspace(network="L1_design", waveform="IMRPhenomD")
src = SourceParams(m1=35., q=0.7, z=0.1, chi1z=0.2, chi2z=0.0)
F = ws.fisher(src)                      # full Fisher, gwfast-native params
res = analyze_subspace(F, ws.par_order, ["chi1z", "chi2z"],
                       method="marginalize", whiten="subcov",
                       sign_ref=[1.0, 0.7])[0]
print(res.best)                          # best-measured spin combination ~ chi_eff
```

## Scripts

```bash
PY=/home/jack.heinzel/.conda/envs/gwjax313/bin/python
$PY scripts/run_grid.py --network L1_design --nq 25 --out data/grid.npz
$PY scripts/plot_grid.py --grid data/grid.npz --out data/grid.png
$PY -m pytest tests/ -q
```

## Layout

- `omp/` — package: `compat`, `detectors`, `params`, `fisher`, `eigen`.
- `scripts/` — `run_grid.py`, `plot_grid.py`.
- `tests/` — regression tests.
