"""omp -- optimal measured parameters of gravitational-wave sources.

Compute the best-measured linear combinations of intrinsic source parameters
(generalisations of chi_eff and chirp mass) from the Fisher information matrix,
as smooth functions of the true source parameters.

Importing this package installs a small compatibility shim (see ``omp.compat``)
so that gwfast 1.1.2 runs against the modern scipy/numpy in the ``gwjax313``
environment.
"""

from . import compat  # noqa: F401  (applies scipy/numpy shims on import)

__all__ = ["compat"]
