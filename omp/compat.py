"""Compatibility shims so gwfast 1.1.2 runs on modern scipy/numpy.

gwfast pins ``numpy<=1.26.4`` and ``scipy`` old enough to still expose
``scipy.integrate.cumtrapz``.  The ``gwjax313`` environment ships scipy >= 1.14
(``cumtrapz`` renamed to ``cumulative_trapezoid``) and numpy >= 2.x.  We avoid
downgrading the environment by aliasing the handful of renamed symbols that
gwfast still calls.  Import this module *before* importing gwfast.
"""

import scipy.integrate as _si

# scipy >= 1.14 removed cumtrapz in favour of cumulative_trapezoid.
if not hasattr(_si, "cumtrapz") and hasattr(_si, "cumulative_trapezoid"):
    _si.cumtrapz = _si.cumulative_trapezoid

# numpy 2.x still provides np.trapz (with a DeprecationWarning); nothing to do
# there yet.  If a future numpy removes it, alias np.trapezoid here.
import numpy as _np

if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid
