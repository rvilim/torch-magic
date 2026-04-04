"""Radial discretization dispatch layer.

Provides a unified interface for radial grid, derivative matrices,
spectral transforms, and integration, regardless of the backend
(Chebyshev or finite differences).

All consumer modules should import from here instead of directly
from chebyshev.py or finite_differences.py.

The scheme is selected once at import time via params.radial_scheme
and is immutable thereafter.
"""

from .params import radial_scheme as _scheme, l_finite_diff

if _scheme == "FD":
    from .finite_differences import (  # noqa: F401
        r, r_cmb, r_icb,
        rMat, drMat, d2rMat, d3rMat, d4rMat,
        drx, ddrx, dddrx,
        rnorm, boundary_fac,
        dr_top, dr_bot,
    )
    from .integration import rInt_R  # noqa: F401

    x_cheb = None  # FD has no Chebyshev points

    def costf(f):
        """No-op for FD (fields are already in physical space)."""
        return f
else:
    # Chebyshev backend
    from .chebyshev import (  # noqa: F401
        r, r_cmb, r_icb, x_cheb,
        rMat, drMat, d2rMat, d3rMat,
        drx, ddrx, dddrx,
        rnorm, boundary_fac,
        dr_top, dr_bot,
    )
    from .cosine_transform import costf  # noqa: F401
    from .integration import rInt_R  # noqa: F401

    d4rMat = None  # Chebyshev doesn't use 4th derivative matrix


def to_physical(coeffs):
    """Convert from spectral coefficients to physical (grid-point) space.

    For Chebyshev: applies DCT-I (costf, self-inverse).
    For FD: identity (fields are already in physical space).
    """
    if _scheme == "FD":
        return coeffs
    else:
        return costf(coeffs)


# IC always uses Chebyshev regardless of OC scheme.
# The IC block solve produces even-Chebyshev coefficients that always
# need a DCT transform to physical space.
from .cosine_transform import costf as ic_costf  # noqa: F401
