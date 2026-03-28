"""Chebyshev (Clenshaw-Curtis) integration matching Fortran's integration.f90.

The integral of a function f(r) over [r_icb, r_cmb] is computed via:
1. Undo mapping: g = f / drx
2. Forward DCT: a = costf(g)
3. Apply Chebyshev integration weights: sum = a[0]*w[0] + sum_{odd n} a[n]*w[n]
4. Normalize: integral = 2 * rnorm * sum

where the weights are: w[0] = boundary_fac, w[odd n>=2] = -1/(n*(n-2)),
using our costf convention.
"""

import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max
from .chebyshev import drx, rnorm, boundary_fac
from .cosine_transform import costf


def _build_cheb_int_weights():
    """Build Chebyshev integration weights.

    Matches radial.f90 lines 787-793 (1-based indexing → 0-based here):
    cheb_int[0] = 1.0
    cheb_int[n] = -1/(n*(n-2)) for odd n (1-based: n=3,5,...)
    cheb_int[n] = 0 for even n > 0

    But in our 0-based indexing, "1-based index k" = "0-based index k-1":
    Fortran cheb_int(1) = 1.0  → our index 0: weight = 1.0
    Fortran cheb_int(3) = -1/(3*1)  → our index 2: weight = -1/3
    Fortran cheb_int(5) = -1/(5*3)  → our index 4: weight = -1/15
    etc.
    """
    w = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
    w[0] = 1.0
    for n in range(2, n_r_max, 2):  # 0-based indices 2, 4, 6, ...
        # In Fortran 1-based: this is index n+1 = 3, 5, 7, ...
        n1 = n + 1  # Fortran 1-based index
        w[n] = -1.0 / (n1 * (n1 - 2))
    return w


_cheb_int = _build_cheb_int_weights()


def rInt_R(f: torch.Tensor) -> torch.Tensor:
    """Chebyshev (Clenshaw-Curtis) radial integration.

    Integrates f(r) over [r_icb, r_cmb].

    Args:
        f: shape (..., n_r_max), function values at radial grid points

    Returns:
        integral: shape (...), the definite integral
    """
    # Undo mapping
    g = f / drx

    # Forward DCT
    a = costf(g)

    # Apply boundary factor to endpoints
    a[..., 0] = a[..., 0] * boundary_fac
    a[..., -1] = a[..., -1] * boundary_fac

    # Sum with integration weights
    result = (a * _cheb_int).sum(dim=-1)

    # Normalize
    result = 2.0 * rnorm * result

    return result
