"""Radial integration matching Fortran's integration.f90.

Two backends:
- Chebyshev: Clenshaw-Curtis via DCT + spectral weights
- FD: Simpson's rule on non-uniform grid
"""

import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, l_finite_diff

if not l_finite_diff:
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


if not l_finite_diff:
    _cheb_int = _build_cheb_int_weights()


def _rInt_R_cheb(f: torch.Tensor) -> torch.Tensor:
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


def simps(f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Simpson's rule on non-uniform grid. Matches integration.f90 simps.

    Args:
        f: shape (..., N), function values
        r: shape (N,), radial grid points (decreasing: r[0]=r_cmb > r[-1]=r_icb)

    Returns:
        integral: shape (...), the definite integral (negated because r decreases)
    """
    N = r.shape[0]

    if N % 2 == 1:
        # Odd number of points: standard composite Simpson
        result = torch.zeros(f.shape[:-1], dtype=f.dtype, device=f.device)
        for n_r in range(1, N - 1, 2):  # 0-based: 1,3,5,...,N-2
            h2 = r[n_r + 1] - r[n_r]
            h1 = r[n_r] - r[n_r - 1]
            result = result + (h1 + h2) / 6.0 * (
                f[..., n_r - 1] * (2.0 * h1 - h2) / h1
                + f[..., n_r] * (h1 + h2) * (h1 + h2) / (h1 * h2)
                + f[..., n_r + 1] * (2.0 * h2 - h1) / h2
            )
        return -result
    else:
        # Even: trapezoidal on first interval + Simpson on rest,
        # then Simpson on all even pairs + trapezoidal on last, average both
        # (Fortran: "twice simpson + trapz on first and last points")
        result = 0.5 * (r[1] - r[0]) * (f[..., 1] + f[..., 0])
        for n_r in range(2, N - 1, 2):  # 0-based: 2,4,...
            h2 = r[n_r + 1] - r[n_r]
            h1 = r[n_r] - r[n_r - 1]
            result = result + (h1 + h2) / 6.0 * (
                f[..., n_r - 1] * (2.0 * h1 - h2) / h1
                + f[..., n_r] * (h1 + h2) * (h1 + h2) / (h1 * h2)
                + f[..., n_r + 1] * (2.0 * h2 - h1) / h2
            )
        result = result + 0.5 * (r[N - 1] - r[N - 2]) * (f[..., N - 1] + f[..., N - 2])
        for n_r in range(1, N - 1, 2):  # 0-based: 1,3,...
            h2 = r[n_r + 1] - r[n_r]
            h1 = r[n_r] - r[n_r - 1]
            result = result + (h1 + h2) / 6.0 * (
                f[..., n_r - 1] * (2.0 * h1 - h2) / h1
                + f[..., n_r] * (h1 + h2) * (h1 + h2) / (h1 * h2)
                + f[..., n_r + 1] * (2.0 * h2 - h1) / h2
            )
        return -0.5 * result


def _rInt_R_fd(f: torch.Tensor) -> torch.Tensor:
    """FD radial integration via Simpson's rule.

    Args:
        f: shape (..., n_r_max), function values at radial grid points

    Returns:
        integral: shape (...), the definite integral
    """
    from .radial_scheme import r
    return simps(f, r)


# Dispatch
rInt_R = _rInt_R_fd if l_finite_diff else _rInt_R_cheb
