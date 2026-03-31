"""Radial derivatives via Chebyshev spectral methods.

Two approaches implemented:
1. Matrix multiply using drMat/d2rMat/d3rMat (GPU-friendly, used by default)
2. DCT-based Chebyshev differentiation (costf → dcheb → costf)

Both give identical results to machine precision.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .chebyshev import drMat, d2rMat, d3rMat, drx, ddrx, dddrx, rnorm, boundary_fac
from .params import n_r_max, n_cheb_max, l_cond_ic
from .cosine_transform import costf


def get_dr(f: torch.Tensor) -> torch.Tensor:
    """First radial derivative via matrix multiply.

    Args:
        f: shape (..., n_r_max), real or complex

    Returns:
        df: shape (..., n_r_max), first derivative df/dr
    """
    # drMat[i, n] contains dT_n/dr evaluated at r[i]
    # f[..., i] is the physical-space value at r[i]
    # In Chebyshev space: f(r_i) = sum_n a_n * T_n(r_i) = rMat[i,:] . a
    # df/dr(r_i) = sum_n a_n * dT_n/dr(r_i) = drMat[i,:] . a
    # But since rMat is invertible: a = rMat^{-1} . f, so df = drMat . rMat^{-1} . f
    # However, for Chebyshev collocation: drMat . rMat^{-1} is the differentiation matrix D
    # We compute D = drMat @ rMat_inv once
    return f @ _D1.T


def get_ddr(f: torch.Tensor):
    """First and second radial derivatives via matrix multiply.

    Returns:
        (df, ddf): both shape (..., n_r_max)
    """
    df = f @ _D1.T
    ddf = f @ _D2.T
    return df, ddf


def get_dddr(f: torch.Tensor):
    """First, second, and third radial derivatives via matrix multiply.

    Returns:
        (df, ddf, dddf): all shape (..., n_r_max)
    """
    df = f @ _D1.T
    ddf = f @ _D2.T
    dddf = f @ _D3.T
    return df, ddf, dddf


def _build_diff_matrices():
    """Build Chebyshev differentiation matrices D1, D2, D3.

    D_k = d^k_rMat @ rMat^{-1} gives the k-th derivative operator
    in physical space.

    When n_cheb_max < n_r_max, high Chebyshev modes are zeroed out
    (spectral truncation), matching the Fortran get_dcheb behavior.
    """
    from .chebyshev import rMat

    rMat_inv = torch.linalg.inv(rMat)

    # Spectral truncation: zero out high-order Chebyshev modes
    if n_cheb_max < n_r_max:
        rMat_inv = rMat_inv.clone()
        rMat_inv[n_cheb_max:, :] = 0.0

    D1 = drMat @ rMat_inv
    D2 = d2rMat @ rMat_inv
    D3 = d3rMat @ rMat_inv
    return D1, D2, D3


_D1_real, _D2_real, _D3_real = _build_diff_matrices()
# Complex versions for use with complex fields
_D1 = _D1_real.to(CDTYPE)
_D2 = _D2_real.to(CDTYPE)
_D3 = _D3_real.to(CDTYPE)


# --- DCT-based derivatives (alternative, matches Fortran get_dr exactly) ---

def _get_dcheb(f: torch.Tensor) -> torch.Tensor:
    """Chebyshev derivative recurrence in spectral space.

    Matches get_dcheb from radial_derivatives.f90.
    Input f is in "costf space" (self-inverse DCT-I coefficients).
    Output df is the derivative in the same space.

    The recurrence (0-indexed, with s[n] = costf coefficients):
        df[N] = 0
        df[N-1] = fac * f[N]    where fac = N (if n_r_max == n_cheb_max) else 2*N
        df[n] = df[n+2] + 2*(n+1) * f[n+1]   for n = N-2, ..., 0
    """
    N = n_r_max - 1
    fac = float(n_cheb_max - 1) if n_r_max == n_cheb_max else 2.0 * float(n_cheb_max - 1)

    df = torch.zeros_like(f)
    df[..., n_cheb_max - 1] = 0.0
    if n_cheb_max >= 2:
        df[..., n_cheb_max - 2] = fac * f[..., n_cheb_max - 1]

    for n in range(n_cheb_max - 3, -1, -1):
        df[..., n] = df[..., n + 2] + 2.0 * (n + 1) * f[..., n + 1]

    return df


def get_dr_costf(f: torch.Tensor) -> torch.Tensor:
    """First radial derivative via costf + Chebyshev recurrence.

    This matches the Fortran get_dr code path exactly:
    1. Forward DCT: physical → spectral
    2. Chebyshev derivative recurrence
    3. Inverse DCT: spectral → physical
    4. Mapping correction
    """
    # Forward transform
    f_cheb = costf(f)
    # Differentiate in spectral space
    df_cheb = _get_dcheb(f_cheb)
    # Inverse transform (costf is self-inverse)
    df = costf(df_cheb)
    # Apply mapping correction
    df = drx * df
    return df


# --- IC (inner core) derivatives using even Chebyshev polynomials ---

def _build_ic_diff_matrices():
    """Build IC differentiation matrices D1_ic, D2_ic.

    Uses the even Chebyshev polynomial matrices: cheb_ic, dcheb_ic, d2cheb_ic.
    D_k = d^k_cheb_ic @ inv(cheb_ic) maps grid values to k-th derivative values.
    """
    from .radial_functions import cheb_ic, dcheb_ic, d2cheb_ic
    cheb_inv = torch.linalg.inv(cheb_ic)
    # D maps grid values to derivatives: df = f @ D
    # f = coeff @ cheb_ic → coeff = f @ inv(cheb_ic)
    # df = coeff @ dcheb_ic = f @ inv(cheb_ic) @ dcheb_ic
    D1 = cheb_inv @ dcheb_ic
    D2 = cheb_inv @ d2cheb_ic
    return D1, D2


if l_cond_ic:
    _D1_ic_real, _D2_ic_real = _build_ic_diff_matrices()
    _D1_ic = _D1_ic_real.to(CDTYPE).to(DEVICE)
    _D2_ic = _D2_ic_real.to(CDTYPE).to(DEVICE)
else:
    _D1_ic = None
    _D2_ic = None


def get_ddr_even(f: torch.Tensor):
    """First and second radial derivatives for IC fields using even Chebyshev.

    Matches get_ddr_even from radial_derivatives_even.f90.

    Args:
        f: shape (..., n_r_ic_max) complex IC field values at grid points

    Returns:
        (df, ddf): both shape (..., n_r_ic_max)
    """
    df = f @ _D1_ic
    ddf = f @ _D2_ic
    return df, ddf
