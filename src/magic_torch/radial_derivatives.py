"""Radial derivatives via Chebyshev spectral methods.

Two approaches implemented:
1. Matrix multiply using drMat/d2rMat/d3rMat (GPU-friendly, used by default)
2. DCT-based Chebyshev differentiation (costf → dcheb → costf)

Both give identical results to machine precision.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .radial_scheme import drMat, d2rMat, d3rMat, drx, ddrx, dddrx, rnorm, boundary_fac
from .params import n_r_max, n_cheb_max, l_cond_ic
from .cosine_transform import costf


def get_dr(f: torch.Tensor) -> torch.Tensor:
    """First radial derivative.

    Uses banded matvec for FD, dense matmul for Chebyshev.
    """
    if _D1_bands is not None:
        return _banded_matvec(_D1_bands, f)
    return f @ _D1.T


def get_ddr(f: torch.Tensor):
    """First and second radial derivatives.

    Returns:
        (df, ddf): both shape (..., n_r_max)
    """
    if _D1_bands is not None:
        return _banded_matvec(_D1_bands, f), _banded_matvec(_D2_bands, f)
    return f @ _D1.T, f @ _D2.T


def get_d4(f: torch.Tensor) -> torch.Tensor:
    """Fourth radial derivative (FD only)."""
    if _D4_bands is not None:
        return _banded_matvec(_D4_bands, f)
    if _D4 is None:
        raise RuntimeError("D4 not available (Chebyshev has no d4rMat)")
    return f @ _D4.T


def get_dddr(f: torch.Tensor):
    """First, second, and third radial derivatives.

    Returns:
        (df, ddf, dddf): all shape (..., n_r_max)
    """
    if _D1_bands is not None:
        return (_banded_matvec(_D1_bands, f),
                _banded_matvec(_D2_bands, f),
                _banded_matvec(_D3_bands, f))
    return f @ _D1.T, f @ _D2.T, f @ _D3.T


def _build_diff_matrices():
    """Build differentiation matrices D1, D2, D3 (and D4 for FD).

    D_k = d^k_rMat @ rMat^{-1} gives the k-th derivative operator
    in physical space. For Chebyshev, rMat is the polynomial matrix.
    For FD, rMat = I, so D_k = d^k_rMat directly.

    When n_cheb_max < n_r_max, high Chebyshev modes are zeroed out
    (spectral truncation), matching the Fortran get_dcheb behavior.
    """
    from .radial_scheme import rMat, d4rMat

    rMat_inv = torch.linalg.inv(rMat)

    # Spectral truncation: zero out high-order Chebyshev modes
    if n_cheb_max < n_r_max:
        rMat_inv = rMat_inv.clone()
        rMat_inv[n_cheb_max:, :] = 0.0

    D1 = drMat @ rMat_inv
    D2 = d2rMat @ rMat_inv
    D3 = d3rMat @ rMat_inv
    D4 = d4rMat @ rMat_inv if d4rMat is not None else None
    return D1, D2, D3, D4


_D1_real, _D2_real, _D3_real, _D4_real = _build_diff_matrices()
# Complex versions for use with complex fields
_D1 = _D1_real.to(CDTYPE)
_D2 = _D2_real.to(CDTYPE)
_D3 = _D3_real.to(CDTYPE)
_D4 = _D4_real.to(CDTYPE) if _D4_real is not None else None


# --- Banded derivative matvec for FD ---

def _extract_bands(D):
    """Extract nonzero diagonals from a matrix as (offset, values) pairs.

    Args:
        D: (N, N) dense matrix

    Returns:
        list of (offset, diag_values) tuples where offset is the diagonal
        index (0=main, +k=k-th superdiag, -k=k-th subdiag) and diag_values
        is a 1-D tensor of the diagonal entries.
    """
    N = D.shape[0]
    bands = []
    for off in range(-(N - 1), N):
        diag = torch.diagonal(D, offset=off)
        if diag.abs().max() > 0:
            bands.append((off, diag.clone()))
    return bands


def _banded_matvec(bands, f):
    """Apply banded matrix D to batched vectors: result = f @ D.T.

    Equivalent to result[b, j] = sum_k D[j, k] * f[b, k], but computed
    as a sum of shifted elementwise multiplies over the nonzero diagonals.

    Args:
        bands: list of (offset, diag) from _extract_bands(D)
        f: (..., N) input tensor

    Returns:
        (..., N) result
    """
    N = f.shape[-1]
    result = torch.zeros_like(f)
    for off, diag in bands:
        # D[j, j+off] = diag[j]  (for valid j)
        # result[..., j] += diag[j] * f[..., j+off]
        if off >= 0:
            # j ranges from 0 to N-1-off; k = j+off ranges from off to N-1
            result[..., :N - off] += diag[:N - off] * f[..., off:]
        else:
            # j ranges from -off to N-1; k = j+off ranges from 0 to N-1+off
            result[..., -off:] += diag * f[..., :N + off]
    return result


# Build banded representations for FD (None for Chebyshev)
from .params import l_finite_diff as _l_fd

_D1_bands = None
_D2_bands = None
_D3_bands = None
_D4_bands = None

if _l_fd:
    _D1_bands = [(off, d.to(CDTYPE).to(DEVICE))
                 for off, d in _extract_bands(_D1_real)]
    _D2_bands = [(off, d.to(CDTYPE).to(DEVICE))
                 for off, d in _extract_bands(_D2_real)]
    _D3_bands = [(off, d.to(CDTYPE).to(DEVICE))
                 for off, d in _extract_bands(_D3_real)]
    if _D4_real is not None:
        _D4_bands = [(off, d.to(CDTYPE).to(DEVICE))
                     for off, d in _extract_bands(_D4_real)]


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
