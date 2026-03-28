"""Chebyshev infrastructure matching Fortran's chebyshev.f90 + chebyshev_polynoms.f90.

Provides:
- Chebyshev-Gauss-Lobatto grid on [r_icb, r_cmb]
- Polynomial evaluation matrices (rMat, drMat, d2rMat, d3rMat)
- Mapping derivatives (drx, ddrx, dddrx)
- Boundary differentiation vectors (dr_top, dr_bot)

All initialization built on CPU to avoid GPU sync overhead, then transferred to DEVICE.
"""

import math
import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, n_cheb_max, radratio
from .constants import pi, one, two, three, four, half


def _cheb_grid(ricb: float, rcmb: float, N: int):
    """Compute Chebyshev-Gauss-Lobatto grid (no mapping).

    Returns:
        r: radial grid points, shape (N+1,), r[0]=rcmb, r[N]=ricb
        x_cheb: Chebyshev points in [-1,1], shape (N+1,), x[0]=1, x[N]=-1
    """
    k = torch.arange(N + 1, dtype=DTYPE, device="cpu")
    x_cheb = torch.cos(pi * k / N)
    bma = half * (rcmb - ricb)
    bpa = half * (ricb + rcmb)
    r = bma * x_cheb + bpa
    return r.to(DEVICE), x_cheb.to(DEVICE)


# Compute grid
r_cmb = one / (one - radratio)  # ~1.538461538...
r_icb = r_cmb - one             # ~0.538461538...

r, x_cheb = _cheb_grid(r_icb, r_cmb, n_r_max - 1)

# Mapping derivatives for the linear (no-map) case
drx = torch.full((n_r_max,), two / (r_cmb - r_icb), dtype=DTYPE, device=DEVICE)
ddrx = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
dddrx = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

# Normalization factor for Chebyshev transforms
rnorm = math.sqrt(two / (n_r_max - 1))

# Boundary factor (halves first/last Chebyshev coefficients)
boundary_fac = half


def _build_der_mats():
    """Build Chebyshev polynomial matrices and their derivatives.

    Matches get_der_mat in chebyshev.f90. Uses the recursion:
        T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    with the chain rule baked in for mapping derivatives.

    Built on CPU to avoid GPU sync from column loop, transferred to DEVICE.

    Returns (rMat, drMat, d2rMat, d3rMat), each shape (n_r_max, n_r_max).
    rMat[i,n] = T_{n-1}(x_cheb[i])  (Fortran convention: 1-indexed columns)
    """
    n = n_r_max
    _x = x_cheb.cpu()
    _drx = drx.cpu()
    _ddrx = ddrx.cpu()
    _dddrx = dddrx.cpu()

    rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    drMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    d2rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    d3rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")

    # Column 0 (T_0 = 1)
    rMat[:, 0] = one
    # drMat[:, 0] = 0 already
    # d2rMat[:, 0] = 0 already
    # d3rMat[:, 0] = 0 already

    # Column 1 (T_1 = x)
    rMat[:, 1] = _x
    drMat[:, 1] = _drx
    d2rMat[:, 1] = _ddrx
    d3rMat[:, 1] = _dddrx

    # Recursion for columns 2..n_r_max-1
    for col in range(2, n):
        rMat[:, col] = two * _x * rMat[:, col - 1] - rMat[:, col - 2]
        drMat[:, col] = (two * _drx * rMat[:, col - 1]
                         + two * _x * drMat[:, col - 1]
                         - drMat[:, col - 2])
        d2rMat[:, col] = (two * _ddrx * rMat[:, col - 1]
                          + four * _drx * drMat[:, col - 1]
                          + two * _x * d2rMat[:, col - 1]
                          - d2rMat[:, col - 2])
        d3rMat[:, col] = (two * _dddrx * rMat[:, col - 1]
                          + 6.0 * _ddrx * drMat[:, col - 1]
                          + 6.0 * _drx * d2rMat[:, col - 1]
                          + two * _x * d3rMat[:, col - 1]
                          - d3rMat[:, col - 2])

    return rMat.to(DEVICE), drMat.to(DEVICE), d2rMat.to(DEVICE), d3rMat.to(DEVICE)


rMat, drMat, d2rMat, d3rMat = _build_der_mats()


def _build_dr_boundary():
    """Build boundary differentiation vectors for robin_bc.

    Matches get_der_mat lines 357-379 in chebyshev.f90.
    Built on CPU to avoid GPU sync from scalar loop.
    """
    n = n_r_max
    N = n - 1  # = n_r_max - 1

    dr_top = torch.zeros(n, dtype=DTYPE, device="cpu")
    dr_top[0] = (two * N * N + one) / 6.0

    for k in range(1, n):
        diff = two * math.sin(half * k * pi / N) ** 2
        coeff = -one if (k + 1) % 2 == 0 else one  # k+1 because 0-indexed
        dr_top[k] = two * coeff / diff

    # Factor half for the last one
    dr_top[n - 1] = half * dr_top[n - 1]

    # dr_bot is reversed and negated (Baltensperger & Trummer trick)
    dr_bot = -dr_top.flip(0)

    # Multiply by mapping factor
    _drx = drx.cpu()
    dr_top = dr_top * _drx[0]
    dr_bot = dr_bot * _drx[n - 1]

    return dr_top.to(DEVICE), dr_bot.to(DEVICE)


dr_top, dr_bot = _build_dr_boundary()
