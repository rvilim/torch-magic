"""Horizontal data: Gauss-Legendre grid, theta/phi arrays, coupling coefficients.

Matches horizontal.f90 for the non-scrambled theta case.
"""

import math
import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import (l_max, m_max, m_min, minc, n_theta_max, n_phi_max,
                     n_m_max, lm_max, ek)
from .blocking import st_lm2l, st_lm2m
from .constants import pi, one, two, half


def _gauleg(x_min: float, x_max: float, n: int):
    """Gauss-Legendre quadrature points and weights.

    Returns theta_ord (colatitudes in radians) and gauss (weights).
    Matches the gauleg subroutine in horizontal.f90 exactly.
    Built on CPU to avoid GPU sync overhead from scalar loops.
    """
    eps = 10.0 * torch.finfo(DTYPE).eps
    theta_ord = torch.zeros(n, dtype=DTYPE, device="cpu")
    gauss = torch.zeros(n, dtype=DTYPE, device="cpu")

    m = (n + 1) // 2
    xm = 0.5 * (x_max + x_min)
    xl = 0.5 * (x_max - x_min)

    for i in range(1, m + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        z1 = z + 10.0 * eps

        while abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp

        theta_ord[i - 1] = math.acos(xm + xl * z)
        theta_ord[n - i] = math.acos(xm - xl * z)
        gauss[i - 1] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        gauss[n - i] = gauss[i - 1]

    return theta_ord.to(DEVICE), gauss.to(DEVICE)


# Compute Gauss-Legendre quadrature
theta_ord, gauss = _gauleg(-1.0, 1.0, n_theta_max)

# Theta-dependent arrays in sorted order (for Plm evaluation etc.)
sinTheta = torch.sin(theta_ord)
cosTheta = torch.cos(theta_ord)
O_sin_theta = 1.0 / sinTheta
O_sin_theta_E2 = O_sin_theta ** 2
sinTheta_E2 = sinTheta ** 2
cosn_theta_E2 = cosTheta * O_sin_theta_E2

# --- Grid-space theta arrays (interleaved N/S layout matching SHT output) ---
# The SHT produces fields in interleaved order: [θ_N[0], θ_S[0], θ_N[1], θ_S[1], ...]
# where θ_S[j] = π - θ_N[j] are paired points. Build a permutation index.
_NHS = n_theta_max // 2
_grid_idx = torch.zeros(n_theta_max, dtype=torch.long, device=DEVICE)
_grid_idx[0::2] = torch.arange(_NHS, device=DEVICE)                       # N hemisphere
_grid_idx[1::2] = torch.arange(n_theta_max - 1, _NHS - 1, -1, device=DEVICE)  # S hemisphere

theta_grid = theta_ord[_grid_idx]
sinTheta_grid = torch.sin(theta_grid)
cosTheta_grid = torch.cos(theta_grid)
O_sin_theta_E2_grid = 1.0 / sinTheta_grid ** 2
cosn_theta_E2_grid = cosTheta_grid * O_sin_theta_E2_grid
gauss_grid = gauss[_grid_idx]  # Gauss weights in interleaved N/S order
O_sin_theta_grid = 1.0 / sinTheta_grid  # 1/sin(theta) in interleaved order

# n_theta_cal2ord: maps interleaved (calculation) order to geographic (sorted) order
# Inverse of _grid_idx: _grid_idx maps geo→interleaved, n_theta_cal2ord maps interleaved→geo
n_theta_cal2ord = torch.zeros(n_theta_max, dtype=torch.long, device=DEVICE)
n_theta_cal2ord[_grid_idx] = torch.arange(n_theta_max, device=DEVICE)

# Phi grid
phi = torch.arange(n_phi_max, dtype=DTYPE, device=DEVICE) * (2.0 * pi / (n_phi_max * minc))

# --- LM-dependent arrays (built on CPU, transferred to DEVICE) ---


def _clm(l, m):
    """Coupling coefficient clm = sqrt((l+m)(l-m) / ((2l-1)(2l+1))). Vectorized."""
    result = torch.zeros_like(l, dtype=DTYPE)
    mask = l > 0
    lm = l[mask].to(DTYPE)
    mm = m[mask].to(DTYPE)
    result[mask] = torch.sqrt((lm + mm) * (lm - mm) / ((2 * lm - 1) * (2 * lm + 1)))
    return result


def _build_lm_arrays():
    """Build LM-dependent coupling arrays — vectorized, on CPU then transfer."""
    l = st_lm2l.cpu().clone()
    m = st_lm2m.cpu().clone()
    lf = l.to(DTYPE)
    mf = m.to(DTYPE)

    clm_l = _clm(l, m)      # clm(l, m)
    clm_lp1 = _clm(l + 1, m)  # clm(l+1, m)

    _dPhi = torch.zeros(lm_max, dtype=CDTYPE, device="cpu")
    _dPhi.imag = mf

    _dLh = lf * (lf + 1)
    _dTheta1S = (lf + 1) * clm_l
    _dTheta1A = lf * clm_lp1
    _dTheta2S = (lf - 1) * clm_l
    _dTheta2A = (lf + 2) * clm_lp1
    _dTheta3S = (lf - 1) * (lf + 1) * clm_l
    _dTheta3A = lf * (lf + 2) * clm_lp1
    _dTheta4S = _dTheta1S * (lf - 1) * lf
    _dTheta4A = _dTheta1A * (lf + 1) * (lf + 2)

    return (_dPhi.to(DEVICE), _dLh.to(DEVICE),
            _dTheta1S.to(DEVICE), _dTheta1A.to(DEVICE),
            _dTheta2S.to(DEVICE), _dTheta2A.to(DEVICE),
            _dTheta3S.to(DEVICE), _dTheta3A.to(DEVICE),
            _dTheta4S.to(DEVICE), _dTheta4A.to(DEVICE))


(dPhi, dLh,
 dTheta1S, dTheta1A,
 dTheta2S, dTheta2A,
 dTheta3S, dTheta3A,
 dTheta4S, dTheta4A) = _build_lm_arrays()

# Hyperdiffusion (all 1.0 for the benchmark — no hyperdiffusion)
hdif_B = torch.ones(l_max + 1, dtype=DTYPE, device=DEVICE)
hdif_V = torch.ones(l_max + 1, dtype=DTYPE, device=DEVICE)
hdif_S = torch.ones(l_max + 1, dtype=DTYPE, device=DEVICE)
hdif_Xi = torch.ones(l_max + 1, dtype=DTYPE, device=DEVICE)
