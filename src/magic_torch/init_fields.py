"""Initial field setup matching init_fields.f90 + startFields.f90.

Implements:
- ps_cond: conduction state for entropy and pressure (Boussinesq)
- initS: entropy initialization with perturbation at (l=4, m=4)
- initB: magnetic field initialization (init_b1=3)

Specialized for the dynamo_benchmark case.
"""

import math
import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import (n_r_max, lm_max, l_max, init_b1, amp_b1,
                     init_s1, amp_s1, radratio)
from .constants import (pi, one, two, three, four, half, third, osq4pi,
                        sq4pi, zero)
from .chebyshev import (r, r_cmb, r_icb, rMat, drMat, d2rMat,
                        rnorm, boundary_fac)
from .radial_functions import or1, rgrav, rho0
from .pre_calculations import BuoFac, opr
from .algebra import prepare_mat, solve_mat_real
from .cosine_transform import costf
from .blocking import st_lm2


def ps_cond():
    """Compute conduction state for entropy and pressure.

    Solves the coupled 2N x 2N system for (s0, p0) in Chebyshev space.
    Boussinesq case: kappa=1, beta=0, dLtemp0=0, dLkappa=0, epsc=0.

    Returns:
        (s0, p0): each shape (n_r_max,) in physical (radial) space
    """
    N = n_r_max
    ps0Mat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=DEVICE)

    # --- Entropy diffusion (entropy equation, NOT l_temperature_diff) ---
    # ps0Mat[n_r, n_r_out] = rnorm * opr * kappa * (d2rMat + (2/r) * drMat)
    # For Boussinesq: kappa=1, opr=1, beta=0, dLtemp0=0, dLkappa=0
    for nr in range(N):
        for nc in range(N):
            ps0Mat[nr, nc] = rnorm * opr * (
                d2rMat[nr, nc] + two * or1[nr] * drMat[nr, nc]
            )
            # No pressure coupling in entropy equation for Boussinesq
            ps0Mat[nr, nc + N] = 0.0

            # Hydrostatic equilibrium: -rho0 * BuoFac * rgrav * rMat
            ps0Mat[nr + N, nc] = -rnorm * rho0[nr] * BuoFac * rgrav[nr] * rMat[nr, nc]

            # Pressure gradient: drMat (beta=0)
            ps0Mat[nr + N, nc + N] = rnorm * drMat[nr, nc]

    # --- Boundary conditions ---
    # CMB (row 0): s0(r_cmb) = tops(0,0), ktops=1 (fixed entropy)
    ps0Mat[0, :N] = rnorm * rMat[0, :]
    ps0Mat[0, N:] = 0.0

    # ICB (row N-1): s0(r_icb) = bots(0,0), kbots=1 (fixed entropy)
    ps0Mat[N - 1, :N] = rnorm * rMat[N - 1, :]
    ps0Mat[N - 1, N:] = 0.0

    # Pressure BC (row N): p0(r_cmb) = 0  (Boussinesq, ViscHeatFac*ThExpNb=0)
    ps0Mat[N, :N] = 0.0
    ps0Mat[N, N:] = rnorm * rMat[0, :]

    # --- Boundary factor scaling (columns 0, N-1, N, 2N-1) ---
    for nr in range(2 * N):
        ps0Mat[nr, 0] *= boundary_fac
        ps0Mat[nr, N - 1] *= boundary_fac
        ps0Mat[nr, N] *= boundary_fac
        ps0Mat[nr, 2 * N - 1] *= boundary_fac

    # --- Row scaling for conditioning ---
    ps0Mat_fac = 1.0 / ps0Mat.abs().max(dim=1).values

    ps0Mat = ps0Mat * ps0Mat_fac.unsqueeze(1)

    # --- LU factorize ---
    a_lu, ip, info = prepare_mat(ps0Mat)
    assert info == 0, f'Singular ps0Mat, info={info}'

    # --- RHS ---
    rhs = torch.zeros(2 * N, dtype=DTYPE, device=DEVICE)
    # Interior: -epsc * epscProf * orho1 = 0 for benchmark (no internal heating)
    # BCs:
    rhs[0] = 0.0       # tops(0,0) = 0
    rhs[N - 1] = sq4pi  # bots(0,0) = sqrt(4*pi)
    rhs[N] = 0.0        # p0(r_cmb) = 0

    # Scale RHS
    rhs = ps0Mat_fac * rhs

    # --- Solve ---
    rhs = solve_mat_real(a_lu, ip, rhs)

    # --- Extract s0 and p0 ---
    s0 = rhs[:N].clone()
    p0 = rhs[N:].clone()

    # --- Transform to physical space ---
    s0 = costf(s0)
    p0 = costf(p0)

    return s0, p0


def initS(s_LMloc, p_LMloc):
    """Initialize entropy field.

    1. Set conduction state in (l=0, m=0) mode
    2. Add perturbation at (l=4, m=4) mode with amp_s1

    Args:
        s_LMloc: (lm_max, n_r_max) complex, modified in-place
        p_LMloc: (lm_max, n_r_max) complex, modified in-place
    """
    # Compute conduction state
    s0, p0 = ps_cond()

    # Store in (l=0, m=0) mode (index 0 in standard ordering)
    s_LMloc[0, :] = s0.to(CDTYPE)
    p_LMloc[0, :] = p0.to(CDTYPE)

    # Radial perturbation function: (1 - x²)³ where x = 2r - r_cmb - r_icb
    x = two * r - r_cmb - r_icb
    s1 = one - three * x ** 2 + three * x ** 4 - x ** 6

    # init_s1 >= 100: initialize specific mode
    # init_s1 = 404 → l=4, m=4
    if init_s1 >= 100:
        l = init_s1 // 100
        m = init_s1 % 100
        lm = st_lm2[l, m].item()
        s_LMloc[lm, :] += (amp_s1 * s1).to(CDTYPE)


def initB(b_LMloc, aj_LMloc):
    """Initialize magnetic field for init_b1=3.

    Sets poloidal b(l=1, m=0) and toroidal aj(l=2, m=0).
    Insulating inner core (sigma_ratio=0).

    Args:
        b_LMloc, aj_LMloc: (lm_max, n_r_max) complex, modified in-place
    """
    assert init_b1 == 3, f'Only init_b1=3 is supported, got {init_b1}'

    # LM indices for l=1,m=0 and l=2,m=0 (standard ordering)
    l1m0 = st_lm2[1, 0].item()
    l2m0 = st_lm2[2, 0].item()

    # Poloidal field: b(l=1, m=0)
    # Insulating IC: b_pol = amp_b1 * sqrt(3π) / 4
    b_pol = amp_b1 * math.sqrt(3.0 * math.pi) / 4.0
    # b(r) = b_pol * (r³ - (4/3)*r_cmb*r² + (1/3)*r_icb⁴/r)
    b_r = b_pol * (r ** 3 - four * third * r_cmb * r ** 2 + third * r_icb ** 4 * or1)
    b_LMloc[l1m0, :] += b_r.to(CDTYPE)

    # Toroidal field: aj(l=2, m=0)
    b_tor = -four * third * amp_b1 * math.sqrt(math.pi / 5.0)
    # aj(r) = b_tor * r * sin(π*(r - r_icb))
    aj_r = b_tor * r * torch.sin(pi * (r - r_icb))
    aj_LMloc[l2m0, :] += aj_r.to(CDTYPE)


def initialize_fields():
    """Full field initialization for the dynamo benchmark.

    Initializes entropy, pressure, and magnetic fields from scratch.
    Returns nothing - modifies the global field arrays in-place.
    """
    from . import fields

    # Zero all fields before initialization (required for re-entrant calls)
    fields.w_LMloc.zero_()
    fields.dw_LMloc.zero_()
    fields.ddw_LMloc.zero_()
    fields.z_LMloc.zero_()
    fields.dz_LMloc.zero_()
    fields.p_LMloc.zero_()
    fields.dp_LMloc.zero_()
    fields.s_LMloc.zero_()
    fields.ds_LMloc.zero_()
    fields.b_LMloc.zero_()
    fields.db_LMloc.zero_()
    fields.ddb_LMloc.zero_()
    fields.aj_LMloc.zero_()
    fields.dj_LMloc.zero_()
    fields.ddj_LMloc.zero_()

    # Initialize entropy and pressure
    initS(fields.s_LMloc, fields.p_LMloc)

    # Initialize magnetic field
    initB(fields.b_LMloc, fields.aj_LMloc)

    # omega_ic = 0, omega_ma = 0 (default for benchmark)
