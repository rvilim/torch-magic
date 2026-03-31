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
from .params import (n_r_max, n_r_ic_max, lm_max, l_max, init_b1, amp_b1,
                     init_s1, amp_s1, radratio, l_cond_ic, l_mag, l_chemical_conv)
from .constants import (pi, one, two, three, four, half, third, osq4pi,
                        sq4pi, zero)
from .chebyshev import (r, r_cmb, r_icb, rMat, drMat, d2rMat,
                        rnorm, boundary_fac)
from .radial_functions import or1, rgrav, rho0, beta, dLtemp0, dLkappa, kappa
from .params import l_anel
if l_anel:
    from .radial_functions import (ViscHeatFac, ThExpNb, ogrun_f64, alpha0_f64,
                                   temp0_f64, rho0_f64, r_f64, beta_f64,
                                   dLtemp0_f64, or1_f64,
                                   rgrav_f64, orho1_f64, ogrun, alpha0)
from .pre_calculations import BuoFac, opr, osc, ChemFac
from .algebra import prepare_mat, solve_mat_real
from .cosine_transform import costf
from .blocking import st_lm2
from .radial_derivatives import get_dr

# Conduction state boundary diagnostics (set by initialize_fields)
topcond = 0.0
botcond = 0.0
deltacond = 0.0


def _build_cheb_mats_f64():
    """Build Chebyshev polynomial matrices in float64 on CPU.

    Needed for anelastic ps_cond where float32 precision is insufficient.
    """
    N = n_r_max
    _rcmb = float(r_cmb)
    _ricb = float(r_icb)
    k = torch.arange(N, dtype=torch.float64)
    x = torch.cos(math.pi * k / (N - 1))
    _drx = 2.0 / (_rcmb - _ricb)

    rM = torch.zeros(N, N, dtype=torch.float64)
    drM = torch.zeros(N, N, dtype=torch.float64)
    d2rM = torch.zeros(N, N, dtype=torch.float64)

    rM[:, 0] = 1.0
    rM[:, 1] = x
    drM[:, 1] = _drx
    for col in range(2, N):
        rM[:, col] = 2.0 * x * rM[:, col - 1] - rM[:, col - 2]
        drM[:, col] = (2.0 * _drx * rM[:, col - 1]
                       + 2.0 * x * drM[:, col - 1]
                       - drM[:, col - 2])
        d2rM[:, col] = (4.0 * _drx * drM[:, col - 1]
                        + 2.0 * x * d2rM[:, col - 1]
                        - d2rM[:, col - 2])
    return rM, drM, d2rM


def _cheb_integ_kernel_f64(N):
    """Build Chebyshev spectral integration kernel (N x N) in float64.

    Computes the matrix K[j,k] such that row N of ps0Mat is K @ work_cheb.
    K[j,k] = (1/(1-(k-j)^2) + 1/(1-(k+j)^2)) * 0.5 * rnorm
    when (j+k) is even, else 0.

    This implements the Chebyshev integral constraint ∫ f(x) T_j(x) dx.
    """
    j = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    jj, kk = torch.meshgrid(j, k, indexing='ij')
    parity = ((jj + kk) % 2 == 0).to(torch.float64)
    denom1 = 1.0 - (kk - jj) ** 2
    denom2 = 1.0 - (kk + jj) ** 2
    safe1 = torch.where(denom1 != 0, 1.0 / denom1, torch.zeros_like(denom1))
    safe2 = torch.where(denom2 != 0, 1.0 / denom2, torch.zeros_like(denom2))
    return parity * (safe1 + safe2) * 0.5 * rnorm


def ps_cond():
    """Compute conduction state for entropy and pressure.

    Solves the coupled 2N x 2N system for (s0, p0) in Chebyshev space.
    Handles both Boussinesq (trivial profiles) and anelastic (polytropic).

    For anelastic: builds in float64 and uses Chebyshev integral BC for
    pressure when ViscHeatFac*ThExpNb != 0 (init_fields.f90 lines 2351-2383).

    Returns:
        (s0, p0): each shape (n_r_max,) in physical (radial) space
    """
    N = n_r_max
    cpu = torch.device("cpu")
    _bfac = boundary_fac

    if l_anel:
        return _ps_cond_anel()

    # --- Boussinesq path (original, DTYPE precision) ---
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1 = or1.to(cpu)
    _rgrav = rgrav.to(cpu)
    _rho0 = rho0.to(cpu)

    _beta = beta.to(cpu)
    _dLtemp0 = dLtemp0.to(cpu)
    _dLkappa = dLkappa.to(cpu)
    _kappa = kappa.to(cpu)

    ps0Mat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=cpu)

    _s_d1_coeff = (_beta + _dLtemp0 + two * _or1 + _dLkappa).unsqueeze(1)
    ps0Mat[:N, :N] = rnorm * opr * _kappa.unsqueeze(1) * (
        _d2rMat + _s_d1_coeff * _drMat
    )
    ps0Mat[:N, N:] = 0.0
    ps0Mat[N:, :N] = -rnorm * (_rho0 * BuoFac * _rgrav).unsqueeze(1) * _rMat
    ps0Mat[N:, N:] = rnorm * (_drMat - _beta.unsqueeze(1) * _rMat)

    ps0Mat[0, :N] = rnorm * _rMat[0, :]
    ps0Mat[0, N:] = 0.0
    ps0Mat[N - 1, :N] = rnorm * _rMat[N - 1, :]
    ps0Mat[N - 1, N:] = 0.0
    ps0Mat[N, :N] = 0.0
    ps0Mat[N, N:] = rnorm * _rMat[0, :]

    for col in [0, N - 1, N, 2 * N - 1]:
        ps0Mat[:, col] *= _bfac

    ps0Mat_fac = 1.0 / ps0Mat.abs().max(dim=1).values
    ps0Mat = ps0Mat * ps0Mat_fac.unsqueeze(1)

    a_lu, ip, info = prepare_mat(ps0Mat)
    assert info == 0, f'Singular ps0Mat, info={info}'

    rhs = torch.zeros(2 * N, dtype=DTYPE, device=cpu)
    rhs[0] = 0.0
    rhs[N - 1] = sq4pi
    rhs[N] = 0.0
    rhs = ps0Mat_fac * rhs

    rhs = solve_mat_real(a_lu, ip, rhs)

    s0 = rhs[:N].clone().to(DEVICE)
    p0 = rhs[N:].clone().to(DEVICE)

    s0 = costf(s0)
    p0 = costf(p0)

    return s0, p0


def _ps_cond_anel():
    """Anelastic conduction state, computed entirely in float64 on CPU.

    Matches init_fields.f90 ps_cond with l_temperature_diff=.false. (entropy
    diffusion) and the Chebyshev integral BC for pressure when
    ViscHeatFac*ThExpNb != 0 and ktopp == 1.
    """
    N = n_r_max
    _bfac = boundary_fac
    f64 = torch.float64

    # Build Chebyshev matrices in float64
    _rMat, _drMat, _d2rMat = _build_cheb_mats_f64()

    # Float64 profiles from radial_functions
    _or1 = or1_f64
    _rgrav = rgrav_f64
    _rho0 = rho0_f64
    _beta = beta_f64
    _dLtemp0 = dLtemp0_f64
    _dLkappa = torch.zeros(N, dtype=f64)  # constant transport: dLkappa=0
    _alpha0 = alpha0_f64
    _ogrun = ogrun_f64
    _temp0 = temp0_f64
    _orho1 = orho1_f64
    _r = r_f64

    # kappa=1, dLkappa=0 for constant transport (nVarDiff=0)
    _kappa = torch.ones(N, dtype=f64)

    ps0Mat = torch.zeros(2 * N, 2 * N, dtype=f64)

    # Entropy diffusion: opr * kappa * (d2 + (beta+dLtemp0+2/r+dLkappa)*d1)
    _s_d1_coeff = (_beta + _dLtemp0 + 2.0 * _or1 + _dLkappa).unsqueeze(1)
    ps0Mat[:N, :N] = rnorm * opr * _kappa.unsqueeze(1) * (
        _d2rMat + _s_d1_coeff * _drMat
    )
    ps0Mat[:N, N:] = 0.0

    # Hydrostatic equilibrium: -rho0 * BuoFac * rgrav * rMat
    ps0Mat[N:, :N] = -rnorm * (_rho0 * BuoFac * _rgrav).unsqueeze(1) * _rMat
    # Pressure: drMat - beta * rMat
    ps0Mat[N:, N:] = rnorm * (_drMat - _beta.unsqueeze(1) * _rMat)

    # Entropy BCs: ktops=1 (fixed s at top), kbots=1 (fixed s at bottom)
    ps0Mat[0, :N] = rnorm * _rMat[0, :]
    ps0Mat[0, N:] = 0.0
    ps0Mat[N - 1, :N] = rnorm * _rMat[N - 1, :]
    ps0Mat[N - 1, N:] = 0.0

    # Pressure BC (row N): Chebyshev integral constraint
    # when ViscHeatFac*ThExpNb != 0 and ktopp == 1
    if ViscHeatFac * ThExpNb != 0.0:
        # work = ThExpNb * ViscHeatFac * ogrun * alpha0 * r^2 (physical space)
        work_phys = ThExpNb * ViscHeatFac * _ogrun * _alpha0 * _r * _r
        work = costf(work_phys)  # costf is self-inverse: physical → spectral
        work = work * rnorm
        work[0] = _bfac * work[0]
        work[N - 1] = _bfac * work[N - 1]

        # work2 = -ThExpNb * alpha0 * temp0 * rho0 * r^2 (physical space)
        work2_phys = -ThExpNb * _alpha0 * _temp0 * _rho0 * _r * _r
        work2 = costf(work2_phys)
        work2 = work2 * rnorm
        work2[0] = _bfac * work2[0]
        work2[N - 1] = _bfac * work2[N - 1]

        # Chebyshev spectral integration kernel
        K = _cheb_integ_kernel_f64(N)

        # Row N: integral constraint
        ps0Mat[N, N:] = K @ work    # pressure columns
        ps0Mat[N, :N] = K @ work2   # entropy columns
    else:
        # Standard pressure BC: p(r_cmb) = 0
        ps0Mat[N, :N] = 0.0
        ps0Mat[N, N:] = rnorm * _rMat[0, :]

    # Boundary factor scaling (columns 0, N-1, N, 2N-1)
    for col in [0, N - 1, N, 2 * N - 1]:
        ps0Mat[:, col] *= _bfac

    # Row scaling for conditioning
    ps0Mat_fac = 1.0 / ps0Mat.abs().max(dim=1).values
    ps0Mat = ps0Mat * ps0Mat_fac.unsqueeze(1)

    # LU factorize
    a_lu, ip, info = prepare_mat(ps0Mat)
    assert info == 0, f'Singular ps0Mat (anelastic), info={info}'

    # RHS: epsc=0 → interior is zero; tops(0,0)=0, bots(0,0)=sq4pi
    rhs = torch.zeros(2 * N, dtype=f64)
    rhs[0] = 0.0           # top: s = tops(0,0) = 0
    rhs[N - 1] = sq4pi     # bot: s = bots(0,0) = sq4pi
    rhs[N] = 0.0           # pressure constraint
    rhs = ps0Mat_fac * rhs

    # Solve
    rhs = solve_mat_real(a_lu, ip, rhs)

    # Extract s0 and p0 (still float64 on CPU)
    s0 = rhs[:N].clone()
    p0 = rhs[N:].clone()

    # Transform to physical space in float64 before casting
    s0 = costf(s0)
    p0 = costf(p0)

    return s0.to(dtype=DTYPE, device=DEVICE), p0.to(dtype=DTYPE, device=DEVICE)


def xi_cond():
    """Compute conduction state for composition.

    Same structure as ps_cond but:
    - Uses osc (1/Schmidt) instead of opr (1/Prandtl)
    - Uses ChemFac instead of BuoFac
    - BC: top=0, bot=sq4pi (same as entropy, set in preCalculations.f90 line 617)

    Returns:
        (xi0, p_xi0): each shape (n_r_max,) in physical (radial) space
    """
    N = n_r_max

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1 = or1.to(cpu)
    _rgrav = rgrav.to(cpu)
    _rho0 = rho0.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    ps0Mat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=cpu)

    # Composition diffusion block
    ps0Mat[:N, :N] = _rnorm * osc * (
        _d2rMat + two * _or1.unsqueeze(1) * _drMat
    )
    # Hydrostatic equilibrium block (ChemFac instead of BuoFac)
    ps0Mat[N:, :N] = -_rnorm * (_rho0 * ChemFac * _rgrav).unsqueeze(1) * _rMat
    # Pressure gradient block
    ps0Mat[N:, N:] = _rnorm * _drMat

    # Boundary conditions
    ps0Mat[0, :N] = _rnorm * _rMat[0, :]
    ps0Mat[0, N:] = 0.0
    ps0Mat[N - 1, :N] = _rnorm * _rMat[N - 1, :]
    ps0Mat[N - 1, N:] = 0.0
    ps0Mat[N, :N] = 0.0
    ps0Mat[N, N:] = _rnorm * _rMat[0, :]

    for col in [0, N - 1, N, 2 * N - 1]:
        ps0Mat[:, col] *= _bfac

    ps0Mat_fac = 1.0 / ps0Mat.abs().max(dim=1).values
    ps0Mat = ps0Mat * ps0Mat_fac.unsqueeze(1)

    a_lu, ip, info = prepare_mat(ps0Mat)
    assert info == 0, f'Singular xi_ps0Mat, info={info}'

    rhs = torch.zeros(2 * N, dtype=DTYPE, device=cpu)
    rhs[0] = 0.0                        # top: xi = 0
    rhs[N - 1] = math.sqrt(4.0 * math.pi)  # bot: xi = sq4pi (preCalculations.f90:617)
    rhs[N] = 0.0       # pressure at top = 0
    rhs = ps0Mat_fac * rhs

    rhs = solve_mat_real(a_lu, ip, rhs)

    xi0 = rhs[:N].clone().to(DEVICE)
    p_xi0 = rhs[N:].clone().to(DEVICE)

    xi0 = costf(xi0)
    p_xi0 = costf(p_xi0)

    return xi0, p_xi0


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


def initB(b_LMloc, aj_LMloc, b_ic=None, aj_ic=None):
    """Initialize magnetic field for init_b1=3.

    Sets poloidal b(l=1, m=0) and toroidal aj(l=2, m=0).
    When l_cond_ic: conducting IC init with different OC profile + IC fields.

    Args:
        b_LMloc, aj_LMloc: (lm_max, n_r_max) complex, modified in-place
        b_ic, aj_ic: (lm_max, n_r_ic_max) complex, modified in-place (if l_cond_ic)
    """
    assert init_b1 == 3, f'Only init_b1=3 is supported, got {init_b1}'

    # LM indices for l=1,m=0 and l=2,m=0 (standard ordering)
    l1m0 = st_lm2[1, 0].item()
    l2m0 = st_lm2[2, 0].item()

    b_tor = -four * third * amp_b1 * math.sqrt(math.pi / 5.0)

    if l_cond_ic:
        from .radial_functions import r_ic

        # Conducting IC: b_pol = amp_b1 * sqrt(3π) / (3 + r_cmb)
        b_pol = amp_b1 * math.sqrt(3.0 * math.pi) / (3.0 + float(r_cmb))

        # OC poloidal: b(r) = b_pol * (r³ - (4/3)*r_cmb*r²)
        b_r = b_pol * (r ** 3 - four * third * r_cmb * r ** 2)
        b_LMloc[l1m0, :] += b_r.to(CDTYPE)

        # IC poloidal: b_ic(r) = b_pol * r_icb² * (r²/(2*r_icb) + r_icb/2 - (4/3)*r_cmb)
        b_ic_r = b_pol * r_icb ** 2 * (
            half * r_ic ** 2 / r_icb + half * r_icb - four * third * r_cmb
        )
        b_ic[l1m0, :] += b_ic_r.to(CDTYPE)

        # OC toroidal: aj(r) = b_tor * r * sin(π*r/r_cmb)
        aj_r = b_tor * r * torch.sin(pi * (r / r_cmb))
        aj_LMloc[l2m0, :] += aj_r.to(CDTYPE)

        # IC toroidal: aj_ic(r) = b_tor * (aj_ic1*r*sin(πr/r_cmb) + aj_ic2*cos(πr/r_cmb))
        arg = pi * float(r_icb) / float(r_cmb)
        sin_arg = math.sin(arg)
        cos_arg = math.cos(arg)
        aj_ic1 = (arg - two * sin_arg * cos_arg) / (arg + sin_arg * cos_arg)
        aj_ic2 = (one - aj_ic1) * float(r_icb) * sin_arg / cos_arg
        aj_ic_r = b_tor * (
            aj_ic1 * r_ic * torch.sin(pi * r_ic / r_cmb)
            + aj_ic2 * torch.cos(pi * r_ic / r_cmb)
        )
        aj_ic[l2m0, :] += aj_ic_r.to(CDTYPE)
    else:
        # Insulating IC: b_pol = amp_b1 * sqrt(3π) / 4
        b_pol = amp_b1 * math.sqrt(3.0 * math.pi) / 4.0
        # b(r) = b_pol * (r³ - (4/3)*r_cmb*r² + (1/3)*r_icb⁴/r)
        b_r = b_pol * (r ** 3 - four * third * r_cmb * r ** 2 + third * r_icb ** 4 * or1)
        b_LMloc[l1m0, :] += b_r.to(CDTYPE)

        # Toroidal field: aj(r) = b_tor * r * sin(π*(r - r_icb))
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

    fields.xi_LMloc.zero_()
    fields.dxi_LMloc.zero_()

    if l_cond_ic:
        fields.b_ic.zero_()
        fields.db_ic.zero_()
        fields.ddb_ic.zero_()
        fields.aj_ic.zero_()
        fields.dj_ic.zero_()
        fields.ddj_ic.zero_()

    # Initialize entropy and pressure
    initS(fields.s_LMloc, fields.p_LMloc)

    # Compute conduction state boundary diagnostics (startFields.f90:172-175)
    # For Boussinesq entropy diffusion: topcond = -osq4pi * ds0(CMB)
    global topcond, botcond, deltacond
    s0_phys = fields.s_LMloc[0, :].real  # l=0,m=0 conduction state in physical space
    ds0 = get_dr(s0_phys.to(CDTYPE)).real
    topcond = -osq4pi * ds0[0].item()
    botcond = -osq4pi * ds0[-1].item()
    deltacond = osq4pi * (s0_phys[-1] - s0_phys[0]).item()

    # Initialize composition field
    if l_chemical_conv:
        xi0, p_xi0 = xi_cond()
        fields.xi_LMloc[0, :] = xi0.to(CDTYPE)
        # Add composition contribution to pressure (l=0,m=0)
        fields.p_LMloc[0, :] += p_xi0.to(CDTYPE)

    # Initialize magnetic field
    if l_mag:
        if l_cond_ic:
            initB(fields.b_LMloc, fields.aj_LMloc, fields.b_ic, fields.aj_ic)
        else:
            initB(fields.b_LMloc, fields.aj_LMloc)

    # omega_ic = 0, omega_ma = 0 (default for benchmark)
