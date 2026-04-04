"""Radial grid and background profiles.

Matches radial.f90 for the non-FD, no-mapping case:
- Boussinesq when l_anel=False (all profiles trivial)
- Polytropic reference state when l_anel=True (strat > 0)
"""

import math

import torch

from .precision import DTYPE, DEVICE
from .params import (n_r_max, n_r_ic_max, radratio, l_cond_ic,
                     l_anel, strat, polind, g0, g1, g2, l_mag,
                     ra, ek, pr, prmag, mode, l_max)
from .radial_scheme import r, r_cmb, r_icb

# Inverse radial functions
or1 = 1.0 / r          # 1/r
or2 = or1 * or1         # 1/r^2
or3 = or2 * or1         # 1/r^3
or4 = or2 * or2         # 1/r^4

# Gravity (default: g0=0, g1=1, g2=0 → rgrav = r/r_cmb)
rgrav = r / r_cmb

# Boussinesq: all background profiles are trivial
rho0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
temp0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
orho1 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
orho2 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)

alpha0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
beta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
dbeta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddbeta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

dLtemp0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLtemp0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
dLalpha0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLalpha0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

ogrun = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

# Transport properties (constant for Boussinesq)
visc = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLvisc = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLvisc = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

kappa = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLkappa = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

lambda_ = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLlambda = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

sigma = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)

# Extra anelastic profiles (always declared)
otemp1 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dentropy0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

# Heating factors (zero for Boussinesq)
ViscHeatFac = 0.0
OhmLossFac = 0.0

# Anelastic scalars
DissNb = 0.0
GrunNb = 0.0
ThExpNb = 1.0  # default from Namelists.f90:1398

if l_anel:
    # --- Polytropic reference state (radial.f90:622-725) ---
    # Compute in float64 on CPU for machine precision, then cast to DTYPE/DEVICE.
    # Use the actual grid from radial_scheme (Chebyshev CGL or FD stretched).
    _rcmb = float(r_cmb)
    _ricb = float(r_icb)
    _r64 = r.to(torch.float64).to('cpu')
    _or1_64 = 1.0 / _r64
    _or2_64 = _or1_64 * _or1_64
    _or3_64 = _or2_64 * _or1_64
    _or4_64 = _or2_64 * _or2_64

    # Gravity: g(r) = g0 + g1*r/r_cmb + g2*r_cmb^2/r^2
    _rgrav64 = g0 + g1 * _r64 / _rcmb + g2 * _rcmb**2 * _or2_64

    # DissNb (radial.f90:689)
    DissNb = float((math.exp(strat / polind) - 1.0)
                   / (_rcmb - _ricb)
                   / (g0 + 0.5 * g1 * (1.0 + radratio) + g2 / radratio))

    # GrunNb (radial.f90:693)
    GrunNb = 1.0 / polind

    # ogrun (radial.f90:694)
    _ogrun64 = torch.full((n_r_max,), polind, dtype=torch.float64)

    # temp0 (radial.f90:695-697)
    _temp064 = (-DissNb * (g0 * _r64 + 0.5 * g1 * _r64**2 / _rcmb - g2 * _rcmb**2 * _or1_64)
                + 1.0 + DissNb * _rcmb * (g0 + 0.5 * g1 - g2))

    # rho0 (radial.f90:698)
    _rho064 = _temp064 ** polind

    # beta = d(ln rho0)/dr (radial.f90:703)
    _beta64 = -polind * DissNb * _rgrav64 / _temp064

    # dgrav/dr
    _dgrav64 = g1 / _rcmb - 2.0 * g2 * _rcmb**2 * _or3_64

    # dbeta (radial.f90:704-706)
    _dbeta64 = -polind * DissNb / _temp064**2 * (
        _dgrav64 * _temp064 + DissNb * _rgrav64**2)

    # ddbeta (radial.f90:707-711)
    _ddbeta64 = -polind * DissNb / _temp064**3 * (
        _temp064 * (6.0 * g2 * _rcmb**2 * _or4_64 * _temp064
                    + DissNb * _rgrav64 * _dgrav64)
        + 2.0 * DissNb * _rgrav64 * (DissNb * _rgrav64**2 + _dgrav64 * _temp064))

    # dtemp0, d2temp0 (radial.f90:712-713)
    _dtemp064 = -DissNb * _rgrav64
    _d2temp064 = -DissNb * _dgrav64

    # dLtemp0, ddLtemp0 (radial.f90:714-715)
    _dLtemp064 = _dtemp064 / _temp064
    _ddLtemp064 = -(_dtemp064 / _temp064)**2 + _d2temp064 / _temp064

    # alpha0 = 1/T (ideal gas, radial.f90:721)
    _alpha064 = 1.0 / _temp064

    # orho1, orho2 (radial.f90:757-758)
    _orho164 = 1.0 / _rho064
    _orho264 = _orho164 * _orho164

    # otemp1 = 1/T (radial.f90:759)
    _otemp164 = 1.0 / _temp064

    # Cast all to DTYPE/DEVICE
    def _cast(t):
        return t.to(dtype=DTYPE, device=DEVICE)

    rgrav = _cast(_rgrav64)
    ogrun = _cast(_ogrun64)
    temp0 = _cast(_temp064)
    rho0 = _cast(_rho064)
    beta = _cast(_beta64)
    dbeta = _cast(_dbeta64)
    ddbeta = _cast(_ddbeta64)
    dLtemp0 = _cast(_dLtemp064)
    ddLtemp0 = _cast(_ddLtemp064)
    alpha0 = _cast(_alpha064)
    orho1 = _cast(_orho164)
    orho2 = _cast(_orho264)
    otemp1 = _cast(_otemp164)

    # ViscHeatFac = DissNb * pr / raScaled (radial.f90:760)
    # raScaled = ra (since n_lScale=0 → lScale=1)
    ViscHeatFac = DissNb * pr / ra

    # OhmLossFac (radial.f90:761-764)
    if l_mag:
        OhmLossFac = ViscHeatFac / (ek * prmag**2)
    else:
        OhmLossFac = 0.0

    # Transport properties: constant (nVarVisc=0, nVarDiff=0)
    # visc=1, dLvisc=0, ddLvisc=0, kappa=1, dLkappa=0 — already set above

    # dentropy0 = 0 for adiabatic polytropic (default nVarEntropyGrad=0)

    # Keep float64 versions on CPU for solvers that need full precision
    rgrav_f64 = _rgrav64
    temp0_f64 = _temp064
    rho0_f64 = _rho064
    beta_f64 = _beta64
    dbeta_f64 = _dbeta64
    ddbeta_f64 = _ddbeta64
    dLtemp0_f64 = _dLtemp064
    ddLtemp0_f64 = _ddLtemp064
    alpha0_f64 = _alpha064
    orho1_f64 = _orho164
    ogrun_f64 = _ogrun64
    otemp1_f64 = _otemp164
    or1_f64 = _or1_64
    or2_f64 = _or2_64
    r_f64 = _r64

# --- CFL grid spacing arrays (preCalculations.f90 lines 305-311) ---
# delxh2: horizontal Courant interval (l_R=l_max for all benchmarks, no l_var_l)
delxh2 = r ** 2 / float(l_max * (l_max + 1))

# delxr2: radial Courant interval (min of adjacent spacings, squared)
_dr = r[:-1] - r[1:]  # r is decreasing, so dr > 0
delxr2 = torch.empty(n_r_max, dtype=DTYPE, device=DEVICE)
delxr2[0] = _dr[0] ** 2
delxr2[-1] = _dr[-1] ** 2
delxr2[1:-1] = torch.minimum(_dr[:-1], _dr[1:]) ** 2

# Volume of outer core and inner core
vol_oc = (4.0 / 3.0) * torch.pi * (r_cmb**3 - r_icb**3)
vol_ic = (4.0 / 3.0) * torch.pi * r_icb**3
surf_cmb = 4.0 * torch.pi * r_cmb**2


# ============================================================
# Inner core radial grid (only populated when l_cond_ic=True)
# ============================================================

def _build_ic_grid():
    """Build IC radial grid using even Chebyshev polynomials on [0, r_icb].

    Matches radial.f90 lines 797-863:
    - Full 2*n_r_ic_max-1 point Chebyshev grid on [-r_icb, r_icb]
    - Only keep top half (r_icb down to 0)
    - Even Chebyshev polynomials T_0, T_2, T_4, ..., T_{2(n-1)} via get_chebs_even
    - dr_top_ic: real-space derivative weights at ICB (Baltensperger-Trummer trick)
    """
    import math

    N = n_r_ic_max
    n_tot = 2 * N - 1  # full symmetric grid size

    # 1. Full Chebyshev-Gauss-Lobatto grid on [-r_icb, r_icb]
    # y(k) = cos(pi * k / (n_tot-1)), computed with math.cos to match Fortran
    y_full = [math.cos(math.pi * k / (n_tot - 1)) for k in range(n_tot)]

    # 2. Keep top half: indices 0..N-1 (r_icb down to ~0), then center=0
    r_ic_arr = torch.zeros(N, dtype=DTYPE)
    for k in range(N - 1):
        r_ic_arr[k] = r_icb * y_full[k]
    r_ic_arr[N - 1] = 0.0  # center

    O_r_ic_arr = torch.zeros(N, dtype=DTYPE)
    for k in range(N - 1):
        O_r_ic_arr[k] = 1.0 / r_ic_arr[k].item()
    # O_r_ic[N-1] = 0 (center)

    # 3. Even Chebyshev polynomials: get_chebs_even
    # map_fac = 2/(b-a) = 2/(2*r_icb) = 1/r_icb
    map_fac = 1.0 / r_icb
    dr_fac = map_fac  # = 1/r_icb

    # y_points: Chebyshev coordinates for the N grid points (top half)
    y_pts = [y_full[k] for k in range(N)]

    # Even Chebyshev polynomials via scalar recursion matching Fortran exactly.
    # cheb[n, k] = T_{2(n-1)}(y_k), with derivatives.
    # Use pure Python floats throughout to match Fortran FP rounding exactly,
    # then convert to torch tensors at the end.
    map_fac_f = float(map_fac)
    cheb_py = [[0.0] * N for _ in range(N)]
    dcheb_py = [[0.0] * N for _ in range(N)]
    d2cheb_py = [[0.0] * N for _ in range(N)]

    for k in range(N):
        yk = y_pts[k]
        cheb_py[0][k] = 1.0
        last_cheb = yk
        dcheb_py[0][k] = 0.0
        last_dcheb = map_fac_f
        d2cheb_py[0][k] = 0.0
        last_d2cheb = 0.0

        for n in range(1, N):
            # Even Cheb T_{2n}
            cheb_py[n][k] = 2.0 * yk * last_cheb - cheb_py[n - 1][k]
            dcheb_py[n][k] = 2.0 * map_fac_f * last_cheb + 2.0 * yk * last_dcheb - dcheb_py[n - 1][k]
            d2cheb_py[n][k] = 4.0 * map_fac_f * last_dcheb + 2.0 * yk * last_d2cheb - d2cheb_py[n - 1][k]

            # Odd Cheb T_{2n+1} (not stored, needed for recursion)
            new_last_cheb = 2.0 * yk * cheb_py[n][k] - last_cheb
            new_last_dcheb = 2.0 * map_fac_f * cheb_py[n][k] + 2.0 * yk * dcheb_py[n][k] - last_dcheb
            new_last_d2cheb = 4.0 * map_fac_f * dcheb_py[n][k] + 2.0 * yk * d2cheb_py[n][k] - last_d2cheb
            last_cheb = new_last_cheb
            last_dcheb = new_last_dcheb
            last_d2cheb = new_last_d2cheb

    cheb = torch.tensor(cheb_py, dtype=DTYPE)
    dcheb = torch.tensor(dcheb_py, dtype=DTYPE)
    d2cheb = torch.tensor(d2cheb_py, dtype=DTYPE)

    # 4. cheb_norm_ic = sqrt(2/(N-1))
    cheb_norm = math.sqrt(2.0 / (N - 1))

    # 5. dr_top_ic: real-space derivative weights at ICB
    # Baltensperger-Trummer differentiation matrix, folded for even symmetry
    dr_top_full = torch.zeros(n_tot, dtype=DTYPE)
    dr_top_full[0] = (2.0 * (n_tot - 1) ** 2 + 1.0) / 6.0
    for nr in range(1, n_tot):
        angle = 0.5 * nr * math.pi / (n_tot - 1)
        diff = 2.0 * math.sin(angle) ** 2
        coeff = -1.0 if (nr + 1) % 2 == 0 else 1.0  # (-1)^(nr+1), Fortran 1-based
        dr_top_full[nr] = 2.0 * coeff / diff
    dr_top_full[n_tot - 1] = 0.5 * dr_top_full[n_tot - 1]

    # Fold: dr_top_ic(nr) = dr_top(nr) + dr_top(2*N - 1 - nr) for nr=0..N-2
    # (Fortran 1-based: dr_top_ic(n_r) = dr_top(n_r) + dr_top(2*N - n_r))
    dr_top_arr = torch.zeros(N, dtype=DTYPE)
    for nr in range(N - 1):
        dr_top_arr[nr] = dr_top_full[nr] + dr_top_full[n_tot - 1 - nr]
    dr_top_arr[N - 1] = dr_top_full[N - 1]

    # Apply map factor
    dr_top_arr = dr_top_arr * dr_fac

    return r_ic_arr, O_r_ic_arr, cheb, dcheb, d2cheb, dr_top_arr, cheb_norm, dr_fac


def _build_ic_radii():
    """Build IC radial grid (r_ic, O_r_ic) — needed even for insulating IC (graph output)."""
    import math
    N = n_r_ic_max
    n_tot = 2 * N - 1
    y_full = [math.cos(math.pi * k / (n_tot - 1)) for k in range(n_tot)]
    r_ic_arr = torch.zeros(N, dtype=DTYPE)
    for k in range(N - 1):
        r_ic_arr[k] = r_icb * y_full[k]
    r_ic_arr[N - 1] = 0.0
    O_r_ic_arr = torch.zeros(N, dtype=DTYPE)
    for k in range(N - 1):
        O_r_ic_arr[k] = 1.0 / r_ic_arr[k].item()
    return r_ic_arr, O_r_ic_arr


if l_cond_ic:
    (r_ic, O_r_ic, cheb_ic, dcheb_ic, d2cheb_ic,
     dr_top_ic, cheb_norm_ic, dr_fac_ic) = _build_ic_grid()
else:
    # Always build IC radii (needed for graph output IC potential field)
    r_ic, O_r_ic = _build_ic_radii()
    cheb_ic = torch.zeros(1, 1, dtype=DTYPE)
    dcheb_ic = torch.zeros(1, 1, dtype=DTYPE)
    d2cheb_ic = torch.zeros(1, 1, dtype=DTYPE)
    dr_top_ic = torch.zeros(n_r_ic_max, dtype=DTYPE)
    cheb_norm_ic = 0.0
    dr_fac_ic = 0.0

# O_r_ic2 = 1/r_ic^2 (needed for graph output IC fields)
O_r_ic2 = O_r_ic * O_r_ic
