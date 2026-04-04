"""Double-curl poloidal velocity solver matching updateWP.f90 (l_double_curl path).

Implements:
- build_w_matrices: N×N matrix for the 4th-order double-curl equation
- updateW: IMEX solve for w only (pressure recovered separately)
- dwdt.old and dwdt.impl formulas for double-curl

The double-curl formulation eliminates pressure by applying curl(curl) to
the momentum equation, yielding a 4th-order ODE for w alone. This gives
a pentadiagonal banded matrix for 2nd-order FD (vs the coupled 2N×2N
dense matrix in the standard formulation).
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max, n_cheb_max, ktopv, kbotv, l_finite_diff
from .constants import two, three, third, four
from .radial_scheme import (
    rMat, drMat, d2rMat, d3rMat, d4rMat, rnorm, boundary_fac, to_physical, r,
)
from .radial_functions import (
    or1, or2, rgrav, rho0,
    beta, dbeta, ddbeta, visc, dLvisc, ddLvisc, orho1,
)
from .horizontal_data import dLh, hdif_V
from .pre_calculations import BuoFac, ChemFac
from .blocking import st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_real, chunked_solve_complex
from .radial_derivatives import get_dr, get_ddr, get_dddr

# --- Precompute broadcast arrays ---
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)          # (lm_max, 1)
_or1_r = or1.unsqueeze(0)                        # (1, n_r_max)
_or2_r = or2.unsqueeze(0)                        # (1, n_r_max)
_orho1_r = orho1.unsqueeze(0).to(CDTYPE)         # (1, n_r_max)
_beta_r = beta.unsqueeze(0).to(CDTYPE)           # (1, n_r_max)
_visc_r = visc.unsqueeze(0).to(CDTYPE)           # (1, n_r_max)
_rgrav_r = rgrav.unsqueeze(0).to(CDTYPE)         # (1, n_r_max)
_rho0_r = rho0.unsqueeze(0).to(CDTYPE)           # (1, n_r_max)
_hdif_lm = hdif_V[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)

_m0_mask = (st_lm2m == 0)

# Unique inverse per l degree: (l_max+1, N, N) — Chebyshev
_w_inv_by_l = None
# Banded LU per l degree — FD (fallback for non-pentadiag)
_w_bands_by_l = None
_w_piv_by_l = None
_w_fac_row_by_l = None
_w_fac_col_by_l = None
_w_kl = 0
_w_ku = 0
# Batched pentadiag: precomputed weights per-lm
_w_penta_w1 = None
_w_penta_w2 = None
_w_penta_inv_d = None
_w_penta_du1 = None
_w_penta_du2 = None
_w_penta_fac_row = None
_w_penta_fac_col = None


def build_w_matrices(wimp_lin0: float):
    """Build and factorize N×N double-curl matrices for l>=1.

    The matrix represents:
        -dLh*or2*orho1*(d2 - beta*d - dLh*or2) * w     [old part]
      + wimp*orho1*hdif*visc*dLh*or2*(d4 + c3*d3 + c2*d2 + c1*d + c0) * w  [impl part]

    Matches get_wMat in updateWP.f90 (lines 2237-2360).

    Must be called whenever dt changes.
    """
    global _w_inv_by_l, _w_bands_by_l, _w_piv_by_l, _w_fac_row_by_l, _w_fac_col_by_l, _w_kl, _w_ku
    global _w_penta_w1, _w_penta_w2, _w_penta_inv_d, _w_penta_du1, _w_penta_du2
    global _w_penta_fac_row, _w_penta_fac_col
    from .params import l_finite_diff
    N = n_r_max

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _d3rMat = d3rMat.to(cpu) if d3rMat is not None else None
    _d4rMat = d4rMat.to(cpu) if d4rMat is not None else None
    _or1 = or1.to(cpu)
    _or2 = or2.to(cpu)
    _beta = beta.to(cpu)
    _dbeta = dbeta.to(cpu)
    _ddbeta = ddbeta.to(cpu)
    _visc = visc.to(cpu)
    _dLvisc = dLvisc.to(cpu)
    _ddLvisc = ddLvisc.to(cpu)
    _orho1 = orho1.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    eye = torch.eye(N, dtype=DTYPE, device=cpu)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)

    # FD: pentadiagonal (4th-order operator, kl=ku=2 for fd_order=2)
    if l_finite_diff:
        from .params import fd_order, fd_order_bound
        # Matches Fortran updateWP.f90 lines 96-100:
        #   if order<=2 and order_boundary<=2: n_bands = order+3
        #   else: n_bands = max(order+3, 2*order_boundary+3)
        if fd_order <= 2 and fd_order_bound <= 2:
            n_bands_w = fd_order + 3
        else:
            n_bands_w = max(fd_order + 3, 2 * fd_order_bound + 3)
        _w_kl = (n_bands_w - 1) // 2
        _w_ku = _w_kl
    n_abd = 2 * _w_kl + _w_ku + 1 if l_finite_diff else 1
    w_abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    w_piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    w_fac_row_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
    w_fac_col_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
    # Pentadiag precompute storage
    _use_batched_w = l_finite_diff and _w_kl == 2 and _w_ku == 2 and fd_order <= 2
    if _use_batched_w:
        w_pw1_all = torch.zeros(l_max + 1, N - 1, dtype=DTYPE, device=cpu)
        w_pw2_all = torch.zeros(l_max + 1, N - 2, dtype=DTYPE, device=cpu)
        w_pinv_d_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
        w_pdu1_all = torch.zeros(l_max + 1, N - 1, dtype=DTYPE, device=cpu)
        w_pdu2_all = torch.zeros(l_max + 1, N - 2, dtype=DTYPE, device=cpu)
    if l_finite_diff:
        # l=0: identity bands
        w_abd_all[0, _w_kl + _w_ku, :] = 1.0
        w_piv_all[0] = torch.arange(1, N + 1, dtype=torch.long)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        dat = torch.zeros(N, N, dtype=DTYPE, device=cpu)

        # Bulk rows (nR=2..N-3 in 0-based, Fortran nR=3..n_r_max-2)
        for nR in range(2, N - 2):
            # "old" part: -dLh*or2*orho1*(d2 - beta*d - dLh*or2) at row nR
            old_part = -dL * _or2[nR] * _orho1[nR] * (
                _d2rMat[nR, :]
                - _beta[nR] * _drMat[nR, :]
                - dL * _or2[nR] * _rMat[nR, :]
            )

            # Implicit diffusion coefficients
            c3 = 2.0 * (_dLvisc[nR] - _beta[nR])
            c2 = (_ddLvisc[nR] - 2.0 * _dbeta[nR]
                  + _dLvisc[nR] ** 2 + _beta[nR] ** 2
                  - 3.0 * _dLvisc[nR] * _beta[nR]
                  - 2.0 * _or1[nR] * (_dLvisc[nR] + _beta[nR])
                  - 2.0 * dL * _or2[nR])
            c1 = (-_ddbeta[nR]
                  - _dbeta[nR] * (2.0 * _dLvisc[nR] - _beta[nR] + 2.0 * _or1[nR])
                  - _ddLvisc[nR] * (_beta[nR] + 2.0 * _or1[nR])
                  + _beta[nR] ** 2 * (_dLvisc[nR] + 2.0 * _or1[nR])
                  - _beta[nR] * (_dLvisc[nR] ** 2 - 2.0 * _or2[nR])
                  - 2.0 * _dLvisc[nR] * _or1[nR] * (_dLvisc[nR] - _or1[nR])
                  + 2.0 * (2.0 * _or1[nR] + _beta[nR] - _dLvisc[nR]) * dL * _or2[nR])
            c0 = dL * _or2[nR] * (
                2.0 * _dbeta[nR] + _ddLvisc[nR]
                + _dLvisc[nR] ** 2 - (2.0 / 3.0) * _beta[nR] ** 2
                + _dLvisc[nR] * _beta[nR]
                + 2.0 * _or1[nR] * (2.0 * _dLvisc[nR] - _beta[nR] - 3.0 * _or1[nR])
                + dL * _or2[nR]
            )

            impl_part = _orho1[nR] * hdif_l * _visc[nR] * dL * _or2[nR] * (
                _d4rMat[nR, :]
                + c3 * _d3rMat[nR, :]
                + c2 * _d2rMat[nR, :]
                + c1 * _drMat[nR, :]
                + c0 * _rMat[nR, :]
            )

            dat[nR, :] = _rnorm * (old_part + wimp_lin0 * impl_part)

        # Boundary conditions
        # Row 0: w=0 at CMB
        dat[0, :] = _rnorm * _rMat[0, :]

        # Row N-1: w=0 at ICB
        dat[N - 1, :] = _rnorm * _rMat[N - 1, :]

        # Row 1: velocity BC at CMB
        if ktopv == 1:  # stress-free
            dat[1, :] = _rnorm * (
                _d2rMat[0, :] - (2.0 * _or1[0] + _beta[0]) * _drMat[0, :])
        else:  # no-slip
            dat[1, :] = _rnorm * _drMat[0, :]

        # Row N-2: velocity BC at ICB
        if kbotv == 1:  # stress-free
            dat[N - 2, :] = _rnorm * (
                _d2rMat[N - 1, :] - (2.0 * _or1[N - 1] + _beta[N - 1]) * _drMat[N - 1, :])
        else:  # no-slip
            dat[N - 2, :] = _rnorm * _drMat[N - 1, :]

        # Zero high modes in boundary rows (Chebyshev truncation)
        if n_cheb_max < N:
            for row in [0, 1, N - 2, N - 1]:
                dat[row, n_cheb_max:] = 0.0

        # Chebyshev boundary factor on first/last columns
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        # Row+column preconditioning
        fac_row = 1.0 / dat.abs().max(dim=1).values
        dat = fac_row.unsqueeze(1) * dat
        fac_col = 1.0 / dat.abs().max(dim=0).values
        dat = dat * fac_col.unsqueeze(0)

        if l_finite_diff:
            from .algebra import dense_to_band_storage, prepare_band
            abd = dense_to_band_storage(dat, _w_kl, _w_ku)
            abd_f, piv, info = prepare_band(abd, N, _w_kl, _w_ku)
            assert info == 0, f"Singular wMat (band) for l={l}, info={info}"
            w_abd_all[l] = abd_f
            w_piv_all[l] = piv
            w_fac_row_all[l] = fac_row
            w_fac_col_all[l] = fac_col
            # Precompute pentadiag weights for batched solve
            if _w_kl == 2 and _w_ku == 2:
                from .algebra import precompute_pentadiag
                dl2 = dat[range(2, N), range(N - 2)]
                dl1 = dat[range(1, N), range(N - 1)]
                d_w = dat.diag()
                du1 = dat[range(N - 1), range(1, N)]
                du2 = dat[range(N - 2), range(2, N)]
                pw1, pw2, pinv_d, pdu1, pdu2 = precompute_pentadiag(dl2, dl1, d_w, du1, du2)
                w_pw1_all[l] = pw1
                w_pw2_all[l] = pw2
                w_pinv_d_all[l] = pinv_d
                w_pdu1_all[l] = pdu1
                w_pdu2_all[l] = pdu2
        else:
            lu, ip, info = prepare_mat(dat)
            assert info == 0, f"Singular wMat for l={l}, info={info}"
            inv_precond = solve_mat_real(lu, ip, eye)
            inv_by_l[l] = fac_col.unsqueeze(1) * inv_precond * fac_row.unsqueeze(0)

    if l_finite_diff:
        _w_bands_by_l = w_abd_all.to(DEVICE)
        _w_piv_by_l = w_piv_all.to(DEVICE)
        _w_fac_row_by_l = w_fac_row_all.to(DEVICE)
        _w_fac_col_by_l = w_fac_col_all.to(DEVICE)
        _w_inv_by_l = None
        if _use_batched_w:
            _lm2l_cpu = st_lm2l.cpu()
            _w_penta_w1 = w_pw1_all[_lm2l_cpu].to(DEVICE)
            _w_penta_w2 = w_pw2_all[_lm2l_cpu].to(DEVICE)
            _w_penta_inv_d = w_pinv_d_all[_lm2l_cpu].to(DEVICE)
            _w_penta_du1 = w_pdu1_all[_lm2l_cpu].to(DEVICE)
            _w_penta_du2 = w_pdu2_all[_lm2l_cpu].to(DEVICE)
            _w_penta_fac_row = w_fac_row_all[_lm2l_cpu].to(DEVICE)
            _w_penta_fac_col = w_fac_col_all[_lm2l_cpu].to(DEVICE)
        else:
            _w_penta_w1 = None
    else:
        _w_inv_by_l = inv_by_l.to(DEVICE)
        _w_bands_by_l = None


def get_w_old(w, dw, ddw):
    """Compute dwdt.old[0] for the double-curl formulation.

    dwdt.old = dL*or2*(-orho1*(ddw - beta*dw - dL*or2*w))

    Matches updateWP.f90 lines 1192-1203.
    """
    result = _dLh_lm * _or2_r * (
        -_orho1_r * (ddw - _beta_r * dw - _dLh_lm * _or2_r * w)
    )
    result[:, 0] = 0.0
    result[:, -1] = 0.0
    return result


def get_w_impl(w, dw, ddw, dddw, ddddw, s, xi=None):
    """Compute dwdt.impl for the double-curl formulation.

    The 4th-order diffusion operator + buoyancy.
    Matches updateWP.f90 lines 1229-1271.

    Args:
        w, dw, ddw, dddw, ddddw: poloidal potential and radial derivatives
        s: entropy field
        xi: composition field (optional)

    Returns:
        (lm_max, n_r_max) complex
    """
    # Diffusion coefficients (same as in build_w_matrices)
    c3 = 2.0 * (dLvisc - beta).unsqueeze(0).to(CDTYPE)
    c2 = (ddLvisc - 2.0 * dbeta
          + dLvisc ** 2 + beta ** 2
          - 3.0 * dLvisc * beta
          - 2.0 * or1 * (dLvisc + beta)
          - 2.0 * _dLh_lm * or2.unsqueeze(0)
          ).to(CDTYPE)
    # c2 has shape issues — dLh_lm is (lm_max,1), or2 is (n_r_max,)
    # Let me restructure using proper broadcasting

    _dLvisc_r = dLvisc.unsqueeze(0).to(CDTYPE)
    _ddLvisc_r = ddLvisc.unsqueeze(0).to(CDTYPE)
    _dbeta_r = dbeta.unsqueeze(0).to(CDTYPE)
    _ddbeta_r = ddbeta.unsqueeze(0).to(CDTYPE)

    c3_r = 2.0 * (_dLvisc_r - _beta_r)  # (1, N)
    c2_r = (_ddLvisc_r - 2.0 * _dbeta_r
            + _dLvisc_r ** 2 + _beta_r ** 2
            - 3.0 * _dLvisc_r * _beta_r
            - 2.0 * _or1_r * (_dLvisc_r + _beta_r)
            - 2.0 * _dLh_lm * _or2_r)  # (lm_max, N)
    c1_r = (-_ddbeta_r
            - _dbeta_r * (2.0 * _dLvisc_r - _beta_r + 2.0 * _or1_r)
            - _ddLvisc_r * (_beta_r + 2.0 * _or1_r)
            + _beta_r ** 2 * (_dLvisc_r + 2.0 * _or1_r)
            - _beta_r * (_dLvisc_r ** 2 - 2.0 * _or2_r)
            - 2.0 * _dLvisc_r * _or1_r * (_dLvisc_r - _or1_r)
            + 2.0 * (2.0 * _or1_r + _beta_r - _dLvisc_r) * _dLh_lm * _or2_r)  # (lm_max, N)
    c0_r = _dLh_lm * _or2_r * (
        2.0 * _dbeta_r + _ddLvisc_r
        + _dLvisc_r ** 2 - (2.0 / 3.0) * _beta_r ** 2
        + _dLvisc_r * _beta_r
        + 2.0 * _or1_r * (2.0 * _dLvisc_r - _beta_r - 3.0 * _or1_r)
        + _dLh_lm * _or2_r)  # (lm_max, N)

    Dif = -_hdif_lm * _dLh_lm * _or2_r * _visc_r * _orho1_r * (
        ddddw
        + c3_r * dddw
        + c2_r * ddw
        + c1_r * dw
        + c0_r * w
    )

    Buo = BuoFac * _dLh_lm * _or2_r * _rgrav_r * s
    if xi is not None:
        Buo = Buo + ChemFac * _dLh_lm * _or2_r * _rgrav_r * xi

    result = Dif + Buo
    result[:, 0] = 0.0
    result[:, -1] = 0.0
    return result


def updateW(s_LMloc, w_LMloc, dw_LMloc, ddw_LMloc,
            dwdt, p_LMloc, dp_LMloc, tscheme,
            xi_LMloc=None):
    """Double-curl poloidal velocity solve + post-processing.

    Solves the N×N 4th-order system for w only. Pressure is NOT co-solved
    (recovered separately when needed for diagnostics).

    The l=0 pressure is still solved via p0Mat (same as standard formulation).

    Modifies w, dw, ddw, dwdt, p, dp in place.
    """
    from .radial_derivatives import get_ddr

    N = n_r_max

    # 1. Assemble IMEX RHS for w
    rhs_w = tscheme.set_imex_rhs(dwdt)  # (lm_max, n_r_max)

    # 2. Add buoyancy coupling: wimp * BuoFac * dLh * or2 * rgrav * s
    wimp_lin0 = tscheme.wimp_lin[0]
    buo_fac_r = (or2[2:N-2] * rgrav[2:N-2]).unsqueeze(0).to(CDTYPE)
    rhs_w[:, 2:N-2] += wimp_lin0 * BuoFac * _dLh_lm * buo_fac_r * s_LMloc[:, 2:N-2]
    if xi_LMloc is not None:
        rhs_w[:, 2:N-2] += wimp_lin0 * ChemFac * _dLh_lm * buo_fac_r * xi_LMloc[:, 2:N-2]

    # 3. Boundary conditions: 4 BCs for 4th-order system
    # Row 0: w=0 at CMB, Row 1: dw/dr=0 at CMB (no-slip)
    # Row N-2: dw/dr=0 at ICB, Row N-1: w=0 at ICB
    rhs_w[:, 0] = 0.0
    rhs_w[:, 1] = 0.0
    rhs_w[:, N - 2] = 0.0
    rhs_w[:, N - 1] = 0.0
    # l=0 has no poloidal equation — zero its RHS
    # (For dense inverse, inv[l=0]=0 gives zero output.
    #  For banded solve, l=0 identity bands would return the RHS unchanged.)
    lm0_idx = (st_lm2l == 0)
    rhs_w[lm0_idx] = 0.0

    # 4. Solve
    if _w_penta_w1 is not None:
        # FD batched pentadiag (no Python loop over l)
        from .algebra import batched_pentadiag_solve_precomp
        rhs_precond = _w_penta_fac_row * rhs_w
        w_phys = batched_pentadiag_solve_precomp(
            _w_penta_w1, _w_penta_w2, _w_penta_inv_d,
            _w_penta_du1, _w_penta_du2, rhs_precond)
        w_phys = _w_penta_fac_col * w_phys  # column post-scaling
        w_phys[_m0_mask] = w_phys[_m0_mask].real.to(CDTYPE)
        w_LMloc[:] = w_phys
    elif _w_bands_by_l is not None:
        # FD fallback: per-l pivoted banded LU
        from .algebra import banded_solve_by_l
        w_phys = banded_solve_by_l(
            _w_bands_by_l, _w_piv_by_l, st_lm2l, rhs_w,
            n=N, kl=_w_kl, ku=_w_ku,
            fac_row_by_l=_w_fac_row_by_l, fac_col_by_l=_w_fac_col_by_l)
        w_phys[_m0_mask] = w_phys[_m0_mask].real.to(CDTYPE)
        w_LMloc[:] = w_phys
    else:
        w_cheb = chunked_solve_complex(_w_inv_by_l, st_lm2l, rhs_w)
        w_cheb[_m0_mask] = w_cheb[_m0_mask].real.to(CDTYPE)
        if n_cheb_max < N:
            w_cheb[:, n_cheb_max:] = 0.0
        from .radial_scheme import costf
        w_LMloc[:] = costf(w_cheb)

    # 7. Compute derivatives: dw, ddw via direct matrices, then
    #    dddw, ddddw via chaining D1(D2(w)), D2(D2(w)) — matches Fortran
    #    get_pol_rhs_imp (updateWP.f90:1172-1176) which calls get_ddr(w)→dw,ddw
    #    then get_ddr(ddw)→dddw,ddddw. For FD, chaining gives different
    #    results than direct D3/D4 matrices.
    dw_new, ddw_new = get_ddr(w_LMloc)
    dw_LMloc[:] = dw_new
    ddw_LMloc[:] = ddw_new
    dddw, ddddw = get_ddr(ddw_LMloc)  # chain: D1(D2(w)), D2(D2(w))

    # 8. l=0 pressure from p0Mat (same as standard formulation)
    from .update_wp import solve_p0, _lm_l0, _l_p0_integ_bc
    lm0 = _lm_l0
    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0
    p0_rhs = torch.zeros(N, dtype=DTYPE, device=DEVICE)
    p0_rhs[1:] = (rho0[1:] * BuoFac * rgrav[1:] * s_LMloc[lm0, 1:].real
                  + dwdt.expl[lm0, 1:, expl_idx].real)
    if xi_LMloc is not None:
        p0_rhs[1:] += rho0[1:] * ChemFac * rgrav[1:] * xi_LMloc[lm0, 1:].real
    # Row 0: integral BC for Chebyshev+anelastic (FD uses Dirichlet, stays 0)
    if _l_p0_integ_bc:
        from .radial_functions import alpha0, temp0, ThExpNb
        from .radial_scheme import r as _r_p0
        from .integration import rInt_R
        work = ThExpNb * alpha0 * temp0 * rho0 * _r_p0 * _r_p0 * s_LMloc[lm0].real
        p0_rhs[0] = rInt_R(work)
    p_LMloc[lm0, :] = solve_p0(p0_rhs).to(CDTYPE)
    # dp not computed in double-curl (Fortran doesn't either — pressure
    # derivative is only needed when l_RMS or l_FluxProfs is active)

    # 9. Rotate IMEX time arrays (w only, NOT dpdt)
    tscheme.rotate_imex(dwdt)

    # 10. Store old state
    if tscheme.store_old:
        dwdt.old[:, :, 0] = get_w_old(w_LMloc, dw_LMloc, ddw_LMloc)

    # 11. Compute implicit term for next step
    idx = tscheme.next_impl_idx
    dwdt.impl[:, :, idx] = get_w_impl(
        w_LMloc, dw_LMloc, ddw_LMloc, dddw, ddddw,
        s_LMloc, xi_LMloc)
