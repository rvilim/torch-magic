"""Implicit composition solver matching updateXi.f90.

Structurally identical to update_s.py but with different coefficients:
- osc (1/Schmidt number) instead of opr (1/Prandtl)
- hdif_Xi instead of hdif_S
- BCs: top=0, bot=sq4pi (same as entropy, set in preCalculations.f90 line 617)

Boussinesq specialization: kappa_xi=1, beta=0, dLkappa_xi=0, orho1=1.
"""

import math

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max, n_cheb_max
from .constants import two
from .radial_scheme import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2, beta, orho1
from .horizontal_data import dLh, hdif_Xi
from .params import l_anel
from .pre_calculations import osc
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex
from .radial_scheme import costf
from .radial_derivatives import get_dr, get_ddr


# --- Boundary condition values ---
# ktopxi=1, kbotxi=1: fixed composition at both boundaries
# topxi=0 for all (l,m); botxi=sq4pi for (l=0,m=0), 0 otherwise
# preCalculations.f90 line 617: botxi(0,0)=sq4pi (overrides init_fields.f90's one)
_topxi = torch.zeros(lm_max, dtype=CDTYPE, device=DEVICE)
_botxi = torch.zeros(lm_max, dtype=CDTYPE, device=DEVICE)
_sq4pi = math.sqrt(4.0 * math.pi)
_botxi[st_lm2[0, 0].item()] = complex(_sq4pi, 0.0)

# m=0 mask for forcing real coefficients
_m0_mask = (st_lm2m == 0)

# Precompute per-lm broadcast arrays for implicit term
_hdif_lm = hdif_Xi[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)                # (lm_max, 1)
_or1_r = or1.unsqueeze(0)                             # (1, n_r_max)
_or2_r = or2.unsqueeze(0)                             # (1, n_r_max)
_beta_r = beta.unsqueeze(0)                           # (1, n_r_max)
_orho1_r = orho1.unsqueeze(0)                         # (1, n_r_max)


# Unique inverse per l degree: (l_max+1, N, N) float64 (Chebyshev)
_xi_inv_by_l = None
# FD: banded LU storage per l for pivoted solve
_xi_bands_by_l = None
_xi_piv_by_l = None
_xi_fac_by_l = None
_xi_kl = 1
_xi_ku = 1


def build_xi_matrices(wimp_lin0: float):
    """Build and LU-factorize composition LHS matrices for each l degree.

    Same structure as entropy matrices but with osc instead of opr,
    and hdif_Xi instead of hdif_S.

    Must be called whenever dt changes.
    """
    global _xi_inv_by_l, _xi_bands_by_l, _xi_piv_by_l, _xi_fac_by_l, _xi_kl, _xi_ku
    from .params import l_finite_diff
    N = n_r_max

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1 = or1.to(cpu)
    _or2 = or2.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac
    or1_col = _or1.unsqueeze(1)
    or2_col = _or2.unsqueeze(1)
    _beta = beta.to(cpu)
    beta_col = _beta.unsqueeze(1)

    eye = torch.eye(N, dtype=DTYPE, device=cpu)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)
    # FD bandwidth (same as sMat: Dirichlet BCs → tridiag for fd_order<=2)
    if l_finite_diff:
        from .params import fd_order, fd_order_bound
        if fd_order <= 2 and fd_order_bound <= 2:
            _xi_kl, _xi_ku = 1, 1
        else:
            hw = max(fd_order // 2, fd_order_bound)
            _xi_kl, _xi_ku = hw, hw
    n_abd = 2 * _xi_kl + _xi_ku + 1 if l_finite_diff else 1
    abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    fac_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)

    for l in range(l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_Xi[l].item()

        dat = _rnorm * (_rMat - wimp_lin0 * osc * hdif_l * (
            _d2rMat + (beta_col + two * or1_col) * _drMat - dL * or2_col * _rMat
        ))

        dat[0, :] = _rnorm * _rMat[0, :]
        dat[N - 1, :] = _rnorm * _rMat[N - 1, :]

        if n_cheb_max < N:
            dat[0, n_cheb_max:N] = 0.0
            dat[N - 1, n_cheb_max:N] = 0.0

        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        fac = 1.0 / dat.abs().max(dim=1).values
        dat_precond = fac.unsqueeze(1) * dat

        if l_finite_diff:
            from .algebra import dense_to_band_storage, prepare_band
            abd = dense_to_band_storage(dat_precond, _xi_kl, _xi_ku)
            abd_f, piv, info = prepare_band(abd, N, _xi_kl, _xi_ku)
            assert info == 0, f"Singular xiMat (band) for l={l}, info={info}"
            abd_all[l] = abd_f
            piv_all[l] = piv
            fac_all[l] = fac
        else:
            lu, ip, info = prepare_mat(dat_precond)
            assert info == 0, f"Singular xiMat for l={l}, info={info}"
            inv_precond = solve_mat_real(lu, ip, eye)
            inv_by_l[l] = inv_precond * fac.unsqueeze(0)

    if l_finite_diff:
        _xi_bands_by_l = abd_all.to(DEVICE)
        _xi_piv_by_l = piv_all.to(DEVICE)
        _xi_fac_by_l = fac_all.to(DEVICE)
        _xi_inv_by_l = None
    else:
        _xi_inv_by_l = inv_by_l.to(DEVICE)
        _xi_bands_by_l = None


def finish_exp_comp(dxi_exp, dVXirLM):
    """Complete explicit composition term by adding radial derivative of dVXirLM.

    dxi_exp = orho1 * (dxi_exp - or2 * d(dVXirLM)/dr)

    Structurally identical to finish_exp_entropy.

    Args:
        dxi_exp: (lm_max, n_r_max) complex — partial explicit term from get_dxidt
        dVXirLM: (lm_max, n_r_max) complex — Q-component of v*Xi from SHT

    Returns:
        dxi_exp: (lm_max, n_r_max) complex — completed explicit term
    """
    d_dVXirLM = get_dr(dVXirLM)
    result = dxi_exp - _or2_r * d_dVXirLM
    if l_anel:
        result = _orho1_r * result
    return result


def updateXi(xi_LMloc, dxi_LMloc, dxidt, tscheme):
    """Composition equation: IMEX RHS assembly, implicit solve, post-processing.

    Structurally identical to updateS but with different coefficients and BCs.

    Modifies xi_LMloc, dxi_LMloc, dxidt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS
    rhs = tscheme.set_imex_rhs(dxidt)  # (lm_max, n_r_max)

    # 2. Batched solve: set BCs, then solve
    rhs[:, 0] = _topxi
    rhs[:, N - 1] = _botxi

    if _xi_bands_by_l is not None:
        from .algebra import banded_solve_by_l
        xi_phys = banded_solve_by_l(
            _xi_bands_by_l, _xi_piv_by_l, st_lm2l, rhs,
            n=N, kl=_xi_kl, ku=_xi_ku, fac_row_by_l=_xi_fac_by_l)
        xi_phys[_m0_mask] = xi_phys[_m0_mask].real.to(CDTYPE)
        xi_LMloc[:] = xi_phys
    else:
        xi_cheb = chunked_solve_complex(_xi_inv_by_l, st_lm2l, rhs)
        xi_cheb[_m0_mask] = xi_cheb[_m0_mask].real.to(CDTYPE)
        if n_cheb_max < N:
            xi_cheb[:, n_cheb_max:] = 0.0
        xi_LMloc[:] = costf(xi_cheb)

    # 5. Compute radial derivatives in physical space
    dxi_new, d2xi = get_ddr(xi_LMloc)
    dxi_LMloc[:] = dxi_new

    # 6. Rotate IMEX time arrays
    tscheme.rotate_imex(dxidt)

    # 7. Store current state as old
    if tscheme.store_old:
        dxidt.old[:, :, 0] = xi_LMloc.clone()

    # 8. Compute implicit diffusion term for next IMEX stage
    # impl = osc * hdif_Xi(l) * (d2xi + (beta+2*or1)*dxi - l(l+1)*or2*xi)
    idx = tscheme.next_impl_idx
    dxidt.impl[:, :, idx] = osc * _hdif_lm * (
        d2xi + (_beta_r + two * _or1_r) * dxi_LMloc - _dLh_lm * _or2_r * xi_LMloc
    )
