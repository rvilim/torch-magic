"""Implicit entropy solver matching updateS.f90.

Implements:
- build_s_matrices: LHS matrix construction + LU factorization per l-degree
- finish_exp_entropy: Complete explicit nonlinear term (radial derivative)
- updateS: Full IMEX solve + post-processing

Handles both Boussinesq (kappa=1, beta=0) and anelastic (variable profiles).
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max, n_cheb_max, l_anel
from .constants import sq4pi, two
from .radial_scheme import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2, beta, dLtemp0, dLkappa, kappa, orho1
from .horizontal_data import dLh, hdif_S
from .pre_calculations import opr
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex
from .radial_scheme import costf
from .radial_derivatives import get_dr, get_ddr


# --- Boundary condition values (constant for the benchmark) ---
# ktops=1, kbots=1: fixed entropy at both boundaries
# tops=0 for all (l,m); bots=sqrt(4*pi) for (l=0,m=0), 0 otherwise
_tops = torch.zeros(lm_max, dtype=CDTYPE, device=DEVICE)
_bots = torch.zeros(lm_max, dtype=CDTYPE, device=DEVICE)
_bots[st_lm2[0, 0].item()] = sq4pi

# m=0 mask for forcing real coefficients
_m0_mask = (st_lm2m == 0)


# --- Precompute per-l LM index groups ---
_l_lm_idx = []
for _l in range(l_max + 1):
    _l_lm_idx.append(torch.where(st_lm2l == _l)[0])

# Precompute per-lm broadcast arrays for implicit term
_hdif_lm = hdif_S[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)                # (lm_max, 1)
_or1_r = or1.unsqueeze(0)                             # (1, n_r_max)
_or2_r = or2.unsqueeze(0)                             # (1, n_r_max)
_orho1_r = orho1.unsqueeze(0)                         # (1, n_r_max)

# Anelastic first-derivative coefficient for implicit term
# (beta + dLtemp0 + 2/r + dLkappa) — Boussinesq: just 2/r
_s_impl_d1 = (beta + dLtemp0 + two * or1 + dLkappa).unsqueeze(0)  # (1, n_r_max)
_kappa_r = kappa.unsqueeze(0)                         # (1, n_r_max)


# --- LU-factored matrices storage (one per l degree) ---
_sMat_lu = [None] * (l_max + 1)
_sMat_ip = [None] * (l_max + 1)
_sMat_fac = [None] * (l_max + 1)

# Unique inverse per l degree: (l_max+1, N, N) float64 (Chebyshev)
_s_inv_by_l = None
# FD: banded LU storage per l for pivoted solve
_s_bands_by_l = None   # (l_max+1, n_abd_rows, N) band LU factors
_s_piv_by_l = None     # (l_max+1, N) pivot indices
_s_fac_by_l = None     # (l_max+1, N) row preconditioning
_s_kl = 1              # bandwidth (set in build_s_matrices for FD)
_s_ku = 1


def build_s_matrices(wimp_lin0: float):
    """Build and LU-factorize entropy LHS matrices for each l degree.

    Matrix in Chebyshev space:
        dat = rnorm * (rMat - wimp * opr * hdif * (d2rMat + 2*or1*drMat - dLh*or2*rMat))
    with Dirichlet BCs at rows 0 and N-1, boundary_fac on columns 0 and N-1,
    and row preconditioning (WITH_PRECOND_S).

    Must be called whenever dt changes.
    """
    global _s_inv_by_l, _s_bands_by_l, _s_piv_by_l, _s_fac_by_l, _s_kl, _s_ku
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

    eye = torch.eye(N, dtype=DTYPE, device=cpu)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)
    # FD bandwidth
    if l_finite_diff:
        from .params import fd_order, fd_order_bound
        if fd_order <= 2 and fd_order_bound <= 2:
            _s_kl, _s_ku = 1, 1
        else:
            hw = max(fd_order // 2, fd_order_bound)
            _s_kl, _s_ku = hw, hw
    n_abd = 2 * _s_kl + _s_ku + 1 if l_finite_diff else 1
    abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    fac_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)

    # Anelastic: first-derivative coefficient includes beta + dLtemp0 + dLkappa
    # Boussinesq: beta=0, dLtemp0=0, dLkappa=0 → just 2/r
    _beta = beta.to(cpu)
    _dLtemp0 = dLtemp0.to(cpu)
    _dLkappa = dLkappa.to(cpu)
    _kappa = kappa.to(cpu)
    d1_coeff = (_beta + _dLtemp0 + two * _or1 + _dLkappa).unsqueeze(1)  # (N, 1)
    kappa_col = _kappa.unsqueeze(1)  # (N, 1)

    for l in range(l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_S[l].item()

        dat = _rnorm * (_rMat - wimp_lin0 * opr * hdif_l * kappa_col * (
            _d2rMat + d1_coeff * _drMat - dL * or2_col * _rMat
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
            abd = dense_to_band_storage(dat_precond, _s_kl, _s_ku)
            abd_f, piv, info = prepare_band(abd, N, _s_kl, _s_ku)
            assert info == 0, f"Singular sMat (band) for l={l}, info={info}"
            abd_all[l] = abd_f
            piv_all[l] = piv
            fac_all[l] = fac
        else:
            lu, ip, info = prepare_mat(dat_precond)
            assert info == 0, f"Singular sMat for l={l}, info={info}"
            inv_precond = solve_mat_real(lu, ip, eye)
            inv_by_l[l] = inv_precond * fac.unsqueeze(0)

    if l_finite_diff:
        _s_bands_by_l = abd_all.to(DEVICE)
        _s_piv_by_l = piv_all.to(DEVICE)
        _s_fac_by_l = fac_all.to(DEVICE)
        _s_inv_by_l = None
    else:
        _s_inv_by_l = inv_by_l.to(DEVICE)
        _s_bands_by_l = None


def finish_exp_entropy(ds_exp, dVSrLM):
    """Complete explicit entropy term by adding radial derivative of dVSrLM.

    Non-anelastic-liquid:
        ds_exp = orho1 * (ds_exp - or2 * d(dVSrLM)/dr - dL*or2*dentropy0*w)
    For Boussinesq: orho1=1, dentropy0=0 → ds_exp - or2*d(dVSrLM)/dr
    For anelastic adiabatic: dentropy0=0 → orho1*(ds_exp - or2*d(dVSrLM)/dr)

    Args:
        ds_exp: (lm_max, n_r_max) complex — partial explicit term from get_dsdt
        dVSrLM: (lm_max, n_r_max) complex — Q-component of v*S from SHT

    Returns:
        ds_exp: (lm_max, n_r_max) complex — completed explicit term
    """
    d_dVSrLM = get_dr(dVSrLM)
    result = ds_exp - _or2_r * d_dVSrLM
    if l_anel:
        result = _orho1_r * result
    return result


def updateS(s_LMloc, ds_LMloc, dsdt, tscheme):
    """Entropy equation: IMEX RHS assembly, implicit solve, post-processing.

    1. Assemble IMEX RHS from time history
    2. Solve Chebyshev-space system for each l degree
    3. Convert to physical space via costf
    4. Compute derivatives and implicit term for next step

    Modifies s_LMloc, ds_LMloc, dsdt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS (physical-space values at grid points)
    rhs = tscheme.set_imex_rhs(dsdt)  # (lm_max, n_r_max)

    # 2. Batched solve: set BCs, then solve
    rhs[:, 0] = _tops
    rhs[:, N - 1] = _bots

    if _s_bands_by_l is not None:
        # FD: pivoted banded LU solve
        from .algebra import banded_solve_by_l
        s_phys = banded_solve_by_l(
            _s_bands_by_l, _s_piv_by_l, st_lm2l, rhs,
            n=N, kl=_s_kl, ku=_s_ku, fac_row_by_l=_s_fac_by_l)
        s_phys[_m0_mask] = s_phys[_m0_mask].real.to(CDTYPE)
        s_LMloc[:] = s_phys  # already physical space for FD
    else:
        s_cheb = chunked_solve_complex(_s_inv_by_l, st_lm2l, rhs)
        s_cheb[_m0_mask] = s_cheb[_m0_mask].real.to(CDTYPE)
        if n_cheb_max < N:
            s_cheb[:, n_cheb_max:] = 0.0
        s_LMloc[:] = costf(s_cheb)

    # 4. Compute radial derivatives in physical space
    ds_new, d2s = get_ddr(s_LMloc)
    ds_LMloc[:] = ds_new

    # 5. Rotate IMEX time arrays (shift explicit history)
    tscheme.rotate_imex(dsdt)

    # 6. Store current state as old (always for CNAB2, only at wrap for DIRK)
    if tscheme.store_old:
        dsdt.old[:, :, 0] = s_LMloc.clone()

    # 7. Compute implicit diffusion term for next IMEX stage
    # impl = opr * hdif_S(l) * kappa * (d2s + (beta+dLtemp0+2/r+dLkappa)*ds - dLh*or2*s)
    idx = tscheme.next_impl_idx
    dsdt.impl[:, :, idx] = opr * _hdif_lm * _kappa_r * (
        d2s + _s_impl_d1 * ds_LMloc - _dLh_lm * _or2_r * s_LMloc
    )
