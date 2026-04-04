"""Implicit magnetic field solver matching updateB.f90.

Implements:
- build_b_matrices: bMat (poloidal) + jMat (toroidal) per l-degree
- finish_exp_mag: Complete explicit induction term (radial derivative)
- updateB: Full IMEX solve + post-processing

Boussinesq: lambda=1, dLlambda=0.
When l_cond_ic: coupled OC+IC solve with n_r_tot x n_r_tot matrices.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import (n_r_max, n_r_ic_max, n_r_tot, n_cheb_max, n_cheb_ic_max,
                     lm_max, l_max, l_cond_ic, sigma_ratio)
from .radial_scheme import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2, lambda_, dLlambda
from .horizontal_data import dLh, hdif_B
from .pre_calculations import opm
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex
from .radial_scheme import costf
from .radial_derivatives import get_dr, get_ddr


# --- Precompute per-l LM index groups ---
_l_lm_idx = []
for _l in range(l_max + 1):
    _l_lm_idx.append(torch.where(st_lm2l == _l)[0])

# m=0 mask for forcing real coefficients
_m0_mask = (st_lm2m == 0)

# Broadcast arrays for implicit terms
_hdif_lm = hdif_B[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)                # (lm_max, 1)
_or2_r = or2.unsqueeze(0)                             # (1, n_r_max)
_lambda_r = lambda_.unsqueeze(0).to(CDTYPE)           # (1, n_r_max)
_dLlambda_r = dLlambda.unsqueeze(0).to(CDTYPE)        # (1, n_r_max)

# l=0 index
_lm_l0 = st_lm2[0, 0].item()

# Precompute ICB constants as Python floats (avoids GPU→CPU sync per step)
_or2_icb_py = or2[n_r_max - 1].item()
# Precompute m*dLh for finish_exp_mag_ic (constant across steps)
_im_dLh = (1j * st_lm2m.to(CDTYPE) * dLh.to(CDTYPE)).unsqueeze(1)  # (lm_max, 1)

# --- LU-factored matrices storage (one per l degree, l>=1) ---
_bMat_lu = [None] * (l_max + 1)
_bMat_ip = [None] * (l_max + 1)
_bMat_fac = [None] * (l_max + 1)

_jMat_lu = [None] * (l_max + 1)
_jMat_ip = [None] * (l_max + 1)
_jMat_fac = [None] * (l_max + 1)

# Unique inverses per l degree — l=0 is zero
_b_inv_by_l = None
_j_inv_by_l = None
# FD insulating: banded LU per l
_b_bands_by_l = None
_b_piv_by_l = None
_b_fac_by_l = None
_j_bands_by_l = None
_j_piv_by_l = None
_j_fac_by_l = None
_bj_kl = 0
_bj_ku = 0
# FD coupled IC: bordered-band per l
_b_bordered_by_l = None
_j_bordered_by_l = None
_b_fac_coupled_by_l = None
_j_fac_coupled_by_l = None

# IC-specific broadcast arrays (only when l_cond_ic)
if l_cond_ic:
    _or2_icb = or2[n_r_max - 1]  # scalar: 1/r_icb^2


def build_b_matrices(wimp_lin0: float):
    """Build and LU-factorize magnetic field LHS matrices for l >= 1.

    When l_cond_ic: builds coupled (n_r_tot x n_r_tot) matrices with
    OC+IC unknowns and ICB continuity conditions.

    Must be called whenever dt changes.
    """
    if l_cond_ic:
        _build_b_matrices_coupled(wimp_lin0)
    else:
        _build_b_matrices_insulating(wimp_lin0)


def _build_b_matrices_insulating(wimp_lin0: float):
    """Build insulating-IC matrices (original code path)."""
    global _b_inv_by_l, _j_inv_by_l
    global _b_bands_by_l, _b_piv_by_l, _b_fac_by_l
    global _j_bands_by_l, _j_piv_by_l, _j_fac_by_l, _bj_kl, _bj_ku
    from .params import l_finite_diff, fd_order, fd_order_bound
    N = n_r_max

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1 = or1.to(cpu)
    _or2 = or2.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac
    _lambda = lambda_.to(cpu)
    _dLlambda = dLlambda.to(cpu)
    or2_col = _or2.unsqueeze(1)
    lambda_col = _lambda.unsqueeze(1)
    dLlambda_col = _dLlambda.unsqueeze(1)

    eye = torch.eye(N, dtype=DTYPE, device=cpu)
    b_inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)
    j_inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)

    # FD: pentadiag for bMat (vacuum BCs use drMat boundary stencil)
    if l_finite_diff:
        _bj_kl = max(fd_order // 2, fd_order_bound)
        _bj_ku = _bj_kl
    n_abd = 2 * _bj_kl + _bj_ku + 1 if l_finite_diff else 1
    b_abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    b_piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    b_fac_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
    j_abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    j_piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    j_fac_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
    if l_finite_diff:
        # l=0 identity bands
        b_abd_all[0, _bj_kl + _bj_ku, :] = 1.0
        b_piv_all[0] = torch.arange(1, N + 1, dtype=torch.long)
        j_abd_all[0, _bj_kl + _bj_ku, :] = 1.0
        j_piv_all[0] = torch.arange(1, N + 1, dtype=torch.long)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_B[l].item()

        # === bMat (poloidal magnetic field) ===
        dat_b = torch.zeros(N, N, dtype=DTYPE, device=cpu)
        dat_b[1:N-1, :] = _rnorm * dL * or2_col[1:N-1] * (
            _rMat[1:N-1] - wimp_lin0 * opm * lambda_col[1:N-1] * hdif_l * (
                _d2rMat[1:N-1] - dL * or2_col[1:N-1] * _rMat[1:N-1]
            )
        )
        dat_b[0, :] = _rnorm * (
            _drMat[0, :] + float(l) * _or1[0] * _rMat[0, :]
        )
        dat_b[N-1, :] = _rnorm * (
            _drMat[N-1, :] - float(l + 1) * _or1[N-1] * _rMat[N-1, :]
        )
        if n_cheb_max < N:
            dat_b[0, n_cheb_max:N] = 0.0
            dat_b[N-1, n_cheb_max:N] = 0.0
        dat_b[:, 0] = dat_b[:, 0] * _bfac
        dat_b[:, N-1] = dat_b[:, N-1] * _bfac
        fac_b = 1.0 / dat_b.abs().max(dim=1).values
        dat_b_precond = fac_b.unsqueeze(1) * dat_b

        if l_finite_diff:
            from .algebra import dense_to_band_storage, prepare_band
            abd = dense_to_band_storage(dat_b_precond, _bj_kl, _bj_ku)
            abd_f, piv, info_b = prepare_band(abd, N, _bj_kl, _bj_ku)
            assert info_b == 0, f"Singular bMat (band) for l={l}, info={info_b}"
            b_abd_all[l] = abd_f
            b_piv_all[l] = piv
            b_fac_all[l] = fac_b
        else:
            lu_b, ip_b, info_b = prepare_mat(dat_b_precond)
            assert info_b == 0, f"Singular bMat for l={l}, info={info_b}"
            b_inv_precond = solve_mat_real(lu_b, ip_b, eye)
            b_inv_by_l[l] = b_inv_precond * fac_b.unsqueeze(0)

        # === jMat (toroidal magnetic field) ===
        dat_j = torch.zeros(N, N, dtype=DTYPE, device=cpu)
        dat_j[1:N-1, :] = _rnorm * dL * or2_col[1:N-1] * (
            _rMat[1:N-1] - wimp_lin0 * opm * lambda_col[1:N-1] * hdif_l * (
                _d2rMat[1:N-1] + dLlambda_col[1:N-1] * _drMat[1:N-1]
                - dL * or2_col[1:N-1] * _rMat[1:N-1]
            )
        )
        dat_j[0, :] = _rnorm * _rMat[0, :]
        dat_j[N-1, :] = _rnorm * _rMat[N-1, :]
        if n_cheb_max < N:
            dat_j[0, n_cheb_max:N] = 0.0
            dat_j[N-1, n_cheb_max:N] = 0.0
        dat_j[:, 0] = dat_j[:, 0] * _bfac
        dat_j[:, N-1] = dat_j[:, N-1] * _bfac
        fac_j = 1.0 / dat_j.abs().max(dim=1).values
        dat_j_precond = fac_j.unsqueeze(1) * dat_j

        if l_finite_diff:
            abd = dense_to_band_storage(dat_j_precond, _bj_kl, _bj_ku)
            abd_f, piv, info_j = prepare_band(abd, N, _bj_kl, _bj_ku)
            assert info_j == 0, f"Singular jMat (band) for l={l}, info={info_j}"
            j_abd_all[l] = abd_f
            j_piv_all[l] = piv
            j_fac_all[l] = fac_j
        else:
            lu_j, ip_j, info_j = prepare_mat(dat_j_precond)
            assert info_j == 0, f"Singular jMat for l={l}, info={info_j}"
            j_inv_precond = solve_mat_real(lu_j, ip_j, eye)
            j_inv_by_l[l] = j_inv_precond * fac_j.unsqueeze(0)

    if l_finite_diff:
        _b_bands_by_l = b_abd_all.to(DEVICE)
        _b_piv_by_l = b_piv_all.to(DEVICE)
        _b_fac_by_l = b_fac_all.to(DEVICE)
        _j_bands_by_l = j_abd_all.to(DEVICE)
        _j_piv_by_l = j_piv_all.to(DEVICE)
        _j_fac_by_l = j_fac_all.to(DEVICE)
        _b_inv_by_l = None
        _j_inv_by_l = None
    else:
        _b_inv_by_l = b_inv_by_l.to(DEVICE)
        _j_inv_by_l = j_inv_by_l.to(DEVICE)
        _b_bands_by_l = None
        _j_bands_by_l = None


def _build_b_matrices_coupled(wimp_lin0: float):
    """Build coupled OC+IC matrices for conducting inner core.

    Matrix structure (n_r_tot x n_r_tot):
    - Rows 0:          CMB BC (OC Chebyshev columns only)
    - Rows 1..N-2:     OC bulk diffusion (OC columns only)
    - Row N-1:         ICB continuity b_OC = b_IC (OC + IC columns)
    - Row N:           ICB derivative match (OC + IC columns)
    - Rows N+1..NT-2:  IC bulk diffusion (IC columns only)
    - Row NT-1:        IC center regularity (IC columns only)

    where N = n_r_max, NT = n_r_tot = n_r_max + n_r_ic_max.
    """
    global _b_inv_by_l, _j_inv_by_l, _b_bordered_by_l, _j_bordered_by_l, _b_fac_coupled_by_l, _j_fac_coupled_by_l
    global _bj_kl, _bj_ku
    from .radial_functions import cheb_ic, dcheb_ic, d2cheb_ic, O_r_ic, cheb_norm_ic
    from .pre_calculations import O_sr
    from .params import l_finite_diff, fd_order, fd_order_bound

    N = n_r_max
    N_ic = n_r_ic_max
    NT = n_r_tot

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1 = or1.to(cpu)
    _or2 = or2.to(cpu)
    _cheb_ic = cheb_ic.to(cpu)       # (N_ic, N_ic): [poly_idx, grid_idx]
    _dcheb_ic = dcheb_ic.to(cpu)
    _d2cheb_ic = d2cheb_ic.to(cpu)
    _O_r_ic = O_r_ic.to(cpu)
    _rnorm = rnorm
    _cnorm = cheb_norm_ic
    _bfac = boundary_fac

    or2_col = _or2.unsqueeze(1)
    or2_icb = _or2[N - 1].item()  # 1/r_icb^2
    _lambda = lambda_.to(cpu)
    _dLlambda = dLlambda.to(cpu)
    lambda_col = _lambda.unsqueeze(1)
    dLlambda_col = _dLlambda.unsqueeze(1)

    if l_finite_diff:
        _bj_kl = max(fd_order // 2, fd_order_bound)
        _bj_ku = _bj_kl
        _b_bordered_by_l = [None] * (l_max + 1)
        _j_bordered_by_l = [None] * (l_max + 1)
        _b_fac_coupled_by_l = torch.ones(l_max + 1, NT, dtype=DTYPE, device=cpu)
        _j_fac_coupled_by_l = torch.ones(l_max + 1, NT, dtype=DTYPE, device=cpu)
        _b_inv_by_l = None
        _j_inv_by_l = None
    else:
        eye = torch.eye(NT, dtype=DTYPE, device=cpu)
        b_inv_by_l = torch.zeros(l_max + 1, NT, NT, dtype=DTYPE, device=cpu)
        j_inv_by_l = torch.zeros(l_max + 1, NT, NT, dtype=DTYPE, device=cpu)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_B[l].item()
        wimp = wimp_lin0

        # ===== bMat =====
        dat_b = torch.zeros(NT, NT, dtype=DTYPE, device=cpu)

        # --- OC bulk (rows 1..N-2, cols 0..N-1) ---
        dat_b[1:N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            _rMat[1:N-1] - wimp * opm * lambda_col[1:N-1] * hdif_l * (
                _d2rMat[1:N-1] - dL * or2_col[1:N-1] * _rMat[1:N-1]
            )
        )

        # --- CMB BC (row 0, cols 0..N-1) ---
        dat_b[0, :N] = _rnorm * (
            _drMat[0, :] + float(l) * _or1[0] * _rMat[0, :]
        )

        # --- ICB continuity (row N-1): b_OC(r_icb) = b_IC(r_icb) ---
        dat_b[N-1, :N] = _rnorm * _rMat[N-1, :]
        dat_b[N-1, N:N+n_cheb_ic_max] = -_cnorm * _cheb_ic[:n_cheb_ic_max, 0]

        # --- ICB derivative match (row N): db/dr_OC = db/dr_IC + (l+1)/r * b_IC ---
        dat_b[N, :N] = _rnorm * _drMat[N-1, :]
        dat_b[N, N:N+n_cheb_ic_max] = -_cnorm * (
            _dcheb_ic[:n_cheb_ic_max, 0] + float(l + 1) * _or1[N-1] * _cheb_ic[:n_cheb_ic_max, 0]
        )

        # --- Zero high OC Chebyshev modes in boundary rows ---
        if n_cheb_max < N:
            dat_b[0, n_cheb_max:N] = 0.0
            dat_b[N-1, n_cheb_max:N] = 0.0
            dat_b[N, n_cheb_max:N] = 0.0

        # --- IC bulk (rows N+1..NT-2, cols N..NT-1) ---
        # IC grid indices 1..N_ic-2 (0-based; excludes ICB surface [0] and center [N_ic-1])
        ic_bulk_slice = slice(1, N_ic - 1)
        # cheb_ic[:, k] = T_{2n}(y_k), shape (N_ic,)
        # We need: dat_b[N+k, N+n] = cnorm * dL * or2_icb * (cheb[n,k] - wimp*...)
        # Vectorized: cheb_ic[:, ic_bulk].T gives (N_ic-2, N_ic)
        ic_eval = _cheb_ic[:, ic_bulk_slice].T     # (N_ic-2, N_ic)
        dic_eval = _dcheb_ic[:, ic_bulk_slice].T   # (N_ic-2, N_ic)
        d2ic_eval = _d2cheb_ic[:, ic_bulk_slice].T # (N_ic-2, N_ic)
        or_ic_col = _O_r_ic[ic_bulk_slice].unsqueeze(1)  # (N_ic-2, 1)

        dat_b[N+1:NT-1, N:] = _cnorm * dL * or2_icb * (
            ic_eval - wimp * opm * O_sr * (
                d2ic_eval + 2.0 * float(l + 1) * or_ic_col * dic_eval
            )
        )

        # --- IC center (row NT-1, cols N..NT-1) ---
        # L'Hopital limit at r=0: (1 + 2*(l+1)) * d2cheb replaces d2cheb + 2*(l+1)/r * dcheb
        dat_b[NT-1, N:] = _cnorm * dL * or2_icb * (
            _cheb_ic[:, N_ic-1] - wimp * opm * O_sr * (
                1.0 + 2.0 * float(l + 1)
            ) * _d2cheb_ic[:, N_ic-1]
        )

        # --- OC boundary factor (cols 0 and N-1) ---
        dat_b[:, 0] *= _bfac
        dat_b[:, N-1] *= _bfac

        # --- IC boundary factor (cols N and NT-1): half-weight endpoints ---
        dat_b[:, N] *= 0.5
        dat_b[:, NT-1] *= 0.5

        # --- Row scaling ---
        fac_b = 1.0 / dat_b.abs().max(dim=1).values
        dat_b = fac_b.unsqueeze(1) * dat_b

        if l_finite_diff:
            from .algebra import prepare_bordered
            _b_bordered_by_l[l] = prepare_bordered(dat_b, N, N_ic, _bj_kl, _bj_ku)
            _b_fac_coupled_by_l[l] = fac_b
        else:
            lu_b, ip_b, info_b = prepare_mat(dat_b)
            assert info_b == 0, f"Singular coupled bMat for l={l}, info={info_b}"
            b_inv_precond = solve_mat_real(lu_b, ip_b, eye)
            b_inv_by_l[l] = b_inv_precond * fac_b.unsqueeze(0)

        # ===== jMat =====
        dat_j = torch.zeros(NT, NT, dtype=DTYPE, device=cpu)

        # --- OC bulk ---
        dat_j[1:N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            _rMat[1:N-1] - wimp * opm * lambda_col[1:N-1] * hdif_l * (
                _d2rMat[1:N-1] + dLlambda_col[1:N-1] * _drMat[1:N-1]
                - dL * or2_col[1:N-1] * _rMat[1:N-1]
            )
        )

        # --- CMB BC (row 0): j=0 at CMB ---
        dat_j[0, :N] = _rnorm * _rMat[0, :]

        # --- ICB continuity (row N-1): j_OC(r_icb) = j_IC(r_icb) ---
        dat_j[N-1, :N] = _rnorm * _rMat[N-1, :]
        dat_j[N-1, N:N+n_cheb_ic_max] = -_cnorm * _cheb_ic[:n_cheb_ic_max, 0]

        # --- ICB derivative match (row N): sigma_ratio * dj/dr_OC = dj/dr_IC + (l+1)/r * j_IC ---
        dat_j[N, :N] = _rnorm * sigma_ratio * _drMat[N-1, :]
        dat_j[N, N:N+n_cheb_ic_max] = -_cnorm * (
            _dcheb_ic[:n_cheb_ic_max, 0] + float(l + 1) * _or1[N-1] * _cheb_ic[:n_cheb_ic_max, 0]
        )

        # --- Zero high OC Chebyshev modes in boundary rows ---
        if n_cheb_max < N:
            dat_j[0, n_cheb_max:N] = 0.0
            dat_j[N-1, n_cheb_max:N] = 0.0
            dat_j[N, n_cheb_max:N] = 0.0

        # --- IC bulk (same as bMat) ---
        dat_j[N+1:NT-1, N:] = _cnorm * dL * or2_icb * (
            ic_eval - wimp * opm * O_sr * (
                d2ic_eval + 2.0 * float(l + 1) * or_ic_col * dic_eval
            )
        )

        # --- IC center (same as bMat) ---
        dat_j[NT-1, N:] = _cnorm * dL * or2_icb * (
            _cheb_ic[:, N_ic-1] - wimp * opm * O_sr * (
                1.0 + 2.0 * float(l + 1)
            ) * _d2cheb_ic[:, N_ic-1]
        )

        # --- Boundary factors ---
        dat_j[:, 0] *= _bfac
        dat_j[:, N-1] *= _bfac
        dat_j[:, N] *= 0.5
        dat_j[:, NT-1] *= 0.5

        # --- Row scaling ---
        fac_j = 1.0 / dat_j.abs().max(dim=1).values
        dat_j = fac_j.unsqueeze(1) * dat_j

        if l_finite_diff:
            from .algebra import prepare_bordered
            _j_bordered_by_l[l] = prepare_bordered(dat_j, N, N_ic, _bj_kl, _bj_ku)
            _j_fac_coupled_by_l[l] = fac_j
        else:
            lu_j, ip_j, info_j = prepare_mat(dat_j)
            assert info_j == 0, f"Singular coupled jMat for l={l}, info={info_j}"
            j_inv_precond = solve_mat_real(lu_j, ip_j, eye)
            j_inv_by_l[l] = j_inv_precond * fac_j.unsqueeze(0)

    if not l_finite_diff:
        _b_inv_by_l = b_inv_by_l.to(DEVICE)
        _j_inv_by_l = j_inv_by_l.to(DEVICE)
    else:
        _b_fac_coupled_by_l = _b_fac_coupled_by_l.to(DEVICE)
        _j_fac_coupled_by_l = _j_fac_coupled_by_l.to(DEVICE)


def finish_exp_mag(dj_exp, dVxBhLM):
    """Complete explicit induction term.

    dj_exp += or2 * d(dVxBhLM)/dr   (l>0 only, but l=0 has dVxBhLM=0)

    Args:
        dj_exp: (lm_max, n_r_max) complex — partial explicit term
        dVxBhLM: (lm_max, n_r_max) complex — horizontal curl from SHT

    Returns:
        dj_exp: completed explicit term
    """
    d_dVxBhLM = get_dr(dVxBhLM)
    return dj_exp + _or2_r * d_dVxBhLM


def updateB(b_LMloc, db_LMloc, ddb_LMloc, aj_LMloc, dj_LMloc, ddj_LMloc,
            dbdt, djdt, tscheme,
            b_ic=None, db_ic=None, ddb_ic=None,
            aj_ic=None, dj_ic=None, ddj_ic=None,
            dbdt_ic=None, djdt_ic=None):
    """Magnetic field: IMEX solve + post-processing.

    - l=0: b=0, j=0 (no monopole)
    - l>=1: separate solves for b (poloidal) and j (toroidal)
    - When l_cond_ic: coupled OC+IC solve with n_r_tot unknowns

    Modifies b, db, ddb, aj, dj, ddj, dbdt, djdt in place.
    When l_cond_ic: also modifies IC field arrays and time arrays.
    """
    if l_cond_ic and b_ic is not None:
        _updateB_coupled(b_LMloc, db_LMloc, ddb_LMloc,
                         aj_LMloc, dj_LMloc, ddj_LMloc,
                         dbdt, djdt, tscheme,
                         b_ic, db_ic, ddb_ic,
                         aj_ic, dj_ic, ddj_ic,
                         dbdt_ic, djdt_ic)
    else:
        _updateB_insulating(b_LMloc, db_LMloc, ddb_LMloc,
                            aj_LMloc, dj_LMloc, ddj_LMloc,
                            dbdt, djdt, tscheme)


def _updateB_insulating(b_LMloc, db_LMloc, ddb_LMloc,
                         aj_LMloc, dj_LMloc, ddj_LMloc,
                         dbdt, djdt, tscheme):
    """Original insulating-IC magnetic field solve."""
    N = n_r_max

    # 1. Assemble IMEX RHS for b and j
    rhs_b = tscheme.set_imex_rhs(dbdt)  # (lm_max, n_r_max)
    rhs_j = tscheme.set_imex_rhs(djdt)  # (lm_max, n_r_max)

    # 2. Batched solve: BCs=0, then solve
    rhs_b[:, 0] = 0.0
    rhs_b[:, N - 1] = 0.0
    rhs_j[:, 0] = 0.0
    rhs_j[:, N - 1] = 0.0

    if _b_bands_by_l is not None:
        from .algebra import banded_solve_by_l
        b_cheb = banded_solve_by_l(
            _b_bands_by_l, _b_piv_by_l, st_lm2l, rhs_b,
            n=N, kl=_bj_kl, ku=_bj_ku, fac_row_by_l=_b_fac_by_l)
        j_cheb = banded_solve_by_l(
            _j_bands_by_l, _j_piv_by_l, st_lm2l, rhs_j,
            n=N, kl=_bj_kl, ku=_bj_ku, fac_row_by_l=_j_fac_by_l)
    else:
        b_cheb = chunked_solve_complex(_b_inv_by_l, st_lm2l, rhs_b)
        j_cheb = chunked_solve_complex(_j_inv_by_l, st_lm2l, rhs_j)

    b_cheb[_m0_mask] = b_cheb[_m0_mask].real.to(CDTYPE)
    j_cheb[_m0_mask] = j_cheb[_m0_mask].real.to(CDTYPE)

    # 3. Truncate high Chebyshev modes (Fortran only stores n_cheb_max modes)
    if n_cheb_max < N:
        b_cheb[:, n_cheb_max:] = 0.0
        j_cheb[:, n_cheb_max:] = 0.0

    # 4. Convert to physical space
    b_LMloc[:] = costf(b_cheb)
    aj_LMloc[:] = costf(j_cheb)

    # 4. Compute derivatives
    db_new, ddb_new = get_ddr(b_LMloc)
    db_LMloc[:] = db_new
    ddb_LMloc[:] = ddb_new

    dj_new, ddj_new = get_ddr(aj_LMloc)
    dj_LMloc[:] = dj_new
    ddj_LMloc[:] = ddj_new

    # 5. Rotate IMEX time arrays
    tscheme.rotate_imex(dbdt)
    tscheme.rotate_imex(djdt)

    # 6. Store old state (always for CNAB2, only at wrap for DIRK)
    if tscheme.store_old:
        dbdt.old[:, :, 0] = _dLh_lm * _or2_r * b_LMloc
        djdt.old[:, :, 0] = _dLh_lm * _or2_r * aj_LMloc

    # 7. Compute implicit terms
    idx = tscheme.next_impl_idx
    dbdt.impl[:, :, idx] = opm * _lambda_r * _hdif_lm * _dLh_lm * _or2_r * (
        ddb_LMloc - _dLh_lm * _or2_r * b_LMloc
    )

    djdt.impl[:, :, idx] = opm * _lambda_r * _hdif_lm * _dLh_lm * _or2_r * (
        ddj_LMloc + _dLlambda_r * dj_LMloc - _dLh_lm * _or2_r * aj_LMloc
    )


def _updateB_coupled(b_LMloc, db_LMloc, ddb_LMloc,
                      aj_LMloc, dj_LMloc, ddj_LMloc,
                      dbdt, djdt, tscheme,
                      b_ic, db_ic, ddb_ic,
                      aj_ic, dj_ic, ddj_ic,
                      dbdt_ic, djdt_ic):
    """Coupled OC+IC magnetic field solve for conducting inner core.

    Solves the combined (n_r_tot x n_r_tot) system for OC Chebyshev
    coefficients and IC even-Chebyshev coefficients simultaneously.
    """
    from .radial_derivatives import get_ddr_even

    N = n_r_max
    N_ic = n_r_ic_max
    NT = n_r_tot

    # 1. Assemble OC IMEX RHS
    oc_rhs_b = tscheme.set_imex_rhs(dbdt)   # (lm_max, n_r_max)
    oc_rhs_j = tscheme.set_imex_rhs(djdt)   # (lm_max, n_r_max)

    # 2. Assemble IC IMEX RHS
    ic_rhs_b = tscheme.set_imex_rhs(dbdt_ic)  # (lm_max, n_r_ic_max)
    ic_rhs_j = tscheme.set_imex_rhs(djdt_ic)  # (lm_max, n_r_ic_max)

    # 3. Build coupled RHS (n_r_tot columns)
    rhs_b = torch.zeros(lm_max, NT, dtype=CDTYPE, device=DEVICE)
    rhs_b[:, 1:N-1] = oc_rhs_b[:, 1:N-1]     # OC bulk
    # Row 0 (CMB BC) = 0
    # Row N-1 (ICB continuity) = 0
    # Row N (ICB derivative match) = 0
    rhs_b[:, N+1:NT] = ic_rhs_b[:, 1:N_ic]    # IC bulk + center

    rhs_j = torch.zeros(lm_max, NT, dtype=CDTYPE, device=DEVICE)
    rhs_j[:, 1:N-1] = oc_rhs_j[:, 1:N-1]     # OC bulk
    rhs_j[:, N+1:NT] = ic_rhs_j[:, 1:N_ic]    # IC bulk + center

    # 4. Solve: bordered-band per-l for FD, dense batched for Chebyshev
    from .params import l_finite_diff
    if l_finite_diff and _b_bordered_by_l is not None:
        from .algebra import solve_bordered
        cpu = torch.device("cpu")
        rhs_b_cpu = rhs_b.to(cpu)
        rhs_j_cpu = rhs_j.to(cpu)
        b_sol_cpu = torch.zeros_like(rhs_b_cpu)
        j_sol_cpu = torch.zeros_like(rhs_j_cpu)
        fac_b_cpu = _b_fac_coupled_by_l.to(cpu)
        fac_j_cpu = _j_fac_coupled_by_l.to(cpu)
        for l in range(1, l_max + 1):
            lm_idx = _l_lm_idx[l]
            if lm_idx.numel() == 0:
                continue
            fac_b = fac_b_cpu[l]
            fac_j = fac_j_cpu[l]
            for i in range(lm_idx.numel()):
                lm = lm_idx[i].item()
                b_sol_cpu[lm] = solve_bordered(
                    _b_bordered_by_l[l], fac_b * rhs_b_cpu[lm])
                j_sol_cpu[lm] = solve_bordered(
                    _j_bordered_by_l[l], fac_j * rhs_j_cpu[lm])
        b_sol_cpu[_m0_mask] = b_sol_cpu[_m0_mask].real.to(CDTYPE)
        j_sol_cpu[_m0_mask] = j_sol_cpu[_m0_mask].real.to(CDTYPE)
        b_sol = b_sol_cpu.to(DEVICE)
        j_sol = j_sol_cpu.to(DEVICE)
    else:
        b_sol = chunked_solve_complex(_b_inv_by_l, st_lm2l, rhs_b)
        b_sol[_m0_mask] = b_sol[_m0_mask].real.to(CDTYPE)

        j_sol = chunked_solve_complex(_j_inv_by_l, st_lm2l, rhs_j)
        j_sol[_m0_mask] = j_sol[_m0_mask].real.to(CDTYPE)

    # 5. Extract and transform OC part (Chebyshev → physical via costf)
    b_oc_cheb = b_sol[:, :N]
    j_oc_cheb = j_sol[:, :N]
    if n_cheb_max < N:
        b_oc_cheb = b_oc_cheb.clone()
        b_oc_cheb[:, n_cheb_max:] = 0.0
        j_oc_cheb = j_oc_cheb.clone()
        j_oc_cheb[:, n_cheb_max:] = 0.0
    b_LMloc[:] = costf(b_oc_cheb)
    aj_LMloc[:] = costf(j_oc_cheb)

    # 6. Extract and transform IC part (even-Chebyshev → physical via costf)
    # Extract all n_r_ic_max coefficients, zero high modes, then full costf
    b_ic_cheb = b_sol[:, N:N+N_ic].clone()
    aj_ic_cheb = j_sol[:, N:N+N_ic].clone()
    if n_cheb_ic_max < N_ic:
        b_ic_cheb[:, n_cheb_ic_max:] = 0.0
        aj_ic_cheb[:, n_cheb_ic_max:] = 0.0
    # IC always uses Chebyshev DCT regardless of OC scheme (FD or Chebyshev).
    from .radial_scheme import ic_costf
    b_ic[:] = ic_costf(b_ic_cheb)
    aj_ic[:] = ic_costf(aj_ic_cheb)

    # 7. Compute OC derivatives
    db_new, ddb_new = get_ddr(b_LMloc)
    db_LMloc[:] = db_new
    ddb_LMloc[:] = ddb_new

    dj_new, ddj_new = get_ddr(aj_LMloc)
    dj_LMloc[:] = dj_new
    ddj_LMloc[:] = ddj_new

    # 8. Compute IC derivatives
    db_ic_new, ddb_ic_new = get_ddr_even(b_ic)
    db_ic[:] = db_ic_new
    ddb_ic[:] = ddb_ic_new

    dj_ic_new, ddj_ic_new = get_ddr_even(aj_ic)
    dj_ic[:] = dj_ic_new
    ddj_ic[:] = ddj_ic_new

    # 9. Rotate IMEX time arrays (OC + IC)
    tscheme.rotate_imex(dbdt)
    tscheme.rotate_imex(djdt)
    tscheme.rotate_imex(dbdt_ic)
    tscheme.rotate_imex(djdt_ic)

    # 10. Store old state (always for CNAB2, only at wrap for DIRK)
    if tscheme.store_old:
        # OC old
        dbdt.old[:, :, 0] = _dLh_lm * _or2_r * b_LMloc
        djdt.old[:, :, 0] = _dLh_lm * _or2_r * aj_LMloc

        # IC old: dLh * or2(r_icb) * field (skip lm=0)
        or2_icb = _or2_icb
        dbdt_ic.old[1:, :, 0] = _dLh_lm[1:] * or2_icb * b_ic[1:]
        djdt_ic.old[1:, :, 0] = _dLh_lm[1:] * or2_icb * aj_ic[1:]

    # 11. Compute OC implicit terms
    idx = tscheme.next_impl_idx
    dbdt.impl[:, :, idx] = opm * _lambda_r * _hdif_lm * _dLh_lm * _or2_r * (
        ddb_LMloc - _dLh_lm * _or2_r * b_LMloc
    )
    djdt.impl[:, :, idx] = opm * _lambda_r * _hdif_lm * _dLh_lm * _or2_r * (
        ddj_LMloc + _dLlambda_r * dj_LMloc - _dLh_lm * _or2_r * aj_LMloc
    )

    # 12. Compute IC implicit terms
    from .pre_calculations import O_sr
    from .radial_functions import O_r_ic

    or2_icb = _or2_icb
    l_arr = st_lm2l.to(DTYPE)
    l_plus_1 = (l_arr + 1.0).unsqueeze(1).to(CDTYPE)  # (lm_max, 1)
    O_r_ic_r = O_r_ic.to(DEVICE).unsqueeze(0)  # (1, n_r_ic_max)

    # IC bulk (indices 1..N_ic-2)
    bulk = slice(1, N_ic - 1)
    impl_b_bulk = opm * O_sr * _dLh_lm * or2_icb * (
        ddb_ic[:, bulk] + 2.0 * l_plus_1 * O_r_ic_r[:, bulk] * db_ic[:, bulk]
    )
    impl_j_bulk = opm * O_sr * _dLh_lm * or2_icb * (
        ddj_ic[:, bulk] + 2.0 * l_plus_1 * O_r_ic_r[:, bulk] * dj_ic[:, bulk]
    )
    dbdt_ic.impl[1:, bulk, idx] = impl_b_bulk[1:]
    djdt_ic.impl[1:, bulk, idx] = impl_j_bulk[1:]

    # IC center (index N_ic-1): L'Hopital limit
    center = N_ic - 1
    fac_center = (1.0 + 2.0 * l_plus_1)  # (lm_max, 1)
    dbdt_ic.impl[1:, center, idx] = (
        opm * O_sr * _dLh_lm[1:, 0] * or2_icb * fac_center[1:, 0] * ddb_ic[1:, center]
    )
    djdt_ic.impl[1:, center, idx] = (
        opm * O_sr * _dLh_lm[1:, 0] * or2_icb * fac_center[1:, 0] * ddj_ic[1:, center]
    )


def get_mag_ic_rhs_imp(b_ic, db_ic, ddb_ic, aj_ic, dj_ic, ddj_ic,
                        dbdt_ic, djdt_ic, istage, l_calc_lin):
    """Compute IC magnetic field derivatives and linear implicit terms.

    Matches get_mag_ic_rhs_imp from updateB.f90 lines 1077-1187.
    Used during setup_initial_state (before first time step).

    1. Compute db_ic, ddb_ic, dj_ic, ddj_ic from b_ic, aj_ic via get_ddr_even
    2. Store old terms: dLh * or2(r_icb) * field (at istage==1)
    3. Store implicit terms: opm/sigma_ratio * dLh * or2(r_icb) * diffusion
       Diffusion = ddf + 2*(l+1)/r * df (bulk), (1+2*(l+1))*ddf at center

    Args:
        b_ic, aj_ic: (lm_max, n_r_ic_max) complex IC fields (modified: not changed)
        db_ic, ddb_ic, dj_ic, ddj_ic: (lm_max, n_r_ic_max) outputs
        dbdt_ic, djdt_ic: TimeArray IC time derivatives
        istage: time scheme stage index (1-based)
        l_calc_lin: whether to compute implicit terms
    """
    from .radial_derivatives import get_ddr_even
    from .radial_functions import O_r_ic
    from .pre_calculations import O_sr

    # 1. Compute IC derivatives
    db_new, ddb_new = get_ddr_even(b_ic)
    db_ic[:] = db_new
    ddb_ic[:] = ddb_new

    dj_new, ddj_new = get_ddr_even(aj_ic)
    dj_ic[:] = dj_new
    ddj_ic[:] = ddj_new

    # Precompute broadcast arrays
    # dLh and or2 at ICB (r_icb = r[n_r_max-1])
    or2_icb = or2[n_r_max - 1]  # scalar
    dLh_lm = dLh.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
    l_arr = st_lm2l.to(DTYPE)  # (lm_max,)

    # 2. Store old terms at istage==1
    if istage == 1:
        # old = dLh * or2(r_icb) * field (skip lm=0)
        dbdt_ic.old[1:, :, 0] = dLh_lm[1:] * or2_icb * b_ic[1:]
        djdt_ic.old[1:, :, 0] = dLh_lm[1:] * or2_icb * aj_ic[1:]

    # 3. Store implicit terms
    if l_calc_lin:
        O_r_ic_r = O_r_ic.to(DEVICE).unsqueeze(0)  # (1, n_r_ic_max)
        l_plus_1 = (l_arr + 1.0).unsqueeze(1).to(CDTYPE)  # (lm_max, 1)

        # Bulk (r indices 1 to n_r_ic_max-2, 0-indexed)
        bulk = slice(1, n_r_ic_max - 1)
        impl_b_bulk = opm * O_sr * dLh_lm * or2_icb * (
            ddb_ic[:, bulk] + 2.0 * l_plus_1 * O_r_ic_r[:, bulk] * db_ic[:, bulk]
        )
        impl_j_bulk = opm * O_sr * dLh_lm * or2_icb * (
            ddj_ic[:, bulk] + 2.0 * l_plus_1 * O_r_ic_r[:, bulk] * dj_ic[:, bulk]
        )
        dbdt_ic.impl[1:, bulk, istage - 1] = impl_b_bulk[1:]
        djdt_ic.impl[1:, bulk, istage - 1] = impl_j_bulk[1:]

        # Center (r=0, index n_r_ic_max-1): L'Hopital limit
        # (1 + 2*(l+1)) * ddf
        center = n_r_ic_max - 1
        fac_center = (1.0 + 2.0 * l_plus_1)  # (lm_max, 1)
        dbdt_ic.impl[1:, center, istage - 1] = (
            opm * O_sr * dLh_lm[1:, 0] * or2_icb * fac_center[1:, 0] * ddb_ic[1:, center]
        )
        djdt_ic.impl[1:, center, istage - 1] = (
            opm * O_sr * dLh_lm[1:, 0] * or2_icb * fac_center[1:, 0] * ddj_ic[1:, center]
        )


def finish_exp_mag_ic(b_ic, aj_ic, omega_ic: float,
                      db_exp_out, dj_exp_out):
    """IC magnetic advection by solid-body rotation (updateB.f90 lines 955-1003).

    When omega_ic != 0, computes:
        db_exp = -omega_ic * or2(ICB) * i*m * l(l+1) * b_ic
        dj_exp = -omega_ic * or2(ICB) * i*m * l(l+1) * aj_ic

    Args:
        b_ic, aj_ic: IC field arrays (lm_max, n_r_ic_max)
        omega_ic: IC rotation rate
        db_exp_out, dj_exp_out: output explicit arrays (lm_max, n_r_ic_max)
    """
    from .params import l_rot_ic
    if omega_ic != 0.0 and l_rot_ic:
        # fac = -omega_ic * or2(n_r_max) * i*m * l*(l+1)
        # Use precomputed _or2_icb_py and _im_dLh to avoid GPU syncs
        fac_2d = (-omega_ic * _or2_icb_py) * _im_dLh  # (lm_max, 1)
        # Apply to all IC radii except r=0 (nR=1 in Fortran = index 0 in Python)
        # Fortran loops n_r=2..n_r_ic_max, Python: indices 1..n_r_ic_max-1
        db_exp_out[:, 1:] = fac_2d * b_ic[:, 1:]
        dj_exp_out[:, 1:] = fac_2d * aj_ic[:, 1:]
        db_exp_out[:, 0] = 0.0
        dj_exp_out[:, 0] = 0.0
    else:
        db_exp_out[:] = 0.0
        dj_exp_out[:] = 0.0
