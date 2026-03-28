"""Implicit poloidal velocity + pressure solver matching updateWP.f90.

Implements:
- build_p0_matrix: l=0 pressure matrix (dp/dr equation)
- build_wp_matrices: 2N×2N coupled (w, p) matrices for l>=1
- finish_exp_pol: Complete explicit poloidal term (radial derivative)
- updateWP: Full IMEX solve + post-processing

Boussinesq: visc=1, beta=0, dLvisc=0, dbeta=0, rho0=1, orho1=1.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .constants import two
from .chebyshev import rMat, drMat, d2rMat, d3rMat, rnorm, boundary_fac
from .radial_functions import or1, or2, or3, rgrav
from .horizontal_data import dLh, hdif_V
from .pre_calculations import BuoFac
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real
from .cosine_transform import costf
from .radial_derivatives import get_dr, get_dddr


# --- Precompute per-l LM index groups ---
_l_lm_idx = []
for _l in range(l_max + 1):
    _l_lm_idx.append(torch.where(st_lm2l == _l)[0])

# m=0 mask for forcing real coefficients
_m0_mask = (st_lm2m == 0)

# Broadcast arrays for implicit terms
_hdif_lm = hdif_V[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)                # (lm_max, 1)
_or1_r = or1.unsqueeze(0)                             # (1, n_r_max)
_or2_r = or2.unsqueeze(0)                             # (1, n_r_max)
_or3_r = or3.unsqueeze(0)                             # (1, n_r_max)
_rgrav_r = rgrav.unsqueeze(0).to(CDTYPE)              # (1, n_r_max)

_lm_l0 = st_lm2[0, 0].item()

# --- l=0 pressure matrix ---
_p0Mat_lu = None
_p0Mat_ip = None

# --- l>=1 coupled (w,p) matrices ---
_wpMat_lu = [None] * (l_max + 1)
_wpMat_ip = [None] * (l_max + 1)
_wpMat_fac_row = [None] * (l_max + 1)  # row preconditioning
_wpMat_fac_col = [None] * (l_max + 1)  # column preconditioning

# Batched combined inverse: (lm_max, 2N, 2N) complex — l=0 rows are zero
_wp_inv_all = None


def build_p0_matrix():
    """Build and LU-factorize the l=0 pressure matrix.

    For Boussinesq (beta=0, ThExpNb*ViscHeatFac=0):
        Row 0: rnorm * rMat[0,:] (Dirichlet: p=0 at CMB)
        Rows 1..N-1: rnorm * drMat[nR,:] (pressure gradient equation)
    """
    global _p0Mat_lu, _p0Mat_ip
    N = n_r_max

    dat = torch.zeros(N, N, dtype=DTYPE, device=DEVICE)

    # Bulk: dp/dr equation (beta=0)
    dat[1:, :] = rnorm * drMat[1:, :]

    # BC: p(r_cmb) = 0
    dat[0, :] = rnorm * rMat[0, :]

    # Boundary factor
    dat[:, 0] = dat[:, 0] * boundary_fac
    dat[:, N - 1] = dat[:, N - 1] * boundary_fac

    lu, ip, info = prepare_mat(dat)
    assert info == 0, "Singular p0Mat"
    _p0Mat_lu = lu
    _p0Mat_ip = ip


def build_wp_matrices(wimp_lin0: float):
    """Build and LU-factorize 2N×2N coupled (w,p) matrices for l>=1.

    Layout: rows/cols 0..N-1 = w block, N..2N-1 = p block.

    Boussinesq simplifications: visc=1, beta=0, dLvisc=0, dbeta=0.

    Must be called whenever dt changes.
    """
    N = n_r_max
    or1_col = or1.unsqueeze(1)
    or2_col = or2.unsqueeze(1)
    or3_col = or3.unsqueeze(1)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        dat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=DEVICE)

        # === W equation (rows 0..N-1, interior rows 1..N-2) ===
        # w-w block: dLh*or2*(rMat - wimp*hdif*(d2rMat - dLh*or2*rMat))
        dat[1:N-1, :N] = rnorm * dL * or2_col[1:N-1] * (
            rMat[1:N-1] - wimp_lin0 * hdif_l * (
                d2rMat[1:N-1] - dL * or2_col[1:N-1] * rMat[1:N-1]
            )
        )

        # w-p block: wimp * drMat (beta=0)
        dat[1:N-1, N:] = rnorm * wimp_lin0 * drMat[1:N-1]

        # === P equation (rows N..2N-1, interior rows N+1..2N-2) ===
        # p-w block: -dLh*or2*drMat + wimp*hdif*dLh*or2*(d3rMat - dLh*or2*drMat + 2*dLh*or3*rMat)
        dat[N+1:2*N-1, :N] = rnorm * dL * or2_col[1:N-1] * (
            -drMat[1:N-1] + wimp_lin0 * hdif_l * (
                d3rMat[1:N-1] - dL * or2_col[1:N-1] * drMat[1:N-1]
                + two * dL * or3_col[1:N-1] * rMat[1:N-1]
            )
        )

        # p-p block: -wimp * dLh * or2 * rMat
        dat[N+1:2*N-1, N:] = -rnorm * wimp_lin0 * dL * or2_col[1:N-1] * rMat[1:N-1]

        # === Boundary conditions ===
        # Row 0 (CMB w=0): rnorm * rMat[0,:]
        dat[0, :N] = rnorm * rMat[0, :]
        # Row N-1 (ICB w=0): rnorm * rMat[N-1,:]
        dat[N-1, :N] = rnorm * rMat[N-1, :]
        # Row N (CMB dw/dr=0, no-slip): rnorm * drMat[0,:]
        dat[N, :N] = rnorm * drMat[0, :]
        # Row 2N-1 (ICB dw/dr=0, no-slip): rnorm * drMat[N-1,:]
        dat[2*N-1, :N] = rnorm * drMat[N-1, :]
        # p columns in BC rows are zero (already)

        # === Boundary factor on first/last Chebyshev columns of each block ===
        for blk_start in [0, N]:
            dat[:, blk_start] = dat[:, blk_start] * boundary_fac
            dat[:, blk_start + N - 1] = dat[:, blk_start + N - 1] * boundary_fac

        # === Two-pass preconditioning (row then column) ===
        # Row scaling
        fac_row = 1.0 / dat.abs().max(dim=1).values
        dat = fac_row.unsqueeze(1) * dat

        # Column scaling
        fac_col = 1.0 / dat.abs().max(dim=0).values
        dat = dat * fac_col.unsqueeze(0)

        # LU factorize
        lu, ip, info = prepare_mat(dat)
        assert info == 0, f"Singular wpMat for l={l}, info={info}"

        _wpMat_lu[l] = lu
        _wpMat_ip[l] = ip
        _wpMat_fac_row[l] = fac_row
        _wpMat_fac_col[l] = fac_col

    # Precompute batched combined inverse (l=0 stays zero → w(l=0)=0)
    # combined_inv = fac_col[:, None] * inv(precondA) * fac_row[None, :]
    global _wp_inv_all
    N = n_r_max
    eye = torch.eye(2 * N, dtype=DTYPE, device=DEVICE)
    wp_inv_by_l = torch.zeros(l_max + 1, 2 * N, 2 * N, dtype=DTYPE, device=DEVICE)
    for l in range(1, l_max + 1):
        inv_precond = solve_mat_real(_wpMat_lu[l], _wpMat_ip[l], eye)
        wp_inv_by_l[l] = _wpMat_fac_col[l].unsqueeze(1) * inv_precond * _wpMat_fac_row[l].unsqueeze(0)
    _wp_inv_all = wp_inv_by_l[st_lm2l]  # (lm_max, 2N, 2N) float64 — kept real for fast bmm


def finish_exp_pol(dw_exp, dVxVhLM):
    """Complete explicit poloidal term.

    dw_exp += or2 * d(dVxVhLM)/dr   (l>0 only, but l=0 has dVxVhLM=0)

    Args:
        dw_exp: (lm_max, n_r_max) complex — partial explicit term from get_dwdt
        dVxVhLM: (lm_max, n_r_max) complex — horizontal divergence from SHT

    Returns:
        dw_exp: completed explicit term
    """
    d_dVxVhLM = get_dr(dVxVhLM)
    return dw_exp + _or2_r * d_dVxVhLM


def updateWP(s_LMloc, w_LMloc, dw_LMloc, ddw_LMloc,
             dwdt, p_LMloc, dp_LMloc, dpdt, tscheme):
    """Poloidal velocity + pressure: IMEX solve + post-processing.

    - l=0: w=0, pressure from buoyancy + explicit terms via p0Mat
    - l>=1: coupled 2N×2N system for (w, p)
    - Buoyancy coupling: wimp * BuoFac * rgrav * s added to w-RHS

    Modifies w, dw, ddw, dwdt, p, dp, dpdt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS for w and p
    rhs_w = tscheme.set_imex_rhs(dwdt)  # (lm_max, n_r_max)
    rhs_p = tscheme.set_imex_rhs(dpdt)  # (lm_max, n_r_max)

    # === l=0: compute p0 RHS (solve after batched solve) ===
    lm0 = _lm_l0
    p0_rhs = torch.zeros(N, dtype=DTYPE, device=DEVICE)
    p0_rhs[1:] = (BuoFac * rgrav[1:] * s_LMloc[lm0, 1:].real
                  + dwdt.expl[lm0, 1:, tscheme.istage].real)

    # === l>=1: batched coupled (w, p) solve ===
    wimp_lin0 = tscheme.wimp_lin[0]

    # Build combined RHS: (lm_max, 2N) — w in rows 0..N-1, p in rows N..2N-1
    rhs_combined = torch.zeros(lm_max, 2 * N, dtype=CDTYPE, device=DEVICE)

    # w rows (interior): IMEX RHS + implicit buoyancy coupling
    rhs_combined[:, 1:N-1] = rhs_w[:, 1:N-1]
    rhs_combined[:, 1:N-1] += (wimp_lin0 * BuoFac *
                                rgrav[1:N-1].unsqueeze(0).to(CDTYPE) *
                                s_LMloc[:, 1:N-1])

    # p rows (interior): IMEX RHS for pressure
    rhs_combined[:, N+1:2*N-1] = rhs_p[:, 1:N-1]

    # BCs: all zero (w=0, dw/dr=0 at both boundaries) — already zero
    # l=0 modes get zero solution (inv is zero for l=0)

    # Batched solve via precomputed inverse
    # Real bmm: inverse is float64, use view_as_real for ~4x speedup
    sol = torch.view_as_complex(torch.bmm(_wp_inv_all, torch.view_as_real(rhs_combined)))
    sol[_m0_mask] = sol[_m0_mask].real.to(CDTYPE)

    # Extract w and p Chebyshev coefficients
    w_cheb = sol[:, :N]
    p_cheb = sol[:, N:]

    # l=0 pressure from p0Mat (overwrite the zero from batched solve)
    p_cheb[lm0, :] = solve_mat_real(_p0Mat_lu, _p0Mat_ip, p0_rhs).to(CDTYPE)

    # 3. Convert to physical space
    w_LMloc[:] = costf(w_cheb)
    p_LMloc[:] = costf(p_cheb)

    # 4. Compute derivatives
    dw_new, ddw_new, dddw = get_dddr(w_LMloc)
    dw_LMloc[:] = dw_new
    ddw_LMloc[:] = ddw_new
    dp_new = get_dr(p_LMloc)
    dp_LMloc[:] = dp_new

    # 5. Rotate IMEX time arrays
    tscheme.rotate_imex(dwdt)
    tscheme.rotate_imex(dpdt)

    # 6. Store old state (istage=1, skip l=0)
    # dwdt.old = dLh * or2 * w
    # dpdt.old = -dLh * or2 * dw
    dwdt.old[:, :, 0] = _dLh_lm * _or2_r * w_LMloc
    dpdt.old[:, :, 0] = -_dLh_lm * _or2_r * dw_LMloc

    # 7. Compute implicit terms (interior points, l>0; l=0 contributes zero via dLh=0)
    # Dif = hdif_V(l) * dLh * or2 * (ddw - dLh * or2 * w)
    # Pre = -dp  (beta=0)
    # Buo = BuoFac * rgrav * s  (rho0=1)
    # dwdt.impl = Pre + Dif + Buo
    Dif_w = _hdif_lm * _dLh_lm * _or2_r * (ddw_LMloc - _dLh_lm * _or2_r * w_LMloc)
    dwdt.impl[:, :, 0] = -dp_LMloc + Dif_w + BuoFac * _rgrav_r * s_LMloc

    # dpdt.impl = dLh*or2*p + hdif_V(l)*dLh*or2*(-dddw + dLh*or2*dw - 2*dLh*or3*w)
    dpdt.impl[:, :, 0] = (_dLh_lm * _or2_r * p_LMloc
                          + _hdif_lm * _dLh_lm * _or2_r * (
                              -dddw + _dLh_lm * _or2_r * dw_LMloc
                              - two * _dLh_lm * _or3_r * w_LMloc
                          ))
