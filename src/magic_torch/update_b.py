"""Implicit magnetic field solver matching updateB.f90.

Implements:
- build_b_matrices: bMat (poloidal) + jMat (toroidal) per l-degree
- finish_exp_mag: Complete explicit induction term (radial derivative)
- updateB: Full IMEX solve + post-processing

Boussinesq: lambda=1, dLlambda=0.
Insulating boundaries: ktopb=1, kbotb=1, conductance_ma=0.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2
from .horizontal_data import dLh, hdif_B
from .pre_calculations import opm
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real
from .cosine_transform import costf
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

# l=0 index
_lm_l0 = st_lm2[0, 0].item()

# --- LU-factored matrices storage (one per l degree, l>=1) ---
_bMat_lu = [None] * (l_max + 1)
_bMat_ip = [None] * (l_max + 1)
_bMat_fac = [None] * (l_max + 1)

_jMat_lu = [None] * (l_max + 1)
_jMat_ip = [None] * (l_max + 1)
_jMat_fac = [None] * (l_max + 1)

# Batched combined inverses: (lm_max, N, N) complex — l=0 rows are zero
_b_inv_all = None
_j_inv_all = None


def build_b_matrices(wimp_lin0: float):
    """Build and LU-factorize magnetic field LHS matrices for l >= 1.

    bMat (poloidal b): potential field matching BCs at CMB/ICB.
    jMat (toroidal j): insulating BCs (j=0) at CMB/ICB.

    Bulk equations (Boussinesq, lambda=1, dLlambda=0):
        bMat = jMat (interior) = rnorm * dLh*or2*(rMat - wimp*opm*hdif*(d2rMat - dLh*or2*rMat))

    Must be called whenever dt changes.
    """
    N = n_r_max
    or1_col = or1.unsqueeze(1)  # (N, 1)
    or2_col = or2.unsqueeze(1)  # (N, 1)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_B[l].item()

        # === bMat (poloidal magnetic field) ===
        dat_b = torch.zeros(N, N, dtype=DTYPE, device=DEVICE)

        # Interior rows (1..N-2): dLh*or2*(I - wimp*opm*hdif*(d2 - dLh*or2))
        dat_b[1:N-1, :] = rnorm * dL * or2_col[1:N-1] * (
            rMat[1:N-1] - wimp_lin0 * opm * hdif_l * (
                d2rMat[1:N-1] - dL * or2_col[1:N-1] * rMat[1:N-1]
            )
        )

        # CMB BC (row 0): potential field matching db/dr + l/r * b = 0
        # ktopb=1, ktopv=2, conductance_ma=0:
        #   drMat(0,:) + l*or1(0)*rMat(0,:)
        dat_b[0, :] = rnorm * (
            drMat[0, :] + float(l) * or1[0] * rMat[0, :]
        )

        # ICB BC (row N-1): potential field matching db/dr - (l+1)/r * b = 0
        # kbotb=1:
        #   drMat(N-1,:) - (l+1)*or1(N-1)*rMat(N-1,:)
        dat_b[N-1, :] = rnorm * (
            drMat[N-1, :] - float(l + 1) * or1[N-1] * rMat[N-1, :]
        )

        # Boundary factor on first/last Chebyshev columns
        dat_b[:, 0] = dat_b[:, 0] * boundary_fac
        dat_b[:, N-1] = dat_b[:, N-1] * boundary_fac

        # Row preconditioning (WITH_PRECOND_BJ)
        fac_b = 1.0 / dat_b.abs().max(dim=1).values
        dat_b = fac_b.unsqueeze(1) * dat_b

        lu_b, ip_b, info_b = prepare_mat(dat_b)
        assert info_b == 0, f"Singular bMat for l={l}, info={info_b}"

        _bMat_lu[l] = lu_b
        _bMat_ip[l] = ip_b
        _bMat_fac[l] = fac_b

        # === jMat (toroidal magnetic field) ===
        dat_j = torch.zeros(N, N, dtype=DTYPE, device=DEVICE)

        # Interior rows: identical to bMat for Boussinesq (dLlambda=0)
        dat_j[1:N-1, :] = rnorm * dL * or2_col[1:N-1] * (
            rMat[1:N-1] - wimp_lin0 * opm * hdif_l * (
                d2rMat[1:N-1] - dL * or2_col[1:N-1] * rMat[1:N-1]
            )
        )

        # CMB BC (row 0): j=0 (insulating, conductance_ma=0)
        dat_j[0, :] = rnorm * rMat[0, :]

        # ICB BC (row N-1): j=0 (insulating)
        dat_j[N-1, :] = rnorm * rMat[N-1, :]

        # Boundary factor
        dat_j[:, 0] = dat_j[:, 0] * boundary_fac
        dat_j[:, N-1] = dat_j[:, N-1] * boundary_fac

        # Row preconditioning
        fac_j = 1.0 / dat_j.abs().max(dim=1).values
        dat_j = fac_j.unsqueeze(1) * dat_j

        lu_j, ip_j, info_j = prepare_mat(dat_j)
        assert info_j == 0, f"Singular jMat for l={l}, info={info_j}"

        _jMat_lu[l] = lu_j
        _jMat_ip[l] = ip_j
        _jMat_fac[l] = fac_j

    # Precompute batched combined inverses (l=0 stays zero → b(l=0)=j(l=0)=0)
    global _b_inv_all, _j_inv_all
    N = n_r_max
    eye = torch.eye(N, dtype=DTYPE, device=DEVICE)
    b_inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=DEVICE)
    j_inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=DEVICE)
    for l in range(1, l_max + 1):
        b_inv_precond = solve_mat_real(_bMat_lu[l], _bMat_ip[l], eye)
        b_inv_by_l[l] = b_inv_precond * _bMat_fac[l].unsqueeze(0)
        j_inv_precond = solve_mat_real(_jMat_lu[l], _jMat_ip[l], eye)
        j_inv_by_l[l] = j_inv_precond * _jMat_fac[l].unsqueeze(0)
    _b_inv_all = b_inv_by_l[st_lm2l]  # (lm_max, N, N) float64 — kept real for fast bmm
    _j_inv_all = j_inv_by_l[st_lm2l]  # (lm_max, N, N) float64


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
            dbdt, djdt, tscheme):
    """Magnetic field: IMEX solve + post-processing.

    - l=0: b=0, j=0 (no monopole)
    - l>=1: separate N×N solves for b (poloidal) and j (toroidal)

    Modifies b, db, ddb, aj, dj, ddj, dbdt, djdt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS for b and j
    rhs_b = tscheme.set_imex_rhs(dbdt)  # (lm_max, n_r_max)
    rhs_j = tscheme.set_imex_rhs(djdt)  # (lm_max, n_r_max)

    # 2. Batched solve: BCs=0, then matmul with precomputed inverse
    # Real bmm: inverses are float64, use view_as_real for ~2.5x speedup
    rhs_b[:, 0] = 0.0
    rhs_b[:, N - 1] = 0.0
    b_cheb = torch.view_as_complex(torch.bmm(_b_inv_all, torch.view_as_real(rhs_b)))
    b_cheb[_m0_mask] = b_cheb[_m0_mask].real.to(CDTYPE)

    rhs_j[:, 0] = 0.0
    rhs_j[:, N - 1] = 0.0
    j_cheb = torch.view_as_complex(torch.bmm(_j_inv_all, torch.view_as_real(rhs_j)))
    j_cheb[_m0_mask] = j_cheb[_m0_mask].real.to(CDTYPE)

    # 3. Convert to physical space
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

    # 6. Store old state (istage=1)
    dbdt.old[:, :, 0] = _dLh_lm * _or2_r * b_LMloc
    djdt.old[:, :, 0] = _dLh_lm * _or2_r * aj_LMloc

    # 7. Compute implicit terms
    # dbdt.impl = opm * hdif_B(l) * dLh * or2 * (ddb - dLh * or2 * b)
    dbdt.impl[:, :, 0] = opm * _hdif_lm * _dLh_lm * _or2_r * (
        ddb_LMloc - _dLh_lm * _or2_r * b_LMloc
    )

    # djdt.impl = opm * hdif_B(l) * dLh * or2 * (ddj - dLh * or2 * j)
    # (dLlambda=0 for Boussinesq, so no dj term)
    djdt.impl[:, :, 0] = opm * _hdif_lm * _dLh_lm * _or2_r * (
        ddj_LMloc - _dLh_lm * _or2_r * aj_LMloc
    )
