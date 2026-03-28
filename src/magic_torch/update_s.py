"""Implicit entropy solver matching updateS.f90.

Implements:
- build_s_matrices: LHS matrix construction + LU factorization per l-degree
- finish_exp_entropy: Complete explicit nonlinear term (radial derivative)
- updateS: Full IMEX solve + post-processing

Boussinesq specialization: kappa=1, beta=0, dLtemp0=0, dLkappa=0, orho1=1.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .constants import sq4pi, two
from .chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2
from .horizontal_data import dLh, hdif_S
from .pre_calculations import opr
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real
from .cosine_transform import costf
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


# --- LU-factored matrices storage (one per l degree) ---
_sMat_lu = [None] * (l_max + 1)
_sMat_ip = [None] * (l_max + 1)
_sMat_fac = [None] * (l_max + 1)

# Batched combined inverse: (lm_max, N, N) complex — precomputed at build time
_s_inv_all = None


def build_s_matrices(wimp_lin0: float):
    """Build and LU-factorize entropy LHS matrices for each l degree.

    Matrix in Chebyshev space:
        dat = rnorm * (rMat - wimp * opr * hdif * (d2rMat + 2*or1*drMat - dLh*or2*rMat))
    with Dirichlet BCs at rows 0 and N-1, boundary_fac on columns 0 and N-1,
    and row preconditioning (WITH_PRECOND_S).

    Must be called whenever dt changes.
    """
    global _s_inv_all
    N = n_r_max
    or1_col = or1.unsqueeze(1)  # (N, 1) for row-dependent broadcasting
    or2_col = or2.unsqueeze(1)

    for l in range(l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_S[l].item()

        # Interior: identity minus implicit diffusion operator
        dat = rnorm * (rMat - wimp_lin0 * opr * hdif_l * (
            d2rMat + two * or1_col * drMat - dL * or2_col * rMat
        ))

        # Dirichlet BCs (ktops=1, kbots=1)
        dat[0, :] = rnorm * rMat[0, :]
        dat[N - 1, :] = rnorm * rMat[N - 1, :]

        # Chebyshev boundary factor on first/last columns
        dat[:, 0] = dat[:, 0] * boundary_fac
        dat[:, N - 1] = dat[:, N - 1] * boundary_fac

        # Row preconditioning (WITH_PRECOND_S)
        fac = 1.0 / dat.abs().max(dim=1).values
        dat = fac.unsqueeze(1) * dat

        # LU factorize
        lu, ip, info = prepare_mat(dat)
        assert info == 0, f"Singular sMat for l={l}, info={info}"

        _sMat_lu[l] = lu
        _sMat_ip[l] = ip
        _sMat_fac[l] = fac

    # Precompute batched combined inverse: inv(precondA) @ diag(fac) for each l,
    # then expand to all lm modes via st_lm2l indexing.
    eye = torch.eye(N, dtype=DTYPE, device=DEVICE)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=DEVICE)
    for l in range(l_max + 1):
        inv_precond = solve_mat_real(_sMat_lu[l], _sMat_ip[l], eye)
        inv_by_l[l] = inv_precond * _sMat_fac[l].unsqueeze(0)
    _s_inv_all = inv_by_l[st_lm2l]  # (lm_max, N, N) float64 — kept real for fast bmm


def finish_exp_entropy(ds_exp, dVSrLM):
    """Complete explicit entropy term by adding radial derivative of dVSrLM.

    ds_exp = ds_exp - or2 * d(dVSrLM)/dr

    Boussinesq: orho1=1, dentropy0=0 (no background entropy gradient).

    Args:
        ds_exp: (lm_max, n_r_max) complex — partial explicit term from get_dsdt
        dVSrLM: (lm_max, n_r_max) complex — Q-component of v*S from SHT

    Returns:
        ds_exp: (lm_max, n_r_max) complex — completed explicit term
    """
    d_dVSrLM = get_dr(dVSrLM)
    return ds_exp - _or2_r * d_dVSrLM


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

    # 2. Batched solve: set BCs, then matmul with precomputed inverse
    # Real bmm: inverse is float64, use view_as_real for ~2.5x speedup
    rhs[:, 0] = _tops
    rhs[:, N - 1] = _bots
    s_cheb = torch.view_as_complex(torch.bmm(_s_inv_all, torch.view_as_real(rhs)))
    s_cheb[_m0_mask] = s_cheb[_m0_mask].real.to(CDTYPE)

    # 3. Convert Chebyshev coefficients to physical space
    s_LMloc[:] = costf(s_cheb)

    # 4. Compute radial derivatives in physical space
    ds_new, d2s = get_ddr(s_LMloc)
    ds_LMloc[:] = ds_new

    # 5. Rotate IMEX time arrays (shift explicit history)
    tscheme.rotate_imex(dsdt)

    # 6. Store current state as old (istage=1)
    dsdt.old[:, :, 0] = s_LMloc.clone()

    # 7. Compute implicit diffusion term for next IMEX stage
    # impl = opr * hdif_S(l) * (d2s + 2*or1*ds - l(l+1)*or2*s)
    dsdt.impl[:, :, 0] = opr * _hdif_lm * (
        d2s + two * _or1_r * ds_LMloc - _dLh_lm * _or2_r * s_LMloc
    )
