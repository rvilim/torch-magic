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
from .chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2
from .horizontal_data import dLh, hdif_Xi
from .pre_calculations import osc
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex
from .cosine_transform import costf
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


# Unique inverse per l degree: (l_max+1, N, N) float64
_xi_inv_by_l = None


def build_xi_matrices(wimp_lin0: float):
    """Build and LU-factorize composition LHS matrices for each l degree.

    Same structure as entropy matrices but with osc instead of opr,
    and hdif_Xi instead of hdif_S.

    Must be called whenever dt changes.
    """
    global _xi_inv_by_l
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

    for l in range(l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_Xi[l].item()

        dat = _rnorm * (_rMat - wimp_lin0 * osc * hdif_l * (
            _d2rMat + two * or1_col * _drMat - dL * or2_col * _rMat
        ))

        dat[0, :] = _rnorm * _rMat[0, :]
        dat[N - 1, :] = _rnorm * _rMat[N - 1, :]

        if n_cheb_max < N:
            dat[0, n_cheb_max:N] = 0.0
            dat[N - 1, n_cheb_max:N] = 0.0

        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        fac = 1.0 / dat.abs().max(dim=1).values
        dat = fac.unsqueeze(1) * dat

        lu, ip, info = prepare_mat(dat)
        assert info == 0, f"Singular xiMat for l={l}, info={info}"

        inv_precond = solve_mat_real(lu, ip, eye)
        inv_by_l[l] = inv_precond * fac.unsqueeze(0)

    _xi_inv_by_l = inv_by_l.to(DEVICE)


def finish_exp_comp(dxi_exp, dVXirLM):
    """Complete explicit composition term by adding radial derivative of dVXirLM.

    dxi_exp = dxi_exp - or2 * d(dVXirLM)/dr

    Structurally identical to finish_exp_entropy.

    Args:
        dxi_exp: (lm_max, n_r_max) complex — partial explicit term from get_dxidt
        dVXirLM: (lm_max, n_r_max) complex — Q-component of v*Xi from SHT

    Returns:
        dxi_exp: (lm_max, n_r_max) complex — completed explicit term
    """
    d_dVXirLM = get_dr(dVXirLM)
    return dxi_exp - _or2_r * d_dVXirLM


def updateXi(xi_LMloc, dxi_LMloc, dxidt, tscheme):
    """Composition equation: IMEX RHS assembly, implicit solve, post-processing.

    Structurally identical to updateS but with different coefficients and BCs.

    Modifies xi_LMloc, dxi_LMloc, dxidt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS
    rhs = tscheme.set_imex_rhs(dxidt)  # (lm_max, n_r_max)

    # 2. Batched solve: set BCs, then chunked matmul with per-l inverses
    rhs[:, 0] = _topxi
    rhs[:, N - 1] = _botxi
    xi_cheb = chunked_solve_complex(_xi_inv_by_l, st_lm2l, rhs)
    xi_cheb[_m0_mask] = xi_cheb[_m0_mask].real.to(CDTYPE)

    # 3. Truncate high Chebyshev modes
    if n_cheb_max < N:
        xi_cheb[:, n_cheb_max:] = 0.0

    # 4. Convert Chebyshev coefficients to physical space
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
    # impl = osc * hdif_Xi(l) * (d2xi + 2*or1*dxi - l(l+1)*or2*xi)
    idx = tscheme.next_impl_idx
    dxidt.impl[:, :, idx] = osc * _hdif_lm * (
        d2xi + two * _or1_r * dxi_LMloc - _dLh_lm * _or2_r * xi_LMloc
    )
