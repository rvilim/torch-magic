"""Implicit toroidal velocity solver matching updateZ.f90.

Implements:
- build_z_matrices: LHS matrix construction + LU factorization per l-degree
- updateZ: Full IMEX solve + post-processing

Boussinesq: visc=1, beta=0, dLvisc=0, dbeta=0.
No inner core/mantle rotation (nRotIC=0, nRotMA=0).
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .constants import two
from .chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
from .radial_functions import or1, or2
from .horizontal_data import dLh, hdif_V
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real
from .cosine_transform import costf
from .radial_derivatives import get_ddr


# --- Precompute per-l LM index groups ---
_l_lm_idx = []
for _l in range(l_max + 1):
    _l_lm_idx.append(torch.where(st_lm2l == _l)[0])

# m=0 mask for forcing real coefficients
_m0_mask = (st_lm2m == 0)

# Precompute broadcast arrays for implicit term
_hdif_lm = hdif_V[st_lm2l].to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)                # (lm_max, 1)
_or2_r = or2.unsqueeze(0)                             # (1, n_r_max)

# l=0 index (spherically symmetric toroidal is forced to zero)
_lm_l0 = st_lm2[0, 0].item()

# --- LU-factored matrices storage (one per l degree, l>=1) ---
_zMat_lu = [None] * (l_max + 1)
_zMat_ip = [None] * (l_max + 1)
_zMat_fac = [None] * (l_max + 1)

# Batched combined inverse: (lm_max, N, N) complex — l=0 rows are zero
_z_inv_all = None


def build_z_matrices(wimp_lin0: float):
    """Build and LU-factorize toroidal velocity LHS matrices for l >= 1.

    Matrix in Chebyshev space:
        dat = rnorm * dLh * or2 * (rMat - wimp * hdif * (d2rMat - dLh * or2 * rMat))
    with no-slip BCs (Dirichlet z=0) at rows 0 and N-1.

    Must be called whenever dt changes.
    """
    global _z_inv_all
    N = n_r_max
    or2_col = or2.unsqueeze(1)  # (N, 1)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        dat = rnorm * dL * or2_col * (
            rMat - wimp_lin0 * hdif_l * (
                d2rMat - dL * or2_col * rMat
            )
        )

        dat[0, :] = rnorm * rMat[0, :]
        dat[N - 1, :] = rnorm * rMat[N - 1, :]

        dat[:, 0] = dat[:, 0] * boundary_fac
        dat[:, N - 1] = dat[:, N - 1] * boundary_fac

        fac = 1.0 / dat.abs().max(dim=1).values
        dat = fac.unsqueeze(1) * dat

        lu, ip, info = prepare_mat(dat)
        assert info == 0, f"Singular zMat for l={l}, info={info}"

        _zMat_lu[l] = lu
        _zMat_ip[l] = ip
        _zMat_fac[l] = fac

    # Precompute batched combined inverse (l=0 stays zero → z(l=0)=0)
    eye = torch.eye(N, dtype=DTYPE, device=DEVICE)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=DEVICE)
    for l in range(1, l_max + 1):
        inv_precond = solve_mat_real(_zMat_lu[l], _zMat_ip[l], eye)
        inv_by_l[l] = inv_precond * _zMat_fac[l].unsqueeze(0)
    _z_inv_all = inv_by_l[st_lm2l]  # (lm_max, N, N) float64 — kept real for fast bmm


def updateZ(z_LMloc, dz_LMloc, dzdt, tscheme):
    """Toroidal velocity: IMEX RHS assembly, implicit solve, post-processing.

    - l=0 mode is forced to zero (no spherically symmetric toroidal component)
    - l>=1 modes solved with no-slip BCs (z=0 at boundaries)
    - No inner core/mantle rotation for the benchmark

    Modifies z_LMloc, dz_LMloc, dzdt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS
    rhs = tscheme.set_imex_rhs(dzdt)  # (lm_max, n_r_max)

    # 2. Batched solve: BCs=0, then matmul with precomputed inverse
    # Real bmm: inverse is float64, use view_as_real for ~2.5x speedup
    rhs[:, 0] = 0.0
    rhs[:, N - 1] = 0.0
    z_cheb = torch.view_as_complex(torch.bmm(_z_inv_all, torch.view_as_real(rhs)))
    z_cheb[_m0_mask] = z_cheb[_m0_mask].real.to(CDTYPE)

    # 3. Convert to physical space
    z_LMloc[:] = costf(z_cheb)

    # 4. Compute derivatives
    dz_new, d2z = get_ddr(z_LMloc)
    dz_LMloc[:] = dz_new

    # 5. Rotate IMEX time arrays
    tscheme.rotate_imex(dzdt)

    # 6. Store old state: dLh * or2 * z (matches Fortran updateZ line 926)
    dzdt.old[:, :, 0] = _dLh_lm * _or2_r * z_LMloc

    # 7. Compute implicit diffusion term
    # impl = hdif_V(l) * l(l+1) * or2 * (d2z - l(l+1) * or2 * z)
    dzdt.impl[:, :, 0] = _hdif_lm * _dLh_lm * _or2_r * (
        d2z - _dLh_lm * _or2_r * z_LMloc
    )
