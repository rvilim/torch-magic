"""Implicit poloidal velocity + pressure solver matching updateWP.f90.

Implements:
- build_p0_matrix: l=0 pressure matrix (dp/dr equation)
- build_wp_matrices: 2N×2N coupled (w, p) matrices for l>=1
- finish_exp_pol: Complete explicit poloidal term (radial derivative)
- updateWP: Full IMEX solve + post-processing

Boussinesq: visc=1, beta=0, dLvisc=0, dbeta=0, rho0=1, orho1=1.
Anelastic: variable profiles, stress-free BCs via ktopv/kbotv.
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max, n_cheb_max, l_anel, ktopv, kbotv, l_finite_diff
from .constants import two, three, third, four
from .radial_scheme import rMat, drMat, d2rMat, d3rMat, rnorm, boundary_fac
from .radial_functions import (
    or1, or2, or3, rgrav, rho0,
    beta, dbeta, ddbeta, visc, dLvisc, ddLvisc, orho1,
    ViscHeatFac, ThExpNb,
)
from .horizontal_data import dLh, hdif_V
from .pre_calculations import BuoFac, ChemFac

# Flag: use integral BC for p0 (anelastic with viscous heating)
# Fortran uses integral BC only for Chebyshev; FD falls back to Dirichlet
# (updateWP.f90 get_p0Mat lines 2592-2607: FD sets dat(1,:)=rMat(1,:))
from .params import l_finite_diff as _l_fd_p0
_l_p0_integ_bc = (ViscHeatFac * ThExpNb != 0.0) and not _l_fd_p0
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex, chunked_lu_solve_complex
from .radial_scheme import costf
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

# Anelastic broadcast arrays for implicit terms
_rho0_r = rho0.unsqueeze(0).to(CDTYPE)                     # (1, n_r_max)
_visc_r = visc.unsqueeze(0).to(CDTYPE)                     # (1, n_r_max)
_orho1_r = orho1.unsqueeze(0).to(CDTYPE)                   # (1, n_r_max)
_beta_r = beta.unsqueeze(0).to(CDTYPE)                     # (1, n_r_max)
_dbeta_r = dbeta.unsqueeze(0).to(CDTYPE)                   # (1, n_r_max)
_dLvisc_r = dLvisc.unsqueeze(0).to(CDTYPE)                 # (1, n_r_max)
_ddLvisc_r = ddLvisc.unsqueeze(0).to(CDTYPE)               # (1, n_r_max)
_ddbeta_r = ddbeta.unsqueeze(0).to(CDTYPE)                 # (1, n_r_max)

# Precomputed combined D2+D3 derivative matrix for batched matmul
from .radial_derivatives import _D2 as _D2_cd_wp, _D3 as _D3_cd_wp
_D23_T_wp = torch.cat([_D2_cd_wp.T, _D3_cd_wp.T], dim=1)  # (N, 2N)

_lm_l0 = st_lm2[0, 0].item()

# --- l=0 pressure matrix ---
_p0Mat_lu = None
_p0Mat_ip = None
_p0Mat_inv = None  # (N, N) float64 on DEVICE — precomputed inverse for GPU solve
_p0Mat_bands = None  # (dl, d, du) tridiagonal bands for FD — on DEVICE

# --- l>=1 coupled (w,p) matrices ---
_wpMat_lu = [None] * (l_max + 1)
_wpMat_ip = [None] * (l_max + 1)
_wpMat_fac_row = [None] * (l_max + 1)  # row preconditioning
_wpMat_fac_col = [None] * (l_max + 1)  # column preconditioning

# Unique inverse per l degree: (l_max+1, 2N, 2N) float64 — l=0 is zero
_wp_inv_by_l = None
# LU factors for anelastic (accurate solve for ill-conditioned wpMat)
_wp_lu_by_l = None
_wp_pivots_by_l = None
_wp_fac_row_by_l = None
_wp_fac_col_by_l = None



def build_p0_matrix():
    """Build and LU-factorize the l=0 pressure matrix.

    Rows 1..N-1: rnorm * (drMat - beta*rMat) (pressure gradient equation)
    Row 0: Dirichlet p=0 (Boussinesq) or Chebyshev integral constraint (anelastic
           with ViscHeatFac*ThExpNb != 0).
    """
    global _p0Mat_lu, _p0Mat_ip, _p0Mat_inv, _p0Mat_bands
    N = n_r_max

    # Build on CPU (scalar loops in prepare_mat)
    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _beta = beta.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    dat = torch.zeros(N, N, dtype=DTYPE, device=cpu)
    # Bulk: dp/dr - beta*p = 0  (anelastic mass conservation)
    dat[1:, :] = _rnorm * (_drMat[1:, :] - _beta[1:].unsqueeze(1) * _rMat[1:, :])

    # FD override: last row uses first-order backward difference.
    # Fortran get_p0Mat only overrides columns N-1 and N-2. The drMat
    # boundary stencil at other columns survives in the dense matrix.
    # But Fortran then extracts into type_bandmat with n_bands=order+1,
    # which silently drops entries outside the band. We must match this
    # by zeroing entries that fall outside the Fortran band.
    if l_finite_diff:
        from .radial_scheme import r as _r
        from .params import fd_order as _fd_order_p0
        _r_cpu = _r.to(cpu)
        delr = _r_cpu[N - 1] - _r_cpu[N - 2]
        # Override the two first-order entries
        dat[N - 1, N - 1] = 1.0 / delr - _beta[N - 1]
        dat[N - 1, N - 2] = -1.0 / delr
        # Zero entries outside the Fortran band (n_bands = order+1, kl = order//2)
        p0_kl = _fd_order_p0 // 2
        dat[N - 1, :N - 1 - p0_kl] = 0.0

    # Row 0 boundary condition
    if _l_p0_integ_bc:
        # Chebyshev integral constraint: ∫ ThExpNb*ViscHeatFac*ogrun*alpha0*r²*p dr = const
        # (updateWP.f90 get_p0Mat lines 2565-2585)
        from .init_fields import _cheb_integ_kernel_f64
        from .radial_functions import ViscHeatFac, ThExpNb, ogrun, alpha0
        from .radial_scheme import r
        work_phys = ThExpNb * ViscHeatFac * ogrun * alpha0 * r * r
        work = costf(work_phys.to(cpu))
        work = work * _rnorm
        work[0] = _bfac * work[0]
        work[N - 1] = _bfac * work[N - 1]
        K = _cheb_integ_kernel_f64(N).to(work.dtype)
        dat[0, :] = (K @ work).to(DTYPE)
    else:
        dat[0, :] = _rnorm * _rMat[0, :]

    # Truncate high Chebyshev modes in BC row
    if n_cheb_max < N:
        dat[0, n_cheb_max:] = 0.0

    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N - 1] = dat[:, N - 1] * _bfac

    if l_finite_diff:
        # FD: match Fortran's solver exactly.
        # Fortran type_bandmat with n_bands=3 uses prepare_tridiag/solve_tridiag.
        # Fortran type_bandmat with n_bands>3 uses prepare_band/solve_band.
        from .params import fd_order as _fd_order_p0_build
        p0_kl = _fd_order_p0_build // 2
        p0_ku = p0_kl
        n_bands_p0 = _fd_order_p0_build + 1
        if n_bands_p0 == 3:
            # Tridiagonal: use prepare_tridiag (matches Fortran exactly)
            from .algebra import prepare_tridiag, extract_tridiag
            dl, d, du = extract_tridiag(dat)
            dl, d, du, du2, pivot, info = prepare_tridiag(dl, d, du)
            assert info == 0, f"Singular p0Mat (tridiag), info={info}"
            _p0Mat_bands = ('tridiag', dl, d, du, du2, pivot)
        else:
            # Wider band: use prepare_band (matches Fortran exactly)
            from .algebra import dense_to_band_storage, prepare_band
            abd = dense_to_band_storage(dat, p0_kl, p0_ku)
            abd_f, piv, info = prepare_band(abd, N, p0_kl, p0_ku)
            assert info == 0, f"Singular p0Mat (banded), info={info}"
            _p0Mat_bands = ('band', abd_f, piv, N, p0_kl, p0_ku)
        _p0Mat_inv = None
        _p0Mat_lu = None
        _p0Mat_ip = None
    else:
        # Chebyshev: dense LU + precomputed inverse
        lu, ip, info = prepare_mat(dat)
        assert info == 0, "Singular p0Mat"
        _p0Mat_lu = lu
        _p0Mat_ip = ip
        eye = torch.eye(N, dtype=DTYPE, device=cpu)
        inv_cols = []
        for i in range(N):
            inv_cols.append(solve_mat_real(lu, ip, eye[:, i]))
        _p0Mat_inv = torch.stack(inv_cols, dim=1).to(device=DEVICE)
        _p0Mat_bands = None


def solve_p0(p0_rhs: torch.Tensor) -> torch.Tensor:
    """Solve the l=0 pressure system.

    FD: banded LU solve (matching Fortran's tridiagonal solver).
    Chebyshev: precomputed dense inverse.

    Args:
        p0_rhs: (N,) real RHS

    Returns:
        (N,) real solution
    """
    if _p0Mat_bands is not None:
        if _p0Mat_bands[0] == 'tridiag':
            from .algebra import solve_tridiag_real
            _, dl, d, du, du2, pivot = _p0Mat_bands
            return solve_tridiag_real(dl, d, du, du2, pivot, p0_rhs)
        else:
            from .algebra import solve_band_real
            _, abd_f, piv, n, kl, ku = _p0Mat_bands
            return solve_band_real(abd_f, n, kl, ku, piv, p0_rhs)
    return _p0Mat_inv @ p0_rhs


def build_wp_matrices(wimp_lin0: float):
    """Build and LU-factorize 2N×2N coupled (w,p) matrices for l>=1.

    Layout: rows/cols 0..N-1 = w block, N..2N-1 = p block.

    Full anelastic terms from updateWP.f90 get_wpMat.
    Boussinesq: visc=1, beta=0, dLvisc=0, dbeta=0 → simplifies.

    Must be called whenever dt changes.
    """
    global _wp_inv_by_l, _wp_lu_by_l, _wp_pivots_by_l, _wp_fac_row_by_l, _wp_fac_col_by_l
    N = n_r_max

    # Build on CPU (scalar loops in prepare_mat/solve_mat_real)
    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _d3rMat = d3rMat.to(cpu)
    _or1 = or1.to(cpu)
    _or2 = or2.to(cpu)
    _or3 = or3.to(cpu)
    _beta = beta.to(cpu)
    _dbeta = dbeta.to(cpu)
    _visc = visc.to(cpu)
    _dLvisc = dLvisc.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac
    or1_col = _or1.unsqueeze(1)
    or2_col = _or2.unsqueeze(1)
    or3_col = _or3.unsqueeze(1)
    beta_col = _beta.unsqueeze(1)
    dbeta_col = _dbeta.unsqueeze(1)
    visc_col = _visc.unsqueeze(1)
    dLvisc_col = _dLvisc.unsqueeze(1)

    eye = torch.eye(2 * N, dtype=DTYPE, device=cpu)
    inv_by_l = torch.zeros(l_max + 1, 2 * N, 2 * N, dtype=DTYPE, device=cpu)
    # For anelastic: store LU factors for accurate solve (precomputed inverse loses
    # digits when cond(wpMat) ~ 1e12 due to variable density/viscosity profiles)
    if l_anel:
        lu_factors = torch.eye(2 * N, dtype=DTYPE, device=cpu).unsqueeze(0).expand(l_max + 1, -1, -1).clone()
        pivots_all = torch.arange(1, 2 * N + 1, dtype=torch.int32, device=cpu).unsqueeze(0).expand(l_max + 1, -1).clone()
        fac_row_all = torch.ones(l_max + 1, 2 * N, dtype=DTYPE, device=cpu)
        fac_col_all = torch.ones(l_max + 1, 2 * N, dtype=DTYPE, device=cpu)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        dat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=cpu)

        # w-row, w-col (Fortran lines 2064-2072):
        # dLh*or2*rMat - wimp*hdif*visc*dLh*or2*(
        #   d2rMat + (2*dLvisc - beta/3)*drMat
        #   - (dLh*or2 + 4/3*(dLvisc*beta + (3*dLvisc+beta)*or1 + dbeta))*rMat
        # )
        dat[1:N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            _rMat[1:N-1] - wimp_lin0 * hdif_l * visc_col[1:N-1] * (
                _d2rMat[1:N-1]
                + (two * dLvisc_col[1:N-1] - third * beta_col[1:N-1])
                * _drMat[1:N-1]
                - (dL * or2_col[1:N-1]
                   + four * third * (dLvisc_col[1:N-1] * beta_col[1:N-1]
                                     + (three * dLvisc_col[1:N-1] + beta_col[1:N-1])
                                     * or1_col[1:N-1]
                                     + dbeta_col[1:N-1]))
                * _rMat[1:N-1]
            )
        )

        # w-row, p-col (Fortran lines 2074-2076):
        # wimp*(drMat - beta*rMat)
        dat[1:N-1, N:] = _rnorm * wimp_lin0 * (
            _drMat[1:N-1] - beta_col[1:N-1] * _rMat[1:N-1]
        )

        # p-row, w-col (Fortran lines 2079-2088):
        # -dLh*or2*drMat - wimp*hdif*visc*dLh*or2*(
        #   -d3rMat + (beta-dLvisc)*d2rMat
        #   + (dLh*or2 + dbeta + dLvisc*beta + 2*(dLvisc+beta)*or1)*drMat
        #   - dLh*or2*(2*or1 + dLvisc + 2/3*beta)*rMat
        # )
        dat[N+1:2*N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            -_drMat[1:N-1] - wimp_lin0 * hdif_l * visc_col[1:N-1] * (
                -_d3rMat[1:N-1]
                + (beta_col[1:N-1] - dLvisc_col[1:N-1]) * _d2rMat[1:N-1]
                + (dL * or2_col[1:N-1] + dbeta_col[1:N-1]
                   + dLvisc_col[1:N-1] * beta_col[1:N-1]
                   + two * (dLvisc_col[1:N-1] + beta_col[1:N-1]) * or1_col[1:N-1])
                * _drMat[1:N-1]
                - dL * or2_col[1:N-1]
                * (two * or1_col[1:N-1] + dLvisc_col[1:N-1]
                   + two * third * beta_col[1:N-1])
                * _rMat[1:N-1]
            )
        )

        # p-row, p-col (Fortran lines 2090-2091):
        # -wimp*dLh*or2*rMat
        dat[N+1:2*N-1, N:] = -_rnorm * wimp_lin0 * dL * or2_col[1:N-1] * _rMat[1:N-1]

        # --- Boundary conditions ---
        # w=0 at CMB and ICB (rows 0 and N-1)
        dat[0, :N] = _rnorm * _rMat[0, :]
        dat[N-1, :N] = _rnorm * _rMat[N-1, :]

        # dw/dr BC at CMB (row N): stress-free or no-slip
        if ktopv == 1:  # stress-free
            dat[N, :N] = _rnorm * (
                _d2rMat[0, :] - (two * _or1[0] + _beta[0]) * _drMat[0, :]
            )
        else:  # no-slip
            dat[N, :N] = _rnorm * _drMat[0, :]

        # dw/dr BC at ICB (row 2N-1): stress-free or no-slip
        if kbotv == 1:  # stress-free
            dat[2*N-1, :N] = _rnorm * (
                _d2rMat[N-1, :] - (two * _or1[N-1] + _beta[N-1]) * _drMat[N-1, :]
            )
        else:  # no-slip
            dat[2*N-1, :N] = _rnorm * _drMat[N-1, :]

        # Zero high OC Chebyshev modes in boundary rows (both W and P blocks)
        if n_cheb_max < N:
            for row in [0, N-1, N, 2*N-1]:
                dat[row, n_cheb_max:N] = 0.0          # W block cols
                dat[row, N+n_cheb_max:2*N] = 0.0      # P block cols

        for blk_start in [0, N]:
            dat[:, blk_start] = dat[:, blk_start] * _bfac
            dat[:, blk_start + N - 1] = dat[:, blk_start + N - 1] * _bfac

        row_max = dat.abs().max(dim=1).values
        row_max[row_max == 0] = 1.0  # protect zero rows from inf
        fac_row = 1.0 / row_max
        dat = fac_row.unsqueeze(1) * dat
        col_max = dat.abs().max(dim=0).values
        col_max[col_max == 0] = 1.0  # protect zero cols from inf
        fac_col = 1.0 / col_max
        dat = dat * fac_col.unsqueeze(0)

        if l_anel:
            # Store LU factors for accurate batched lu_solve
            lu_f, piv = torch.linalg.lu_factor(dat)
            lu_factors[l] = lu_f
            pivots_all[l] = piv
            fac_row_all[l] = fac_row
            fac_col_all[l] = fac_col
        else:
            lu, ip, info = prepare_mat(dat)
            assert info == 0, f"Singular wpMat for l={l}, info={info}"
            inv_precond = solve_mat_real(lu, ip, eye)
            inv_by_l[l] = fac_col.unsqueeze(1) * inv_precond * fac_row.unsqueeze(0)

    if l_anel:
        _wp_lu_by_l = lu_factors.to(DEVICE)
        _wp_pivots_by_l = pivots_all.to(DEVICE)
        _wp_fac_row_by_l = fac_row_all.to(DEVICE)
        _wp_fac_col_by_l = fac_col_all.to(DEVICE)
        _wp_inv_by_l = None
    else:
        _wp_inv_by_l = inv_by_l.to(DEVICE)
        _wp_lu_by_l = None
        _wp_pivots_by_l = None
        _wp_fac_row_by_l = None
        _wp_fac_col_by_l = None


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
             dwdt, p_LMloc, dp_LMloc, dpdt, tscheme,
             xi_LMloc=None):
    """Poloidal velocity + pressure: IMEX solve + post-processing.

    - l=0: w=0, pressure from buoyancy + explicit terms via p0Mat
    - l>=1: coupled 2N×2N system for (w, p)
    - Buoyancy coupling: wimp * BuoFac * rgrav * s + wimp * ChemFac * rgrav * xi

    Modifies w, dw, ddw, dwdt, p, dp, dpdt in place.
    """
    N = n_r_max

    # 1. Assemble IMEX RHS for w and p (batched: 1 set of ops instead of 2)
    rhs_wp = tscheme.set_imex_rhs_multi([dwdt, dpdt])  # (2*lm_max, n_r_max)
    rhs_w = rhs_wp[:lm_max]
    rhs_p = rhs_wp[lm_max:]

    # === l=0: compute p0 RHS (solve after batched solve) ===
    lm0 = _lm_l0
    p0_rhs = torch.zeros(N, dtype=DTYPE, device=DEVICE)
    # expl index: CNAB2 istage=0→0, BPR353 istage=1..4→0..3
    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0
    p0_rhs[1:] = (rho0[1:] * BuoFac * rgrav[1:] * s_LMloc[lm0, 1:].real
                  + dwdt.expl[lm0, 1:, expl_idx].real)
    if xi_LMloc is not None:
        p0_rhs[1:] += rho0[1:] * ChemFac * rgrav[1:] * xi_LMloc[lm0, 1:].real

    # Row 0: integral BC or Dirichlet
    if _l_p0_integ_bc:
        from .radial_functions import alpha0, temp0
        from .radial_scheme import r
        from .integration import rInt_R
        work = ThExpNb * alpha0 * temp0 * rho0 * r * r * s_LMloc[lm0].real
        p0_rhs[0] = rInt_R(work)

    # === l>=1: batched coupled (w, p) solve ===
    wimp_lin0 = tscheme.wimp_lin[0]

    # Build combined RHS: (lm_max, 2N) — w in rows 0..N-1, p in rows N..2N-1
    rhs_combined = torch.zeros(lm_max, 2 * N, dtype=CDTYPE, device=DEVICE)

    # w rows (interior): IMEX RHS + implicit buoyancy coupling
    # Non-double-curl: wimp * rho0 * BuoFac * rgrav * s (Fortran lines 538-545)
    rhs_combined[:, 1:N-1] = rhs_w[:, 1:N-1]
    rho0_rgrav_int = (rho0[1:N-1] * rgrav[1:N-1]).unsqueeze(0).to(CDTYPE)
    rhs_combined[:, 1:N-1] += wimp_lin0 * BuoFac * rho0_rgrav_int * s_LMloc[:, 1:N-1]
    if xi_LMloc is not None:
        rhs_combined[:, 1:N-1] += wimp_lin0 * ChemFac * rho0_rgrav_int * xi_LMloc[:, 1:N-1]

    # p rows (interior): IMEX RHS for pressure
    rhs_combined[:, N+1:2*N-1] = rhs_p[:, 1:N-1]

    # BCs: all zero (w=0, dw/dr=0 at both boundaries) — already zero
    # l=0 modes get zero solution (inv is zero for l=0)

    # Chunked batched solve: LU for anelastic (ill-conditioned), inverse for Boussinesq
    if _wp_lu_by_l is not None:
        sol = chunked_lu_solve_complex(
            _wp_lu_by_l, _wp_pivots_by_l,
            _wp_fac_row_by_l, _wp_fac_col_by_l,
            st_lm2l, rhs_combined)
        # l=0 has no WP solve (dLh=0); precomputed-inverse path gives zero via
        # zero inv_by_l[0], but LU path returns RHS unchanged. Zero explicitly.
        sol[st_lm2l == 0] = 0.0
    else:
        sol = chunked_solve_complex(_wp_inv_by_l, st_lm2l, rhs_combined)
    sol[_m0_mask] = sol[_m0_mask].real.to(CDTYPE)

    # Extract w and p Chebyshev coefficients
    w_cheb = sol[:, :N]
    p_cheb = sol[:, N:]

    # l=0 pressure from p0Mat (GPU matmul, no CPU↔GPU sync)
    p_cheb[lm0, :] = solve_p0(p0_rhs).to(CDTYPE)

    # 3. Truncate high Chebyshev modes (Fortran only stores n_cheb_max modes)
    if n_cheb_max < N:
        w_cheb[:, n_cheb_max:] = 0.0
        p_cheb[:, n_cheb_max:] = 0.0

    # 4. Convert to physical space (batched: 1 FFT instead of 2)
    wp_cheb = torch.cat([w_cheb, p_cheb], dim=0)
    wp_phys = costf(wp_cheb)
    w_LMloc[:] = wp_phys[:lm_max]
    p_LMloc[:] = wp_phys[lm_max:]

    # 4b. Compute derivatives (batch D1 for w+p; combined D2+D3 for w)
    from .radial_derivatives import _D1 as _D1_cd, _D2 as _D2_cd, _D3 as _D3_cd
    d1_wp = wp_phys @ _D1_cd.T  # (2*lm_max, N) — 1 matmul instead of 2
    dw_LMloc[:] = d1_wp[:lm_max]
    dp_LMloc[:] = d1_wp[lm_max:]
    # Combined D2+D3 in one matmul (saves 1 dispatch)
    dd_w = w_LMloc @ _D23_T_wp  # (lm_max, 2N)
    ddw_LMloc[:] = dd_w[:, :n_r_max]
    dddw = dd_w[:, n_r_max:]

    # 5. Rotate IMEX time arrays
    tscheme.rotate_imex(dwdt)
    tscheme.rotate_imex(dpdt)

    # 6. Store old state (always for CNAB2, only at wrap for DIRK)
    if tscheme.store_old:
        dwdt.old[:, :, 0] = _dLh_lm * _or2_r * w_LMloc
        dpdt.old[:, :, 0] = -_dLh_lm * _or2_r * dw_LMloc

    # 7. Compute implicit terms (interior points, l>0; l=0 contributes zero via dLh=0)
    # Non-double-curl formulation (updateWP.f90 lines 1310-1333)
    idx = tscheme.next_impl_idx

    # Dif = hdif*dL*or2*visc*(ddw + (2*dLvisc - beta/3)*dw
    #     - (dL*or2 + 4/3*(dbeta + dLvisc*beta + (3*dLvisc+beta)*or1))*w)
    Dif_w = _hdif_lm * _dLh_lm * _or2_r * _visc_r * (
        ddw_LMloc
        + (two * _dLvisc_r - third * _beta_r) * dw_LMloc
        - (_dLh_lm * _or2_r
           + four * third * (_dbeta_r + _dLvisc_r * _beta_r
                             + (three * _dLvisc_r + _beta_r) * _or1_r))
        * w_LMloc
    )
    # Pre = -dp + beta*p
    Pre = -dp_LMloc + _beta_r * p_LMloc
    # Buo = BuoFac*rho0*rgrav*s + ChemFac*rho0*rgrav*xi
    Buo = BuoFac * _rho0_r * _rgrav_r * s_LMloc
    if xi_LMloc is not None:
        Buo = Buo + ChemFac * _rho0_r * _rgrav_r * xi_LMloc
    dwdt.impl[:, :, idx] = Pre + Dif_w + Buo

    # dpdt.impl = dL*or2*p + hdif*visc*dL*or2*(
    #   -dddw + (beta-dLvisc)*ddw
    #   + (dL*or2 + dLvisc*beta + dbeta + 2*(dLvisc+beta)*or1)*dw
    #   - dL*or2*(2*or1 + 2/3*beta + dLvisc)*w
    # )
    dpdt.impl[:, :, idx] = (
        _dLh_lm * _or2_r * p_LMloc
        + _hdif_lm * _visc_r * _dLh_lm * _or2_r * (
            -dddw
            + (_beta_r - _dLvisc_r) * ddw_LMloc
            + (_dLh_lm * _or2_r + _dLvisc_r * _beta_r + _dbeta_r
               + two * (_dLvisc_r + _beta_r) * _or1_r) * dw_LMloc
            - _dLh_lm * _or2_r
            * (two * _or1_r + two * third * _beta_r + _dLvisc_r)
            * w_LMloc
        )
    )
