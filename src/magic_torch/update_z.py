"""Implicit toroidal velocity solver matching updateZ.f90.

Implements:
- build_z_matrices: LHS matrix construction + LU factorization per l-degree
- updateZ: Full IMEX solve + post-processing

Handles both Boussinesq (visc=1, beta=0) and anelastic (variable profiles).
Supports stress-free BCs (ktopv=1, kbotv=1) and inner core rotation.
"""

import torch

import math

from .precision import DTYPE, CDTYPE, DEVICE
from .params import (n_r_max, lm_max, l_max, l_rot_ic, n_cheb_max, ktopv, kbotv,
                     l_anel, l_correct_AMz, l_correct_AMe)
from .constants import two, third, four
from .radial_scheme import rMat, drMat, d2rMat, rnorm, boundary_fac, r
from .radial_functions import or1, or2, visc, beta, dbeta, dLvisc, rho0
from .horizontal_data import dLh, hdif_V
from .blocking import st_lm2, st_lm2l, st_lm2m
from .algebra import prepare_mat, solve_mat_complex, solve_mat_real, chunked_solve_complex
from .radial_scheme import costf
from .radial_derivatives import get_ddr
from .integration import rInt_R
from .pre_calculations import (l_z10mat, c_z10_omega_ic, c_dt_z10_ic,
                                c_lorentz_ic, gammatau_gravi,
                                y10_norm, y11_norm, c_moi_oc, c_moi_ic,
                                c_moi_ma, AMstart)


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

# Anelastic coefficients for implicit term (updateZ.f90 line 965-969)
# d1 = dLvisc - beta; z0 = dLvisc*beta + 2*dLvisc/r + dLh*or2 + dbeta + 2*beta/r
_z_impl_visc = visc.unsqueeze(0)                       # (1, n_r_max)
_z_impl_d1 = (dLvisc - beta).unsqueeze(0)              # (1, n_r_max)
# zeroth-order without the dLh*or2 part (which depends on l):
_z_impl_z0_nol = (dLvisc * beta + two * dLvisc * or1
                  + dbeta + two * beta * or1).unsqueeze(0)  # (1, n_r_max)

# l=0 index (spherically symmetric toroidal is forced to zero)
_lm_l0 = st_lm2[0, 0].item()

# l=1, m=0 index (for z10 angular momentum coupling)
_l1m0 = st_lm2[1, 0].item()

# l=1, m=1 index (for equatorial AM correction)
_l1m1 = st_lm2[1, 1].item() if l_max >= 1 else -1

# --- Angular momentum correction precomputed profiles ---
# These are (n_r_max,) tensors used in the correction subtraction.
# For Boussinesq: rho0=1, beta=0, dbeta=0 → _am_z = r², _am_dz = 2*r, _am_d2z = 2
_am_r2 = r * r  # (n_r_max,) on DEVICE
_am_z_profile = rho0 * _am_r2  # rho0 * r²
_am_dz_profile = rho0 * (two * r + _am_r2 * beta)  # rho0 * (2r + r²*beta)
_am_d2z_profile = rho0 * (two + four * beta * r + dbeta * _am_r2
                          + beta * beta * _am_r2)  # rho0 * (2 + 4β*r + dβ*r² + β²*r²)

# Precompute ICB scalar constants as Python floats (avoids GPU→CPU sync per step)
_visc_icb = visc[n_r_max - 1].item()
_or1_icb = or1[n_r_max - 1].item()

# --- LU-factored matrices storage (one per l degree, l>=1) ---
_zMat_lu = [None] * (l_max + 1)
_zMat_ip = [None] * (l_max + 1)
_zMat_fac = [None] * (l_max + 1)

# Unique inverse per l degree: (l_max+1, N, N) float64 — l=0 is zero
_z_inv_by_l = None
# FD: banded LU storage per l
_z_bands_by_l = None
_z_piv_by_l = None
_z_fac_by_l = None
_z_kl = 0
_z_ku = 0

# Separate z10Mat inverse for l=1,m=0 when l_z10mat
_z10_inv = None


def build_z_matrices(wimp_lin0: float):
    """Build and LU-factorize toroidal velocity LHS matrices for l >= 1.

    Matrix in Chebyshev space:
        dat = rnorm * dLh * or2 * (rMat - wimp * hdif * (d2rMat - dLh * or2 * rMat))
    with no-slip BCs (Dirichlet z=0) at rows 0 and N-1.

    When l_z10mat, also builds a separate z10Mat for l=1,m=0 with IC angular
    momentum coupling at the ICB boundary row.

    Must be called whenever dt changes.
    """
    global _z_inv_by_l, _z10_inv, _z_bands_by_l, _z_piv_by_l, _z_fac_by_l, _z_kl, _z_ku
    from .params import l_finite_diff
    N = n_r_max

    cpu = torch.device("cpu")
    _rMat = rMat.to(cpu)
    _drMat = drMat.to(cpu)
    _d2rMat = d2rMat.to(cpu)
    _or1_cpu = or1.to(cpu)
    _or2 = or2.to(cpu)
    _rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac
    or2_col = _or2.unsqueeze(1)

    eye = torch.eye(N, dtype=DTYPE, device=cpu)
    inv_by_l = torch.zeros(l_max + 1, N, N, dtype=DTYPE, device=cpu)
    # FD bandwidth: depends on BCs (no-slip → narrower, stress-free → wider)
    if l_finite_diff:
        from .params import fd_order, fd_order_bound
        if ktopv != 1 and kbotv != 1 and fd_order <= 2 and fd_order_bound <= 2:
            _z_kl, _z_ku = fd_order // 2, fd_order // 2
        else:
            hw = max(fd_order // 2, fd_order_bound)
            _z_kl, _z_ku = hw, hw
    n_abd = 2 * _z_kl + _z_ku + 1 if l_finite_diff else 1
    abd_all = torch.zeros(l_max + 1, max(n_abd, 1), N, dtype=DTYPE, device=cpu)
    piv_all = torch.zeros(l_max + 1, N, dtype=torch.long, device=cpu)
    fac_all = torch.ones(l_max + 1, N, dtype=DTYPE, device=cpu)
    if l_finite_diff:
        # l=0: identity bands
        abd_all[0, _z_kl + _z_ku, :] = 1.0
        piv_all[0] = torch.arange(1, N + 1, dtype=torch.long)

    # Anelastic profiles for bulk equation
    _beta = beta.to(cpu)
    _dbeta = dbeta.to(cpu)
    _dLvisc = dLvisc.to(cpu)
    _visc = visc.to(cpu)
    beta_col = _beta.unsqueeze(1)
    dbeta_col = _dbeta.unsqueeze(1)
    dLvisc_col = _dLvisc.unsqueeze(1)
    visc_col = _visc.unsqueeze(1)
    or1_col = _or1_cpu.unsqueeze(1)

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        # Bulk: rnorm * dLh*or2 * (rMat - wimp*hdif*visc*(d2rMat + (dLvisc-beta)*drMat
        #   - (dLvisc*beta + 2*dLvisc/r + dLh*or2 + dbeta + 2*beta/r)*rMat))
        dat = _rnorm * dL * or2_col * (
            _rMat - wimp_lin0 * hdif_l * visc_col * (
                _d2rMat
                + (dLvisc_col - beta_col) * _drMat
                - (dLvisc_col * beta_col + two * dLvisc_col * or1_col
                   + dL * or2_col + dbeta_col + two * beta_col * or1_col) * _rMat
            )
        )

        # Boundary conditions
        if ktopv == 1:  # stress-free
            dat[0, :] = _rnorm * (_drMat[0, :] - (two * _or1_cpu[0] + _beta[0]) * _rMat[0, :])
        else:  # no-slip
            dat[0, :] = _rnorm * _rMat[0, :]

        if kbotv == 1:  # stress-free
            dat[N - 1, :] = _rnorm * (_drMat[N - 1, :] - (two * _or1_cpu[N - 1] + _beta[N - 1]) * _rMat[N - 1, :])
        else:  # no-slip
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
            abd = dense_to_band_storage(dat_precond, _z_kl, _z_ku)
            abd_f, piv, info = prepare_band(abd, N, _z_kl, _z_ku)
            assert info == 0, f"Singular zMat (band) for l={l}, info={info}"
            abd_all[l] = abd_f
            piv_all[l] = piv
            fac_all[l] = fac
        else:
            lu, ip, info = prepare_mat(dat_precond)
            assert info == 0, f"Singular zMat for l={l}, info={info}"
            inv_precond = solve_mat_real(lu, ip, eye)
            inv_by_l[l] = inv_precond * fac.unsqueeze(0)

    if l_finite_diff:
        _z_bands_by_l = abd_all.to(DEVICE)
        _z_piv_by_l = piv_all.to(DEVICE)
        _z_fac_by_l = fac_all.to(DEVICE)
        _z_inv_by_l = None
    else:
        _z_inv_by_l = inv_by_l.to(DEVICE)
        _z_bands_by_l = None

    # --- Build z10Mat for l=1,m=0 (angular momentum coupling at ICB) ---
    if l_z10mat:
        l = 1
        dL = 2.0
        hdif_l = hdif_V[l].item()
        # visc is 1.0 for Boussinesq, beta=0
        v_icb = visc[N - 1].item()

        # Bulk rows: same as zMat for l=1 (Fortran get_z10Mat lines 1781-1793)
        dat = _rnorm * dL * or2_col * (
            _rMat - wimp_lin0 * hdif_l * visc_col * (
                _d2rMat
                + (dLvisc_col - beta_col) * _drMat
                - (dLvisc_col * beta_col + two * dLvisc_col * or1_col
                   + dL * or2_col + dbeta_col + two * beta_col * or1_col) * _rMat
            )
        )

        # CMB row (row 0): no-slip, no mantle rotation → just z=0
        dat[0, :] = _rnorm * _rMat[0, :]

        # ICB row (row N-1): depends on IC rotation mode
        from .params import l_SRIC
        if l_SRIC:
            # Prescribed IC rotation: simple Dirichlet z10(ICB) = omega_ic / c_z10_omega_ic
            dat[N - 1, :] = _rnorm * c_z10_omega_ic * _rMat[N - 1, :]
        else:
            # Time-integrated IC rotation: angular momentum equation
            dat[N - 1, :] = _rnorm * (
                c_dt_z10_ic * _rMat[N - 1, :]
                + wimp_lin0 * (
                    v_icb * ((two * _or1_cpu[N - 1] + _beta[N - 1]) * _rMat[N - 1, :]
                             - _drMat[N - 1, :])
                    + gammatau_gravi * c_lorentz_ic * c_z10_omega_ic * _rMat[N - 1, :]
                )
            )

        # Zero high-order Chebyshev modes in boundary rows
        if n_cheb_max < N:
            dat[0, n_cheb_max:N] = 0.0
            dat[N - 1, n_cheb_max:N] = 0.0

        # Boundary normalization
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        # Row preconditioning
        fac = 1.0 / dat.abs().max(dim=1).values
        dat = fac.unsqueeze(1) * dat

        lu, ip, info = prepare_mat(dat)
        assert info == 0, f"Singular z10Mat, info={info}"

        inv_precond = solve_mat_real(lu, ip, eye)
        _z10_inv = (inv_precond * fac.unsqueeze(0)).to(DEVICE)


def get_angular_moment(z10, z11, omega_ic=0.0, omega_ma=0.0):
    """Compute total angular momentum vector (x, y, z).

    Matches outRot.f90 get_angular_moment (lines 485-543).

    Args:
        z10: (n_r_max,) complex — z(l=1,m=0) radial profile
        z11: (n_r_max,) complex — z(l=1,m=1) radial profile
        omega_ic, omega_ma: IC/mantle rotation rates (scalars)

    Returns:
        (AM_x, AM_y, AM_z): total angular momentum components (real scalars)
    """
    # Radial integrands: f(r) = r² * z_component
    f_x = _am_r2 * z11.real
    f_y = _am_r2 * z11.imag
    f_z = _am_r2 * z10.real

    # Radial integration + normalization (outRot.f90 lines 524-533)
    fac = 8.0 * third * math.pi
    AM_oc_x = two * fac * y11_norm * rInt_R(f_x).item()
    AM_oc_y = -two * fac * y11_norm * rInt_R(f_y).item()
    AM_oc_z = fac * y10_norm * rInt_R(f_z).item()

    # IC and mantle contributions (only z-component)
    AM_x = AM_oc_x
    AM_y = AM_oc_y
    AM_z = AM_oc_z + c_moi_ic * omega_ic + c_moi_ma * omega_ma

    return AM_x, AM_y, AM_z


def updateZ(z_LMloc, dz_LMloc, dzdt, tscheme,
            domega_ic_dt=None, omega_ic_ref=None):
    """Toroidal velocity: IMEX RHS assembly, implicit solve, post-processing.

    - l=0 mode is forced to zero (no spherically symmetric toroidal component)
    - l>=1 modes solved with no-slip BCs (z=0 at boundaries)
    - When l_z10mat: l=1,m=0 uses special z10Mat with IC angular momentum coupling

    Args:
        z_LMloc, dz_LMloc, dzdt: toroidal velocity state (modified in place)
        tscheme: time scheme
        domega_ic_dt: IC angular momentum TimeScalar (when l_rot_ic)
        omega_ic_ref: list [omega_ic] for output (modified in place)

    Returns:
        omega_ic: IC rotation rate (float, only when l_z10mat, else 0.0)
    """
    N = n_r_max

    # 1. Assemble IMEX RHS for z
    rhs = tscheme.set_imex_rhs(dzdt)  # (lm_max, n_r_max)

    # 1b. For z10 with IC rotation: set boundary RHS to dom_ic (IMEX-assembled scalar)
    from .params import l_SRIC
    if l_z10mat:
        if l_SRIC:
            # Prescribed IC rotation: use omega_ic directly as Dirichlet BC
            from . import fields as _fields_z
            rhs[_l1m0, N - 1] = complex(_fields_z.omega_ic, 0.0)
            rhs[_l1m0, 0] = 0.0
        elif domega_ic_dt is not None:
            dom_ic = tscheme.set_imex_rhs_scalar(domega_ic_dt)
            rhs[_l1m0, N - 1] = complex(dom_ic, 0.0)
            rhs[_l1m0, 0] = 0.0

    # 2. Zero boundary conditions for all other modes
    if l_z10mat:
        # Zero boundaries for all except l=1,m=0 (already set above)
        mask_not_l1m0 = torch.ones(lm_max, dtype=torch.bool, device=DEVICE)
        mask_not_l1m0[_l1m0] = False
        rhs[mask_not_l1m0, 0] = 0.0
        rhs[mask_not_l1m0, N - 1] = 0.0
    else:
        rhs[:, 0] = 0.0
        rhs[:, N - 1] = 0.0

    # 3. Solve: batched for all modes, then override l=1,m=0 with z10Mat
    if _z_bands_by_l is not None:
        from .algebra import banded_solve_by_l
        z_cheb = banded_solve_by_l(
            _z_bands_by_l, _z_piv_by_l, st_lm2l, rhs,
            n=N, kl=_z_kl, ku=_z_ku, fac_row_by_l=_z_fac_by_l)
    else:
        z_cheb = chunked_solve_complex(_z_inv_by_l, st_lm2l, rhs)

    if l_z10mat:
        # Override l=1,m=0 with z10Mat solve
        rhs_z10 = rhs[_l1m0, :]  # (N,) complex
        z10_cheb = torch.mv(_z10_inv.to(CDTYPE), rhs_z10)
        z_cheb[_l1m0, :] = z10_cheb

    z_cheb[_m0_mask] = z_cheb[_m0_mask].real.to(CDTYPE)

    # 3b. Truncate high Chebyshev modes (Fortran only stores n_cheb_max modes)
    if n_cheb_max < N:
        z_cheb[:, n_cheb_max:] = 0.0

    # 4. Convert to physical space
    z_LMloc[:] = costf(z_cheb)

    # 5. Extract omega_ic from z10 at ICB (after costf)
    # For l_SRIC: omega_ic is prescribed, NOT extracted from z10
    omega_ic = 0.0
    if l_z10mat and not l_SRIC:
        z10_icb = z_LMloc[_l1m0, N - 1].real.item()
        omega_ic = c_z10_omega_ic * z10_icb
        if omega_ic_ref is not None:
            omega_ic_ref[0] = omega_ic

    # 6. Compute derivatives
    dz_new, d2z = get_ddr(z_LMloc)
    dz_LMloc[:] = dz_new

    # 6b. Angular momentum corrections (updateZ.f90 lines 860-933)
    # Correct z, dz, d2z so that total AM is conserved (axial) / zero (equatorial)
    if l_correct_AMz:
        z10 = z_LMloc[_l1m0, :]
        z11_zero = torch.zeros_like(z10)
        AM_x, AM_y, AM_z = get_angular_moment(z10, z11_zero, omega_ic, 0.0)

        # Denominator: stress-free BCs → nomi = c_moi_oc * y10_norm
        # (no l_rot_ma/l_rot_ic contributions for ktopv=1/kbotv=1)
        nomi = c_moi_oc * y10_norm
        corr_l1m0 = complex(AM_z - AMstart, 0.0) / nomi

        z_LMloc[_l1m0, :] -= _am_z_profile * corr_l1m0
        dz_LMloc[_l1m0, :] -= _am_dz_profile * corr_l1m0
        d2z[_l1m0, :] -= _am_d2z_profile * corr_l1m0

    if l_correct_AMe and _l1m1 >= 0:
        # Use updated z10 (post-AMz correction) for equatorial AM computation
        z10_now = z_LMloc[_l1m0, :]
        z11 = z_LMloc[_l1m1, :]
        AM_x, AM_y, AM_z = get_angular_moment(z10_now, z11, omega_ic, 0.0)

        corr_l1m1 = complex(AM_x, -AM_y) / (two * y11_norm * c_moi_oc)

        z_LMloc[_l1m1, :] -= _am_z_profile * corr_l1m1
        dz_LMloc[_l1m1, :] -= _am_dz_profile * corr_l1m1
        d2z[_l1m1, :] -= _am_d2z_profile * corr_l1m1

    # 7. Rotate IMEX time arrays
    tscheme.rotate_imex(dzdt)
    if l_z10mat and domega_ic_dt is not None and not l_SRIC:
        tscheme.rotate_imex_scalar(domega_ic_dt)

    # 8. Store old state: dLh * or2 * z (matches Fortran updateZ line 926)
    if tscheme.store_old:
        dzdt.old[:, :, 0] = _dLh_lm * _or2_r * z_LMloc

    # 8b. IC rotation old state (kbotv=2 no-slip, NOT for SRIC)
    if l_z10mat and domega_ic_dt is not None and tscheme.store_old and not l_SRIC:
        domega_ic_dt.old[0] = c_dt_z10_ic * z10_icb

    # 9. Compute implicit diffusion term
    idx = tscheme.next_impl_idx
    dzdt.impl[:, :, idx] = _hdif_lm * _dLh_lm * _or2_r * _z_impl_visc * (
        d2z + _z_impl_d1 * dz_LMloc
        - (_z_impl_z0_nol + _dLh_lm * _or2_r) * z_LMloc
    )

    # 9b. IC rotation implicit term (NOT for SRIC — prescribed rotation)
    if l_z10mat and domega_ic_dt is not None and not l_SRIC:
        dz10_icb_val = dz_LMloc[_l1m0, N - 1].real.item()
        _beta_icb = beta[N - 1].item()
        domega_ic_dt.impl[idx] = (
            -_visc_icb * ((two * _or1_icb + _beta_icb) * z10_icb - dz10_icb_val)
            - gammatau_gravi * c_lorentz_ic * omega_ic
        )

    return omega_ic


def finish_exp_tor(omega_ic: float, lorentz_torque_ic: float) -> float:
    """Compute explicit IC torque contribution (updateZ.f90 lines 1667-1696).

    Returns domega_ic_dt_exp = c_lorentz_ic * (lorentz_torque_ic + gammatau * omega_ma).
    Since omega_ma=0 (no mantle rotation) and gammatau=0, this simplifies to:
        c_lorentz_ic * lorentz_torque_ic
    """
    return c_lorentz_ic * (lorentz_torque_ic + gammatau_gravi * 0.0)  # omega_ma=0
