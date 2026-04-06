"""Main time stepping driver matching step_time.f90 + rIter.f90 + LMLoop.f90.

Implements:
- setup_initial_state: compute derivatives + initial old/impl from fields
- radial_loop: inverse SHT → get_nl → forward SHT → get_td
- lm_loop: finish explicit assembly + implicit solves
- one_step: single time step (CNAB2 or multi-stage DIRK)
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .profiler import prof
from .params import (n_r_max, lm_max, l_max, n_theta_max, n_phi_max, alpha,
                     l_cond_ic, l_rot_ic, l_mag, l_chemical_conv, l_anel,
                     l_correct_AMz, l_correct_AMe, l_heat, l_SRIC,
                     l_double_curl, l_finite_diff, radial_chunk_size)
from .courant import courant_check, dt_courant
from .radial_functions import or2, or4 as _or4_full, lambda_, dLlambda
from .horizontal_data import dLh, hdif_S, hdif_V, hdif_B, hdif_Xi
from .pre_calculations import opr, opm, BuoFac, ChemFac, CorFac, LFfac, l_z10mat, osc
from .blocking import st_lm2l, st_lm2m
from .radial_derivatives import get_dr, get_ddr, get_dddr
from .radial_scheme import costf
from .time_scheme import tscheme
from . import fields
from . import dt_fields
from .sht import (scal_to_spat, scal_to_SH, torpol_to_spat,
                  torpol_to_curl_spat, spat_to_sphertor,
                  pol_to_grad_spat, pol_to_curlr_spat, torpol_to_dphspat)
from .get_nl import get_nl
from .get_td import get_dwdt, get_dwdt_double_curl, get_dzdt, get_dpdt, get_dsdt, get_dbdt
from .update_s import (build_s_matrices, finish_exp_entropy, updateS)
from .update_z import (build_z_matrices, updateZ, finish_exp_tor)
from .update_wp import (build_p0_matrix, build_wp_matrices, updateWP, finish_exp_pol)
from .update_b import (build_b_matrices, finish_exp_mag, updateB, finish_exp_mag_ic)
from .update_w_doublecurl import (build_w_matrices, updateW, get_w_old, get_w_impl)
from .constants import two, three, third, four
import math

# Conditional imports for composition
if l_chemical_conv:
    from .update_xi import (build_xi_matrices, finish_exp_comp, updateXi)
    from .get_td import get_dxidt


# dLh for torpol_to_spat Q input: Fortran wrapper multiplies by dLh
_dLh_1d = dLh.to(CDTYPE)  # (lm_max,)

# Pre-allocated velocity buffers for radial_loop (no-slip: boundaries stay zero)
# Always allocated: needed by non-chunked path, anelastic path, and movie output.
# Curl-velocity buffers only needed by non-chunked Boussinesq path.
_vrc = torch.zeros(n_r_max, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
_vtc = torch.zeros_like(_vrc)
_vpc = torch.zeros_like(_vrc)
if radial_chunk_size == 0:
    _cvrc = torch.zeros_like(_vrc)
    _cvtc = torch.zeros_like(_vrc)
    _cvpc = torch.zeros_like(_vrc)
else:
    _cvrc = _cvtc = _cvpc = None

# Grid-space fields persisted for movie output (set by radial_loop)
last_brc = None  # (n_r, n_theta, n_phi)
last_btc = None
last_bpc = None
last_sc = None

# Pre-computed radial constants for chunked radial_loop (get_nl + courant_check)
if radial_chunk_size > 0:
    _or4_3_full = _or4_full.reshape(n_r_max, 1, 1)
    _or2_3_full = or2.reshape(n_r_max, 1, 1)
    from .radial_functions import orho1 as _orho1_full, orho2 as _orho2_full
    from .radial_functions import delxr2 as _delxr2, delxh2 as _delxh2
    _orho1_3_full = _orho1_full.reshape(n_r_max, 1, 1)
    _orho2_3_full = _orho2_full.reshape(n_r_max, 1, 1)
else:
    _or4_3_full = _or2_3_full = None
    _orho1_3_full = _orho2_3_full = None
    _delxr2 = _delxh2 = None

# Precompute broadcast arrays for initial rhs_imp
_hdif_S_lm = hdif_S[st_lm2l].to(CDTYPE).unsqueeze(1)
_hdif_V_lm = hdif_V[st_lm2l].to(CDTYPE).unsqueeze(1)
_hdif_B_lm = hdif_B[st_lm2l].to(CDTYPE).unsqueeze(1)
_hdif_Xi_lm = hdif_Xi[st_lm2l].to(CDTYPE).unsqueeze(1)
_dLh_lm = dLh.to(CDTYPE).unsqueeze(1)
from .radial_functions import or1, or3
_or1_r = or1.unsqueeze(0).to(CDTYPE)
_or2_r = or2.unsqueeze(0).to(CDTYPE)
_or3_r = or3.unsqueeze(0).to(CDTYPE)
from .radial_functions import rgrav
_rgrav_r = rgrav.unsqueeze(0).to(CDTYPE)

# Pre-computed broadcast arrays for radial_loop input assembly
_dLh_2d = _dLh_1d.unsqueeze(1)                           # (lm_max, 1)
_or2_2d = or2.unsqueeze(0).to(CDTYPE)                    # (1, N)
_or2_bulk_2d = or2[1:n_r_max - 1].unsqueeze(0).to(CDTYPE)  # (1, Nb)

# Precompute ICB scalar constants for radial_loop (avoids GPU→CPU sync per stage)
if l_cond_ic or l_rot_ic:
    from .radial_functions import or4, r, orho1
    _r_icb_py = r[n_r_max - 1].item()
    _orho1_icb_py = orho1[n_r_max - 1].item()
    _or4_icb_py = or4[n_r_max - 1].item()

# === Batched scalar solver infrastructure ===
# Reduces MPS dispatch overhead by combining S + [Xi] + Z into single bmm/costf/matmul calls

from .radial_derivatives import _D1 as _D1_cd, _D2 as _D2_cd, _D3 as _D3_cd
from .algebra import chunked_solve_complex
from .params import n_cheb_max

_m0_mask = (st_lm2m == 0)  # (lm_max,) bool

# Lorentz torque on IC from most recent radial_loop (for rot.TAG output)
lorentz_torque_ic_last = 0.0

# Combined D matrices as float64 for DGEMM (view_as_real trick)
from .radial_derivatives import _D1_real_T, _D2_real_T, _D3_real_T
_D12_T = torch.cat([_D1_real_T, _D2_real_T], dim=1)    # (N, 2N) float64
_D123_T = torch.cat([_D1_real_T, _D2_real_T, _D3_real_T], dim=1)  # (N, 3N) float64

# Precomputed combined coefficients for impl terms (saves multiplications)
_two_or1_r = (two * _or1_r)  # (1, N) complex
_dLh_or2 = _dLh_lm * _or2_r  # (lm_max, N) complex
_opr_hdif_S = opr * _hdif_S_lm  # (lm_max, 1) complex
_osc_hdif_Xi = osc * _hdif_Xi_lm  # (lm_max, 1) complex
_hdifV_dLh_or2 = _hdif_V_lm * _dLh_lm * _or2_r  # (lm_max, N) complex
_hdifV_dLh_or2_sq = _hdifV_dLh_or2 * _dLh_or2  # Z impl: hdif*dLh*or2 * dLh*or2
_BuoFac_rgrav = BuoFac * _rgrav_r  # W impl: BuoFac*rgrav
_ChemFac_rgrav = ChemFac * _rgrav_r if l_chemical_conv else None
_two_dLh_or3 = two * _dLh_lm * _or3_r  # P impl: 2*dLh*or3

# WP buoyancy coupling constants (precomputed for fused lm_loop)
_rgrav_interior = rgrav[1:n_r_max - 1].unsqueeze(0).to(CDTYPE)  # (1, N-2)
_rgrav_1d = rgrav  # (N,) for p0 RHS
_BuoFac_rgrav_int = (BuoFac * _rgrav_interior)  # (1, N-2) precomputed
_ChemFac_rgrav_int = (ChemFac * _rgrav_interior) if l_chemical_conv else None
_BuoFac_rgrav_1d = (BuoFac * _rgrav_1d[1:])  # (N-1,) for p0 RHS
_ChemFac_rgrav_1d = (ChemFac * _rgrav_1d[1:]) if l_chemical_conv else None

# Pre-allocated p0 RHS buffer (avoid per-call torch.zeros)
_p0_rhs = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

# Stacked inverse matrices and l_index for batched scalar solve
# Initialized by _init_batched_scalar() after matrix build
_scalar_inv = None    # (nS * (l_max+1), N, N) float64
_scalar_lidx = None   # (nS * lm_max,) long
_scalar_m0 = None     # (nS * lm_max,) bool — stacked m=0 mask


def _init_batched_scalar():
    """Stack scalar solver inverses for batched bmm.

    Called by build_all_matrices after individual matrices are built.
    Combines S + [Xi] + Z inverses into one tensor so chunked_solve_complex
    processes all scalar solvers in a single bmm call.
    """
    global _scalar_inv, _scalar_lidx, _scalar_m0
    from . import update_s, update_z
    invs = [update_s._s_inv_by_l]
    if l_chemical_conv:
        from . import update_xi
        invs.append(update_xi._xi_inv_by_l)
    invs.append(update_z._z_inv_by_l)
    nS = len(invs)
    L = l_max + 1
    _scalar_inv = torch.cat(invs, dim=0)  # (nS*L, N, N)
    _scalar_lidx = torch.cat([st_lm2l + i * L for i in range(nS)])
    _scalar_m0 = _m0_mask.repeat(nS)


def _assemble_rhs_mega(mega_old, mega_expl, mega_impl):
    """Assemble IMEX RHS from contiguous mega-tensors (no torch.cat needed).

    Works for both CNAB2 and BPR353 time schemes.
    Uses in-place add_ with alpha for fused multiply-add (1 dispatch per term).
    """
    if tscheme.nstages > 1:  # BPR353
        k = tscheme.istage
        exp_row = tscheme._exp_py[k]
        imp_row = tscheme._imp_py[k]
        # Start from first nonzero explicit term (avoids clone of old)
        # BPR353: exp_row[0] is always nonzero for all stages
        rhs = exp_row[0] * mega_expl[:, :, 0]  # new tensor, no clone
        rhs.add_(mega_old[:, :, 0])
        for j in range(1, k):
            w = exp_row[j]
            if w != 0.0:
                rhs.add_(mega_expl[:, :, j], alpha=w)
        for j in range(k):
            w = imp_row[j]
            if w != 0.0:
                rhs.add_(mega_impl[:, :, j], alpha=w)
        return rhs
    else:  # CNAB2
        rhs = tscheme._wimp_py[0] * mega_old[:, :, 0]
        for n_o in range(tscheme.nimp):
            w = tscheme._wimp_lin_py[n_o + 1]
            if w != 0.0:
                rhs.add_(mega_impl[:, :, n_o], alpha=w)
        for n_o in range(tscheme.nexp):
            w = tscheme._wexp_py[n_o]
            if w != 0.0:
                rhs.add_(mega_expl[:, :, n_o], alpha=w)
        return rhs


def _fused_lm_loop_fast():
    """Fully fused lm_loop for non-IC-rotation case.

    Combines scalar solvers (S + [Xi] + Z) and WP solver into a single
    optimized pipeline that minimizes MPS dispatch overhead:
    - Cat-free RHS assembly via mega-tensor views
    - Deferred Z costf batched with W+P
    - Single D1+D2+D3 matmul for all 5 field derivatives
    """
    from . import update_wp
    f = fields
    d = dt_fields
    N = n_r_max
    LM = lm_max
    nS = d._nS  # 2 or 3 (S+Z or S+Xi+Z)

    # === 1. Unified RHS assembly for ALL 5 dfdts (single pass, no torch.cat) ===
    rhs_all = _assemble_rhs_mega(d._all_old, d._all_expl, d._all_impl)
    rhs_scalar = rhs_all[:nS * LM]  # free view
    rhs_wp_raw = rhs_all[nS * LM:]  # free view

    # BCs: S
    from .update_s import _tops, _bots
    rhs_scalar[:LM, 0] = _tops
    rhs_scalar[:LM, N - 1] = _bots
    if l_chemical_conv:
        from .update_xi import _topxi, _botxi
        rhs_scalar[LM:2 * LM, 0] = _topxi
        rhs_scalar[LM:2 * LM, N - 1] = _botxi
    # Z: no-slip
    rhs_scalar[(nS - 1) * LM:, 0] = 0.0
    rhs_scalar[(nS - 1) * LM:, N - 1] = 0.0

    # === 2. Scalar solve (single bmm) ===
    prof.start("lm_loop.scalar_solve")
    sol_scalar = chunked_solve_complex(_scalar_inv, _scalar_lidx, rhs_scalar)
    sol_scalar[_scalar_m0] = sol_scalar[_scalar_m0].real.to(CDTYPE)
    if n_cheb_max < N:
        sol_scalar[:, n_cheb_max:] = 0.0

    # === 3. costf for S+Xi only (WP needs physical s, xi for buoyancy) ===
    nSXi = nS - 1  # S + Xi (without Z)
    sxi_phys = costf(sol_scalar[:nSXi * LM])
    f.s_LMloc[:] = sxi_phys[:LM]
    if l_chemical_conv:
        f.xi_LMloc[:] = sxi_phys[LM:2 * LM]

    # === 4. WP RHS build (already assembled in step 1) ===
    rhs_w = rhs_wp_raw[:LM]
    rhs_p = rhs_wp_raw[LM:]

    # Build combined (lm_max, 2N) RHS
    rhs_combined = torch.zeros(LM, 2 * N, dtype=CDTYPE, device=DEVICE)
    rhs_combined[:, 1:N - 1] = rhs_w[:, 1:N - 1]
    wimp_lin0 = tscheme.wimp_lin[0]
    rhs_combined[:, 1:N - 1] += wimp_lin0 * _BuoFac_rgrav_int * f.s_LMloc[:, 1:N - 1]
    if l_chemical_conv:
        rhs_combined[:, 1:N - 1] += wimp_lin0 * _ChemFac_rgrav_int * f.xi_LMloc[:, 1:N - 1]
    rhs_combined[:, N + 1:2 * N - 1] = rhs_p[:, 1:N - 1]

    # p0 RHS (l=0 pressure) — use pre-allocated buffer
    lm0 = update_wp._lm_l0
    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0
    p0_rhs = _p0_rhs
    p0_rhs[0] = 0.0
    p0_rhs[1:] = _BuoFac_rgrav_1d * f.s_LMloc[lm0, 1:].real + d.dwdt.expl[lm0, 1:, expl_idx].real
    if l_chemical_conv:
        p0_rhs[1:] += _ChemFac_rgrav_1d * f.xi_LMloc[lm0, 1:].real

    # === 5. WP solve ===
    prof.stop("lm_loop.scalar_solve")
    prof.start("lm_loop.wp_solve")
    sol_wp = chunked_solve_complex(update_wp._wp_inv_by_l, st_lm2l, rhs_combined)
    sol_wp[_m0_mask] = sol_wp[_m0_mask].real.to(CDTYPE)
    w_cheb = sol_wp[:, :N]
    p_cheb = sol_wp[:, N:]
    # l=0 pressure from p0Mat (GPU matmul, no CPU↔GPU sync)
    p_cheb[lm0, :] = update_wp.solve_p0(p0_rhs).to(CDTYPE)
    if n_cheb_max < N:
        w_cheb[:, n_cheb_max:] = 0.0
        p_cheb[:, n_cheb_max:] = 0.0

    # === 6. Batched costf for Z+W+P (1 FFT instead of 2) ===
    prof.stop("lm_loop.wp_solve")
    z_cheb = sol_scalar[(nS - 1) * LM:]  # Z is the last scalar block
    zwp_cheb = torch.cat([z_cheb, w_cheb, p_cheb])  # single cat, 3 fields
    zwp_phys = costf(zwp_cheb)
    f.z_LMloc[:] = zwp_phys[:LM]
    f.w_LMloc[:] = zwp_phys[LM:2 * LM]
    f.p_LMloc[:] = zwp_phys[2 * LM:]

    # === 7. Radial derivatives via DCT pipeline: O(N log N) ===
    prof.start("lm_loop.d123_matmul")  # keep profiler name for comparison
    # S+[Xi]: d1+d2
    f.ds_LMloc[:], d2s = get_ddr(sxi_phys[:LM])
    if l_chemical_conv:
        f.dxi_LMloc[:], d2xi = get_ddr(sxi_phys[LM:2 * LM])
    # Z: d1+d2
    f.dz_LMloc[:], d2z = get_ddr(zwp_phys[:LM])
    # W: d1+d2+d3
    f.dw_LMloc[:], f.ddw_LMloc[:], dddw = get_dddr(zwp_phys[LM:2 * LM])
    dddw = dddw  # keep name for downstream use
    # P: d1 only
    f.dp_LMloc[:] = get_dr(zwp_phys[2 * LM:])

    prof.stop("lm_loop.d123_matmul")
    # === 8. Rotate IMEX (BPR353: no-op; CNAB2: shift history) ===
    tscheme.rotate_imex(d.dsdt)
    if l_chemical_conv:
        tscheme.rotate_imex(d.dxidt)
    tscheme.rotate_imex(d.dzdt)
    tscheme.rotate_imex(d.dwdt)
    tscheme.rotate_imex(d.dpdt)

    # === 9. Store old state ===
    if tscheme.store_old:
        d.dsdt.old[:, :, 0] = f.s_LMloc.clone()
        if l_chemical_conv:
            d.dxidt.old[:, :, 0] = f.xi_LMloc.clone()
        d.dzdt.old[:, :, 0] = _dLh_or2 * f.z_LMloc
        d.dwdt.old[:, :, 0] = _dLh_or2 * f.w_LMloc
        d.dpdt.old[:, :, 0] = -_dLh_or2 * f.dw_LMloc

    # === 10. Implicit terms ===
    impl_idx = tscheme.next_impl_idx
    # S: opr*hdif_S * (d2s + 2*or1*ds - dLh*or2*s)
    d.dsdt.impl[:, :, impl_idx] = _opr_hdif_S * (
        d2s + _two_or1_r * f.ds_LMloc - _dLh_or2 * f.s_LMloc
    )
    if l_chemical_conv:
        d.dxidt.impl[:, :, impl_idx] = _osc_hdif_Xi * (
            d2xi + _two_or1_r * f.dxi_LMloc - _dLh_or2 * f.xi_LMloc
        )
    # Z: hdif_V*dLh*or2 * d2z - hdif_V*(dLh*or2)^2 * z
    d.dzdt.impl[:, :, impl_idx] = _hdifV_dLh_or2 * d2z - _hdifV_dLh_or2_sq * f.z_LMloc
    # W: -dp + hdif*dLh*or2*(ddw - dLh*or2*w) + BuoFac*rgrav*s [+ ChemFac*rgrav*xi]
    Dif_w = _hdifV_dLh_or2 * (f.ddw_LMloc - _dLh_or2 * f.w_LMloc)
    d.dwdt.impl[:, :, impl_idx] = -f.dp_LMloc + Dif_w + _BuoFac_rgrav * f.s_LMloc
    if l_chemical_conv:
        d.dwdt.impl[:, :, impl_idx] += _ChemFac_rgrav * f.xi_LMloc
    # P: dLh*or2*p + hdif*dLh*or2*(-dddw + dLh*or2*dw - 2*dLh*or3*w)
    d.dpdt.impl[:, :, impl_idx] = (
        _dLh_or2 * f.p_LMloc
        + _hdifV_dLh_or2 * (
            -dddw + _dLh_or2 * f.dw_LMloc
            - _two_dLh_or3 * f.w_LMloc
        )
    )
    # Fortran get_pol_rhs_imp loops over l>0, interior radial points only.
    # l=0 zeroing is load-bearing: the WP batched solve processes all lm,
    # and nonzero l=0 impl would corrupt w_cheb[lm0] (p0Mat only fixes p).
    # Boundary zeroing is cosmetic (rhs_combined[:, 1:N-1] skips them).
    d.dwdt.impl[update_wp._lm_l0, :, impl_idx] = 0
    d.dpdt.impl[update_wp._lm_l0, :, impl_idx] = 0
    d.dwdt.impl[:, 0, impl_idx] = 0
    d.dwdt.impl[:, -1, impl_idx] = 0
    d.dpdt.impl[:, 0, impl_idx] = 0
    d.dpdt.impl[:, -1, impl_idx] = 0



def setup_initial_state():
    """Compute derivatives and initial old/impl arrays from initialized fields.

    Must be called after init_fields.initialize_fields().
    Matches get_*_rhs_imp calls in startFields.f90.

    For BPR353: impl is stored at index 0 (the "slot 1" in Fortran), which
    holds L*u_n for the first stage's RHS assembly. old is at index 0.
    """
    f = fields
    d = dt_fields

    # Zero dt_fields for re-entrant calls
    tas = [d.dsdt, d.dwdt, d.dzdt, d.dpdt]
    if l_chemical_conv:
        tas.append(d.dxidt)
    if l_mag:
        tas += [d.dbdt, d.djdt]
        if l_cond_ic:
            tas += [d.dbdt_ic, d.djdt_ic]
    for ta in tas:
        ta.old.zero_()
        ta.impl.zero_()
        ta.expl.zero_()

    # --- Entropy: compute ds, d2s ---
    if l_heat:
        ds, d2s = get_ddr(f.s_LMloc)
        f.ds_LMloc[:] = ds

        # dsdt.old = s
        d.dsdt.old[:, :, 0] = f.s_LMloc.clone()

        # dsdt.impl = opr * hdif_S * kappa * (d2s + (beta+dLtemp0+2/r+dLkappa)*ds - dLh*or2*s)
        if l_anel:
            from .update_s import _kappa_r, _s_impl_d1
            d.dsdt.impl[:, :, 0] = opr * _hdif_S_lm * _kappa_r * (
                d2s + _s_impl_d1 * ds - _dLh_lm * _or2_r * f.s_LMloc
            )
        else:
            d.dsdt.impl[:, :, 0] = opr * _hdif_S_lm * (
                d2s + two * _or1_r * ds - _dLh_lm * _or2_r * f.s_LMloc
            )

    # --- Toroidal velocity: compute dz, d2z ---
    dz, d2z = get_ddr(f.z_LMloc)
    f.dz_LMloc[:] = dz

    # dzdt.old = dLh * or2 * z  (matches Fortran updateZ line 926)
    d.dzdt.old[:, :, 0] = _dLh_lm * _or2_r * f.z_LMloc

    # dzdt.impl = hdif*dLh*or2*visc*(d2z + (dLvisc-beta)*dz - (...)*z)
    if l_anel:
        from .update_z import _z_impl_visc, _z_impl_d1, _z_impl_z0_nol
        d.dzdt.impl[:, :, 0] = _hdif_V_lm * _dLh_lm * _or2_r * _z_impl_visc * (
            d2z + _z_impl_d1 * dz
            - (_z_impl_z0_nol + _dLh_lm * _or2_r) * f.z_LMloc
        )
    else:
        d.dzdt.impl[:, :, 0] = _hdif_V_lm * _dLh_lm * _or2_r * (
            d2z - _dLh_lm * _or2_r * f.z_LMloc
        )

    # --- Poloidal velocity + pressure: compute dw, ddw, dp ---
    dw, ddw = get_ddr(f.w_LMloc)
    f.dw_LMloc[:] = dw
    f.ddw_LMloc[:] = ddw
    if not l_double_curl:
        # Non-double-curl needs dddw for dpdt.impl, dp for dwdt.impl
        dddw = get_dddr(f.w_LMloc)[2]  # 3rd output of get_dddr
        dp = get_dr(f.p_LMloc)
        f.dp_LMloc[:] = dp

    if l_double_curl:
        # Double-curl: dddw, ddddw via chaining D1(D2(w)), D2(D2(w))
        # Matches Fortran get_pol_rhs_imp (get_ddr(w)→dw,ddw; get_ddr(ddw)→dddw,ddddw)
        dddw, ddddw = get_ddr(ddw)
        d.dwdt.old[:, :, 0] = get_w_old(f.w_LMloc, dw, ddw)
        d.dwdt.impl[:, :, 0] = get_w_impl(
            f.w_LMloc, dw, ddw, dddw, ddddw,
            f.s_LMloc, f.xi_LMloc if l_chemical_conv else None)
        # l=0 and boundary zeroing
        from .update_wp import _lm_l0
        d.dwdt.impl[_lm_l0, :, 0] = 0
        d.dwdt.impl[:, 0, 0] = 0
        d.dwdt.impl[:, -1, 0] = 0
        # dpdt is not time-stepped in double-curl — leave at zero
    else:
        # Standard coupled w+p formulation
        # dwdt.old = dLh * or2 * w
        d.dwdt.old[:, :, 0] = _dLh_lm * _or2_r * f.w_LMloc

        # dpdt.old = -dLh * or2 * dw
        d.dpdt.old[:, :, 0] = -_dLh_lm * _or2_r * dw

        # dwdt.impl and dpdt.impl: full anelastic terms from update_wp.py
        if l_anel:
            from .update_wp import (
                _visc_r as _wp_visc, _beta_r as _wp_beta,
                _dLvisc_r as _wp_dLvisc, _dbeta_r as _wp_dbeta,
                _rho0_r as _wp_rho0,
            )
            Dif_w = _hdif_V_lm * _dLh_lm * _or2_r * _wp_visc * (
                ddw
                + (two * _wp_dLvisc - third * _wp_beta) * dw
                - (_dLh_lm * _or2_r
                   + four * third * (_wp_dbeta + _wp_dLvisc * _wp_beta
                                     + (three * _wp_dLvisc + _wp_beta) * _or1_r))
                * f.w_LMloc
            )
            Pre = -dp + _wp_beta * f.p_LMloc
            Buo = BuoFac * _wp_rho0 * _rgrav_r * f.s_LMloc
            if l_chemical_conv:
                Buo = Buo + ChemFac * _wp_rho0 * _rgrav_r * f.xi_LMloc
            d.dwdt.impl[:, :, 0] = Pre + Dif_w + Buo

            d.dpdt.impl[:, :, 0] = (
                _dLh_lm * _or2_r * f.p_LMloc
                + _hdif_V_lm * _wp_visc * _dLh_lm * _or2_r * (
                    -dddw
                    + (_wp_beta - _wp_dLvisc) * ddw
                    + (_dLh_lm * _or2_r + _wp_dLvisc * _wp_beta + _wp_dbeta
                       + two * (_wp_dLvisc + _wp_beta) * _or1_r) * dw
                    - _dLh_lm * _or2_r
                    * (two * _or1_r + two * third * _wp_beta + _wp_dLvisc)
                    * f.w_LMloc
                )
            )
        else:
            Dif_w = _hdif_V_lm * _dLh_lm * _or2_r * (ddw - _dLh_lm * _or2_r * f.w_LMloc)
            d.dwdt.impl[:, :, 0] = -dp + Dif_w + BuoFac * _rgrav_r * f.s_LMloc
            if l_chemical_conv:
                d.dwdt.impl[:, :, 0] += ChemFac * _rgrav_r * f.xi_LMloc

            d.dpdt.impl[:, :, 0] = (_dLh_lm * _or2_r * f.p_LMloc
                                    + _hdif_V_lm * _dLh_lm * _or2_r * (
                                        -dddw + _dLh_lm * _or2_r * dw
                                        - two * _dLh_lm * _or3_r * f.w_LMloc
                                    ))

        # l=0 and boundary zeroing
        from .update_wp import _lm_l0
        d.dwdt.impl[_lm_l0, :, 0] = 0
        d.dpdt.impl[_lm_l0, :, 0] = 0
        d.dwdt.impl[:, 0, 0] = 0
        d.dwdt.impl[:, -1, 0] = 0
        d.dpdt.impl[:, 0, 0] = 0
        d.dpdt.impl[:, -1, 0] = 0

    # --- Composition: compute dxi, d2xi ---
    if l_chemical_conv:
        dxi, d2xi = get_ddr(f.xi_LMloc)
        f.dxi_LMloc[:] = dxi

        d.dxidt.old[:, :, 0] = f.xi_LMloc.clone()
        from .update_xi import _beta_r as _xi_beta_r
        d.dxidt.impl[:, :, 0] = osc * _hdif_Xi_lm * (
            d2xi + (_xi_beta_r + two * _or1_r) * dxi
            - _dLh_lm * _or2_r * f.xi_LMloc
        )

    # --- Magnetic field: compute db, ddb, dj, ddj ---
    if l_mag:
        db, ddb = get_ddr(f.b_LMloc)
        f.db_LMloc[:] = db
        f.ddb_LMloc[:] = ddb
        dj, ddj = get_ddr(f.aj_LMloc)
        f.dj_LMloc[:] = dj
        f.ddj_LMloc[:] = ddj

        # dbdt.old = dLh * or2 * b
        d.dbdt.old[:, :, 0] = _dLh_lm * _or2_r * f.b_LMloc

        # djdt.old = dLh * or2 * aj
        d.djdt.old[:, :, 0] = _dLh_lm * _or2_r * f.aj_LMloc

        # dbdt.impl = opm * lambda * hdif_B * dLh * or2 * (ddb - dLh * or2 * b)
        # Fortran only computes interior (n_r_cmb+1 to n_r_icb-1); boundaries stay zero.
        or2_int = _or2_r[:, 1:-1]
        lambda_int = lambda_.unsqueeze(0).to(CDTYPE)[:, 1:-1]
        dLlambda_int = dLlambda.unsqueeze(0).to(CDTYPE)[:, 1:-1]
        d.dbdt.impl[:, 1:-1, 0] = opm * lambda_int * _hdif_B_lm * _dLh_lm * or2_int * (
            ddb[:, 1:-1] - _dLh_lm * or2_int * f.b_LMloc[:, 1:-1]
        )

        # djdt.impl = opm * lambda * hdif_B * dLh * or2 * (ddj + dLlambda*dj - dLh * or2 * aj)
        d.djdt.impl[:, 1:-1, 0] = opm * lambda_int * _hdif_B_lm * _dLh_lm * or2_int * (
            ddj[:, 1:-1] + dLlambda_int * dj[:, 1:-1]
            - _dLh_lm * or2_int * f.aj_LMloc[:, 1:-1]
        )

        # --- IC magnetic field (if conducting inner core) ---
        if l_cond_ic:
            from .update_b import get_mag_ic_rhs_imp
            get_mag_ic_rhs_imp(
                f.b_ic, f.db_ic, f.ddb_ic, f.aj_ic, f.dj_ic, f.ddj_ic,
                d.dbdt_ic, d.djdt_ic, istage=1, l_calc_lin=True
            )

    # --- IC rotation initial state ---
    if l_z10mat:
        from .pre_calculations import c_dt_z10_ic, c_z10_omega_ic
        from .blocking import st_lm2 as _st_lm2
        _l1m0 = _st_lm2[1, 0].item()
        z10_icb = f.z_LMloc[_l1m0, n_r_max - 1].real.item()

        if l_SRIC:
            # Prescribed IC rotation: omega_ic is fixed, no domega_ic_dt evolution
            f.omega_ic = f.omega_ic  # already set by init_fields
        else:
            # old = c_dt_z10_ic * z10(ICB)
            d.domega_ic_dt.old[0] = c_dt_z10_ic * z10_icb
            # impl = -visc*((2*or1 + beta)*z10 - dz10) at ICB
            from .radial_functions import visc as _visc, beta as _beta_rf
            v_icb = _visc[n_r_max - 1].item()
            beta_icb = _beta_rf[n_r_max - 1].item()
            dz10_icb = f.dz_LMloc[_l1m0, n_r_max - 1].real.item()
            d.domega_ic_dt.impl[0] = -v_icb * ((two * or1[n_r_max - 1].item() + beta_icb) * z10_icb - dz10_icb)
            # omega_ic = c_z10_omega_ic * z10(ICB) — should be 0 at init
            f.omega_ic = c_z10_omega_ic * z10_icb


def _radial_loop_chunked():
    """Chunked version of radial_loop for reduced GPU memory.

    Processes radial levels in chunks of size C instead of all N at once.
    Grid-space memory is O(C) instead of O(N). Spectral arrays remain full (lm, N).

    Operations INSIDE the chunk loop (independent per radial level):
      - Inverse SHT, get_nl, forward SHT
    Operations OUTSIDE (need full spectral arrays):
      - get_td, finish_exp_*, CFL check, IC rotation, movie captures
    """
    f = fields
    N = n_r_max
    C = radial_chunk_size

    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0

    # Pre-allocate spectral output arrays (lm, N) with zeros
    AdvrLM = torch.zeros(lm_max, N, dtype=CDTYPE, device=DEVICE)
    dVSrLM = torch.zeros_like(AdvrLM)
    AdvtLM = torch.zeros_like(AdvrLM)
    AdvpLM = torch.zeros_like(AdvrLM)
    VStLM = torch.zeros_like(AdvrLM)
    VxBrLM = torch.zeros_like(AdvrLM) if l_mag else None
    VxBtLM = torch.zeros_like(AdvrLM) if l_mag else None
    VxBpLM = torch.zeros_like(AdvrLM) if l_mag else None
    dVXirLM = torch.zeros_like(AdvrLM) if l_chemical_conv else None
    VXitLM = torch.zeros_like(AdvrLM) if l_chemical_conv else None

    # Running CFL minimums and ICB slices
    dtrkc_min = 1e10
    dthkc_min = 1e10
    brc_icb = None  # (nθ, nφ) slice at ICB for IC coupling
    bpc_icb = None

    for start in range(0, N, C):
        end = min(start + C, N)
        actual_C = end - start
        need_pad = actual_C < C

        # --- 1. Inverse SHT (spectral[:, start:end] → grid (actual_C, nθ, nφ)) ---
        prof.start("radial_loop.inv_sht")
        # Scalars
        if l_heat or l_chemical_conv:
            scal_inv_list = []
            if l_heat:
                s_chunk = f.s_LMloc[:, start:end]
                if need_pad:
                    s_chunk = torch.nn.functional.pad(s_chunk, (0, C - actual_C))
                scal_inv_list.append(s_chunk)
            if l_chemical_conv:
                xi_chunk = f.xi_LMloc[:, start:end]
                if need_pad:
                    xi_chunk = torch.nn.functional.pad(xi_chunk, (0, C - actual_C))
                scal_inv_list.append(xi_chunk)
            scal_inv_all = scal_to_spat(torch.cat(scal_inv_list, dim=1))
            off = 0
            if l_heat:
                sc_chunk = scal_inv_all[:C]
                off = C
            else:
                sc_chunk = torch.zeros(C, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
            if l_chemical_conv:
                xic_chunk = scal_inv_all[off:off + C]
            else:
                xic_chunk = torch.zeros(C, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        else:
            sc_chunk = torch.zeros(C, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
            xic_chunk = torch.zeros(C, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)

        # Vectors (magnetic + velocity + curls)
        Q_parts = []
        S_parts = []
        T_parts = []

        or2_chunk_2d = or2[start:end].unsqueeze(0).to(CDTYPE)  # (1, actual_C)
        if need_pad:
            or2_chunk_2d = torch.nn.functional.pad(or2_chunk_2d, (0, C - actual_C))

        if l_mag:
            b_chunk = f.b_LMloc[:, start:end]
            db_chunk = f.db_LMloc[:, start:end]
            aj_chunk = f.aj_LMloc[:, start:end]
            dj_chunk = f.dj_LMloc[:, start:end]
            ddb_chunk = f.ddb_LMloc[:, start:end]
            if need_pad:
                b_chunk = torch.nn.functional.pad(b_chunk, (0, C - actual_C))
                db_chunk = torch.nn.functional.pad(db_chunk, (0, C - actual_C))
                aj_chunk = torch.nn.functional.pad(aj_chunk, (0, C - actual_C))
                dj_chunk = torch.nn.functional.pad(dj_chunk, (0, C - actual_C))
                ddb_chunk = torch.nn.functional.pad(ddb_chunk, (0, C - actual_C))
            Q_parts.append(_dLh_2d * b_chunk)
            S_parts.append(db_chunk)
            T_parts.append(aj_chunk)
            Q_parts.append(_dLh_2d * aj_chunk)
            S_parts.append(dj_chunk)
            T_parts.append(or2_chunk_2d * _dLh_2d * b_chunk - ddb_chunk)

        w_chunk = f.w_LMloc[:, start:end]
        dw_chunk = f.dw_LMloc[:, start:end]
        z_chunk = f.z_LMloc[:, start:end]
        dz_chunk = f.dz_LMloc[:, start:end]
        ddw_chunk = f.ddw_LMloc[:, start:end]
        if need_pad:
            w_chunk = torch.nn.functional.pad(w_chunk, (0, C - actual_C))
            dw_chunk = torch.nn.functional.pad(dw_chunk, (0, C - actual_C))
            z_chunk = torch.nn.functional.pad(z_chunk, (0, C - actual_C))
            dz_chunk = torch.nn.functional.pad(dz_chunk, (0, C - actual_C))
            ddw_chunk = torch.nn.functional.pad(ddw_chunk, (0, C - actual_C))
        Q_parts.append(_dLh_2d * w_chunk)
        S_parts.append(dw_chunk)
        T_parts.append(z_chunk)
        Q_parts.append(_dLh_2d * z_chunk)
        S_parts.append(dz_chunk)
        T_parts.append(or2_chunk_2d * _dLh_2d * w_chunk - ddw_chunk)

        Q_all = torch.cat(Q_parts, dim=1)
        S_all = torch.cat(S_parts, dim=1)
        T_all = torch.cat(T_parts, dim=1)
        all_r, all_t, all_p = torpol_to_spat(Q_all, S_all, T_all)

        # Split results
        offset = 0
        if l_mag:
            brc_chunk = all_r[offset:offset + C]
            btc_chunk = all_t[offset:offset + C]
            bpc_chunk = all_p[offset:offset + C]
            offset += C
            cbrc_chunk = all_r[offset:offset + C]
            cbtc_chunk = all_t[offset:offset + C]
            cbpc_chunk = all_p[offset:offset + C]
            offset += C
        else:
            _zeros_chunk = torch.zeros(C, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
            brc_chunk = btc_chunk = bpc_chunk = _zeros_chunk
            cbrc_chunk = cbtc_chunk = cbpc_chunk = _zeros_chunk

        vrc_chunk = all_r[offset:offset + C]
        vtc_chunk = all_t[offset:offset + C]
        vpc_chunk = all_p[offset:offset + C]
        offset += C
        cvrc_chunk = all_r[offset:offset + C]
        cvtc_chunk = all_t[offset:offset + C]
        cvpc_chunk = all_p[offset:offset + C]

        # --- 2. Nonlinear products (chunk-sized) ---
        prof.stop("radial_loop.inv_sht")
        prof.start("radial_loop.get_nl")
        or4_chunk = _or4_3_full[start:end]
        or2_chunk_3 = _or2_3_full[start:end]
        if need_pad:
            or4_chunk = torch.nn.functional.pad(or4_chunk, (0, 0, 0, 0, 0, C - actual_C))
            or2_chunk_3 = torch.nn.functional.pad(or2_chunk_3, (0, 0, 0, 0, 0, C - actual_C))
        nl_result = get_nl(
            vrc_chunk, vtc_chunk, vpc_chunk, cvrc_chunk, cvtc_chunk, cvpc_chunk,
            sc_chunk, brc_chunk, btc_chunk, bpc_chunk, cbrc_chunk, cbtc_chunk, cbpc_chunk,
            xic_chunk, or4_chunk, or2_chunk_3)
        Advr, Advt, Advp, VSr, VSt, VSp, VxBr, VxBt, VxBp, VXir, VXit, VXip = nl_result
        prof.stop("radial_loop.get_nl")

        # --- Per-chunk CFL check (on unpadded data) ---
        dtrkc_c, dthkc_c = courant_check(
            vrc_chunk[:actual_C], vtc_chunk[:actual_C], vpc_chunk[:actual_C],
            brc_chunk[:actual_C] if l_mag else None,
            btc_chunk[:actual_C] if l_mag else None,
            bpc_chunk[:actual_C] if l_mag else None,
            courfac=tscheme.courfac, alffac=tscheme.alffac,
            or4_3=_or4_3_full[start:end],
            or2_3=_or2_3_full[start:end],
            orho1_3=_orho1_3_full[start:end],
            orho2_3=_orho2_3_full[start:end],
            delxr2_in=_delxr2[start:end],
            delxh2_in=_delxh2[start:end])
        dtrkc_min = min(dtrkc_min, dtrkc_c.min().item())
        dthkc_min = min(dthkc_min, dthkc_c.min().item())

        # --- Save ICB slices for IC coupling ---
        if l_mag and (l_cond_ic or l_rot_ic) and start <= N - 1 < end:
            local_icb = N - 1 - start
            brc_icb = brc_chunk[local_icb].clone()
            bpc_icb = bpc_chunk[local_icb].clone()

        # --- 3. Forward SHT on bulk portion of chunk ---
        prof.start("radial_loop.fwd_sht")
        bulk_start = max(1, start)
        bulk_end = min(N - 1, end)
        if bulk_end > bulk_start:
            # Local indices within the chunk
            local_bs = bulk_start - start
            local_be = bulk_end - start
            Nb_chunk = local_be - local_bs

            # Scalar forward SHT
            scal_fwd_list = [Advr[local_bs:local_be], VSr[local_bs:local_be]]
            if l_mag:
                scal_fwd_list.append(VxBr[local_bs:local_be])
            if l_chemical_conv:
                scal_fwd_list.append(VXir[local_bs:local_be])
            scal_in = torch.cat(scal_fwd_list, dim=0)
            scal_out = scal_to_SH(scal_in)

            s_offset = 0
            AdvrLM[:, bulk_start:bulk_end] = scal_out[:, s_offset:s_offset + Nb_chunk]; s_offset += Nb_chunk
            dVSrLM[:, bulk_start:bulk_end] = scal_out[:, s_offset:s_offset + Nb_chunk]; s_offset += Nb_chunk
            if l_mag:
                VxBrLM[:, bulk_start:bulk_end] = scal_out[:, s_offset:s_offset + Nb_chunk]; s_offset += Nb_chunk
            if l_chemical_conv:
                dVXirLM[:, bulk_start:bulk_end] = scal_out[:, s_offset:s_offset + Nb_chunk]; s_offset += Nb_chunk

            # Vector forward SHT
            vt_fwd_list = [Advt[local_bs:local_be], VSt[local_bs:local_be]]
            vp_fwd_list = [Advp[local_bs:local_be], VSp[local_bs:local_be]]
            if l_mag:
                vt_fwd_list.append(VxBt[local_bs:local_be])
                vp_fwd_list.append(VxBp[local_bs:local_be])
            if l_chemical_conv:
                vt_fwd_list.append(VXit[local_bs:local_be])
                vp_fwd_list.append(VXip[local_bs:local_be])
            vt_in = torch.cat(vt_fwd_list, dim=0)
            vp_in = torch.cat(vp_fwd_list, dim=0)
            S_out, T_out = spat_to_sphertor(vt_in, vp_in)

            v_offset = 0
            AdvtLM[:, bulk_start:bulk_end] = S_out[:, v_offset:v_offset + Nb_chunk]
            AdvpLM[:, bulk_start:bulk_end] = T_out[:, v_offset:v_offset + Nb_chunk]
            v_offset += Nb_chunk
            VStLM[:, bulk_start:bulk_end] = S_out[:, v_offset:v_offset + Nb_chunk]
            v_offset += Nb_chunk
            if l_mag:
                VxBtLM[:, bulk_start:bulk_end] = S_out[:, v_offset:v_offset + Nb_chunk]
                VxBpLM[:, bulk_start:bulk_end] = T_out[:, v_offset:v_offset + Nb_chunk]
                v_offset += Nb_chunk
            if l_chemical_conv:
                VXitLM[:, bulk_start:bulk_end] = S_out[:, v_offset:v_offset + Nb_chunk]
                v_offset += Nb_chunk
        prof.stop("radial_loop.fwd_sht")

    # === After all chunks ===
    # Movie captures: not available in chunked mode (grid arrays are chunk-local)
    global last_brc, last_btc, last_bpc, last_sc
    last_brc = last_btc = last_bpc = last_sc = None

    # === 3b. Boundary VxBt from rigid rotation (conducting IC, uses saved ICB slice) ===
    if l_mag and l_cond_ic and f.omega_ic != 0.0:
        from .horizontal_data import sinTheta_grid
        icb_nr = N - 1
        vpc_icb_grid = _r_icb_py ** 2 * _orho1_icb_py * sinTheta_grid ** 2 * f.omega_ic
        VxBt_grid_icb = (_or4_icb_py * _orho1_icb_py
                         * vpc_icb_grid.unsqueeze(1) * brc_icb)
        VxBtLM_icb_S, _ = spat_to_sphertor(
            VxBt_grid_icb.unsqueeze(0),
            torch.zeros_like(VxBt_grid_icb).unsqueeze(0)
        )
        VxBtLM[:, icb_nr] = VxBtLM_icb_S[:, 0]

    # === 4. Time derivative assembly (full spectral arrays) ===
    prof.start("radial_loop.get_td")
    if l_double_curl:
        dwdt_expl, dVxVhLM = get_dwdt_double_curl(
            AdvrLM, AdvtLM, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
            f.z_LMloc, f.dz_LMloc)
        dwdt_expl = finish_exp_pol(dwdt_expl, dVxVhLM)
    else:
        dwdt_expl = get_dwdt(AdvrLM, f.dw_LMloc, f.z_LMloc)
    dt_fields.dwdt.expl[:, :, expl_idx] = dwdt_expl

    dzdt_expl = get_dzdt(AdvpLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
    dt_fields.dzdt.expl[:, :, expl_idx] = dzdt_expl

    if not l_double_curl:
        dpdt_expl = get_dpdt(AdvtLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
        dt_fields.dpdt.expl[:, :, expl_idx] = dpdt_expl

    if l_heat:
        dsdt_partial, dVSrLM_out = get_dsdt(VStLM, dVSrLM)
        dsdt_expl = finish_exp_entropy(dsdt_partial, dVSrLM_out)
        dt_fields.dsdt.expl[:, :, expl_idx] = dsdt_expl

    if l_chemical_conv:
        dxidt_partial, dVXirLM_out = get_dxidt(VXitLM, dVXirLM)
        dxidt_expl = finish_exp_comp(dxidt_partial, dVXirLM_out)
        dt_fields.dxidt.expl[:, :, expl_idx] = dxidt_expl

    if l_mag:
        dbdt_expl, djdt_partial, dVxBhLM = get_dbdt(VxBrLM, VxBtLM, VxBpLM)
        djdt_expl = finish_exp_mag(djdt_partial, dVxBhLM)
        dt_fields.dbdt.expl[:, :, expl_idx] = dbdt_expl
        dt_fields.djdt.expl[:, :, expl_idx] = djdt_expl

    prof.stop("radial_loop.get_td")
    # === 5. IC rotation (uses saved ICB slices) ===
    global lorentz_torque_ic_last
    if l_mag and l_rot_ic and l_cond_ic:
        from .horizontal_data import gauss_grid
        fac_lt = LFfac * two * math.pi / float(n_phi_max)
        lorentz_torque_ic = fac_lt * (gauss_grid.unsqueeze(1) * brc_icb * bpc_icb).sum().item()
        lorentz_torque_ic_last = lorentz_torque_ic
        domega_ic_exp = finish_exp_tor(f.omega_ic, lorentz_torque_ic)
        dt_fields.domega_ic_dt.expl[expl_idx] = domega_ic_exp

    # === 6. IC magnetic advection ===
    if l_mag and l_cond_ic:
        finish_exp_mag_ic(
            f.b_ic, f.aj_ic, f.omega_ic,
            dt_fields.dbdt_ic.expl[:, :, expl_idx],
            dt_fields.djdt_ic.expl[:, :, expl_idx]
        )

    return dtrkc_min, dthkc_min


def radial_loop():
    """Inverse SHT → nonlinear products → forward SHT → time derivative assembly.

    For the Boussinesq benchmark with l_adv_curl=.true., no-slip BCs.
    At boundaries (nR=0 and nR=N-1), velocity is zero (omega=0)
    and nonlinear terms are not computed (get_td zeros boundary terms).

    All SHT calls are batched over radial levels to minimize Python overhead.
    Multiple transforms are combined into single calls where possible.
    Dynamic batching based on l_mag and l_chemical_conv flags.

    When radial_chunk_size > 0, dispatches to _radial_loop_chunked() to
    reduce peak GPU memory by processing radial levels in chunks.
    """
    if 0 < radial_chunk_size < n_r_max:
        return _radial_loop_chunked()
    f = fields
    N = n_r_max
    Nb = N - 2
    bulk = slice(1, N - 1)  # interior radial levels

    # Determine expl storage index:
    # CNAB2: istage=0, expl at 0; BPR353: istage=1..4, expl at istage-1
    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0

    # === 1. Inverse SHT (batched) ===
    prof.start("radial_loop.inv_sht")
    # Scalars: entropy (all N levels), plus composition if active
    if l_heat or l_chemical_conv:
        scal_inv_list = []
        if l_heat:
            scal_inv_list.append(f.s_LMloc)
        if l_chemical_conv:
            scal_inv_list.append(f.xi_LMloc)
        scal_inv_all = scal_to_spat(torch.cat(scal_inv_list, dim=1))
        offset_sc = 0
        if l_heat:
            sc = scal_inv_all[:N]
            offset_sc = N
        else:
            sc = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        if l_chemical_conv:
            xic = scal_inv_all[offset_sc:offset_sc + N]
        else:
            xic = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
    else:
        sc = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        xic = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)

    # Build torpol_to_spat inputs dynamically
    Q_parts = []
    S_parts = []
    T_parts = []
    part_sizes = []  # track sizes for splitting

    if l_mag:
        # Magnetic field (all N levels)
        Q_parts.append(_dLh_2d * f.b_LMloc)
        S_parts.append(f.db_LMloc)
        T_parts.append(f.aj_LMloc)
        part_sizes.append(N)

        # Curl of magnetic field (all N levels)
        Q_parts.append(_dLh_2d * f.aj_LMloc)
        S_parts.append(f.dj_LMloc)
        T_parts.append(_or2_2d * _dLh_2d * f.b_LMloc - f.ddb_LMloc)
        part_sizes.append(N)

    # Velocity (all N levels — need boundaries for CFL check, e.g. ICB with l_SRIC)
    Q_parts.append(_dLh_2d * f.w_LMloc)
    S_parts.append(f.dw_LMloc)
    T_parts.append(f.z_LMloc)
    part_sizes.append(N)

    # Curl of velocity (all N levels)
    Q_parts.append(_dLh_2d * f.z_LMloc)
    S_parts.append(f.dz_LMloc)
    T_parts.append(_or2_2d * _dLh_2d * f.w_LMloc - f.ddw_LMloc)
    part_sizes.append(N)

    Q_all = torch.cat(Q_parts, dim=1)
    S_all = torch.cat(S_parts, dim=1)
    T_all = torch.cat(T_parts, dim=1)

    all_r, all_t, all_p = torpol_to_spat(Q_all, S_all, T_all)

    # Split results based on part_sizes
    offset = 0
    if l_mag:
        brc = all_r[offset:offset + N]
        btc = all_t[offset:offset + N]
        bpc = all_p[offset:offset + N]
        offset += N

        cbrc = all_r[offset:offset + N]
        cbtc = all_t[offset:offset + N]
        cbpc = all_p[offset:offset + N]
        offset += N
    else:
        # Zero magnetic fields when mode=1
        _zeros_grid = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        brc = btc = bpc = _zeros_grid
        cbrc = cbtc = cbpc = _zeros_grid

    # Fill velocity buffers (all levels including boundaries for CFL check)
    _vrc[:] = all_r[offset:offset + N]
    _vtc[:] = all_t[offset:offset + N]
    _vpc[:] = all_p[offset:offset + N]
    offset += N
    _cvrc[:] = all_r[offset:offset + N]
    _cvtc[:] = all_t[offset:offset + N]
    _cvpc[:] = all_p[offset:offset + N]

    # === 2. Nonlinear products (all radial levels at once) ===
    prof.stop("radial_loop.inv_sht")
    prof.start("radial_loop.get_nl")
    nl_result = get_nl(
        _vrc, _vtc, _vpc, _cvrc, _cvtc, _cvpc,
        sc, brc, btc, bpc, cbrc, cbtc, cbpc, xic)
    Advr, Advt, Advp, VSr, VSt, VSp, VxBr, VxBt, VxBp, VXir, VXit, VXip = nl_result
    prof.stop("radial_loop.get_nl")

    # Persist grid-space fields for movie output
    global last_brc, last_btc, last_bpc, last_sc
    if l_mag:
        last_brc = brc
        last_btc = btc
        last_bpc = bpc
    last_sc = sc

    # === 3. Forward SHT (batched) ===
    prof.start("radial_loop.fwd_sht")
    # For FD: all N levels (Fortran nBc=0 at boundaries, needed for p0 RHS)
    # For Chebyshev: interior only (bulk), boundaries stay zero (nBc != 0)
    scal_fwd_list = [Advr[bulk], VSr[bulk]]
    if l_mag:
        scal_fwd_list.append(VxBr[bulk])
    if l_chemical_conv:
        scal_fwd_list.append(VXir[bulk])
    scal_in = torch.cat(scal_fwd_list, dim=0)
    scal_out = scal_to_SH(scal_in)

    # Build vector (S,T-component) batch
    vt_fwd_list = [Advt[bulk], VSt[bulk]]
    vp_fwd_list = [Advp[bulk], VSp[bulk]]
    if l_mag:
        vt_fwd_list.append(VxBt[bulk])
        vp_fwd_list.append(VxBp[bulk])
    if l_chemical_conv:
        vt_fwd_list.append(VXit[bulk])
        vp_fwd_list.append(VXip[bulk])
    vt_in = torch.cat(vt_fwd_list, dim=0)
    vp_in = torch.cat(vp_fwd_list, dim=0)
    S_out, T_out = spat_to_sphertor(vt_in, vp_in)

    # Spectral work arrays: (lm_max, n_r_max) — boundaries zero for Chebyshev
    AdvrLM = torch.zeros(lm_max, N, dtype=CDTYPE, device=DEVICE)
    AdvtLM = torch.zeros_like(AdvrLM)
    AdvpLM = torch.zeros_like(AdvrLM)
    dVSrLM = torch.zeros_like(AdvrLM)
    VStLM = torch.zeros_like(AdvrLM)

    # Scatter batched scalar results
    s_offset = 0
    AdvrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    dVSrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb

    if l_mag:
        VxBrLM = torch.zeros_like(AdvrLM)
        VxBrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    if l_chemical_conv:
        dVXirLM = torch.zeros_like(AdvrLM)
        dVXirLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb

    # Scatter batched vector results
    v_offset = 0
    AdvtLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
    AdvpLM[:, bulk] = T_out[:, v_offset:v_offset + Nb]
    v_offset += Nb

    VStLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
    v_offset += Nb

    if l_mag:
        VxBtLM = torch.zeros_like(AdvrLM)
        VxBpLM = torch.zeros_like(AdvrLM)
        VxBtLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
        VxBpLM[:, bulk] = T_out[:, v_offset:v_offset + Nb]
        v_offset += Nb

    if l_chemical_conv:
        VXitLM = torch.zeros_like(AdvrLM)
        VXitLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
        v_offset += Nb

    # === 3b. Boundary VxBt from rigid rotation (conducting IC/mantle) ===
    if l_mag and l_cond_ic and f.omega_ic != 0.0:
        from .horizontal_data import sinTheta_grid
        icb_nr = N - 1
        vpc_icb = _r_icb_py ** 2 * _orho1_icb_py * sinTheta_grid ** 2 * f.omega_ic
        VxBt_grid_icb = (_or4_icb_py * _orho1_icb_py
                         * vpc_icb.unsqueeze(1) * brc[icb_nr])
        VxBtLM_icb_S, _ = spat_to_sphertor(
            VxBt_grid_icb.unsqueeze(0),
            torch.zeros_like(VxBt_grid_icb).unsqueeze(0)
        )
        VxBtLM[:, icb_nr] = VxBtLM_icb_S[:, 0]

    # === 4. Time derivative assembly ===
    prof.stop("radial_loop.fwd_sht")
    prof.start("radial_loop.get_td")
    # Poloidal velocity
    if l_double_curl:
        dwdt_expl, dVxVhLM = get_dwdt_double_curl(
            AdvrLM, AdvtLM, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
            f.z_LMloc, f.dz_LMloc)
        dwdt_expl = finish_exp_pol(dwdt_expl, dVxVhLM)
    else:
        dwdt_expl = get_dwdt(AdvrLM, f.dw_LMloc, f.z_LMloc)
    dt_fields.dwdt.expl[:, :, expl_idx] = dwdt_expl

    # Toroidal velocity
    dzdt_expl = get_dzdt(AdvpLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
    dt_fields.dzdt.expl[:, :, expl_idx] = dzdt_expl

    # Pressure (not computed in double-curl — recovered post-hoc)
    if not l_double_curl:
        dpdt_expl = get_dpdt(AdvtLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
        dt_fields.dpdt.expl[:, :, expl_idx] = dpdt_expl

    # Entropy (partial + finish)
    if l_heat:
        dsdt_partial, dVSrLM_out = get_dsdt(VStLM, dVSrLM)
        dsdt_expl = finish_exp_entropy(dsdt_partial, dVSrLM_out)
        dt_fields.dsdt.expl[:, :, expl_idx] = dsdt_expl

    # Composition (partial + finish)
    if l_chemical_conv:
        dxidt_partial, dVXirLM_out = get_dxidt(VXitLM, dVXirLM)
        dxidt_expl = finish_exp_comp(dxidt_partial, dVXirLM_out)
        dt_fields.dxidt.expl[:, :, expl_idx] = dxidt_expl

    # Magnetic field
    if l_mag:
        dbdt_expl, djdt_partial, dVxBhLM = get_dbdt(VxBrLM, VxBtLM, VxBpLM)
        djdt_expl = finish_exp_mag(djdt_partial, dVxBhLM)
        dt_fields.dbdt.expl[:, :, expl_idx] = dbdt_expl
        dt_fields.djdt.expl[:, :, expl_idx] = djdt_expl

    # === 5. IC rotation: Lorentz torque + explicit torque ===
    global lorentz_torque_ic_last
    if l_mag and l_rot_ic and l_cond_ic:
        from .horizontal_data import gauss_grid
        icb_idx = N - 1
        brc_icb = brc[icb_idx]
        bpc_icb = bpc[icb_idx]
        fac_lt = LFfac * two * math.pi / float(n_phi_max)
        lorentz_torque_ic = fac_lt * (gauss_grid.unsqueeze(1) * brc_icb * bpc_icb).sum().item()
        lorentz_torque_ic_last = lorentz_torque_ic

        domega_ic_exp = finish_exp_tor(f.omega_ic, lorentz_torque_ic)
        dt_fields.domega_ic_dt.expl[expl_idx] = domega_ic_exp

    # === 6. IC magnetic advection by solid-body rotation ===
    if l_mag and l_cond_ic:
        finish_exp_mag_ic(
            f.b_ic, f.aj_ic, f.omega_ic,
            dt_fields.dbdt_ic.expl[:, :, expl_idx],
            dt_fields.djdt_ic.expl[:, :, expl_idx]
        )

    # === 7. CFL Courant condition ===
    prof.stop("radial_loop.get_td")
    dtrkc, dthkc = courant_check(
        _vrc, _vtc, _vpc,
        brc if l_mag else None,
        btc if l_mag else None,
        bpc if l_mag else None,
        courfac=tscheme.courfac,
        alffac=tscheme.alffac,
    )
    return dtrkc.min().item(), dthkc.min().item()


def radial_loop_anel():
    """Inverse SHT → non-curl advection + Lorentz + viscous heating → forward SHT → time derivative assembly.

    Anelastic path (l_anel=True, l_adv_curl=False).
    Uses u·∇u advection formulation with 11 grid-space velocity fields.
    Includes magnetic field (Lorentz force + induction) when l_mag=True.
    """
    from .get_nl_anel import get_nl_anel
    from .radial_functions import temp0 as _temp0_rf

    f = fields
    N = n_r_max
    Nb = N - 2
    bulk = slice(1, N - 1)

    expl_idx = max(0, tscheme.istage - 1) if tscheme.nstages > 1 else 0

    # === 1. Inverse SHT (entropy + composition) ===
    scal_inv_list = [f.s_LMloc]
    if l_chemical_conv:
        scal_inv_list.append(f.xi_LMloc)
    scal_inv_all = scal_to_spat(torch.cat(scal_inv_list, dim=1))
    sc = scal_inv_all[:N]
    if l_chemical_conv:
        xic = scal_inv_all[N:2 * N]
    else:
        xic = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)

    # === 1b. Inverse SHT for magnetic field (all N levels) ===
    if l_mag:
        Q_mag_parts = [_dLh_2d * f.b_LMloc, _dLh_2d * f.aj_LMloc]
        S_mag_parts = [f.db_LMloc, f.dj_LMloc]
        T_mag_parts = [f.aj_LMloc, _or2_2d * _dLh_2d * f.b_LMloc - f.ddb_LMloc]
        Q_mag = torch.cat(Q_mag_parts, dim=1)
        S_mag = torch.cat(S_mag_parts, dim=1)
        T_mag = torch.cat(T_mag_parts, dim=1)
        mag_r, mag_t, mag_p = torpol_to_spat(Q_mag, S_mag, T_mag)
        brc = mag_r[:N]
        btc = mag_t[:N]
        bpc = mag_p[:N]
        cbrc = mag_r[N:2 * N]
        cbtc = mag_t[N:2 * N]
        cbpc = mag_p[N:2 * N]
    else:
        _zeros_grid = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        brc = btc = bpc = _zeros_grid
        cbrc = cbtc = cbpc = _zeros_grid

    # === 2. Inverse SHT for velocity (bulk only) ===
    Q_vel = _dLh_2d * f.w_LMloc[:, bulk]
    S_vel = f.dw_LMloc[:, bulk]
    T_vel = f.z_LMloc[:, bulk]
    vrc_bulk, vtc_bulk, vpc_bulk = torpol_to_spat(Q_vel, S_vel, T_vel)

    # Fill pre-allocated buffers (boundaries stay zero = stress-free vr=0)
    _vrc[bulk] = vrc_bulk
    _vtc[bulk] = vtc_bulk
    _vpc[bulk] = vpc_bulk

    # === 3. Additional SHTs for non-curl advection (bulk only) ===
    # torpol_to_spat(dw, ddw, dz) → dvrdrc, dvtdrc, dvpdrc
    Q_dv = _dLh_2d * f.dw_LMloc[:, bulk]
    S_dv = f.ddw_LMloc[:, bulk]
    T_dv = f.dz_LMloc[:, bulk]
    dvrdrc_bulk, dvtdrc_bulk, dvpdrc_bulk = torpol_to_spat(Q_dv, S_dv, T_dv)

    # pol_to_curlr_spat(z) → cvrc
    cvrc_bulk = pol_to_curlr_spat(f.z_LMloc[:, bulk])

    # pol_to_grad_spat(w) → dvrdtc, dvrdpc  (batched over radial levels)
    dvrdtc_bulk, dvrdpc_bulk = pol_to_grad_spat(f.w_LMloc[:, bulk])

    # torpol_to_dphspat(dw, z) → dvtdpc, dvpdpc  (inline via batched torpol_to_spat)
    from .horizontal_data import dPhi, O_sin_theta_E2_grid
    dPhi_2d = dPhi.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
    Slm_dph = dPhi_2d * f.dw_LMloc[:, bulk]
    Tlm_dph = dPhi_2d * f.z_LMloc[:, bulk]
    Qlm_dph = torch.zeros_like(Slm_dph)
    _, dvtdpc_raw, dvpdpc_raw = torpol_to_spat(Qlm_dph, Slm_dph, Tlm_dph)
    # Multiply by 1/sin²θ (grid layout)
    Ost2_grid = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)  # (1, n_theta, 1)
    dvtdpc_bulk = dvtdpc_raw * Ost2_grid
    dvpdpc_bulk = dvpdpc_raw * Ost2_grid

    # Pre-allocated derivative buffers (boundaries zero)
    dvrdrc = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
    dvtdrc = torch.zeros_like(dvrdrc)
    dvpdrc = torch.zeros_like(dvrdrc)
    dvrdtc = torch.zeros_like(dvrdrc)
    dvrdpc = torch.zeros_like(dvrdrc)
    dvtdpc = torch.zeros_like(dvrdrc)
    dvpdpc = torch.zeros_like(dvrdrc)
    cvrc_full = torch.zeros_like(dvrdrc)

    dvrdrc[bulk] = dvrdrc_bulk
    dvtdrc[bulk] = dvtdrc_bulk
    dvpdrc[bulk] = dvpdrc_bulk
    dvrdtc[bulk] = dvrdtc_bulk
    dvrdpc[bulk] = dvrdpc_bulk
    dvtdpc[bulk] = dvtdpc_bulk
    dvpdpc[bulk] = dvpdpc_bulk
    cvrc_full[bulk] = cvrc_bulk

    # === 4. Nonlinear products ===
    nl_result = get_nl_anel(
        _vrc, _vtc, _vpc,
        dvrdrc, dvtdrc, dvpdrc,
        dvrdtc, dvrdpc,
        dvtdpc, dvpdpc,
        cvrc_full,
        sc, xic,
        brc, btc, bpc, cbrc, cbtc, cbpc)
    (Advr, Advt, Advp, VSr, VSt, VSp,
     VxBr, VxBt, VxBp, VXir, VXit, VXip, heatTerms) = nl_result

    # === 5. Forward SHT (bulk only) ===
    # Scalars: Advr, VSr, [VxBr], [VXir], heatTerms
    scal_fwd_list = [Advr[bulk], VSr[bulk]]
    if l_mag:
        scal_fwd_list.append(VxBr[bulk])
    if l_chemical_conv:
        scal_fwd_list.append(VXir[bulk])
    scal_fwd_list.append(heatTerms[bulk])
    scal_in = torch.cat(scal_fwd_list, dim=0)
    scal_out = scal_to_SH(scal_in)

    # Vectors: Advt/Advp, VSt/VSp, [VxBt/VxBp], [VXit/VXip]
    vt_fwd_list = [Advt[bulk], VSt[bulk]]
    vp_fwd_list = [Advp[bulk], VSp[bulk]]
    if l_mag:
        vt_fwd_list.append(VxBt[bulk])
        vp_fwd_list.append(VxBp[bulk])
    if l_chemical_conv:
        vt_fwd_list.append(VXit[bulk])
        vp_fwd_list.append(VXip[bulk])
    vt_in = torch.cat(vt_fwd_list, dim=0)
    vp_in = torch.cat(vp_fwd_list, dim=0)
    S_out, T_out = spat_to_sphertor(vt_in, vp_in)

    # Spectral work arrays: (lm_max, n_r_max) — boundaries stay zero
    AdvrLM = torch.zeros(lm_max, N, dtype=CDTYPE, device=DEVICE)
    AdvtLM = torch.zeros_like(AdvrLM)
    AdvpLM = torch.zeros_like(AdvrLM)
    dVSrLM = torch.zeros_like(AdvrLM)
    VStLM = torch.zeros_like(AdvrLM)
    heatTermsLM = torch.zeros_like(AdvrLM)

    # Scatter batched scalar results
    s_offset = 0
    AdvrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    dVSrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    if l_mag:
        VxBrLM = torch.zeros_like(AdvrLM)
        VxBrLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    if l_chemical_conv:
        dVXirLM = torch.zeros_like(AdvrLM)
        dVXirLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb
    heatTermsLM[:, bulk] = scal_out[:, s_offset:s_offset + Nb]; s_offset += Nb

    # Scatter batched vector results
    v_offset = 0
    AdvtLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
    AdvpLM[:, bulk] = T_out[:, v_offset:v_offset + Nb]
    v_offset += Nb

    VStLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
    v_offset += Nb

    if l_mag:
        VxBtLM = torch.zeros_like(AdvrLM)
        VxBpLM = torch.zeros_like(AdvrLM)
        VxBtLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
        VxBpLM[:, bulk] = T_out[:, v_offset:v_offset + Nb]
        v_offset += Nb

    if l_chemical_conv:
        VXitLM = torch.zeros_like(AdvrLM)
        VXitLM[:, bulk] = S_out[:, v_offset:v_offset + Nb]
        v_offset += Nb

    # === 5b. Boundary VxBt from rigid rotation (conducting IC/mantle) ===
    if l_mag and l_cond_ic and f.omega_ic != 0.0:
        from .horizontal_data import sinTheta_grid
        icb_nr = N - 1
        vpc_icb = _r_icb_py ** 2 * _orho1_icb_py * sinTheta_grid ** 2 * f.omega_ic
        VxBt_grid_icb = (_or4_icb_py * _orho1_icb_py
                         * vpc_icb.unsqueeze(1) * brc[icb_nr])
        VxBtLM_icb_S, _ = spat_to_sphertor(
            VxBt_grid_icb.unsqueeze(0),
            torch.zeros_like(VxBt_grid_icb).unsqueeze(0)
        )
        VxBtLM[:, icb_nr] = VxBtLM_icb_S[:, 0]

    # === 6. Time derivative assembly ===
    if l_double_curl:
        dwdt_expl, dVxVhLM = get_dwdt_double_curl(
            AdvrLM, AdvtLM, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
            f.z_LMloc, f.dz_LMloc)
        dwdt_expl = finish_exp_pol(dwdt_expl, dVxVhLM)
    else:
        dwdt_expl = get_dwdt(AdvrLM, f.dw_LMloc, f.z_LMloc)
    dt_fields.dwdt.expl[:, :, expl_idx] = dwdt_expl

    dzdt_expl = get_dzdt(AdvpLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
    dt_fields.dzdt.expl[:, :, expl_idx] = dzdt_expl

    if not l_double_curl:
        dpdt_expl = get_dpdt(AdvtLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
        dt_fields.dpdt.expl[:, :, expl_idx] = dpdt_expl

    # Entropy: dsdt + heatTermsLM (get_td.f90:485,499)
    dsdt_partial, dVSrLM_out = get_dsdt(VStLM, dVSrLM)
    dsdt_partial = dsdt_partial + heatTermsLM  # viscous heating source
    dsdt_expl = finish_exp_entropy(dsdt_partial, dVSrLM_out)
    dt_fields.dsdt.expl[:, :, expl_idx] = dsdt_expl

    # Composition
    if l_chemical_conv:
        dxidt_partial, dVXirLM_out = get_dxidt(VXitLM, dVXirLM)
        dxidt_expl = finish_exp_comp(dxidt_partial, dVXirLM_out)
        dt_fields.dxidt.expl[:, :, expl_idx] = dxidt_expl

    # Magnetic field
    if l_mag:
        dbdt_expl, djdt_partial, dVxBhLM = get_dbdt(VxBrLM, VxBtLM, VxBpLM)
        djdt_expl = finish_exp_mag(djdt_partial, dVxBhLM)
        dt_fields.dbdt.expl[:, :, expl_idx] = dbdt_expl
        dt_fields.djdt.expl[:, :, expl_idx] = djdt_expl

    # === 6b. IC rotation: Lorentz torque + explicit torque ===
    global lorentz_torque_ic_last
    if l_mag and l_rot_ic and l_cond_ic:
        from .horizontal_data import gauss_grid
        icb_idx = N - 1
        brc_icb = brc[icb_idx]
        bpc_icb = bpc[icb_idx]
        fac_lt = LFfac * two * math.pi / float(n_phi_max)
        lorentz_torque_ic = fac_lt * (gauss_grid.unsqueeze(1) * brc_icb * bpc_icb).sum().item()
        lorentz_torque_ic_last = lorentz_torque_ic

        domega_ic_exp = finish_exp_tor(f.omega_ic, lorentz_torque_ic)
        dt_fields.domega_ic_dt.expl[expl_idx] = domega_ic_exp

    # === 6c. IC magnetic advection by solid-body rotation ===
    if l_mag and l_cond_ic:
        finish_exp_mag_ic(
            f.b_ic, f.aj_ic, f.omega_ic,
            dt_fields.dbdt_ic.expl[:, :, expl_idx],
            dt_fields.djdt_ic.expl[:, :, expl_idx]
        )

    # === 7. CFL Courant condition ===
    dtrkc, dthkc = courant_check(
        _vrc, _vtc, _vpc,
        brc if l_mag else None,
        btc if l_mag else None,
        bpc if l_mag else None,
        courfac=tscheme.courfac,
        alffac=tscheme.alffac,
    )
    return dtrkc.min().item(), dthkc.min().item()


def lm_loop():
    """Implicit solves: S → Xi → Z → WP → B.

    Uses fully fused path when l_z10mat is False: all scalar + WP solvers
    combined with cat-free RHS, deferred Z costf, and unified D123 matmul.
    Falls back to sequential calls when l_z10mat is True (IC rotation coupling).
    """
    f = fields
    d = dt_fields

    prof.start("lm_loop.solve_szw")
    if (l_heat and not l_z10mat and not l_anel
            and not l_correct_AMz and not l_correct_AMe and not l_double_curl):
        # Fused path: S + [Xi] + Z + WP all in one optimized pipeline
        # Not used for anelastic, AM corrections, or double-curl
        _fused_lm_loop_fast()
    else:
        # Sequential path
        if l_heat:
            updateS(f.s_LMloc, f.ds_LMloc, d.dsdt, tscheme)
        if l_chemical_conv:
            updateXi(f.xi_LMloc, f.dxi_LMloc, d.dxidt, tscheme)
        omega_ic_ref = [f.omega_ic]
        updateZ(f.z_LMloc, f.dz_LMloc, d.dzdt, tscheme,
                domega_ic_dt=d.domega_ic_dt, omega_ic_ref=omega_ic_ref)
        f.omega_ic = omega_ic_ref[0]
        if l_double_curl:
            updateW(f.s_LMloc, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
                    d.dwdt, f.p_LMloc, f.dp_LMloc, tscheme,
                    xi_LMloc=f.xi_LMloc if l_chemical_conv else None)
        else:
            updateWP(f.s_LMloc, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
                     d.dwdt, f.p_LMloc, f.dp_LMloc, d.dpdt, tscheme,
                     xi_LMloc=f.xi_LMloc if l_chemical_conv else None)
    prof.stop("lm_loop.solve_szw")

    prof.start("lm_loop.solve_b")
    if l_mag:
        if l_cond_ic:
            updateB(f.b_LMloc, f.db_LMloc, f.ddb_LMloc,
                    f.aj_LMloc, f.dj_LMloc, f.ddj_LMloc,
                    d.dbdt, d.djdt, tscheme,
                    f.b_ic, f.db_ic, f.ddb_ic,
                    f.aj_ic, f.dj_ic, f.ddj_ic,
                    d.dbdt_ic, d.djdt_ic)
        else:
            updateB(f.b_LMloc, f.db_LMloc, f.ddb_LMloc,
                    f.aj_LMloc, f.dj_LMloc, f.ddj_LMloc,
                    d.dbdt, d.djdt, tscheme)
    prof.stop("lm_loop.solve_b")


def build_all_matrices():
    """Build and factorize all implicit matrices for current dt."""
    wimp_lin0 = tscheme.wimp_lin[0].item()
    if l_heat:
        build_s_matrices(wimp_lin0)
    if l_chemical_conv:
        build_xi_matrices(wimp_lin0)
    build_z_matrices(wimp_lin0)
    build_p0_matrix()
    if l_double_curl:
        build_w_matrices(wimp_lin0)
    else:
        build_wp_matrices(wimp_lin0)
    if l_mag:
        build_b_matrices(wimp_lin0)
    # Build stacked inverses for batched scalar solver (only when heat is active)
    # Skip for FD (uses banded solve, not batched inverse)
    if l_heat and not l_finite_diff:
        _init_batched_scalar()


_dtMax = 0.0  # set by initialize_dt; used as CFL upper bound


def initialize_dt(dt: float):
    """Initialize dt array to dtMax, matching startFields.f90 line 298.

    Must be called before the first time step. Fortran sets tscheme%dt(:) = dtMax
    when starting from scratch (not from a checkpoint).
    Also stores dtMax for CFL clamping.

    Applies intfac clamp (preCalculations.f90:190):
        if rotating: dtMax = min(dtMax, intfac * ekScaled)
    """
    global _dtMax
    from .pre_calculations import ekScaled
    from .params import ek
    l_non_rot = ek < 0.0
    if not l_non_rot:
        dt = min(dt, tscheme.intfac * ekScaled)
    tscheme.dt[:] = dt
    _dtMax = dt


def one_step(n_time_step: int, dt: float) -> float:
    """Execute one time step (CNAB2 single-stage or multi-stage DIRK).

    Args:
        n_time_step: current step number (1-based)
        dt: time step size

    Returns:
        dt_actual: the time step size actually used (may differ from input if CFL triggers)
    """
    if tscheme.nstages == 1:
        return _one_step_cnab2(n_time_step, dt)
    else:
        return _one_step_dirk(n_time_step, dt)


def _radial_loop_dispatch():
    """Dispatch to appropriate radial loop based on physics.

    Returns:
        (dtrkc, dthkc): per-level Courant time steps
    """
    if l_anel:
        return radial_loop_anel()
    else:
        return radial_loop()


def _one_step_cnab2(n_time_step: int, dt: float) -> float:
    """Single-stage CNAB2 time step.

    Ordering matches Fortran step_time.f90 lines 573-801:
    radialLoopG → dt_courant → set_dt_array → set_weights → matrix rebuild → LMLoop

    Returns:
        dt_new: the time step size used for the implicit solve
    """
    # 1. Explicit terms + CFL arrays
    prof.start("radial_loop")
    dtrkc, dthkc = _radial_loop_dispatch()
    prof.stop("radial_loop")

    # 2. CFL decision
    l_new_dt, dt_new = dt_courant(dt, _dtMax, dtrkc, dthkc)

    # 3. Roll dt history
    dt_old = tscheme.dt[0].item()
    tscheme.dt[0] = dt_new
    tscheme.dt[1] = dt_old

    # 4. Set IMEX weights from new dt ratio
    tscheme.set_weights()

    # 5. Build matrices on first step or when dt changes
    if n_time_step == 1 or l_new_dt:
        build_all_matrices()

    # 6. Implicit solve
    prof.start("lm_loop")
    lm_loop()
    prof.stop("lm_loop")

    prof.step_done()
    return dt_new


def _one_step_dirk(n_time_step: int, dt: float) -> float:
    """Multi-stage DIRK time step (e.g., BPR353 with 4 stages).

    CFL check is done at istage==1 only (first radial_loop call).

    Returns:
        dt_new: the time step size used for the implicit solve
    """
    # Reset stage counter (1-based, matching Fortran)
    tscheme.istage = 1
    dt_new = dt

    for n_stage in range(1, tscheme.nstages + 1):
        # Compute explicit terms (skip for stages where l_exp_calc is False)
        if tscheme.l_exp_calc[n_stage - 1]:
            prof.start("radial_loop")
            dtrkc, dthkc = _radial_loop_dispatch()
            prof.stop("radial_loop")

            # CFL only at first stage
            if n_stage == 1:
                l_new_dt, dt_new = dt_courant(dt, _dtMax, dtrkc, dthkc)

                # Set dt and weights (after CFL decision on first stage)
                tscheme.dt[0] = dt_new
                tscheme.set_weights()

                # Build matrices on first step or when dt changes
                if n_time_step == 1 or l_new_dt:
                    build_all_matrices()

        # Implicit solve
        prof.start("lm_loop")
        lm_loop()
        prof.stop("lm_loop")

        # Advance stage counter
        tscheme.istage += 1

    prof.step_done()
    return dt_new
