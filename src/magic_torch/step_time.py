"""Main time stepping driver matching step_time.f90 + rIter.f90 + LMLoop.f90.

Implements:
- setup_initial_state: compute derivatives + initial old/impl from fields
- radial_loop: inverse SHT → get_nl → forward SHT → get_td
- lm_loop: finish explicit assembly + implicit solves
- one_step: single CNAB2 time step
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max, n_theta_max, n_phi_max, alpha
from .radial_functions import or2
from .horizontal_data import dLh, hdif_S, hdif_V, hdif_B
from .pre_calculations import opr, opm, BuoFac, CorFac
from .blocking import st_lm2l
from .radial_derivatives import get_dr, get_ddr, get_dddr
from .cosine_transform import costf
from .time_scheme import tscheme
from . import fields
from . import dt_fields
from .sht import (scal_to_spat, scal_to_SH, torpol_to_spat,
                  torpol_to_curl_spat, spat_to_sphertor)
from .get_nl import get_nl
from .get_td import get_dwdt, get_dzdt, get_dpdt, get_dsdt, get_dbdt
from .update_s import (build_s_matrices, finish_exp_entropy, updateS)
from .update_z import (build_z_matrices, updateZ)
from .update_wp import (build_p0_matrix, build_wp_matrices, updateWP)
from .update_b import (build_b_matrices, finish_exp_mag, updateB)
from .constants import two


# dLh for torpol_to_spat Q input: Fortran wrapper multiplies by dLh
_dLh_1d = dLh.to(CDTYPE)  # (lm_max,)

# Pre-allocated velocity buffers for radial_loop (no-slip: boundaries stay zero)
_vrc = torch.zeros(n_r_max, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
_vtc = torch.zeros_like(_vrc)
_vpc = torch.zeros_like(_vrc)
_cvrc = torch.zeros_like(_vrc)
_cvtc = torch.zeros_like(_vrc)
_cvpc = torch.zeros_like(_vrc)

# Precompute broadcast arrays for initial rhs_imp
_hdif_S_lm = hdif_S[st_lm2l].to(CDTYPE).unsqueeze(1)
_hdif_V_lm = hdif_V[st_lm2l].to(CDTYPE).unsqueeze(1)
_hdif_B_lm = hdif_B[st_lm2l].to(CDTYPE).unsqueeze(1)
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


def setup_initial_state():
    """Compute derivatives and initial old/impl arrays from initialized fields.

    Must be called after init_fields.initialize_fields().
    Matches get_*_rhs_imp calls in startFields.f90.
    """
    f = fields
    d = dt_fields

    # Zero dt_fields for re-entrant calls
    for ta in [d.dsdt, d.dwdt, d.dzdt, d.dpdt, d.dbdt, d.djdt]:
        ta.old.zero_()
        ta.impl.zero_()
        ta.expl.zero_()

    # --- Entropy: compute ds, d2s ---
    ds, d2s = get_ddr(f.s_LMloc)
    f.ds_LMloc[:] = ds

    # dsdt.old = s
    d.dsdt.old[:, :, 0] = f.s_LMloc.clone()

    # dsdt.impl = opr * hdif_S * (d2s + 2*or1*ds - dLh*or2*s)
    d.dsdt.impl[:, :, 0] = opr * _hdif_S_lm * (
        d2s + two * _or1_r * ds - _dLh_lm * _or2_r * f.s_LMloc
    )

    # --- Toroidal velocity: compute dz, d2z ---
    dz, d2z = get_ddr(f.z_LMloc)
    f.dz_LMloc[:] = dz

    # dzdt.old = dLh * or2 * z  (matches Fortran updateZ line 926)
    d.dzdt.old[:, :, 0] = _dLh_lm * _or2_r * f.z_LMloc

    # dzdt.impl = hdif_V * dLh * or2 * (d2z - dLh * or2 * z)
    d.dzdt.impl[:, :, 0] = _hdif_V_lm * _dLh_lm * _or2_r * (
        d2z - _dLh_lm * _or2_r * f.z_LMloc
    )

    # --- Poloidal velocity + pressure: compute dw, ddw, dddw, dp ---
    dw, ddw, dddw = get_dddr(f.w_LMloc)
    f.dw_LMloc[:] = dw
    f.ddw_LMloc[:] = ddw
    dp = get_dr(f.p_LMloc)
    f.dp_LMloc[:] = dp

    # dwdt.old = dLh * or2 * w
    d.dwdt.old[:, :, 0] = _dLh_lm * _or2_r * f.w_LMloc

    # dpdt.old = -dLh * or2 * dw
    d.dpdt.old[:, :, 0] = -_dLh_lm * _or2_r * dw

    # dwdt.impl = -dp + hdif_V*dLh*or2*(ddw - dLh*or2*w) + BuoFac*rgrav*s
    Dif_w = _hdif_V_lm * _dLh_lm * _or2_r * (ddw - _dLh_lm * _or2_r * f.w_LMloc)
    d.dwdt.impl[:, :, 0] = -dp + Dif_w + BuoFac * _rgrav_r * f.s_LMloc

    # dpdt.impl = dLh*or2*p + hdif_V*dLh*or2*(-dddw + dLh*or2*dw - 2*dLh*or3*w)
    d.dpdt.impl[:, :, 0] = (_dLh_lm * _or2_r * f.p_LMloc
                            + _hdif_V_lm * _dLh_lm * _or2_r * (
                                -dddw + _dLh_lm * _or2_r * dw
                                - two * _dLh_lm * _or3_r * f.w_LMloc
                            ))

    # --- Magnetic field: compute db, ddb, dj, ddj ---
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

    # dbdt.impl = opm * hdif_B * dLh * or2 * (ddb - dLh * or2 * b)
    d.dbdt.impl[:, :, 0] = opm * _hdif_B_lm * _dLh_lm * _or2_r * (
        ddb - _dLh_lm * _or2_r * f.b_LMloc
    )

    # djdt.impl = opm * hdif_B * dLh * or2 * (ddj - dLh * or2 * aj)
    d.djdt.impl[:, :, 0] = opm * _hdif_B_lm * _dLh_lm * _or2_r * (
        ddj - _dLh_lm * _or2_r * f.aj_LMloc
    )


def radial_loop():
    """Inverse SHT → nonlinear products → forward SHT → time derivative assembly.

    For the Boussinesq benchmark with l_adv_curl=.true., no-slip BCs.
    At boundaries (nR=0 and nR=N-1), velocity is zero (omega=0)
    and nonlinear terms are not computed (get_td zeros boundary terms).

    All SHT calls are batched over radial levels to minimize Python overhead.
    Multiple transforms are combined into single calls where possible.
    """
    f = fields
    N = n_r_max
    Nb = N - 2
    bulk = slice(1, N - 1)  # interior radial levels

    # === 1. Inverse SHT (batched) ===
    # Entropy: all N levels — (lm_max, N) → (N, n_theta, n_phi)
    sc = scal_to_spat(f.s_LMloc)

    # Prepare all 4 torpol_to_spat inputs and batch into single call
    # (magnetic direct + magnetic curl + velocity direct + velocity curl)
    # Magnetic direct: Q=dLh*b, S=db, T=aj (N levels)
    # Magnetic curl:   Q=dLh*j, S=dj, T=or2*dLh*b - ddb (N levels)
    # Velocity direct: Q=dLh*w, S=dw, T=z (Nb levels)
    # Velocity curl:   Q=dLh*z, S=dz, T=or2*dLh*w - ddw (Nb levels)
    Q_all = torch.cat([
        _dLh_2d * f.b_LMloc,
        _dLh_2d * f.aj_LMloc,
        _dLh_2d * f.w_LMloc[:, bulk],
        _dLh_2d * f.z_LMloc[:, bulk],
    ], dim=1)
    S_all = torch.cat([
        f.db_LMloc,
        f.dj_LMloc,
        f.dw_LMloc[:, bulk],
        f.dz_LMloc[:, bulk],
    ], dim=1)
    T_all = torch.cat([
        f.aj_LMloc,
        _or2_2d * _dLh_2d * f.b_LMloc - f.ddb_LMloc,
        f.z_LMloc[:, bulk],
        _or2_bulk_2d * _dLh_2d * f.w_LMloc[:, bulk] - f.ddw_LMloc[:, bulk],
    ], dim=1)

    all_r, all_t, all_p = torpol_to_spat(Q_all, S_all, T_all)

    # Split results: magnetic(N) + curl_mag(N) + velocity(Nb) + curl_vel(Nb)
    brc = all_r[:N]
    btc = all_t[:N]
    bpc = all_p[:N]
    cbrc = all_r[N:2 * N]
    cbtc = all_t[N:2 * N]
    cbpc = all_p[N:2 * N]

    # Fill pre-allocated velocity buffers (boundaries stay zero from init)
    _vrc[bulk] = all_r[2 * N:2 * N + Nb]
    _vtc[bulk] = all_t[2 * N:2 * N + Nb]
    _vpc[bulk] = all_p[2 * N:2 * N + Nb]
    _cvrc[bulk] = all_r[2 * N + Nb:]
    _cvtc[bulk] = all_t[2 * N + Nb:]
    _cvpc[bulk] = all_p[2 * N + Nb:]

    # === 2. Nonlinear products (all radial levels at once) ===
    (Advr, Advt, Advp, VSr, VSt, VSp,
     VxBr, VxBt, VxBp) = get_nl(
        _vrc, _vtc, _vpc, _cvrc, _cvtc, _cvpc,
        sc, brc, btc, bpc, cbrc, cbtc, cbpc)

    # === 3. Forward SHT (batched, bulk only) ===
    # Batch all 3 scal_to_SH calls into one (3*Nb levels)
    scal_in = torch.cat([Advr[bulk], VSr[bulk], VxBr[bulk]], dim=0)
    scal_out = scal_to_SH(scal_in)  # (lm_max, 3*Nb)

    # Batch all 3 spat_to_sphertor calls into one (3*Nb levels)
    vt_in = torch.cat([Advt[bulk], VSt[bulk], VxBt[bulk]], dim=0)
    vp_in = torch.cat([Advp[bulk], VSp[bulk], VxBp[bulk]], dim=0)
    S_out, T_out = spat_to_sphertor(vt_in, vp_in)  # (lm_max, 3*Nb)

    # Spectral work arrays: (lm_max, n_r_max) — boundaries stay zero
    AdvrLM = torch.zeros(lm_max, N, dtype=CDTYPE, device=DEVICE)
    AdvtLM = torch.zeros_like(AdvrLM)
    AdvpLM = torch.zeros_like(AdvrLM)
    dVSrLM = torch.zeros_like(AdvrLM)
    VStLM = torch.zeros_like(AdvrLM)
    VxBrLM = torch.zeros_like(AdvrLM)
    VxBtLM = torch.zeros_like(AdvrLM)
    VxBpLM = torch.zeros_like(AdvrLM)

    # Scatter batched results into per-field arrays
    AdvrLM[:, bulk] = scal_out[:, :Nb]
    dVSrLM[:, bulk] = scal_out[:, Nb:2 * Nb]
    VxBrLM[:, bulk] = scal_out[:, 2 * Nb:]

    AdvtLM[:, bulk] = S_out[:, :Nb]
    AdvpLM[:, bulk] = T_out[:, :Nb]
    VStLM[:, bulk] = S_out[:, Nb:2 * Nb]
    VxBtLM[:, bulk] = S_out[:, 2 * Nb:]
    VxBpLM[:, bulk] = T_out[:, 2 * Nb:]

    # === 4. Time derivative assembly ===

    istage = tscheme.istage if hasattr(tscheme, 'istage') else 0

    # Poloidal velocity
    dwdt_expl = get_dwdt(AdvrLM, f.dw_LMloc, f.z_LMloc)
    dt_fields.dwdt.expl[:, :, 0] = dwdt_expl

    # Toroidal velocity
    dzdt_expl = get_dzdt(AdvpLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
    dt_fields.dzdt.expl[:, :, 0] = dzdt_expl

    # Pressure
    dpdt_expl = get_dpdt(AdvtLM, f.w_LMloc, f.dw_LMloc, f.z_LMloc)
    dt_fields.dpdt.expl[:, :, 0] = dpdt_expl

    # Entropy (partial + finish)
    dsdt_partial, dVSrLM_out = get_dsdt(VStLM, dVSrLM)
    dsdt_expl = finish_exp_entropy(dsdt_partial, dVSrLM_out)
    dt_fields.dsdt.expl[:, :, 0] = dsdt_expl

    # Magnetic field
    dbdt_expl, djdt_partial, dVxBhLM = get_dbdt(VxBrLM, VxBtLM, VxBpLM)
    djdt_expl = finish_exp_mag(djdt_partial, dVxBhLM)
    dt_fields.dbdt.expl[:, :, 0] = dbdt_expl
    dt_fields.djdt.expl[:, :, 0] = djdt_expl


def lm_loop():
    """Implicit solves: S → Z → WP → B."""
    f = fields
    d = dt_fields

    updateS(f.s_LMloc, f.ds_LMloc, d.dsdt, tscheme)
    updateZ(f.z_LMloc, f.dz_LMloc, d.dzdt, tscheme)
    updateWP(f.s_LMloc, f.w_LMloc, f.dw_LMloc, f.ddw_LMloc,
             d.dwdt, f.p_LMloc, f.dp_LMloc, d.dpdt, tscheme)
    updateB(f.b_LMloc, f.db_LMloc, f.ddb_LMloc,
            f.aj_LMloc, f.dj_LMloc, f.ddj_LMloc,
            d.dbdt, d.djdt, tscheme)


def build_all_matrices():
    """Build and factorize all implicit matrices for current dt."""
    wimp_lin0 = tscheme.wimp_lin[0].item()
    build_s_matrices(wimp_lin0)
    build_z_matrices(wimp_lin0)
    build_p0_matrix()
    build_wp_matrices(wimp_lin0)
    build_b_matrices(wimp_lin0)


def initialize_dt(dt: float):
    """Initialize dt array to dtMax, matching startFields.f90 line 298.

    Must be called before the first time step. Fortran sets tscheme%dt(:) = dtMax
    when starting from scratch (not from a checkpoint).
    """
    tscheme.dt[:] = dt


def one_step(n_time_step: int, dt: float):
    """Execute one CNAB2 time step.

    Args:
        n_time_step: current step number (1-based)
        dt: time step size
    """
    # Fortran set_dt_array: roll dt, then set dt(1) = dt_new
    # cshift(dt, shift=nexp-1=1) rotates: [dt0, dt1] -> [dt1, dt0]
    # Then dt(1) = dt_new. For constant dt this is a no-op.
    dt_old = tscheme.dt[0].item()
    tscheme.dt[0] = dt
    tscheme.dt[1] = dt_old

    tscheme.set_weights()

    # Build matrices (always for step 1, or when dt changes)
    if n_time_step == 1:
        build_all_matrices()

    # Radial loop: SHT + nonlinear + get_td
    radial_loop()

    # LM loop: implicit solves
    lm_loop()
