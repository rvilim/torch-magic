"""G_1.TAG binary graph file output matching out_graph_file.f90.

Writes 3D grid-space fields (velocity, entropy, pressure, magnetic field)
as big-endian float32 in Fortran stream (no record markers) format.
"""

import struct

import torch

from .precision import CDTYPE, DEVICE
from .params import (n_r_max, n_r_ic_max, n_theta_max, n_phi_max, n_phi_tot,
                     l_max, l_mag, l_heat, l_chemical_conv, l_cond_ic,
                     minc, lm_max, ra, ek, pr, prmag, radratio, raxi, sc,
                     sigma_ratio)
from .chebyshev import r, r_icb
from .radial_functions import or1, or2, orho1, r_ic, O_r_ic, O_r_ic2
from .horizontal_data import (theta_ord, O_sin_theta_grid, n_theta_cal2ord,
                              dLh)
from .blocking import st_lm2l
from .sht import torpol_to_spat, scal_to_spat
from .pre_calculations import vScale


# Fortran l_PressGraph is always True (Namelists.f90:1641)
_l_PressGraph = True
# Stefan number (zero for Boussinesq benchmark)
_stef = 0.0


def _to_be_f32(tensor):
    """Convert tensor to big-endian float32 bytes (C order for 1-D arrays)."""
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    return arr.astype('>f4').tobytes()


def _to_be_f32_field(tensor):
    """Convert 2-D field (n_theta, n_phi) to big-endian float32 in Fortran column-major order."""
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    return arr.astype('>f4').tobytes(order='F')


def _to_be_i32(val):
    """Convert integer to big-endian int32 bytes."""
    return struct.pack('>i', val)


def _to_be_f32_scalar(val):
    """Convert float to big-endian float32 bytes."""
    return struct.pack('>f', float(val))


def write_graph_file(path_or_fobj, time, w, dw, z, s, p, b, db, aj):
    """Write G_1.TAG binary graph file matching Fortran format.

    Args:
        path_or_fobj: output file path (str/Path) or writable binary file object
        time: simulation time (already scaled)
        w, dw, z: velocity spectral fields (lm_max, n_r_max)
        s: entropy spectral field (lm_max, n_r_max)
        p: pressure spectral field (lm_max, n_r_max)
        b, db, aj: magnetic spectral fields (lm_max, n_r_max)
    """
    if hasattr(path_or_fobj, 'write'):
        _write_graph_to(path_or_fobj, time, w, dw, z, s, p, b, db, aj)
    else:
        with open(path_or_fobj, 'wb') as f:
            _write_graph_to(f, time, w, dw, z, s, p, b, db, aj)


def _write_graph_to(f, time, w, dw, z, s, p, b, db, aj):
    _write_header(f, time)
    _write_oc_data(f, w, dw, z, s, p, b, db, aj)
    if l_mag and n_r_ic_max > 1:
        _write_ic_data(f, b)


def _write_header(f, time):
    """Write graph file header (448 bytes for benchmark)."""
    f.write(_to_be_i32(14))

    runid = b'Benchmark 1'
    f.write(runid + b' ' * (64 - len(runid)))

    f.write(_to_be_f32_scalar(time))

    for val in [ra, pr, raxi, sc, ek, _stef, prmag, radratio, sigma_ratio]:
        f.write(_to_be_f32_scalar(val))

    for val in [n_r_max, n_theta_max, n_phi_tot, minc, n_r_ic_max]:
        f.write(_to_be_i32(val))

    for val in [l_heat, l_chemical_conv, False, l_mag, _l_PressGraph, l_cond_ic]:
        f.write(_to_be_i32(1 if val else 0))

    f.write(_to_be_f32(theta_ord))
    f.write(_to_be_f32(r))

    if l_mag and n_r_ic_max > 1:
        f.write(_to_be_f32(r_ic))


def _write_oc_data(f, w, dw, z, s, p, b, db, aj):
    """Write outer core grid-space fields for all radial levels.

    Batches ALL SHT calls across radial levels, then writes sequentially.
    """
    cal2ord = n_theta_cal2ord
    O_sintheta = O_sin_theta_grid  # (n_theta_max,) interleaved order

    dLh_c = dLh.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)

    # --- Batch velocity SHT: all n_r_max levels at once ---
    Qlm_v = dLh_c * w    # (lm_max, n_r_max)
    vr_all, vt_all, vp_all = torpol_to_spat(Qlm_v, dw, z)
    # vr_all: (n_r_max, n_theta_max, n_phi_max)

    # --- Batch scalar SHTs for entropy and pressure ---
    sr_all = scal_to_spat(s) if l_heat else None
    pr_all = scal_to_spat(p) if _l_PressGraph else None

    # --- Batch magnetic SHT ---
    if l_mag:
        Qlm_b = dLh_c * b
        br_all, bt_all, bp_all = torpol_to_spat(Qlm_b, db, aj)

    # --- Write per radial level (sequential for binary output) ---
    or2_cpu = or2.cpu()
    or1_cpu = or1.cpu()
    orho1_cpu = orho1.cpu()

    # No-slip boundaries: nR=0 (CMB) and nR=n_r_max-1 (ICB) have nBc=2.
    # Fortran zeros velocity via v_rigid_boundary; we must do the same.
    _noslip = {0, n_r_max - 1}

    for nR in range(n_r_max):
        if nR in _noslip:
            # No-slip: velocity is exactly zero (matching Fortran v_rigid_boundary)
            zero_field = torch.zeros(n_theta_max, n_phi_max,
                                     dtype=vr_all.dtype, device=vr_all.device)
            f.write(_to_be_f32_field(zero_field))
            f.write(_to_be_f32_field(zero_field))
            f.write(_to_be_f32_field(zero_field))
        else:
            # Velocity scaling
            fac_r = float(or2_cpu[nR]) * vScale * float(orho1_cpu[nR])
            fac_t = float(or1_cpu[nR]) * vScale * float(orho1_cpu[nR])

            vr_out = fac_r * vr_all[nR]
            vt_out = fac_t * O_sintheta.unsqueeze(1) * vt_all[nR]
            vp_out = fac_t * O_sintheta.unsqueeze(1) * vp_all[nR]

            f.write(_to_be_f32_field(vr_out[cal2ord]))
            f.write(_to_be_f32_field(vt_out[cal2ord]))
            f.write(_to_be_f32_field(vp_out[cal2ord]))

        if l_heat:
            f.write(_to_be_f32_field(sr_all[nR][cal2ord]))

        if _l_PressGraph:
            f.write(_to_be_f32_field(pr_all[nR][cal2ord]))

        if l_mag:
            fac_br = float(or2_cpu[nR])
            fac_bt = float(or1_cpu[nR])

            br_out = fac_br * br_all[nR]
            bt_out = fac_bt * O_sintheta.unsqueeze(1) * bt_all[nR]
            bp_out = fac_bt * O_sintheta.unsqueeze(1) * bp_all[nR]

            f.write(_to_be_f32_field(br_out[cal2ord]))
            f.write(_to_be_f32_field(bt_out[cal2ord]))
            f.write(_to_be_f32_field(bp_out[cal2ord]))


def _write_ic_data(f, b_oc):
    """Write inner core grid-space fields for insulating IC.

    For insulating IC (l_cond_ic=False): potential field from OC b at ICB.
    Batches Q/S/T construction, calls torpol_to_spat per level (IC levels
    have different rDep scaling, so batching would require building per-level
    spectral arrays — we batch the expensive SHT instead when possible).
    """
    cal2ord = n_theta_cal2ord
    O_sintheta = O_sin_theta_grid
    dLh_cpu = dLh.cpu().to(torch.float64)
    st_lm2l_cpu = st_lm2l.cpu()
    l_lm_cpu = st_lm2l_cpu.to(torch.float64)  # (lm_max,)

    bICB = b_oc[:, -1].cpu().to(torch.complex128)  # (lm_max,)
    r_icb_f = float(r_icb)

    # Build rDep/rDep2 for ALL IC levels at once on CPU: (l_max+1, n_r_ic_max)
    rRatio = r_ic.cpu().to(torch.float64) / r_icb_f  # (n_r_ic_max,)
    l_arr = torch.arange(l_max + 1, dtype=torch.float64)  # (l_max+1,)

    # rRatio^l via outer product exponentiation
    log_rr = torch.where(rRatio > 0, torch.log(rRatio),
                         torch.tensor(-1e30, dtype=torch.float64))
    rRatio_pow_l = torch.exp(l_arr.unsqueeze(1) * log_rr.unsqueeze(0))
    rRatio_pow_l[:, rRatio == 0] = 0.0

    rDep = rRatio_pow_l * rRatio.unsqueeze(0)  # rRatio^(l+1)
    rDep2 = rRatio_pow_l / r_icb_f              # rRatio^l / r_ICB

    # Expand to (lm_max, n_r_ic_max) via st_lm2l
    rDep_lm = rDep[st_lm2l_cpu]    # (lm_max, n_r_ic_max)
    rDep2_lm = rDep2[st_lm2l_cpu]  # (lm_max, n_r_ic_max)

    # Build Q, S, T for all IC levels: (lm_max, n_r_ic_max)
    bICB_2d = bICB.unsqueeze(1)  # (lm_max, 1)
    dLh_2d = dLh_cpu.unsqueeze(1).to(torch.complex128)  # (lm_max, 1)
    l_lm_2d = l_lm_cpu.unsqueeze(1)  # (lm_max, 1)

    Qlm_ic = (rDep_lm * dLh_2d) * bICB_2d
    Slm_ic = (rDep2_lm * (l_lm_2d + 1.0)) * bICB_2d
    Tlm_ic = torch.zeros_like(Qlm_ic)

    # Move to DEVICE for SHT
    Qlm_ic = Qlm_ic.to(dtype=CDTYPE, device=DEVICE)
    Slm_ic = Slm_ic.to(dtype=CDTYPE, device=DEVICE)
    Tlm_ic = Tlm_ic.to(dtype=CDTYPE, device=DEVICE)

    # Batch SHT for all IC levels at once
    BrB_all, BtB_all, BpB_all = torpol_to_spat(Qlm_ic, Slm_ic, Tlm_ic)
    # (n_r_ic_max, n_theta_max, n_phi_max)

    # Write per IC level
    O_r_ic2_cpu = O_r_ic2.cpu()
    O_r_ic_cpu = O_r_ic.cpu()

    for nR in range(n_r_ic_max):
        fac_r = float(O_r_ic2_cpu[nR])
        fac_t = float(O_r_ic_cpu[nR])

        Br_out = fac_r * BrB_all[nR]
        Bt_out = fac_t * O_sintheta.unsqueeze(1) * BtB_all[nR]
        Bp_out = fac_t * O_sintheta.unsqueeze(1) * BpB_all[nR]

        f.write(_to_be_f32_field(Br_out[cal2ord]))
        f.write(_to_be_f32_field(Bt_out[cal2ord]))
        f.write(_to_be_f32_field(Bp_out[cal2ord]))
