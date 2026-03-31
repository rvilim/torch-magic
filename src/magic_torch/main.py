"""Main driver for the MagIC dynamo benchmark in PyTorch.

Supports YAML-driven config, CSV energy logging, and checkpointing.
"""

import csv
import os
import time as timeit

import torch

import math

from .init_fields import initialize_fields
from .step_time import setup_initial_state, one_step, initialize_dt, build_all_matrices
from .output import (get_e_kin, get_e_mag, get_e_kin_full, get_e_mag_oc_full,
                     get_e_mag_ic, write_e_kin_line, write_e_mag_oc_line,
                     write_e_mag_ic_line, get_e_kin_radial, get_e_mag_radial,
                     get_dipole, RadialAccumulator, write_radius_file,
                     write_signal_file, write_timestep_line, write_eKinR_file,
                     write_eMagR_file, write_dipole_line,
                     MeanSD, update_heat_means, get_heat_data,
                     write_heat_line, write_heatR_file,
                     get_dlm, get_par_data, get_elsAnel, get_e_mag_cmb,
                     update_par_means, write_par_line, write_parR_file)
from .graph_output import write_graph_file
from .log_output import (write_log_header, write_log_scheme_info,
                         write_log_boundary_info, write_log_namelists,
                         write_log_dtmax_info, write_log_physical_info,
                         write_log_start, write_log_step, write_log_store,
                         write_log_end_energies, write_log_avg_energies,
                         write_log_avg_properties, write_log_timing,
                         write_log_stop)
from . import fields
from . import dt_fields
from .params import (n_time_steps, dtmax, l_cond_ic, radratio, l_heat, l_mag,
                     l_chemical_conv, n_r_max, kbotb, ktopb)
from .precision import CDTYPE, DEVICE
from .time_scheme import tscheme
from .pre_calculations import tScale


def _get_energies():
    """Compute all four energy components (backward-compatible)."""
    e_kin_pol, e_kin_tor = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    if l_mag:
        e_mag_pol, e_mag_tor = get_e_mag(
            fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
            fields.aj_LMloc, fields.dj_LMloc)
    else:
        e_mag_pol, e_mag_tor = 0.0, 0.0
    return e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor


def _get_energies_full():
    """Compute full energy decompositions for file output."""
    ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
    if l_mag:
        em = get_e_mag_oc_full(fields.b_LMloc, fields.db_LMloc,
                               fields.aj_LMloc)
        eic = get_e_mag_ic(fields.b_LMloc)
    else:
        em, eic = None, None
    return ek, em, eic


def _translate_entropy_lm00():
    """Translate s(l=0,m=0) from legacy boundary convention.

    Matches Fortran startFields.f90 lines 254-263: if the l=0,m=0 mode at
    CMB and ICB matches the conduction profile topval/botval, subtract topval
    from the entire radial profile. This shifts the reference state so s=0
    at CMB.
    """
    r_cmb = 1.0 / (1.0 - radratio)
    r_icb = radratio / (1.0 - radratio)
    sq4pi = math.sqrt(4.0 * math.pi)
    topval = -r_icb**2 / (r_icb**2 + r_cmb**2) * sq4pi
    botval = r_cmb**2 / (r_icb**2 + r_cmb**2) * sq4pi
    eps = torch.finfo(torch.float64).eps
    s00 = fields.s_LMloc[0, :]  # l=0, m=0 is index 0 in standard ordering
    if (abs(s00[0].real.item() - topval) <= 10000 * eps and
            abs(s00[-1].real.item() - botval) <= 10000 * eps):
        fields.s_LMloc[0, :] -= topval


def _translate_xi_lm00():
    """Translate xi(l=0,m=0) from legacy boundary convention.

    Matches Fortran startFields.f90 lines 269-275: identical topval/botval
    formula as entropy. If xi(lm00) at CMB/ICB matches conduction profile,
    subtract topval to shift reference so xi=0 at CMB.
    """
    r_cmb = 1.0 / (1.0 - radratio)
    r_icb = radratio / (1.0 - radratio)
    sq4pi = math.sqrt(4.0 * math.pi)
    topval = -r_icb**2 / (r_icb**2 + r_cmb**2) * sq4pi
    botval = r_cmb**2 / (r_icb**2 + r_cmb**2) * sq4pi
    eps = torch.finfo(torch.float64).eps
    xi00 = fields.xi_LMloc[0, :]
    if (abs(xi00[0].real.item() - topval) <= 10000 * eps and
            abs(xi00[-1].real.item() - botval) <= 10000 * eps):
        fields.xi_LMloc[0, :] -= topval


def load_fortran_checkpoint(path: str) -> float:
    """Load a Fortran MagIC checkpoint into Python fields.

    Reads the binary checkpoint, copies field data to fields.*, sets omega_ic,
    and returns the simulation time from the checkpoint.

    Args:
        path: Path to the Fortran checkpoint file.

    Returns:
        sim_time: Simulation time from the checkpoint.
    """
    from .checkpoint_io import read_checkpoint

    ck = read_checkpoint(path)

    # Copy OC fields (checkpoint is (lm_max_ck, n_r_max) in st_map order)
    fields.w_LMloc[:] = torch.from_numpy(ck.w).to(dtype=CDTYPE, device=DEVICE)
    fields.z_LMloc[:] = torch.from_numpy(ck.z).to(dtype=CDTYPE, device=DEVICE)
    if ck.p is not None:
        fields.p_LMloc[:] = torch.from_numpy(ck.p).to(dtype=CDTYPE, device=DEVICE)
    if ck.s is not None:
        fields.s_LMloc[:] = torch.from_numpy(ck.s).to(dtype=CDTYPE, device=DEVICE)
        # Fortran getStartFields translates s(l=0,m=0) from legacy (topval,botval)
        # to (0, botval-topval) convention — see startFields.f90 lines 254-263
        if l_heat:
            _translate_entropy_lm00()
    if l_chemical_conv and ck.xi is not None:
        fields.xi_LMloc[:] = torch.from_numpy(ck.xi).to(dtype=CDTYPE, device=DEVICE)
        _translate_xi_lm00()
    if ck.b is not None:
        fields.b_LMloc[:] = torch.from_numpy(ck.b).to(dtype=CDTYPE, device=DEVICE)
    if ck.aj is not None:
        fields.aj_LMloc[:] = torch.from_numpy(ck.aj).to(dtype=CDTYPE, device=DEVICE)

    # Copy IC fields
    if l_cond_ic and ck.b_ic is not None:
        fields.b_ic[:] = torch.from_numpy(ck.b_ic).to(dtype=CDTYPE, device=DEVICE)
    if l_cond_ic and ck.aj_ic is not None:
        fields.aj_ic[:] = torch.from_numpy(ck.aj_ic).to(dtype=CDTYPE, device=DEVICE)

    # Set omega_ic from checkpoint (Fortran finish_start_fields logic)
    # For nRotIC=1 with omega_ic1=0 in namelist: omega_ic = omega_ic1Old (from checkpoint)
    fields.omega_ic = ck.omega_ic1

    return ck.time


_FIELD_NAMES = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "p_LMloc", "dp_LMloc",
    "s_LMloc", "ds_LMloc",
    "xi_LMloc", "dxi_LMloc",
    "b_LMloc", "db_LMloc", "ddb_LMloc",
    "aj_LMloc", "dj_LMloc", "ddj_LMloc",
]

_DT_NAMES = ["dsdt", "dxidt", "dwdt", "dzdt", "dpdt", "dbdt", "djdt"]


def save_checkpoint(path, step, sim_time, cfg):
    """Save simulation state to a .pt file."""
    state = {
        "step": step,
        "sim_time": sim_time,
        "cfg": cfg,
        "tscheme_dt": tscheme.dt.clone(),
    }
    # Field tensors
    for name in _FIELD_NAMES:
        state[f"fields.{name}"] = getattr(fields, name).clone()
    state["fields.omega_ic"] = fields.omega_ic
    state["fields.omega_ma"] = fields.omega_ma

    # TimeArray objects
    for name in _DT_NAMES:
        ta = getattr(dt_fields, name)
        state[f"dt.{name}.impl"] = ta.impl.clone()
        state[f"dt.{name}.expl"] = ta.expl.clone()
        state[f"dt.{name}.old"] = ta.old.clone()

    torch.save(state, path)


def load_checkpoint(path):
    """Load simulation state from a .pt file.

    Returns (step, sim_time).
    """
    state = torch.load(path, weights_only=False)

    # Restore fields
    for name in _FIELD_NAMES:
        getattr(fields, name).copy_(state[f"fields.{name}"])
    fields.omega_ic = state["fields.omega_ic"]
    fields.omega_ma = state["fields.omega_ma"]

    # Restore dt arrays
    for name in _DT_NAMES:
        ta = getattr(dt_fields, name)
        ta.impl.copy_(state[f"dt.{name}.impl"])
        ta.expl.copy_(state[f"dt.{name}.expl"])
        ta.old.copy_(state[f"dt.{name}.old"])

    # Restore time scheme dt
    tscheme.dt.copy_(state["tscheme_dt"])
    tscheme.set_weights()

    # Rebuild implicit matrices for the restored dt
    build_all_matrices()

    return state["step"], state["sim_time"]


def _zero_mag_boundaries(accum):
    """Zero toroidal energy at CMB/ICB for insulating BCs (magnetic_energy.f90:408-414)."""
    if ktopb == 1:
        accum.profiles[2][0] = 0.0   # e_t_r at CMB
        accum.profiles[3][0] = 0.0   # e_t_as_r at CMB
    if kbotb == 1:
        accum.profiles[2][-1] = 0.0  # e_t_r at ICB
        accum.profiles[3][-1] = 0.0  # e_t_as_r at ICB


def run(cfg=None):
    """Run the dynamo benchmark.

    Args:
        cfg: dict from YAML config, or None for legacy defaults.
    """
    if cfg is None:
        cfg = {}

    n_steps = cfg.get("n_steps", n_time_steps)
    dt = cfg.get("dt", dtmax)
    log_every = cfg.get("log_every", 100)
    checkpoint_every = cfg.get("checkpoint_every", 0)
    output_dir = cfg.get("output_dir", "./output")
    restart = cfg.get("restart", None)
    fortran_restart = cfg.get("fortran_restart", None)
    n_graphs = cfg.get("n_graphs", 1)  # number of graph files to write

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize or restore
    if fortran_restart:
        sim_time = load_fortran_checkpoint(fortran_restart)
        start_step = 0
        setup_initial_state()
        initialize_dt(dt)
        print(f"Loaded Fortran checkpoint: sim_time={sim_time:.6e}")
    elif restart:
        start_step, sim_time = load_checkpoint(restart)
        print(f"Restored from checkpoint: step {start_step}, sim_time={sim_time:.6e}")
    else:
        start_step = 0
        sim_time = 0.0
        initialize_fields()
        setup_initial_state()
        initialize_dt(dt)

    # Open CSV log
    log_path = os.path.join(output_dir, "log.csv")
    csv_columns = [
        "step", "sim_time", "e_kin_pol", "e_kin_tor",
        "e_mag_pol", "e_mag_tor", "wall_elapsed", "ms_per_step",
    ]
    appending = restart and os.path.exists(log_path)
    log_file = open(log_path, "a" if appending else "w", newline="")
    writer = csv.writer(log_file)
    if not appending:
        writer.writerow(csv_columns)

    # Open Fortran-format energy files
    tag = cfg.get("tag", "torch")
    e_kin_mode = "a" if appending else "w"
    f_ekin = open(os.path.join(output_dir, f"e_kin.{tag}"), e_kin_mode)
    f_emag_oc = open(os.path.join(output_dir, f"e_mag_oc.{tag}"), e_kin_mode) if l_mag else None
    f_emag_ic = open(os.path.join(output_dir, f"e_mag_ic.{tag}"), e_kin_mode) if l_mag else None

    # Open dipole file
    f_dipole = open(os.path.join(output_dir, f"dipole.{tag}"), e_kin_mode) if l_mag else None

    # Open timestep file
    f_timestep = open(os.path.join(output_dir, f"timestep.{tag}"), e_kin_mode)

    # Write once-at-init files (only on fresh start)
    if not appending:
        write_radius_file(os.path.join(output_dir, f"radius.{tag}"))
        write_signal_file(os.path.join(output_dir, f"signal.{tag}"))
        write_timestep_line(f_timestep, 0.0, dt)
        f_timestep.flush()

    # Create radial energy accumulators
    kin_accum = RadialAccumulator(4, n_r_max)
    mag_accum = RadialAccumulator(5, n_r_max) if l_mag else None

    # Heat diagnostics
    f_heat = open(os.path.join(output_dir, f"heat.{tag}"), e_kin_mode) if l_heat else None
    smean_r = MeanSD(n_r_max) if l_heat else None
    tmean_r = MeanSD(n_r_max) if l_heat else None
    pmean_r = MeanSD(n_r_max) if l_heat else None
    rhomean_r = MeanSD(n_r_max) if l_heat else None
    ximean_r = MeanSD(n_r_max) if l_heat else None
    heat_n_calls = 0

    # Par diagnostics
    f_par = open(os.path.join(output_dir, f"par.{tag}"), e_kin_mode)
    par_rm_ms = MeanSD(n_r_max)
    par_rol_ms = MeanSD(n_r_max)
    par_urol_ms = MeanSD(n_r_max)
    par_dlv_ms = MeanSD(n_r_max)
    par_dlvc_ms = MeanSD(n_r_max)
    par_dlpp_ms = MeanSD(n_r_max)
    par_n_calls = 0

    # --- log.TAG file and time-averaging accumulators ---
    from .radial_functions import vol_oc as _vol_oc, vol_ic as _vol_ic
    f_log = open(os.path.join(output_dir, f"log.{tag}"), e_kin_mode)
    if not appending:
        write_log_header(f_log)
        write_log_scheme_info(f_log, tscheme)
        write_log_boundary_info(f_log)
        write_log_namelists(f_log, cfg)
        write_log_dtmax_info(f_log, dt)
        write_log_physical_info(f_log)
        write_log_start(f_log, 0.0, 0, dt)

    # Energy time-averaging accumulators (output.f90:814-817)
    log_e_kin_p_sum = 0.0
    log_e_kin_t_sum = 0.0
    log_e_mag_p_sum = 0.0
    log_e_mag_t_sum = 0.0
    # Property time-averaging (output.f90:793-826)
    log_rm_sum = 0.0
    log_el_sum = 0.0
    log_el_cmb_sum = 0.0
    log_rol_sum = 0.0
    log_dip_sum = 0.0
    log_dip_cmb_sum = 0.0
    log_dlv_sum = 0.0
    log_dmv_sum = 0.0
    log_dlb_sum = 0.0
    log_dmb_sum = 0.0
    log_dlvc_sum = 0.0
    log_dlv_polpeak_sum = 0.0
    log_rel_a_sum = 0.0
    log_rel_z_sum = 0.0
    log_rel_m_sum = 0.0
    log_rel_na_sum = 0.0
    log_time_norm = 0.0

    def write_row(step, sim_time, wall_elapsed, steps_done, accumulate=False):
        ek, em, eic = _get_energies_full()
        e_kin_pol, e_kin_tor = ek.e_p, ek.e_t
        e_mag_pol = em.e_p if em else 0.0
        e_mag_tor = em.e_t if em else 0.0
        ms = (wall_elapsed / steps_done * 1000) if steps_done > 0 else 0.0
        writer.writerow([
            step, f"{sim_time:.10e}",
            f"{e_kin_pol:.15e}", f"{e_kin_tor:.15e}",
            f"{e_mag_pol:.15e}", f"{e_mag_tor:.15e}",
            f"{wall_elapsed:.3f}", f"{ms:.3f}",
        ])
        log_file.flush()
        # Write Fortran-format energy files
        t_out = sim_time * tScale
        write_e_kin_line(f_ekin, t_out, ek)
        f_ekin.flush()
        if em is not None:
            write_e_mag_oc_line(f_emag_oc, t_out, em)
            f_emag_oc.flush()
        if eic is not None:
            write_e_mag_ic_line(f_emag_ic, t_out, eic)
            f_emag_ic.flush()
        # Write dipole line
        dip_cols = None
        if em is not None and f_dipole is not None:
            dip_cols = get_dipole(fields.b_LMloc, fields.db_LMloc,
                                  fields.aj_LMloc, em.e_p, em.e_t)
            write_dipole_line(f_dipole, t_out, dip_cols)
            f_dipole.flush()

        # Par diagnostics (getDlm.f90, outPar.f90, output.f90)
        nonlocal par_n_calls
        par_n_calls += 1
        time_passed = dt
        time_norm_par = par_n_calls * dt

        dlm_v = get_dlm(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc, 'V')
        dlm_b = get_dlm(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc, 'B') if l_mag else None

        elsAnel_val = get_elsAnel(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc) if l_mag else 0.0
        e_mag_cmb_val = get_e_mag_cmb(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc) if l_mag else 0.0

        par_cols = get_par_data(ek, em, dip_cols, dlm_v, dlm_b, elsAnel_val, e_mag_cmb_val)
        write_par_line(f_par, t_out, par_cols)
        f_par.flush()

        # Log time-averaging accumulation (output.f90:793-826)
        nonlocal log_e_kin_p_sum, log_e_kin_t_sum, log_e_mag_p_sum, log_e_mag_t_sum
        nonlocal log_rm_sum, log_el_sum, log_el_cmb_sum, log_rol_sum
        nonlocal log_dip_sum, log_dip_cmb_sum
        nonlocal log_dlv_sum, log_dmv_sum, log_dlb_sum, log_dmb_sum
        nonlocal log_dlvc_sum, log_dlv_polpeak_sum
        nonlocal log_rel_a_sum, log_rel_z_sum, log_rel_m_sum, log_rel_na_sum
        nonlocal log_time_norm
        log_time_norm += time_passed
        log_e_kin_p_sum += time_passed * e_kin_pol
        log_e_kin_t_sum += time_passed * e_kin_tor
        log_e_mag_p_sum += time_passed * e_mag_pol
        log_e_mag_t_sum += time_passed * e_mag_tor
        # par_cols indices: [Rm, El, Rol, Geos, Dip, DipCMB,
        #   dlV, dmV, dpV, dzV, lvDiss, lbDiss, dlB, dmB, ElCmb, ...]
        log_rm_sum += time_passed * par_cols[0]
        log_el_sum += time_passed * par_cols[1]
        log_rol_sum += time_passed * par_cols[2]
        log_dip_sum += time_passed * par_cols[4]
        log_dip_cmb_sum += time_passed * par_cols[5]
        log_dlv_sum += time_passed * par_cols[6]
        log_dmv_sum += time_passed * par_cols[7]
        log_dlb_sum += time_passed * par_cols[12]
        log_dmb_sum += time_passed * par_cols[13]
        log_el_cmb_sum += time_passed * par_cols[14]
        log_dlvc_sum += time_passed * par_cols[16]
        log_dlv_polpeak_sum += time_passed * par_cols[17]
        # Relative energy ratios (output.f90:802-811)
        e_kin = e_kin_pol + e_kin_tor
        if e_kin > 0:
            log_rel_a_sum += time_passed * (ek.e_p_as + ek.e_t_as) / e_kin
            log_rel_z_sum += time_passed * ek.e_t_as / e_kin
            log_rel_m_sum += time_passed * ek.e_p_as / e_kin
            log_rel_na_sum += time_passed * (e_kin - ek.e_p_as - ek.e_t_as) / e_kin

        # ParR accumulation
        kin_r = get_e_kin_radial(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
        ekinR_factored = 0.5 * (kin_r[0] + kin_r[2])  # Apply 0.5 factor
        update_par_means(par_rm_ms, par_rol_ms, par_urol_ms, par_dlv_ms,
                         par_dlvc_ms, par_dlpp_ms, ekinR_factored,
                         dlm_v[1], dlm_v[5], dlm_v[6],
                         time_passed, time_norm_par)

        # Heat diagnostics (outMisc.f90 outHeat)
        if l_heat and f_heat is not None:
            nonlocal heat_n_calls
            heat_n_calls += 1
            time_passed = dt  # timePassed = dt for n_log_step=1
            time_norm = heat_n_calls * dt  # timeNorm accumulates

            s00 = fields.s_LMloc[0, :].real
            p00 = fields.p_LMloc[0, :].real

            update_heat_means(smean_r, tmean_r, pmean_r, rhomean_r, ximean_r,
                              s00, p00, time_passed, time_norm)

            heat_cols = get_heat_data(s00, fields.ds_LMloc[0, :].real, p00)
            write_heat_line(f_heat, t_out, heat_cols)
            f_heat.flush()

        # Accumulate radial profiles (step 0 IS included, matching Fortran)
        if accumulate:
            kin_profs = get_e_kin_radial(fields.w_LMloc, fields.dw_LMloc,
                                         fields.z_LMloc)
            kin_accum.accumulate(sim_time, *kin_profs)
            if l_mag and mag_accum is not None:
                mag_profs = get_e_mag_radial(fields.b_LMloc, fields.db_LMloc,
                                              fields.aj_LMloc)
                mag_accum.accumulate(sim_time, *mag_profs)
                _zero_mag_boundaries(mag_accum)
        print(f"Step {step:6d}: e_kin={e_kin_pol + e_kin_tor:.6e} "
              f"(pol={e_kin_pol:.6e} tor={e_kin_tor:.6e}) "
              f"e_mag={e_mag_pol + e_mag_tor:.6e} "
              f"(pol={e_mag_pol:.6e} tor={e_mag_tor:.6e}) "
              f"dt={dt:.6e} t={sim_time:.6e} wall={wall_elapsed:.1f}s")

    # Step 0 row (only if fresh start)
    if not restart:
        write_row(0, 0.0, 0.0, 0, accumulate=True)

    # Time integration
    t_start = timeit.time()

    for n in range(start_step + 1, n_steps + 1):
        dt_actual = one_step(n, dt)
        if dt_actual != dt:
            print(f"  dt changed: {dt:.6e} -> {dt_actual:.6e} at step {n}")
            write_timestep_line(f_timestep, sim_time + dt_actual, dt_actual)
            f_timestep.flush()
        dt = dt_actual
        sim_time += dt
        steps_done = n - start_step
        elapsed = timeit.time() - t_start

        should_log = (n % log_every == 0) or n == start_step + 1 or n == n_steps
        if should_log:
            write_row(n, sim_time, elapsed, steps_done, accumulate=True)
            write_log_step(f_log, n, elapsed / steps_done)
            f_log.flush()

        if checkpoint_every and (n % checkpoint_every == 0 or n == n_steps):
            cp_path = os.path.join(output_dir, f"checkpoint_{n:06d}.pt")
            save_checkpoint(cp_path, n, sim_time, cfg)
            print(f"  Saved checkpoint: {cp_path}")

    # Write graph file at end of run (n_graphs=1 → output at last step)
    if n_graphs >= 1 and n_steps > 0:
        t_out = sim_time * tScale
        graph_path = os.path.join(output_dir, f"G_1.{tag}")
        graph_filename = f"G_1.{tag}"
        write_graph_file(graph_path, t_out,
                         fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc,
                         fields.s_LMloc, fields.p_LMloc,
                         fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
        print(f"  Wrote graph file: {graph_path}")
        write_log_store(f_log, "graphic", t_out, n_steps + 1, graph_filename)

    # Write Fortran-format checkpoint at end of run
    if n_steps > 0:
        from .checkpoint_io import write_checkpoint_fortran
        ckpt_filename = f"checkpoint_end.{tag}"
        ckpt_path = os.path.join(output_dir, ckpt_filename)
        write_checkpoint_fortran(ckpt_path, sim_time, n_steps + 1)
        print(f"  Wrote Fortran checkpoint: {ckpt_path}")

        # Write checkpoint store notice to log
        write_log_store(f_log, "checkpoint", sim_time * tScale,
                        n_steps + 1, ckpt_filename)

    # Write end-of-run log sections
    if n_steps > 0:
        ek_final, em_final, eic_final = _get_energies_full()
        write_log_end_energies(f_log, ek_final, em_final, eic_final,
                               _vol_oc, _vol_ic)

        if log_time_norm > 0:
            energy_means = {
                "e_kin_p": log_e_kin_p_sum / log_time_norm,
                "e_kin_t": log_e_kin_t_sum / log_time_norm,
                "e_mag_p": log_e_mag_p_sum / log_time_norm,
                "e_mag_t": log_e_mag_t_sum / log_time_norm,
            }
            write_log_avg_energies(f_log, energy_means, _vol_oc)

            property_means = {
                "Rm": log_rm_sum / log_time_norm,
                "El": log_el_sum / log_time_norm,
                "ElCmb": log_el_cmb_sum / log_time_norm,
                "Rol": log_rol_sum / log_time_norm,
                "Dip": log_dip_sum / log_time_norm,
                "DipCMB": log_dip_cmb_sum / log_time_norm,
                "dlV": log_dlv_sum / log_time_norm,
                "dmV": log_dmv_sum / log_time_norm,
                "dlB": log_dlb_sum / log_time_norm,
                "dmB": log_dmb_sum / log_time_norm,
                "dlVc": log_dlvc_sum / log_time_norm,
                "dlVPolPeak": log_dlv_polpeak_sum / log_time_norm,
                "rel_a": log_rel_a_sum / log_time_norm,
                "rel_z": log_rel_z_sum / log_time_norm,
                "rel_m": log_rel_m_sum / log_time_norm,
                "rel_na": log_rel_na_sum / log_time_norm,
            }
            write_log_avg_properties(f_log, property_means)

    # Write end-of-run radial profile files
    if kin_accum.n_e_sets > 1:
        write_eKinR_file(os.path.join(output_dir, f"eKinR.{tag}"), kin_accum)
    if l_mag and mag_accum is not None and mag_accum.n_e_sets > 1:
        write_eMagR_file(os.path.join(output_dir, f"eMagR.{tag}"), mag_accum)

    # Write parR.TAG at end of run
    if par_rm_ms.n_calls > 1:
        time_norm_par = par_n_calls * dt
        for ms in [par_rm_ms, par_rol_ms, par_urol_ms,
                   par_dlv_ms, par_dlvc_ms, par_dlpp_ms]:
            ms.finalize(time_norm_par)
        write_parR_file(os.path.join(output_dir, f"parR.{tag}"),
                        par_rm_ms, par_rol_ms, par_urol_ms,
                        par_dlv_ms, par_dlvc_ms, par_dlpp_ms)

    # Write heatR.TAG at end of run
    if l_heat and smean_r is not None and smean_r.n_calls > 1:
        time_norm = heat_n_calls * dt
        smean_r.finalize(time_norm)
        tmean_r.finalize(time_norm)
        pmean_r.finalize(time_norm)
        rhomean_r.finalize(time_norm)
        ximean_r.finalize(time_norm)
        write_heatR_file(os.path.join(output_dir, f"heatR.{tag}"),
                         smean_r, tmean_r, pmean_r, rhomean_r, ximean_r)

    # Write timing + stop to log.TAG
    elapsed = timeit.time() - t_start
    steps_done = n_steps - start_step
    if steps_done > 0:
        write_log_timing(f_log, elapsed / steps_done, elapsed)
        write_log_stop(f_log, sim_time * tScale, n_steps + 1, steps_done)
    f_log.close()

    log_file.close()
    f_ekin.close()
    f_timestep.close()
    if f_emag_oc is not None:
        f_emag_oc.close()
    if f_emag_ic is not None:
        f_emag_ic.close()
    if f_dipole is not None:
        f_dipole.close()
    if f_heat is not None:
        f_heat.close()
    f_par.close()
    if steps_done > 0:
        print(f"\nDone: {steps_done} steps in {elapsed:.1f}s "
              f"({elapsed / steps_done * 1000:.1f}ms/step)")

    return _get_energies()


if __name__ == "__main__":
    run()
