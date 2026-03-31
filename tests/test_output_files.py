"""Test new output files against Fortran reference.

Tests: radius.TAG, signal.TAG, timestep.TAG, eKinR.TAG, eMagR.TAG, dipole.TAG.
Compares Python output against reference files in samples/dynamo_benchmark/.
"""

import pytest
import numpy as np
import io
from pathlib import Path

from magic_torch import fields
from magic_torch.params import dtmax, n_r_max
from magic_torch.output import (
    get_e_kin_radial, get_e_mag_radial, get_dipole,
    get_e_kin_full, get_e_mag_oc_full, get_e_mag_ic,
    RadialAccumulator, write_radius_file, write_signal_file,
    write_timestep_line, write_eKinR_file, write_eMagR_file,
    write_dipole_line,
    MeanSD, update_heat_means, get_heat_data, write_heat_line,
    write_heatR_file,
    get_dlm, get_par_data, get_elsAnel, get_e_mag_cmb,
    update_par_means, write_par_line, write_parR_file,
)
from magic_torch.pre_calculations import tScale

_SAMPLES = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark"


# --- Module-level simulation: run 3 steps, collect everything ---
_dipole_results = {}  # step -> list of 19 floats
_heat_results = {}  # step -> list of 16 floats
_kin_accum = None
_mag_accum = None
_smean_r = None
_tmean_r = None
_pmean_r = None
_rhomean_r = None
_ximean_r = None
_heat_n_calls = 0
_par_results = {}  # step -> list of 19 floats
_ek_results = {}   # step -> EKin namedtuple
_em_results = {}   # step -> EMagOC namedtuple
_eic_results = {}  # step -> EMagIC namedtuple
_par_rm_ms = None
_par_rol_ms = None
_par_urol_ms = None
_par_dlv_ms = None
_par_dlvc_ms = None
_par_dlpp_ms = None
_par_n_calls = 0
_ran = False


def _run_all():
    global _ran, _kin_accum, _mag_accum
    global _smean_r, _tmean_r, _pmean_r, _rhomean_r, _ximean_r, _heat_n_calls
    global _par_rm_ms, _par_rol_ms, _par_urol_ms, _par_dlv_ms, _par_dlvc_ms, _par_dlpp_ms, _par_n_calls
    if _ran:
        return
    _ran = True

    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
    from magic_torch.main import _zero_mag_boundaries

    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)

    _kin_accum = RadialAccumulator(4, n_r_max)
    _mag_accum = RadialAccumulator(5, n_r_max)
    _smean_r = MeanSD(n_r_max)
    _tmean_r = MeanSD(n_r_max)
    _pmean_r = MeanSD(n_r_max)
    _rhomean_r = MeanSD(n_r_max)
    _ximean_r = MeanSD(n_r_max)
    _par_rm_ms = MeanSD(n_r_max)
    _par_rol_ms = MeanSD(n_r_max)
    _par_urol_ms = MeanSD(n_r_max)
    _par_dlv_ms = MeanSD(n_r_max)
    _par_dlvc_ms = MeanSD(n_r_max)
    _par_dlpp_ms = MeanSD(n_r_max)

    def _do_par_step(ek, em, dip_cols):
        global _par_n_calls
        _par_n_calls += 1
        time_passed = dtmax
        time_norm_par = _par_n_calls * dtmax

        dlm_v = get_dlm(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc, 'V')
        dlm_b = get_dlm(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc, 'B')
        elsAnel_val = get_elsAnel(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
        e_mag_cmb_val = get_e_mag_cmb(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
        par_cols = get_par_data(ek, em, dip_cols, dlm_v, dlm_b, elsAnel_val, e_mag_cmb_val)

        kin_r = get_e_kin_radial(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
        ekinR_factored = 0.5 * (kin_r[0] + kin_r[2])
        update_par_means(_par_rm_ms, _par_rol_ms, _par_urol_ms, _par_dlv_ms,
                         _par_dlvc_ms, _par_dlpp_ms, ekinR_factored,
                         dlm_v[1], dlm_v[5], dlm_v[6],
                         time_passed, time_norm_par)
        return par_cols

    # Step 0: compute dipole AND accumulate (step 0 IS included in Fortran)
    ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
    em = get_e_mag_oc_full(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
    eic = get_e_mag_ic(fields.b_LMloc)
    _ek_results[0] = ek
    _em_results[0] = em
    _eic_results[0] = eic
    dip_cols_0 = get_dipole(
        fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc, em.e_p, em.e_t)
    _dipole_results[0] = dip_cols_0

    sim_time = 0.0
    kin_profs = get_e_kin_radial(fields.w_LMloc, fields.dw_LMloc,
                                 fields.z_LMloc)
    _kin_accum.accumulate(sim_time, *kin_profs)
    mag_profs = get_e_mag_radial(fields.b_LMloc, fields.db_LMloc,
                                  fields.aj_LMloc)
    _mag_accum.accumulate(sim_time, *mag_profs)
    _zero_mag_boundaries(_mag_accum)

    # Par diagnostics step 0
    _par_results[0] = _do_par_step(ek, em, dip_cols_0)

    # Heat diagnostics step 0
    _heat_n_calls += 1
    s00 = fields.s_LMloc[0, :].real
    p00 = fields.p_LMloc[0, :].real
    update_heat_means(_smean_r, _tmean_r, _pmean_r, _rhomean_r, _ximean_r,
                      s00, p00, dtmax, _heat_n_calls * dtmax)
    _heat_results[0] = get_heat_data(s00, fields.ds_LMloc[0, :].real, p00)

    # Steps 1-3
    for step in range(1, 4):
        one_step(n_time_step=step, dt=dtmax)
        sim_time += dtmax

        # Compute energies
        ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
        em = get_e_mag_oc_full(fields.b_LMloc, fields.db_LMloc,
                                fields.aj_LMloc)
        eic = get_e_mag_ic(fields.b_LMloc)
        _ek_results[step] = ek
        _em_results[step] = em
        _eic_results[step] = eic
        dip_cols_step = get_dipole(
            fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc, em.e_p, em.e_t)
        _dipole_results[step] = dip_cols_step

        # Accumulate radial profiles
        kin_profs = get_e_kin_radial(fields.w_LMloc, fields.dw_LMloc,
                                     fields.z_LMloc)
        _kin_accum.accumulate(sim_time, *kin_profs)

        mag_profs = get_e_mag_radial(fields.b_LMloc, fields.db_LMloc,
                                      fields.aj_LMloc)
        _mag_accum.accumulate(sim_time, *mag_profs)
        _zero_mag_boundaries(_mag_accum)

        # Par diagnostics
        _par_results[step] = _do_par_step(ek, em, dip_cols_step)

        # Heat diagnostics
        _heat_n_calls += 1
        s00 = fields.s_LMloc[0, :].real
        p00 = fields.p_LMloc[0, :].real
        update_heat_means(_smean_r, _tmean_r, _pmean_r, _rhomean_r, _ximean_r,
                          s00, p00, dtmax, _heat_n_calls * dtmax)
        _heat_results[step] = get_heat_data(
            s00, fields.ds_LMloc[0, :].real, p00)


# --- radius.TAG ---

def test_radius():
    """Compare radius.TAG against Fortran reference (exact byte match)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        path = f.name
    write_radius_file(path)
    with open(path) as f:
        py_lines = f.readlines()
    ref = np.loadtxt(_SAMPLES / "radius.test")
    assert len(py_lines) == n_r_max
    for i, line in enumerate(py_lines):
        parts = line.split()
        assert int(parts[0]) == i + 1, f"Row {i}: index mismatch"
        py_r = float(parts[1])
        assert abs(py_r - ref[i, 1]) < 1e-15 * abs(ref[i, 1]) + 1e-30, \
            f"Row {i}: r mismatch: {py_r} vs {ref[i, 1]}"
    import os
    os.unlink(path)


# --- signal.TAG ---

def test_signal():
    """signal.TAG should contain 'NOT'."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        path = f.name
    write_signal_file(path)
    with open(path) as f:
        content = f.read()
    assert content == "NOT\n"
    import os
    os.unlink(path)


# --- timestep.TAG ---

def test_timestep_format():
    """Verify timestep.TAG format matches Fortran (ES20.12, ES16.8)."""
    buf = io.StringIO()
    write_timestep_line(buf, 0.0, 1e-4)
    line = buf.getvalue()
    # Total: 20 + 16 + 1(newline) = 37
    assert len(line) == 37, f"Expected 37 chars, got {len(line)}"
    parts = line.split()
    assert abs(float(parts[0])) < 1e-30
    assert abs(float(parts[1]) - 1e-4) < 1e-15


def test_timestep_values():
    """Compare timestep.TAG initial line against reference."""
    ref_line = (_SAMPLES / "timestep.test").read_text().strip()
    ref_parts = ref_line.split()
    buf = io.StringIO()
    write_timestep_line(buf, 0.0, dtmax)
    py_parts = buf.getvalue().strip().split()
    assert abs(float(py_parts[0]) - float(ref_parts[0])) < 1e-30
    assert abs(float(py_parts[1]) - float(ref_parts[1])) < 1e-15


# --- eKinR.TAG ---

def test_eKinR():
    """Compare eKinR.TAG against Fortran reference."""
    _run_all()
    import tempfile, os
    path = tempfile.mktemp(suffix='.dat')
    write_eKinR_file(path, _kin_accum)

    py_data = np.loadtxt(path)
    ref_data = np.loadtxt(_SAMPLES / "eKinR.test")
    os.unlink(path)

    assert py_data.shape == ref_data.shape, \
        f"Shape mismatch: {py_data.shape} vs {ref_data.shape}"

    for nR in range(n_r_max):
        for col in range(9):
            py_val = py_data[nR, col]
            ref_val = ref_data[nR, col]
            if abs(ref_val) < 1e-20:
                assert abs(py_val) < 1e-15, \
                    f"eKinR[{nR},{col}]: expected ~0, got {py_val:.6e}"
            else:
                rel_err = abs(py_val - ref_val) / abs(ref_val)
                assert rel_err < 1e-6, \
                    f"eKinR[{nR},{col}]: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- eMagR.TAG ---

def test_eMagR():
    """Compare eMagR.TAG against Fortran reference."""
    _run_all()
    import tempfile, os
    path = tempfile.mktemp(suffix='.dat')
    write_eMagR_file(path, _mag_accum)

    py_data = np.loadtxt(path)
    ref_data = np.loadtxt(_SAMPLES / "eMagR.test")
    os.unlink(path)

    assert py_data.shape == ref_data.shape, \
        f"Shape mismatch: {py_data.shape} vs {ref_data.shape}"

    for nR in range(n_r_max):
        for col in range(10):
            py_val = py_data[nR, col]
            ref_val = ref_data[nR, col]
            if abs(ref_val) < 1e-20:
                assert abs(py_val) < 1e-15, \
                    f"eMagR[{nR},{col}]: expected ~0, got {py_val:.6e}"
            else:
                rel_err = abs(py_val - ref_val) / abs(ref_val)
                assert rel_err < 1e-6, \
                    f"eMagR[{nR},{col}]: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- dipole.TAG ---

_DIPOLE_RTOL = 1e-5  # ES14.6 precision


@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_dipole(step):
    """Compare dipole.TAG against Fortran reference for each step."""
    _run_all()
    ref_data = np.loadtxt(_SAMPLES / "dipole.test")
    ref_row = ref_data[step]  # 20 columns: time + 19 values

    py_cols = _dipole_results[step]  # 19 values (cols 2-20)
    py_time = step * dtmax * tScale

    # Check time (col 1)
    assert abs(py_time - ref_row[0]) < 1e-15 * max(1.0, abs(ref_row[0])), \
        f"step {step}: time mismatch: py={py_time} ref={ref_row[0]}"

    # Check cols 2-20 (our indices 0-18 in py_cols, reference cols 1-19)
    col_names = [
        "theta_dip", "phi_dip", "Dip", "DipCMB", "ax_dip/geo",
        "dip/tot", "dip_cmb/cmb", "dip_cmb/geo",
        "e_dip_cmb", "e_dip_ax_cmb", "e_dipole", "e_dipole_ax",
        "e_cmb", "e_geo", "e_p_e_ratio",
        "EA_cmb", "nonax_cmb", "EA_geo", "nonax_geo",
    ]
    for i, name in enumerate(col_names):
        py_val = py_cols[i]
        ref_val = ref_row[i + 1]

        if abs(ref_val) < 1e-20:
            assert abs(py_val) < 1e-10, \
                f"step {step} {name}: expected ~0, got {py_val:.6e}"
        else:
            rel_err = abs(py_val - ref_val) / abs(ref_val)
            assert rel_err < _DIPOLE_RTOL, \
                f"step {step} {name}: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- dipole synthetic test ---

def test_dipole_synthetic_theta():
    """Verify theta_dip/phi_dip with known b10/b11 coefficients."""
    import torch, math
    from magic_torch.params import lm_max
    from magic_torch.output import _lm_l1m0, _lm_l1m1

    b = torch.zeros(lm_max, n_r_max, dtype=torch.complex128)
    db = torch.zeros_like(b)
    aj = torch.zeros_like(b)

    # Pure axial dipole pointing south: b10 < 0, b11 = 0
    b[_lm_l1m0, :] = -1.0 + 0.0j
    db[_lm_l1m0, :] = 0.1  # nonzero so total energy > 0
    cols = get_dipole(b, db, aj, 1.0, 0.0)
    assert abs(cols[0] - 180.0) < 1e-10, \
        f"South axial dipole: theta_dip={cols[0]}, expected 180"
    assert abs(cols[1]) < 1e-10, \
        f"South axial dipole: phi_dip={cols[1]}, expected 0"

    # Pure axial dipole pointing north: b10 > 0, b11 = 0
    b[_lm_l1m0, :] = 1.0 + 0.0j
    cols2 = get_dipole(b, db, aj, 1.0, 0.0)
    assert abs(cols2[0]) < 1e-10, \
        f"North axial dipole: theta_dip={cols2[0]}, expected 0"


# --- Format tests ---

def test_eKinR_line_width():
    """Verify eKinR line width: ES20.10 + 8*ES15.7 = 20 + 120 = 140."""
    _run_all()
    import tempfile, os
    path = tempfile.mktemp(suffix='.dat')
    write_eKinR_file(path, _kin_accum)
    with open(path) as f:
        line = f.readline()
    os.unlink(path)
    # 20 + 8*15 + 1(newline) = 141
    assert len(line) == 141, f"Expected 141 chars, got {len(line)}"


def test_eMagR_line_width():
    """Verify eMagR line width: ES20.10 + 9*ES15.7 = 20 + 135 = 155."""
    _run_all()
    import tempfile, os
    path = tempfile.mktemp(suffix='.dat')
    write_eMagR_file(path, _mag_accum)
    with open(path) as f:
        line = f.readline()
    os.unlink(path)
    # 20 + 9*15 + 1(newline) = 156
    assert len(line) == 156, f"Expected 156 chars, got {len(line)}"


def test_dipole_line_width():
    """Verify dipole line width: ES20.12 + 19*ES14.6 = 20 + 266 = 286."""
    _run_all()
    buf = io.StringIO()
    write_dipole_line(buf, 0.0, _dipole_results[0])
    line = buf.getvalue()
    # 20 + 19*14 + 1(newline) = 287
    assert len(line) == 287, f"Expected 287 chars, got {len(line)}"


# --- heat.TAG ---

@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_heat(step):
    """Compare heat.TAG against Fortran reference for each step."""
    _run_all()
    ref_data = np.loadtxt(_SAMPLES / "heat.test")
    ref_row = ref_data[step]  # 17 columns: time + 16 values

    py_cols = _heat_results[step]  # 16 values
    py_time = step * dtmax * tScale

    # Check time (col 0)
    assert abs(py_time - ref_row[0]) < 1e-15 * max(1.0, abs(ref_row[0])), \
        f"step {step}: time mismatch: py={py_time} ref={ref_row[0]}"

    col_names = [
        "botnuss", "topnuss", "deltanuss",
        "bottemp", "toptemp", "botentropy", "topentropy",
        "botflux", "topflux", "toppres", "mass",
        "topsherwood", "botsherwood", "deltasherwood", "botxi", "topxi",
    ]
    for i, name in enumerate(col_names):
        py_val = py_cols[i]
        ref_val = ref_row[i + 1]

        if abs(ref_val) < 1e-10:
            assert abs(py_val) < 1e-10, \
                f"step {step} {name}: expected ~0, got {py_val:.6e}"
        else:
            rel_err = abs(py_val - ref_val) / abs(ref_val)
            assert rel_err < 1e-7, \
                f"step {step} {name}: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- heatR.TAG ---

def test_heatR():
    """Compare heatR.TAG against Fortran reference."""
    _run_all()
    import tempfile, os

    # Finalize accumulators
    time_norm = _heat_n_calls * dtmax
    # Clone accumulators to avoid modifying shared state
    import copy
    sm = copy.deepcopy(_smean_r)
    tm = copy.deepcopy(_tmean_r)
    pm = copy.deepcopy(_pmean_r)
    rm = copy.deepcopy(_rhomean_r)
    xm = copy.deepcopy(_ximean_r)
    sm.finalize(time_norm)
    tm.finalize(time_norm)
    pm.finalize(time_norm)
    rm.finalize(time_norm)
    xm.finalize(time_norm)

    path = tempfile.mktemp(suffix='.dat')
    write_heatR_file(path, sm, tm, pm, rm, xm)

    py_data = np.loadtxt(path)
    ref_data = np.loadtxt(_SAMPLES / "heatR.test")
    os.unlink(path)

    assert py_data.shape == ref_data.shape, \
        f"Shape mismatch: {py_data.shape} vs {ref_data.shape}"

    for nR in range(n_r_max):
        for col in range(11):
            py_val = py_data[nR, col]
            ref_val = ref_data[nR, col]
            if abs(ref_val) < 1e-10:
                assert abs(py_val) < 1e-10, \
                    f"heatR[{nR},{col}]: expected ~0, got {py_val:.6e}"
            else:
                rel_err = abs(py_val - ref_val) / abs(ref_val)
                assert rel_err < 1e-5, \
                    f"heatR[{nR},{col}]: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- heat format tests ---

def test_heat_line_width():
    """Verify heat line width: ES20.12 + 16*ES16.8 = 20 + 256 = 276."""
    _run_all()
    buf = io.StringIO()
    write_heat_line(buf, 0.0, _heat_results[0])
    line = buf.getvalue()
    # 20 + 16*16 + 1(newline) = 277
    assert len(line) == 277, f"Expected 277 chars, got {len(line)}"


def test_heatR_line_width():
    """Verify heatR line width: ES20.10 + 5*ES15.7 + 5*ES13.5 = 20+75+65 = 160."""
    _run_all()
    import tempfile, os, copy

    time_norm = _heat_n_calls * dtmax
    sm = copy.deepcopy(_smean_r)
    tm = copy.deepcopy(_tmean_r)
    pm = copy.deepcopy(_pmean_r)
    rm = copy.deepcopy(_rhomean_r)
    xm = copy.deepcopy(_ximean_r)
    sm.finalize(time_norm)
    tm.finalize(time_norm)
    pm.finalize(time_norm)
    rm.finalize(time_norm)
    xm.finalize(time_norm)

    path = tempfile.mktemp(suffix='.dat')
    write_heatR_file(path, sm, tm, pm, rm, xm)
    with open(path) as f:
        line = f.readline()
    os.unlink(path)
    # 20 + 5*15 + 5*13 + 1(newline) = 161
    assert len(line) == 161, f"Expected 161 chars, got {len(line)}"


# --- par.TAG ---

_PAR_RTOL = 1e-7  # ES16.8 precision


@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_par(step):
    """Compare par.TAG against Fortran reference for each step."""
    _run_all()
    ref_data = np.loadtxt(_SAMPLES / "par.test")
    ref_row = ref_data[step]  # 20 columns: time + 19 values

    py_cols = _par_results[step]  # 19 values
    py_time = step * dtmax * tScale

    # Check time (col 0)
    assert abs(py_time - ref_row[0]) < 1e-15 * max(1.0, abs(ref_row[0])), \
        f"step {step}: time mismatch: py={py_time} ref={ref_row[0]}"

    col_names = [
        "Rm", "El", "Rol", "Geos", "Dip", "DipCMB",
        "dlV", "dmV", "dpV", "dzV", "lvDiss", "lbDiss",
        "dlB", "dmB", "ElCmb", "RolC", "dlVc", "dlVPolPeak", "ReEquat",
    ]
    for i, name in enumerate(col_names):
        py_val = py_cols[i]
        ref_val = ref_row[i + 1]

        if abs(ref_val) < 1e-20:
            assert abs(py_val) < 1e-10, \
                f"step {step} {name}: expected ~0, got {py_val:.6e}"
        else:
            rel_err = abs(py_val - ref_val) / abs(ref_val)
            assert rel_err < _PAR_RTOL, \
                f"step {step} {name}: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- parR.TAG ---

def test_parR():
    """Compare parR.TAG against Fortran reference."""
    _run_all()
    import tempfile, os, copy

    time_norm_par = _par_n_calls * dtmax
    rm = copy.deepcopy(_par_rm_ms)
    rol = copy.deepcopy(_par_rol_ms)
    urol = copy.deepcopy(_par_urol_ms)
    dlv = copy.deepcopy(_par_dlv_ms)
    dlvc = copy.deepcopy(_par_dlvc_ms)
    dlpp = copy.deepcopy(_par_dlpp_ms)
    for ms in [rm, rol, urol, dlv, dlvc, dlpp]:
        ms.finalize(time_norm_par)

    path = tempfile.mktemp(suffix='.dat')
    write_parR_file(path, rm, rol, urol, dlv, dlvc, dlpp)

    py_data = np.loadtxt(path)
    ref_data = np.loadtxt(_SAMPLES / "parR.test")
    os.unlink(path)

    assert py_data.shape == ref_data.shape, \
        f"Shape mismatch: {py_data.shape} vs {ref_data.shape}"

    for nR in range(n_r_max):
        for col in range(13):
            py_val = py_data[nR, col]
            ref_val = ref_data[nR, col]
            if abs(ref_val) < 1e-10:
                assert abs(py_val) < 1e-10, \
                    f"parR[{nR},{col}]: expected ~0, got {py_val:.6e}"
            else:
                rel_err = abs(py_val - ref_val) / abs(ref_val)
                assert rel_err < 1e-5, \
                    f"parR[{nR},{col}]: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


# --- par format tests ---

def test_par_line_width():
    """Verify par line width: ES20.12 + 19*ES16.8 = 20 + 304 = 324."""
    _run_all()
    buf = io.StringIO()
    write_par_line(buf, 0.0, _par_results[0])
    line = buf.getvalue()
    # 20 + 19*16 + 1(newline) = 325
    assert len(line) == 325, f"Expected 325 chars, got {len(line)}"


def test_parR_line_width():
    """Verify parR line width: ES20.10 + 6*ES15.7 + 6*ES13.5 = 20+90+78 = 188."""
    _run_all()
    import tempfile, os, copy

    time_norm_par = _par_n_calls * dtmax
    rm = copy.deepcopy(_par_rm_ms)
    rol = copy.deepcopy(_par_rol_ms)
    urol = copy.deepcopy(_par_urol_ms)
    dlv = copy.deepcopy(_par_dlv_ms)
    dlvc = copy.deepcopy(_par_dlvc_ms)
    dlpp = copy.deepcopy(_par_dlpp_ms)
    for ms in [rm, rol, urol, dlv, dlvc, dlpp]:
        ms.finalize(time_norm_par)

    path = tempfile.mktemp(suffix='.dat')
    write_parR_file(path, rm, rol, urol, dlv, dlvc, dlpp)
    with open(path) as f:
        line = f.readline()
    os.unlink(path)
    # 20 + 6*15 + 6*13 + 1(newline) = 189
    assert len(line) == 189, f"Expected 189 chars, got {len(line)}"


# --- G_1.TAG (binary graph file) ---

# Cache the graph file bytes so we only generate once
_graph_py_data = None
_graph_ref_data = None


def _graph_data():
    """Generate Python G_1 and load Fortran reference (cached)."""
    global _graph_py_data, _graph_ref_data
    if _graph_py_data is not None:
        return _graph_ref_data, _graph_py_data

    _run_all()

    from magic_torch.graph_output import write_graph_file
    from magic_torch.params import n_theta_max, n_phi_max, n_r_max

    sim_time = 3 * dtmax
    t_out = sim_time * tScale

    buf = io.BytesIO()
    write_graph_file(buf, t_out,
                     fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc,
                     fields.s_LMloc, fields.p_LMloc,
                     fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
    _graph_py_data = buf.getvalue()

    ref_path = _SAMPLES / "G_1.test"
    _graph_ref_data = ref_path.read_bytes()

    return _graph_ref_data, _graph_py_data


def test_graph_file_size():
    """G_1.TAG must have exact file size."""
    ref, py = _graph_data()
    assert len(py) == len(ref), f"Size mismatch: {len(py)} vs {len(ref)}"


def test_graph_header():
    """Header bytes must match (except tiny float32 rounding in r/r_ic)."""
    ref, py = _graph_data()
    # First 152 bytes: version + runid + time + params + ints + logicals
    # Then theta_ord (96 bytes) + r (132 bytes) + r_ic (68 bytes) = 448 total
    # Allow up to 12 byte diffs (Chebyshev grid float32 rounding)
    header = 448
    n_diff = sum(1 for a, b in zip(ref[:header], py[:header]) if a != b)
    assert n_diff <= 12, f"Too many header byte diffs: {n_diff}"


def _read_field_colmajor(data, offset, n_theta, n_phi):
    """Read a 2D field from Fortran column-major binary (BE float32)."""
    sz = n_theta * n_phi * 4
    flat = np.frombuffer(data[offset:offset + sz], dtype='>f4')
    return flat.reshape(n_phi, n_theta).T  # Fortran col-major → C row-major


@pytest.mark.parametrize("field_name,field_idx", [
    ("vr", 0), ("vt", 1), ("vp", 2), ("sr", 3), ("pr", 4),
    ("br", 5), ("bt", 6), ("bp", 7),
])
def test_graph_oc_field(field_name, field_idx):
    """OC fields: ≤1 ULP float32 tolerance (no fudge factors).

    On CPU the SHT matches Fortran to ~1e-15. After float32 truncation,
    values can differ by at most 1 ULP. We use atol = eps32 * global_field_max
    (the field max across ALL radial levels) — this is the tightest tolerance
    consistent with the output format. Near-zero boundary values (e.g. sr at
    CMB) are within tolerance because they're zero to within 1 ULP of the
    field's dynamic range.
    """
    ref, py = _graph_data()
    from magic_torch.params import n_theta_max, n_phi_max, n_r_max
    eps32 = np.finfo(np.float32).eps  # 1.19e-7
    header = 448
    field_size = n_theta_max * n_phi_max * 4
    n_fields = 8

    # Compute global max across all radial levels for this field
    global_max = 0.0
    for nR in range(n_r_max):
        base = header + nR * n_fields * field_size
        offset = base + field_idx * field_size
        ref_f = _read_field_colmajor(ref, offset, n_theta_max, n_phi_max)
        global_max = max(global_max, np.abs(ref_f).max())
    atol = eps32 * max(global_max, 1e-30)

    for nR in range(n_r_max):
        base = header + nR * n_fields * field_size
        offset = base + field_idx * field_size
        ref_f = _read_field_colmajor(ref, offset, n_theta_max, n_phi_max)
        py_f = _read_field_colmajor(py, offset, n_theta_max, n_phi_max)
        np.testing.assert_allclose(py_f, ref_f, rtol=0, atol=atol,
                                   err_msg=f"{field_name} nR={nR}")


@pytest.mark.parametrize("field_name,field_idx", [
    ("br_ic", 0), ("bt_ic", 1), ("bp_ic", 2),
])
def test_graph_ic_field(field_name, field_idx):
    """IC fields: ≤1 ULP float32 tolerance (no fudge factors)."""
    ref, py = _graph_data()
    from magic_torch.params import n_theta_max, n_phi_max, n_r_max, n_r_ic_max
    eps32 = np.finfo(np.float32).eps
    header = 448
    field_size = n_theta_max * n_phi_max * 4
    n_fields_oc = 8
    oc_end = header + n_r_max * n_fields_oc * field_size

    global_max = 0.0
    for nR in range(n_r_ic_max):
        base = oc_end + nR * 3 * field_size
        offset = base + field_idx * field_size
        ref_f = _read_field_colmajor(ref, offset, n_theta_max, n_phi_max)
        global_max = max(global_max, np.abs(ref_f).max())
    atol = eps32 * max(global_max, 1e-30)

    for nR in range(n_r_ic_max):
        base = oc_end + nR * 3 * field_size
        offset = base + field_idx * field_size
        ref_f = _read_field_colmajor(ref, offset, n_theta_max, n_phi_max)
        py_f = _read_field_colmajor(py, offset, n_theta_max, n_phi_max)
        np.testing.assert_allclose(py_f, ref_f, rtol=0, atol=atol,
                                   err_msg=f"{field_name} nR_ic={nR}")


# --- log.TAG tests ---


_log_content = None
_log_ek = {}     # step -> EKin
_log_em = {}     # step -> EMagOC
_log_eic = {}    # step -> EMagIC
_log_par = {}    # step -> par_cols list


def _build_log():
    """Build log.TAG content from 3-step simulation, parse reference."""
    global _log_content
    if _log_content is not None:
        return
    _run_all()

    from magic_torch.log_output import (
        write_log_header, write_log_scheme_info, write_log_boundary_info,
        write_log_namelists, write_log_dtmax_info, write_log_physical_info,
        write_log_start, write_log_step, write_log_store,
        write_log_end_energies, write_log_avg_energies,
        write_log_avg_properties, write_log_timing, write_log_stop,
    )
    from magic_torch.time_scheme import tscheme
    from magic_torch.radial_functions import vol_oc, vol_ic

    buf = io.StringIO()
    cfg = {"tag": "test", "n_steps": 3}
    write_log_header(buf)
    write_log_scheme_info(buf, tscheme)
    write_log_boundary_info(buf)
    write_log_namelists(buf, cfg)
    write_log_dtmax_info(buf, dtmax)
    write_log_physical_info(buf)
    write_log_start(buf, 0.0, 0, dtmax)

    # Simulate the accumulation done in main.py
    time_norm = 0.0
    ek_p_sum = 0.0; ek_t_sum = 0.0; em_p_sum = 0.0; em_t_sum = 0.0
    rm_sum = 0.0; el_sum = 0.0; el_cmb_sum = 0.0; rol_sum = 0.0
    dip_sum = 0.0; dip_cmb_sum = 0.0
    dlv_sum = 0.0; dmv_sum = 0.0; dlb_sum = 0.0; dmb_sum = 0.0
    dlvc_sum = 0.0; dlpp_sum = 0.0
    rel_a_sum = 0.0; rel_z_sum = 0.0; rel_m_sum = 0.0; rel_na_sum = 0.0

    for step in range(4):  # steps 0-3
        ek = _ek_results[step]
        em = _em_results[step]
        eic = _eic_results[step]

        par_cols = _par_results[step]

        tp = dtmax
        time_norm += tp
        ek_p_sum += tp * ek.e_p; ek_t_sum += tp * ek.e_t
        em_p_sum += tp * em.e_p; em_t_sum += tp * em.e_t
        rm_sum += tp * par_cols[0]
        el_sum += tp * par_cols[1]
        rol_sum += tp * par_cols[2]
        dip_sum += tp * par_cols[4]
        dip_cmb_sum += tp * par_cols[5]
        dlv_sum += tp * par_cols[6]
        dmv_sum += tp * par_cols[7]
        dlb_sum += tp * par_cols[12]
        dmb_sum += tp * par_cols[13]
        el_cmb_sum += tp * par_cols[14]
        dlvc_sum += tp * par_cols[16]
        dlpp_sum += tp * par_cols[17]

        e_kin = ek.e_p + ek.e_t
        if e_kin > 0:
            rel_a_sum += tp * (ek.e_p_as + ek.e_t_as) / e_kin
            rel_z_sum += tp * ek.e_t_as / e_kin
            rel_m_sum += tp * ek.e_p_as / e_kin
            rel_na_sum += tp * (e_kin - ek.e_p_as - ek.e_t_as) / e_kin

    for step in range(1, 4):
        write_log_step(buf, step, step * 0.01)

    t_out = 3 * dtmax * tScale
    write_log_store(buf, "graphic", t_out, 4, "G_1.test")
    write_log_store(buf, "checkpoint", t_out, 4, "checkpoint_end.test")

    # Use last-step energies for end energies
    ek_last = _ek_results[3]
    em_last = _em_results[3]
    eic_last = _eic_results[3]
    write_log_end_energies(buf, ek_last, em_last, eic_last, vol_oc, vol_ic)

    energy_means = {
        "e_kin_p": ek_p_sum / time_norm,
        "e_kin_t": ek_t_sum / time_norm,
        "e_mag_p": em_p_sum / time_norm,
        "e_mag_t": em_t_sum / time_norm,
    }
    write_log_avg_energies(buf, energy_means, vol_oc)

    property_means = {
        "Rm": rm_sum / time_norm, "El": el_sum / time_norm,
        "ElCmb": el_cmb_sum / time_norm, "Rol": rol_sum / time_norm,
        "Dip": dip_sum / time_norm, "DipCMB": dip_cmb_sum / time_norm,
        "dlV": dlv_sum / time_norm, "dmV": dmv_sum / time_norm,
        "dlB": dlb_sum / time_norm, "dmB": dmb_sum / time_norm,
        "dlVc": dlvc_sum / time_norm, "dlVPolPeak": dlpp_sum / time_norm,
        "rel_a": rel_a_sum / time_norm, "rel_z": rel_z_sum / time_norm,
        "rel_m": rel_m_sum / time_norm, "rel_na": rel_na_sum / time_norm,
    }
    write_log_avg_properties(buf, property_means)
    write_log_timing(buf, 0.01, 0.03)
    write_log_stop(buf, t_out, 4, 3)

    _log_content = buf.getvalue()


def _parse_log_line(prefix):
    """Find and return lines matching a prefix from log content."""
    _build_log()
    return [l for l in _log_content.splitlines() if prefix in l]


def _parse_ref_line(prefix):
    """Find and return lines matching prefix from Fortran reference log."""
    ref = (_SAMPLES / "log.test").read_text()
    return [l for l in ref.splitlines() if prefix in l]


def test_log_physical_info():
    """Compare MOI, volume, surface values against Fortran reference."""
    _build_log()
    ref_text = (_SAMPLES / "log.test").read_text()
    ref_lines = ref_text.splitlines()

    # Find the physical info section in reference
    checks = {
        "OC moment of inertia": None,
        "IC moment of inertia": None,
        "MA moment of inertia": None,
        "IC volume": None,
        "OC volume": None,
        "IC surface": None,
        "OC surface": None,
    }
    for label in checks:
        for line in ref_lines:
            if label in line:
                # Extract the number after the last ':'
                val_str = line.split(":")[-1].strip()
                checks[label] = float(val_str)
                break

    py_lines = _log_content.splitlines()
    for label in checks:
        ref_val = checks[label]
        assert ref_val is not None, f"Could not find '{label}' in reference"
        py_val = None
        for line in py_lines:
            if label in line:
                val_str = line.split(":")[-1].strip()
                py_val = float(val_str)
                break
        assert py_val is not None, f"Could not find '{label}' in Python log"
        rel_err = abs(py_val - ref_val) / abs(ref_val)
        assert rel_err < 1e-6, \
            f"{label}: py={py_val:.10e} ref={ref_val:.10e} rel={rel_err:.2e}"


def test_log_end_energies():
    """Compare end-of-run energy lines against Fortran reference."""
    _build_log()
    ref_text = (_SAMPLES / "log.test").read_text()

    # Parse 3 energy lines from reference
    ref_kin = ref_mag = ref_ic = None
    for line in ref_text.splitlines():
        if "Kinetic energies:" in line and "averaged" not in ref_text.splitlines()[
                ref_text.splitlines().index(line) - 2]:
            ref_kin = line
        if "OC mag. energies:" in line and "averaged" not in ref_text.splitlines()[
                ref_text.splitlines().index(line) - 2]:
            ref_mag = line
        if "IC mag. energies:" in line:
            ref_ic = line

    # More robust: find lines after "Energies at end of time integration"
    ref_lines = ref_text.splitlines()
    for i, line in enumerate(ref_lines):
        if "Energies at end of time integration" in line:
            ref_kin = ref_lines[i + 2]  # skip "(total,poloidal,...)"
            ref_mag = ref_lines[i + 3]
            ref_ic = ref_lines[i + 4]
            break

    py_lines = _log_content.splitlines()
    for i, line in enumerate(py_lines):
        if "Energies at end of time integration" in line:
            py_kin = py_lines[i + 2]
            py_mag = py_lines[i + 3]
            py_ic = py_lines[i + 4]
            break

    def _parse_energy_line(line):
        # Extract 4 numbers after the label
        parts = line.split(":")[-1].split()
        return [float(x) for x in parts]

    for label, ref_line, py_line in [
        ("Kinetic", ref_kin, py_kin),
        ("OC mag", ref_mag, py_mag),
        ("IC mag", ref_ic, py_ic),
    ]:
        ref_vals = _parse_energy_line(ref_line)
        py_vals = _parse_energy_line(py_line)
        assert len(ref_vals) == 4, f"{label}: expected 4 values, got {len(ref_vals)}"
        assert len(py_vals) == 4, f"{label}: expected 4 values, got {len(py_vals)}"
        for j, (rv, pv) in enumerate(zip(ref_vals, py_vals)):
            if abs(rv) < 1e-20:
                assert abs(pv) < 1e-10, f"{label}[{j}]: expected ~0, got {pv}"
            else:
                rel_err = abs(pv - rv) / abs(rv)
                assert rel_err < 1e-6, \
                    f"{label}[{j}]: py={pv:.10e} ref={rv:.10e} rel={rel_err:.2e}"


def test_log_avg_energies():
    """Compare time-averaged energy lines against Fortran reference."""
    _build_log()
    ref_text = (_SAMPLES / "log.test").read_text()
    ref_lines = ref_text.splitlines()

    for i, line in enumerate(ref_lines):
        if "Time averaged energies" in line:
            ref_kin = ref_lines[i + 2]
            ref_mag = ref_lines[i + 3]
            break

    py_lines = _log_content.splitlines()
    for i, line in enumerate(py_lines):
        if "Time averaged energies" in line:
            py_kin = py_lines[i + 2]
            py_mag = py_lines[i + 3]
            break

    def _parse_energy_line(line):
        parts = line.split(":")[-1].split()
        return [float(x) for x in parts]

    for label, ref_line, py_line in [
        ("Kin avg", ref_kin, py_kin),
        ("Mag avg", ref_mag, py_mag),
    ]:
        ref_vals = _parse_energy_line(ref_line)
        py_vals = _parse_energy_line(py_line)
        for j, (rv, pv) in enumerate(zip(ref_vals, py_vals)):
            if abs(rv) < 1e-20:
                assert abs(pv) < 1e-10, f"{label}[{j}]: expected ~0, got {pv}"
            else:
                rel_err = abs(pv - rv) / abs(rv)
                assert rel_err < 1e-5, \
                    f"{label}[{j}]: py={pv:.10e} ref={rv:.10e} rel={rel_err:.2e}"


def test_log_avg_properties():
    """Compare time-averaged property lines against Fortran reference."""
    _build_log()
    ref_text = (_SAMPLES / "log.test").read_text()
    ref_lines = ref_text.splitlines()
    py_lines = _log_content.splitlines()

    # Find the "Time averaged property parameters" section
    def _find_props(lines):
        props = {}
        in_section = False
        for line in lines:
            if "Time averaged property parameters" in line:
                in_section = True
                continue
            if in_section:
                # Stop at lines without "!" or at unrelated sections
                if "No of stored" in line or "Mean wall" in line:
                    break
                if ":" in line and "!" in line:
                    parts = line.split(":")
                    label = parts[0].replace("!", "").strip()
                    vals = parts[-1].strip().split()
                    if vals:
                        try:
                            props[label] = [float(v) for v in vals]
                        except ValueError:
                            break
                elif "!" not in line:
                    break
        return props

    ref_props = _find_props(ref_lines)
    py_props = _find_props(py_lines)

    for label, ref_vals in ref_props.items():
        assert label in py_props, f"Missing property: {label}"
        py_vals = py_props[label]
        assert len(py_vals) == len(ref_vals), \
            f"{label}: expected {len(ref_vals)} values, got {len(py_vals)}"
        for j, (rv, pv) in enumerate(zip(ref_vals, py_vals)):
            if abs(rv) < 1e-20:
                assert abs(pv) < 1e-6, \
                    f"{label}[{j}]: expected ~0, got {pv}"
            else:
                rel_err = abs(pv - rv) / abs(rv)
                assert rel_err < 1e-4, \
                    f"{label}[{j}]: py={pv:.10e} ref={rv:.10e} rel={rel_err:.2e}"


def test_log_step_messages():
    """Verify 3 'Time step finished' messages with N=1,2,3."""
    _build_log()
    step_lines = [l for l in _log_content.splitlines()
                  if "Time step finished:" in l]
    assert len(step_lines) == 3, f"Expected 3 step messages, got {len(step_lines)}"
    for i, line in enumerate(step_lines, 1):
        n = int(line.split(":")[-1].strip())
        assert n == i, f"Expected step {i}, got {n}"


def test_log_start_stop():
    """Verify start and stop blocks have correct step numbers."""
    _build_log()
    lines = _log_content.splitlines()

    # Find start block
    start_step = None
    for line in lines:
        if "step no    =" in line:
            start_step = int(line.split("=")[-1].strip())
            break
    assert start_step == 0, f"start step={start_step}, expected 0"

    # Find stop block
    stop_step = None
    steps_gone = None
    for line in lines:
        if "stop step =" in line:
            stop_step = int(line.split("=")[-1].strip())
        if "steps gone=" in line:
            steps_gone = int(line.split("=")[-1].strip())
    assert stop_step == 4, f"stop step={stop_step}, expected 4"
    assert steps_gone == 3, f"steps gone={steps_gone}, expected 3"


def test_log_grid_params():
    """Verify grid parameters in log match expected values."""
    _build_log()
    lines = _log_content.splitlines()

    # Extract only the grid parameters section (between "Grid parameters:" and next blank line)
    grid_lines = []
    in_grid = False
    for line in lines:
        if "Grid parameters:" in line:
            in_grid = True
            continue
        if in_grid:
            if line.strip() == "":
                break
            grid_lines.append(line)

    expected = {"n_r_max": 33, "l_max": 16, "lm_max": 153}
    for key, val in expected.items():
        found = False
        for line in grid_lines:
            # Match exact key (avoid l_max matching l_max_cmb)
            stripped = line.strip()
            if stripped.startswith(key) and "=" in stripped:
                parts = line.split("=")
                num = int(parts[1].strip().split()[0])
                assert num == val, f"{key}: got {num}, expected {val}"
                found = True
                break
        assert found, f"Could not find {key} in Grid parameters section"


# --- Checkpoint tests ---

_SAMPLES_BB = Path(__file__).parent.parent.parent / "samples" / "boussBenchSat"


def test_checkpoint_read_big_endian():
    """Read big-endian dynamo_benchmark checkpoint, verify header."""
    from magic_torch.checkpoint_io import read_checkpoint
    ck = read_checkpoint(str(_SAMPLES / "checkpoint_end.test"))
    assert ck.version == 5
    assert ck.endian == ">"
    assert abs(ck.time - 3e-4) < 1e-10
    assert ck.ra == 1e5
    assert ck.n_r_max == 33
    assert ck.l_max == 16
    assert ck.family == "MULTISTEP"
    assert ck.n_time_step == 4
    assert ck.l_heat is True
    assert ck.l_mag is True
    assert ck.w is not None
    assert ck.w.shape == (153, 33)


def test_checkpoint_read_little_endian():
    """Read little-endian boussBenchSat checkpoint, verify header and fields."""
    from magic_torch.checkpoint_io import read_checkpoint
    ck = read_checkpoint(str(_SAMPLES_BB / "checkpoint_end.start"))
    assert ck.version == 4
    assert ck.endian == "<"
    assert ck.l_max == 64
    assert ck.w is not None
    assert ck.w.shape[0] > 0
    assert np.isfinite(ck.w).all()
    assert np.isfinite(ck.z).all()


def test_checkpoint_roundtrip():
    """Write Fortran checkpoint → read back → verify fields match."""
    _run_all()
    import tempfile, os
    from magic_torch.checkpoint_io import write_checkpoint_fortran, read_checkpoint
    from magic_torch import fields

    path = tempfile.mktemp(suffix=".ckpt")
    try:
        write_checkpoint_fortran(path, 3 * dtmax, 4, endian=">")
        ck = read_checkpoint(path)

        assert ck.version == 5
        assert ck.n_time_step == 4
        assert abs(ck.time - 3 * dtmax) < 1e-15

        # Compare fields
        import torch
        def _check(name, ck_arr, field_tensor):
            py = field_tensor.detach().cpu().numpy()
            np.testing.assert_allclose(ck_arr, py, rtol=0, atol=0,
                                       err_msg=f"{name} roundtrip mismatch")

        _check("w", ck.w, fields.w_LMloc)
        _check("z", ck.z, fields.z_LMloc)
        _check("p", ck.p, fields.p_LMloc)
        _check("s", ck.s, fields.s_LMloc)
        _check("b", ck.b, fields.b_LMloc)
        _check("aj", ck.aj, fields.aj_LMloc)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_checkpoint_header_exact():
    """Verify checkpoint header matches Fortran reference byte-for-byte.

    The header (everything before the field data) must match exactly,
    except the radial grid r[:] which can differ by 1 ULP due to
    Chebyshev grid construction differences.
    """
    _run_all()
    import tempfile, os
    from magic_torch.checkpoint_io import write_checkpoint_fortran

    path = tempfile.mktemp(suffix=".ckpt")
    try:
        write_checkpoint_fortran(path, 3 * dtmax, 4, endian=">")

        ref_path = str(_SAMPLES / "checkpoint_end.test")
        with open(ref_path, "rb") as f:
            ref_bytes = f.read()
        with open(path, "rb") as f:
            py_bytes = f.read()

        assert len(py_bytes) == len(ref_bytes), \
            f"Size mismatch: {len(py_bytes)} vs {len(ref_bytes)}"

        # Header layout: pre-r (0:258), r (258:522), post-r (522:658)
        r_start = 258
        r_end = 258 + n_r_max * 8  # 522
        # post-r header: scalars(16) + omegas(96) + logicals(24) = 136
        field_start = r_end + 16 + 96 + 24  # 658

        # Pre-r header: must match exactly
        n_pre_r_diff = sum(1 for i in range(r_start)
                           if ref_bytes[i] != py_bytes[i])
        assert n_pre_r_diff == 0, \
            f"Pre-r header: {n_pre_r_diff} differing bytes"

        # r array: at most 1 ULP per value
        r_ref = np.frombuffer(ref_bytes[r_start:r_end], dtype=">f8")
        r_py = np.frombuffer(py_bytes[r_start:r_end], dtype=">f8")
        np.testing.assert_allclose(r_py, r_ref, rtol=0,
                                   atol=np.finfo(np.float64).eps,
                                   err_msg="r array > 1 ULP")

        # Post-r header: must match exactly
        n_post_r_diff = sum(1 for i in range(r_end, field_start)
                            if ref_bytes[i] != py_bytes[i])
        assert n_post_r_diff == 0, \
            f"Post-r header: {n_post_r_diff} differing bytes"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_checkpoint_field_comparison():
    """Verify checkpoint fields match Fortran reference to machine precision."""
    _run_all()
    from magic_torch.checkpoint_io import read_checkpoint

    ck = read_checkpoint(str(_SAMPLES / "checkpoint_end.test"))

    from magic_torch import fields
    import torch

    for name, ck_name in [("w_LMloc", "w"), ("z_LMloc", "z"),
                           ("p_LMloc", "p"), ("s_LMloc", "s"),
                           ("b_LMloc", "b"), ("aj_LMloc", "aj")]:
        ref_arr = getattr(ck, ck_name)
        py_arr = getattr(fields, name).detach().cpu().numpy()
        max_abs = np.max(np.abs(ref_arr - py_arr))
        # p has higher tolerance due to condition number
        tol = 1e-7 if ck_name == "p" else 1e-12
        assert max_abs < tol, \
            f"{ck_name}: max_abs_err={max_abs:.2e} exceeds tol={tol:.0e}"


_MAGIC_EXE = Path(__file__).parent.parent.parent / "src" / "magic.exe"


@pytest.mark.skipif(not _MAGIC_EXE.exists(),
                    reason="magic.exe not found at src/magic.exe")
def test_fortran_reads_python_checkpoint(tmp_path):
    """Write Python checkpoint after 3 steps, have Fortran read it back,
    and verify the initial-state energies match Python's step-3 state.

    This proves that Python-written checkpoints are valid Fortran input.
    We use n_time_steps=1 (no integration, output-only) to avoid CFL
    dt adjustments on restart.
    """
    _run_all()  # runs 3 steps, sets up fields
    import subprocess

    from magic_torch.checkpoint_io import write_checkpoint_fortran
    from magic_torch import fields

    # --- 1. Get Python energies at step 3 (cached from _run_all) ---
    ek_py = _ek_results[3]
    em_py = _em_results[3]

    # --- 2. Write Python checkpoint ---
    ckpt_path = str(tmp_path / "checkpoint_end.test")
    write_checkpoint_fortran(ckpt_path, 3 * dtmax, 4, endian=">")

    # --- 3. Set up Fortran restart (n_time_steps=1: output-only, no integration) ---
    nml_content = """\
&grid
 n_r_max     =33,
 n_cheb_max  =33,
 l_max       =16,
 m_max       =16,
 n_r_ic_max  =17,
 n_cheb_ic_max=17,
 nalias      =20,
 minc        =1,
/
&control
 mode        =0,
 tag         ="test",
 n_time_steps=1,
 courfac     =2.5D0,
 alffac      =1.0D0,
 dtmax       =1.0D-4,
 alpha       =0.6D0,
 runHours    =12,
 runMinutes  =00,
 anelastic_flavour='ENT',
/
&phys_param
 ra          =1.0D5,
 ek          =1.0D-3,
 pr          =1.0D0,
 prmag       =5.0D0,
 radratio    =0.35D0,
 ktops       =1,
 kbots       =1,
 ktopv       =2,
 kbotv       =2,
/
&start_field
 l_start_file=.true.,
 start_file  ="checkpoint_end.test",
 init_b1     =3,
 amp_b1      =5,
 init_s1     =0404,
 amp_s1      =0.1,
/
&output_control
 n_log_step  =1,
 n_graphs    =0,
 n_rsts      =0,
 n_stores    =0,
 runid       ="restart test",
 l_movie     =.false.,
 l_RMS       =.false.,
/
&mantle
 nRotMa      =0
/
&inner_core
 sigma_ratio =0.d0,
 nRotIC      =0,
/
"""
    (tmp_path / "input.nml").write_text(nml_content)
    (tmp_path / "fortran_dumps").mkdir()

    # --- 4. Run Fortran ---
    result = subprocess.run(
        [str(_MAGIC_EXE), "input.nml"],
        cwd=str(tmp_path),
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, \
        f"Fortran failed:\nstdout={result.stdout.decode()[-2000:]}\nstderr={result.stderr.decode()[-2000:]}"

    # --- 5. Parse Fortran e_kin.test (initial-state output) ---
    ekin_file = tmp_path / "e_kin.test"
    assert ekin_file.exists(), (
        f"e_kin.test not found in {tmp_path}\n"
        f"Files: {list(tmp_path.iterdir())}\n"
        f"stdout: {result.stdout.decode()[-3000:]}\n"
        f"stderr: {result.stderr.decode()[-1000:]}"
    )
    lines = ekin_file.read_text().strip().splitlines()
    # With n_time_steps=1, only one line: the checkpoint state
    last_line = lines[-1].split()
    # Columns: time, e_kin_pol, e_kin_tor, ...
    f_ekin_pol = float(last_line[1])
    f_ekin_tor = float(last_line[2])

    # --- 6. Compare checkpoint-state energies ---
    py_ekin_pol = float(ek_py.e_p)
    py_ekin_tor = float(ek_py.e_t)

    rel_pol = abs(f_ekin_pol - py_ekin_pol) / max(abs(f_ekin_pol), 1e-30)
    rel_tor = abs(f_ekin_tor - py_ekin_tor) / max(abs(f_ekin_tor), 1e-30)

    assert rel_pol < 1e-8, \
        f"ekin_pol: Fortran={f_ekin_pol:.15e}, Python={py_ekin_pol:.15e}, rel={rel_pol:.2e}"
    assert rel_tor < 1e-8, \
        f"ekin_tor: Fortran={f_ekin_tor:.15e}, Python={py_ekin_tor:.15e}, rel={rel_tor:.2e}"
