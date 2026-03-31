"""Test energy decompositions against Fortran reference files.

Compares get_e_kin_full, get_e_mag_oc_full, get_e_mag_ic against the
Fortran-generated e_kin.test, e_mag_oc.test, e_mag_ic.test files in
samples/dynamo_benchmark/ for steps 0-3.

Also includes synthetic ES/EA mask tests that catch parity bugs the
benchmark reference can't (early steps have trivially zero EA).
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from magic_torch import fields
from magic_torch.params import dtmax, lm_max, l_max, n_r_max
from magic_torch.blocking import st_lm2l, st_lm2m
from magic_torch.output import (
    get_e_kin_full, get_e_mag_oc_full, get_e_mag_ic,
    write_e_kin_line, write_e_mag_oc_line, write_e_mag_ic_line,
    _cc2real,
)

# --- Fortran reference data ---
_SAMPLES = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark"


def _load_ref_energies():
    """Load and parse the 3 Fortran energy reference files."""
    ekin = np.loadtxt(_SAMPLES / "e_kin.test")
    emag_oc = np.loadtxt(_SAMPLES / "e_mag_oc.test")
    emag_ic = np.loadtxt(_SAMPLES / "e_mag_ic.test")
    return ekin, emag_oc, emag_ic


# --- Run simulation once, collect energies for steps 0-3 ---
_ekin_results = {}
_emag_oc_results = {}
_emag_ic_results = {}
_ran = False


def _run_all():
    global _ran
    if _ran:
        return
    _ran = True

    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import setup_initial_state, one_step, initialize_dt

    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)

    # Step 0: initial state
    ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
    em = get_e_mag_oc_full(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
    eic = get_e_mag_ic(fields.b_LMloc)
    _ekin_results[0] = ek
    _emag_oc_results[0] = em
    _emag_ic_results[0] = eic

    # Steps 1-3
    for step in range(1, 4):
        one_step(n_time_step=step, dt=dtmax)
        ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
        em = get_e_mag_oc_full(fields.b_LMloc, fields.db_LMloc,
                                fields.aj_LMloc)
        eic = get_e_mag_ic(fields.b_LMloc)
        _ekin_results[step] = ek
        _emag_oc_results[step] = em
        _emag_ic_results[step] = eic


# ---- e_kin tests ----
# Columns: time, e_p, e_t, e_p_as, e_t_as, e_p_es, e_t_es, e_p_eas, e_t_eas

_EKIN_FIELDS = ['e_p', 'e_t', 'e_p_as', 'e_t_as',
                'e_p_es', 'e_t_es', 'e_p_eas', 'e_t_eas']

# Fortran reference uses ES16.8 (8 significant digits), so reference precision
# is ~5e-9. Use 1e-8 tolerance to account for this.
_REF_RTOL = 1e-8


@pytest.mark.parametrize("step", [0, 1, 2, 3])
@pytest.mark.parametrize("col_idx,field_name", list(enumerate(_EKIN_FIELDS)),
                         ids=_EKIN_FIELDS)
def test_e_kin(step, col_idx, field_name):
    """Compare e_kin decomposition against Fortran reference."""
    _run_all()
    ref_data, _, _ = _load_ref_energies()
    ref_val = ref_data[step, col_idx + 1]  # col 0 is time
    py_val = getattr(_ekin_results[step], field_name)

    if ref_val == 0.0:
        assert abs(py_val) < 1e-15, f"step {step} {field_name}: expected 0, got {py_val}"
    else:
        rel_err = abs(py_val - ref_val) / abs(ref_val)
        assert rel_err < _REF_RTOL, (
            f"step {step} {field_name}: py={py_val:.15e} ref={ref_val:.15e} "
            f"rel_err={rel_err:.2e}")


# ---- e_mag_oc tests ----
# Columns: time, e_p, e_t, e_p_as, e_t_as, e_p_os, e_p_as_os,
#           e_p_es, e_t_es, e_p_eas, e_t_eas, e_p_e, e_p_as_e

_EMAG_OC_FIELDS = ['e_p', 'e_t', 'e_p_as', 'e_t_as', 'e_p_os', 'e_p_as_os',
                   'e_p_es', 'e_t_es', 'e_p_eas', 'e_t_eas', 'e_p_e', 'e_p_as_e']


@pytest.mark.parametrize("step", [0, 1, 2, 3])
@pytest.mark.parametrize("col_idx,field_name",
                         list(enumerate(_EMAG_OC_FIELDS)),
                         ids=_EMAG_OC_FIELDS)
def test_e_mag_oc(step, col_idx, field_name):
    """Compare e_mag_oc decomposition against Fortran reference."""
    _run_all()
    _, ref_data, _ = _load_ref_energies()
    ref_val = ref_data[step, col_idx + 1]  # col 0 is time
    py_val = getattr(_emag_oc_results[step], field_name)

    if ref_val == 0.0:
        assert abs(py_val) < 1e-15, f"step {step} {field_name}: expected 0, got {py_val}"
    else:
        rel_err = abs(py_val - ref_val) / abs(ref_val)
        assert rel_err < _REF_RTOL, (
            f"step {step} {field_name}: py={py_val:.15e} ref={ref_val:.15e} "
            f"rel_err={rel_err:.2e}")


# ---- e_mag_ic tests ----
# Columns: time, e_p, e_t, e_p_as, e_t_as

_EMAG_IC_FIELDS = ['e_p', 'e_t', 'e_p_as', 'e_t_as']


@pytest.mark.parametrize("step", [0, 1, 2, 3])
@pytest.mark.parametrize("col_idx,field_name",
                         list(enumerate(_EMAG_IC_FIELDS)),
                         ids=_EMAG_IC_FIELDS)
def test_e_mag_ic(step, col_idx, field_name):
    """Compare e_mag_ic decomposition against Fortran reference."""
    _run_all()
    _, _, ref_data = _load_ref_energies()
    ref_val = ref_data[step, col_idx + 1]  # col 0 is time
    py_val = getattr(_emag_ic_results[step], field_name)

    if ref_val == 0.0:
        assert abs(py_val) < 1e-15, f"step {step} {field_name}: expected 0, got {py_val}"
    else:
        rel_err = abs(py_val - ref_val) / abs(ref_val)
        assert rel_err < _REF_RTOL, (
            f"step {step} {field_name}: py={py_val:.15e} ref={ref_val:.15e} "
            f"rel_err={rel_err:.2e}")


# ---- Format tests ----

def test_e_kin_line_width():
    """Verify e_kin line is exactly 148 chars (matching Fortran ES20.12,8ES16.8)."""
    _run_all()
    import io
    buf = io.StringIO()
    write_e_kin_line(buf, 0.0, _ekin_results[0])
    line = buf.getvalue()
    # Line includes trailing newline
    assert len(line) == 149, f"Expected 149 chars (148 + newline), got {len(line)}"


def test_e_mag_oc_line_width():
    """Verify e_mag_oc line is exactly 212 chars."""
    _run_all()
    import io
    buf = io.StringIO()
    write_e_mag_oc_line(buf, 0.0, _emag_oc_results[0])
    line = buf.getvalue()
    assert len(line) == 213, f"Expected 213 chars (212 + newline), got {len(line)}"


def test_e_mag_ic_line_width():
    """Verify e_mag_ic line is exactly 84 chars."""
    _run_all()
    import io
    buf = io.StringIO()
    write_e_mag_ic_line(buf, 0.0, _emag_ic_results[0])
    line = buf.getvalue()
    assert len(line) == 85, f"Expected 85 chars (84 + newline), got {len(line)}"


# ---- Synthetic ES/EA test ----
# Early benchmark steps have e_p_es ≈ e_p and EA ≈ 0 because the field is
# nearly axisymmetric. This synthetic test constructs modes with known parity
# to verify the mask logic catches all 4 conventions.

def test_es_ea_synthetic_kinetic():
    """Verify kinetic ES/EA decomposition with synthetic modes.

    Construct w/dw/z with energy concentrated in specific (l,m) modes,
    then check that the decomposition correctly separates ES/EA.
    """
    from magic_torch.output import get_e_kin_full

    # Create fields with a single mode at l=2, m=1: (l+m)=3 is odd
    # For kinetic: this is EA-poloidal (not ES-poloidal) and ES-toroidal
    w = torch.zeros(lm_max, n_r_max, dtype=torch.complex128, device='cpu')
    dw = torch.zeros_like(w)
    z = torch.zeros_like(w)

    # Find the lm index for l=2, m=1
    mask_21 = (st_lm2l == 2) & (st_lm2m == 1)
    idx_21 = mask_21.nonzero(as_tuple=True)[0].item()

    # Put energy in w at this mode (fill all radial points with a nonzero value)
    w[idx_21, :] = 1.0 + 0.1j
    dw[idx_21, :] = 0.5 + 0.05j
    z[idx_21, :] = 0.3 + 0.03j

    ek = get_e_kin_full(w, dw, z)

    # (l+m)=3 odd: poloidal is EA (not ES), toroidal is ES (not EA)
    # So e_p_es should be 0 (all poloidal energy is EA)
    assert abs(ek.e_p_es) < 1e-20, f"e_p_es should be 0 for (l+m) odd mode, got {ek.e_p_es}"
    assert ek.e_p > 0, "e_p should be nonzero"
    # Toroidal ES should equal total toroidal (only one mode, and it IS ES for tor)
    assert abs(ek.e_t_es - ek.e_t) < 1e-20 * max(1, abs(ek.e_t)), \
        f"e_t_es should equal e_t for (l+m) odd mode: {ek.e_t_es} vs {ek.e_t}"

    # Now test l=2, m=0: (l+m)=2 is even → ES-poloidal, EA-toroidal
    w2 = torch.zeros_like(w)
    dw2 = torch.zeros_like(w)
    z2 = torch.zeros_like(w)
    mask_20 = (st_lm2l == 2) & (st_lm2m == 0)
    idx_20 = mask_20.nonzero(as_tuple=True)[0].item()
    w2[idx_20, :] = 1.0
    dw2[idx_20, :] = 0.5
    z2[idx_20, :] = 0.3

    ek2 = get_e_kin_full(w2, dw2, z2)
    # (l+m)=2 even: poloidal is ES, toroidal is EA
    assert abs(ek2.e_p_es - ek2.e_p) < 1e-20 * max(1, abs(ek2.e_p)), \
        "e_p_es should equal e_p for (l+m) even mode"
    assert abs(ek2.e_t_es) < 1e-20, \
        f"e_t_es should be 0 for (l+m) even mode, got {ek2.e_t_es}"


def test_es_ea_synthetic_magnetic():
    """Verify magnetic ES/EA decomposition with synthetic modes.

    Magnetic uses OPPOSITE parity from kinetic:
    - mag ES-poloidal: (l+m) odd
    - mag ES-toroidal: (l+m) even
    """
    from magic_torch.output import get_e_mag_oc_full

    # Mode l=2, m=1: (l+m)=3 odd → mag ES-poloidal, mag EA-toroidal
    b = torch.zeros(lm_max, n_r_max, dtype=torch.complex128, device='cpu')
    db = torch.zeros_like(b)
    aj = torch.zeros_like(b)

    mask_21 = (st_lm2l == 2) & (st_lm2m == 1)
    idx_21 = mask_21.nonzero(as_tuple=True)[0].item()
    b[idx_21, :] = 1.0 + 0.1j
    db[idx_21, :] = 0.5 + 0.05j
    aj[idx_21, :] = 0.3 + 0.03j

    em = get_e_mag_oc_full(b, db, aj)

    # (l+m)=3 odd: mag ES-poloidal (yes), mag ES-toroidal (no)
    assert abs(em.e_p_es - em.e_p) < 1e-20 * max(1, abs(em.e_p)), \
        "mag e_p_es should equal e_p for (l+m) odd mode"
    assert abs(em.e_t_es) < 1e-20, \
        f"mag e_t_es should be 0 for (l+m) odd mode, got {em.e_t_es}"

    # Mode l=2, m=0: (l+m)=2 even → mag EA-poloidal, mag ES-toroidal
    b2 = torch.zeros_like(b)
    db2 = torch.zeros_like(b)
    aj2 = torch.zeros_like(b)
    mask_20 = (st_lm2l == 2) & (st_lm2m == 0)
    idx_20 = mask_20.nonzero(as_tuple=True)[0].item()
    b2[idx_20, :] = 1.0
    db2[idx_20, :] = 0.5
    aj2[idx_20, :] = 0.3

    em2 = get_e_mag_oc_full(b2, db2, aj2)
    # (l+m)=2 even: mag EA-poloidal (not ES), mag ES-toroidal (yes)
    assert abs(em2.e_p_es) < 1e-20, \
        f"mag e_p_es should be 0 for (l+m) even mode, got {em2.e_p_es}"
    assert abs(em2.e_t_es - em2.e_t) < 1e-20 * max(1, abs(em2.e_t)), \
        "mag e_t_es should equal e_t for (l+m) even mode"


def test_cc2real_m0():
    """Verify _cc2real excludes imaginary part for m=0."""
    c = torch.tensor([1.0 + 0.5j], dtype=torch.complex128)
    m = torch.tensor([0], dtype=torch.long)
    result = _cc2real(c, m)
    expected = 1.0 ** 2  # only real part
    assert abs(result.item() - expected) < 1e-15, \
        f"_cc2real(m=0) should be real^2={expected}, got {result.item()}"

    m_pos = torch.tensor([1], dtype=torch.long)
    result_pos = _cc2real(c, m_pos)
    expected_pos = 2.0 * (1.0 ** 2 + 0.5 ** 2)
    assert abs(result_pos.item() - expected_pos) < 1e-15, \
        f"_cc2real(m=1) should be {expected_pos}, got {result_pos.item()}"
