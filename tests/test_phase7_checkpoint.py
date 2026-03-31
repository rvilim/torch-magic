"""Phase 7: Fortran checkpoint reader — verify correct parsing of boussBenchSat checkpoint.

The checkpoint file (checkpoint_end.start) was written by Fortran's storeCheckPoints
in stream-access format. This test verifies that our Python reader correctly parses
every byte of the binary file and produces fields matching the expected physical state.

Fortran comparison:
  - The file IS the Fortran reference (written by Fortran, binary exact)
  - EOF check verifies we consumed every byte correctly
  - Header values must match the Fortran input.nml parameters
  - omega_ic must match the known Christensen benchmark saturated value (~-2.66)
  - m=0 spectral coefficients must be real (imaginary part = 0) for real spatial fields
"""

import pytest
import numpy as np
from pathlib import Path

from magic_torch.checkpoint_io import read_checkpoint, _compute_lm_max

CKPT = Path(__file__).parent.parent.parent / "samples" / "boussBenchSat" / "checkpoint_end.start"


@pytest.fixture(scope="module")
def ck():
    """Load the checkpoint once for all tests."""
    return read_checkpoint(str(CKPT))


# --- Header tests ---

def test_version(ck):
    assert ck.version == 4

def test_time_scheme(ck):
    assert ck.family == "DIRK"
    assert ck.nexp == 4
    assert ck.nimp == 4
    assert ck.nold == 1

def test_dt(ck):
    assert len(ck.dt) == 1  # DIRK stores 1 dt value
    assert ck.dt[0] == pytest.approx(2.0e-4, rel=1e-14)

def test_physical_params(ck):
    assert ck.ra == pytest.approx(1.1e5, rel=1e-14)
    assert ck.pr == pytest.approx(1.0, rel=1e-14)
    assert ck.prmag == pytest.approx(5.0, rel=1e-14)
    assert ck.ek == pytest.approx(1.0e-3, rel=1e-14)
    assert ck.radratio == pytest.approx(0.35, rel=1e-14)
    assert ck.sigma_ratio == pytest.approx(1.0, rel=1e-14)

def test_grid_params(ck):
    assert ck.n_r_max == 33
    assert ck.n_theta_max == 96
    assert ck.n_phi_tot == 192
    assert ck.minc == 4
    assert ck.n_r_ic_max == 17
    assert ck.l_max == 64
    assert ck.m_min == 0
    assert ck.m_max == 64
    assert ck.n_cheb_max == 31

def test_logicals(ck):
    assert ck.l_heat is True
    assert ck.l_chemical_conv is False
    assert ck.l_phase_field is False
    assert ck.l_mag is True
    assert ck.l_press_store is True
    assert ck.l_cond_ic is True


# --- Radial grid tests ---

def test_radial_grid(ck):
    r_cmb = 1.0 / (1.0 - ck.radratio)
    r_icb = ck.radratio / (1.0 - ck.radratio)
    assert ck.r[0] == pytest.approx(r_cmb, rel=1e-14)
    assert ck.r[-1] == pytest.approx(r_icb, rel=1e-14)
    assert len(ck.r) == ck.n_r_max
    # Grid should be monotonically decreasing (CMB to ICB)
    assert np.all(np.diff(ck.r) < 0)


# --- Rotation ---

def test_omega_ic(ck):
    """Christensen benchmark saturated omega_ic ≈ -2.66."""
    assert ck.omega_ic1 == pytest.approx(-2.6578397088063173, rel=1e-12)


# --- Field shape tests ---

def test_lm_max(ck):
    lm_max = _compute_lm_max(ck.l_max, ck.m_max, ck.minc)
    assert lm_max == 561

def test_oc_field_shapes(ck):
    lm_max = 561
    for name in ["w", "z", "p", "s", "b", "aj"]:
        arr = getattr(ck, name)
        assert arr is not None, f"{name} should not be None"
        assert arr.shape == (lm_max, ck.n_r_max), f"{name} shape mismatch"
        assert arr.dtype == np.complex128, f"{name} dtype mismatch"

def test_ic_field_shapes(ck):
    lm_max = 561
    for name in ["b_ic", "aj_ic"]:
        arr = getattr(ck, name)
        assert arr is not None, f"{name} should not be None"
        assert arr.shape == (lm_max, ck.n_r_ic_max), f"{name} shape mismatch"
        assert arr.dtype == np.complex128, f"{name} dtype mismatch"


# --- Physical sanity checks ---

def test_no_nan_inf(ck):
    """All fields must be finite."""
    for name in ["w", "z", "p", "s", "b", "aj", "b_ic", "aj_ic"]:
        arr = getattr(ck, name)
        assert np.all(np.isfinite(arr)), f"{name} contains NaN/Inf"

def test_m0_modes_real(ck):
    """m=0 spectral coefficients must be real for real spatial fields."""
    # For minc=4, st_map ordering: m=0 modes are the first (l_max+1) entries
    n_m0 = ck.l_max + 1
    for name in ["w", "z", "s", "b", "aj"]:
        arr = getattr(ck, name)
        max_imag = np.abs(arr[:n_m0, :].imag).max()
        assert max_imag < 1e-14, f"{name} m=0 imag part = {max_imag}"

def test_fields_nonzero(ck):
    """Saturated dynamo: all fields should have nontrivial amplitudes."""
    for name in ["w", "z", "s", "b", "aj"]:
        arr = getattr(ck, name)
        assert np.abs(arr).max() > 1e-3, f"{name} is unexpectedly small"

def test_b_ic_nonzero(ck):
    """Conducting IC: b_ic and aj_ic should be nontrivial."""
    assert np.abs(ck.b_ic).max() > 1e-2
    assert np.abs(ck.aj_ic).max() > 1e-2

def test_b_ic_l0_zero(ck):
    """l=0, m=0 mode of IC magnetic field should be zero (no monopole)."""
    assert np.abs(ck.b_ic[0, :]).max() < 1e-30
    assert np.abs(ck.aj_ic[0, :]).max() < 1e-30


# --- File integrity ---

def test_file_fully_consumed(ck):
    """The reader must consume every byte — no leftover data.

    This is verified by the EOF check in read_checkpoint().
    If we get here, it already passed.
    """
    # Re-read to explicitly verify
    ck2 = read_checkpoint(str(CKPT))
    assert ck2.version == ck.version  # Just verify it didn't raise
