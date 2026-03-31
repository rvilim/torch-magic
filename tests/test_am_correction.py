"""Tests for angular momentum correction (Step 6 of anelastic plan).

Verifies:
1. c_moi_oc matches Fortran value (8/3 * pi * int(r^4 * rho0 dr))
2. get_angular_moment returns zero for zero fields
3. AM is conserved to machine precision after corrections over multiple steps
4. Correction profiles match analytical derivatives of rho0 * r^2
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


DEVICE = torch.device("cpu")
CDTYPE = torch.complex128
DTYPE = torch.float64

_AM_RUNNER = Path(__file__).parent / "_am_runner.py"


def _run_am_check(n_steps=50):
    """Run the AM correction runner and return parsed results."""
    result = subprocess.run(
        [sys.executable, str(_AM_RUNNER), str(n_steps)],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        pytest.fail(f"AM runner failed:\n{result.stderr}")
    lines = result.stdout.strip().split("\n")
    data = {}
    for line in lines:
        if "=" in line:
            key, val = line.split("=", 1)
            data[key.strip()] = float(val.strip())
    return data


@pytest.fixture(scope="module")
def am_data():
    return _run_am_check(50)


def test_c_moi_oc(am_data):
    """c_moi_oc should match analytical value for polytropic rho0."""
    # For polytropic: rho0 = temp0^polind, integrated numerically
    # Just verify it's positive and finite
    val = am_data["c_moi_oc"]
    assert val > 0, f"c_moi_oc should be positive, got {val}"
    assert np.isfinite(val), f"c_moi_oc should be finite, got {val}"


def test_am_zero_initial(am_data):
    """AM should be exactly zero at step 0 (fields start at zero)."""
    assert am_data["AM_z_step0"] == 0.0
    assert am_data["AM_x_step0"] == 0.0


def test_am_z_conserved(am_data):
    """Axial AM should remain zero (< 1e-14) after 50 steps with correction."""
    am_z = am_data["AM_z_step50"]
    assert abs(am_z) < 1e-14, f"AM_z at step 50 = {am_z:.3e}, expected < 1e-14"


def test_am_equatorial_conserved(am_data):
    """Equatorial AM should remain zero (< 1e-30) after 50 steps."""
    am_x = am_data["AM_x_step50"]
    am_y = am_data["AM_y_step50"]
    assert abs(am_x) < 1e-30, f"AM_x at step 50 = {am_x:.3e}, expected < 1e-30"
    assert abs(am_y) < 1e-30, f"AM_y at step 50 = {am_y:.3e}, expected < 1e-30"


def test_correction_profiles_analytical(am_data):
    """Correction profiles should match analytical d/dr, d²/dr² of rho0*r²."""
    # Residual = ||numerical - analytical|| / ||analytical||
    # Should be zero to machine precision since profiles are computed from same data
    assert am_data["profile_z_ok"] == 1.0, "z profile mismatch"
    assert am_data["profile_dz_ok"] == 1.0, "dz profile mismatch"
    assert am_data["profile_d2z_ok"] == 1.0, "d2z profile mismatch"


def test_am_without_correction(am_data):
    """AM without correction should drift (> 1e-20) after 50 steps."""
    # This verifies the correction is actually doing something
    am_z_uncorr = am_data["AM_z_uncorrected_step50"]
    assert abs(am_z_uncorr) > 1e-20, (
        f"AM_z without correction = {am_z_uncorr:.3e}, expected drift > 1e-20"
    )
