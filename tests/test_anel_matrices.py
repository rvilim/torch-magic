"""Tests for anelastic implicit matrix roundtrip quality.

Verifies that the precomputed matrix inverses (_s_inv_by_l, _z_inv_by_l,
_wp_inv_by_l, _p0Mat_inv) satisfy A @ A^{-1} ≈ I for each implicit solver.

The original matrix A is reconstructed from the same coefficient formulas
used in build_*_matrices(), and multiplied by the stored inverse. The
max absolute deviation from the identity matrix is checked.

Runs in a subprocess to isolate anelastic env vars (MAGIC_STRAT=5.0, etc.)
from other test configurations.

Tolerance rationale (measured roundtrip errors, 2026-03-29):
  - sMat: max |A @ inv - I| < 8e-15 — well-conditioned, near machine epsilon.
  - zMat: max |A @ inv - I| < 4e-13 — slightly worse due to dL*or2 scaling.
  - wpMat: max |A @ inv - I| < 1.6e-10 — 2N x 2N coupled system, condition
    number ~1.7e12 for anelastic profiles, so cond(A)*eps ≈ 4e-4 bounds
    the solve error; the roundtrip itself is much tighter.
  - p0Mat: max |A @ inv - I| < 1e-9 — pressure equation with anelastic beta
    profile; condition number drives the error.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


# Run the matrix runner once for all tests in this module
_results = {}


@pytest.fixture(scope="module", autouse=True)
def run_matrix_runner():
    """Run the anelastic matrix runner subprocess and parse results."""
    global _results
    runner = Path(__file__).parent / "_anel_matrix_runner.py"
    result = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        pytest.fail(f"Anel matrix runner failed:\n{result.stderr}")
    _results = json.loads(result.stdout.strip())


# Tolerances: ~3x measured worst-case to allow FP variability across platforms.
# Measured values documented in module docstring.
_SMAT_TOL = 3e-14
_ZMAT_TOL = 1e-12
_WPMAT_TOL = 5e-10
_P0MAT_TOL = 3e-9


@pytest.mark.parametrize("l", [1, 5, 10, 16])
def test_sMat_roundtrip(l):
    """sMat (entropy): A @ A^{-1} ≈ I for degree l."""
    err = _results["sMat"][str(l)]
    assert err < _SMAT_TOL, (
        f"sMat l={l}: roundtrip error {err:.3e} exceeds tolerance {_SMAT_TOL:.0e}"
    )


@pytest.mark.parametrize("l", [1, 5, 10, 16])
def test_zMat_roundtrip(l):
    """zMat (toroidal velocity): A @ A^{-1} ≈ I for degree l."""
    err = _results["zMat"][str(l)]
    assert err < _ZMAT_TOL, (
        f"zMat l={l}: roundtrip error {err:.3e} exceeds tolerance {_ZMAT_TOL:.0e}"
    )


@pytest.mark.parametrize("l", [1, 5, 10, 16])
def test_wpMat_roundtrip(l):
    """wpMat (poloidal velocity + pressure): A @ A^{-1} ≈ I for degree l."""
    err = _results["wpMat"][str(l)]
    assert err < _WPMAT_TOL, (
        f"wpMat l={l}: roundtrip error {err:.3e} exceeds tolerance {_WPMAT_TOL:.0e}"
    )


def test_p0Mat_roundtrip():
    """p0Mat (l=0 pressure): A @ A^{-1} ≈ I."""
    err = _results["p0Mat"]["0"]
    assert err < _P0MAT_TOL, (
        f"p0Mat: roundtrip error {err:.3e} exceeds tolerance {_P0MAT_TOL:.0e}"
    )
