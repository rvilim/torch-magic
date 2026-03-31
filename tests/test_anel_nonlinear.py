"""Tests for anelastic nonlinear terms against Fortran reference data.

Verifies solver RHS terms (dwdt/dzdt old/impl/expl, w/z/p_imex_rhs) after
one radial_loop_anel() call on the initial anelastic state (hydro_bench_anel_lowres).

Runs in a subprocess to isolate anelastic-specific env vars.

Measured error magnitudes (2026-03-29):
  dwdt_old:  0 (exact match — initial state has zero old slot)
  dzdt_old:  0 (exact match — same)
  dzdt_impl: 0 (exact match — initial z is zero)
  dzdt_expl: 0 (exact match — initial velocity is zero, no nonlinear terms)
  dwdt_impl: abs=1.3e-4, rel=1.2e-9 (ref_max=1.09e5, WP conditioning)
  dwdt_expl: 0 (exact match — initial velocity is zero)
  z_imex_rhs: 0 (exact match)
  w_imex_rhs: abs=5.2e-9, rel=1.2e-9 (from dwdt_impl propagation)
  p_imex_rhs: 0 (exact match)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import _compute_snake_to_standard_perm

DEVICE = torch.device("cpu")
CDTYPE = torch.complex128
DTYPE = torch.float64

# Fortran reference data
ANEL_REF = Path(__file__).parent.parent.parent / "samples" / "hydro_bench_anel_lowres" / "fortran_ref"

# Snake-to-standard permutation (l_max=16, minc=1)
_SNAKE2ST = _compute_snake_to_standard_perm(l_max=16, minc=1)


def _load_anel_ref(name: str) -> torch.Tensor:
    """Load Fortran reference array with snake-to-standard LM reordering."""
    arr = np.load(ANEL_REF / f"{name}.npy")
    if np.issubdtype(arr.dtype, np.complexfloating):
        t = torch.from_numpy(arr.copy()).to(CDTYPE)
    else:
        t = torch.from_numpy(arr.copy()).to(DTYPE)
    # Reorder LM dimension (first axis with size lm_max=153)
    if t.ndim >= 1 and t.shape[0] == len(_SNAKE2ST):
        result = torch.zeros_like(t)
        result[_SNAKE2ST] = t
        return result
    return t


# Run the subprocess once for all tests
_test_data = {}


@pytest.fixture(scope="module", autouse=True)
def run_anel_nl():
    """Run the anelastic nonlinear runner subprocess and load results."""
    global _test_data
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Path(__file__).parent / "_anel_nl_runner.py"
        result = subprocess.run(
            [sys.executable, str(runner), tmpdir],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode != 0:
            pytest.fail(f"Anel NL runner failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        for name in [
            "dwdt_old", "dwdt_impl", "dwdt_expl",
            "dzdt_old", "dzdt_impl", "dzdt_expl",
            "w_imex_rhs", "z_imex_rhs", "p_imex_rhs",
        ]:
            arr = np.load(f"{tmpdir}/{name}.npy")
            _test_data[name] = torch.from_numpy(arr.copy()).to(CDTYPE)


def _compare(name: str, atol: float, rtol: float):
    """Compare test data against Fortran reference."""
    ref_path = ANEL_REF / f"{name}.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference: {name}")
    test = _test_data[name]
    ref = _load_anel_ref(name)
    ref_max = ref.abs().max().item()
    abs_err = (test - ref).abs().max().item()
    rel_err = abs_err / ref_max if ref_max > 0 else abs_err
    # Print diagnostics for tolerance calibration
    print(f"\n  {name}: abs_err={abs_err:.3e}, rel_err={rel_err:.3e}, ref_max={ref_max:.3e}")
    torch.testing.assert_close(test, ref, atol=atol, rtol=rtol)


# === dt_field component tests ===

def test_dzdt_old():
    """dzdt old slot: copy of initial implicit terms. Expected ~1e-13."""
    _compare("dzdt_old", atol=1e-12, rtol=1e-12)


def test_dzdt_impl():
    """dzdt implicit: linear diffusion operator on initial fields. Expected ~1e-12."""
    _compare("dzdt_impl", atol=1e-11, rtol=1e-11)


def test_dzdt_expl():
    """dzdt explicit: SHT → nonlinear → SHT chain. Expected ~1e-5 to 1e-9."""
    _compare("dzdt_expl", atol=1e-4, rtol=1e-4)


def test_dwdt_old():
    """dwdt old slot: copy of initial implicit terms. Expected ~1e-13."""
    _compare("dwdt_old", atol=1e-12, rtol=1e-12)


def test_dwdt_impl():
    """dwdt implicit: WP conditioning amplifies error.
    Measured: abs_err=1.3e-4, rel_err=1.2e-9 (ref_max=1.09e5).
    Some near-zero elements have inf element-wise rel_err, so use generous atol."""
    _compare("dwdt_impl", atol=5e-4, rtol=1e-8)


def test_dwdt_expl():
    """dwdt explicit: full nonlinear + Coriolis + buoyancy chain. Expected ~1e-5."""
    _compare("dwdt_expl", atol=1e-4, rtol=1e-4)


# === IMEX RHS tests ===

def test_z_imex_rhs():
    """z IMEX RHS: weighted sum of dzdt components."""
    _compare("z_imex_rhs", atol=1e-12, rtol=1e-12)


def test_w_imex_rhs():
    """w IMEX RHS: weighted sum of dwdt components. WP error propagates."""
    _compare("w_imex_rhs", atol=1e-4, rtol=1e-4)


def test_p_imex_rhs():
    """p IMEX RHS: weighted sum of dpdt components."""
    _compare("p_imex_rhs", atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    # Allow running standalone for debugging
    run_anel_nl()
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Anelastic nonlinear tests done")
