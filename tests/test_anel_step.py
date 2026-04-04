"""Tests for anelastic hydrodynamic benchmark (hydro_bench_anel_lowres).

Runs the anelastic step runner in a subprocess (isolated env vars)
and compares all fields against Fortran reference data for steps 1-3.

Tolerance rationale (measured error magnitudes, 2026-03-29):
  - s, ds, p, dp: rel_err < 1e-7 — limited only by float64 accumulation
  - w, dw, ddw: rel_err 1e-6 to 4e-3 — WP matrix condition number ~1.7e12
    with precomputed inverse approach amplifies rounding error. Roundtrip
    error of the WP matrix itself is ~1.6e-11, but cond(A)*eps ≈ 1.7e-4
    bounds the worst-case solve error.
  - z, dz: rel_err ~1e-4 to 2e-3 — same WP conditioning propagates through
    Coriolis coupling from w into z.
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
    if t.ndim >= 1 and t.shape[0] == len(_SNAKE2ST):
        result = torch.zeros_like(t)
        result[_SNAKE2ST] = t
        return result
    return t


# Run the step runner once for all tests
_N_STEPS = 3
_test_data = {}


@pytest.fixture(scope="module", autouse=True)
def run_anel_steps():
    """Run the anelastic step runner subprocess and load results."""
    global _test_data
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Path(__file__).parent / "_anel_step_runner.py"
        result = subprocess.run(
            [sys.executable, str(runner), tmpdir, str(_N_STEPS)],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode != 0:
            pytest.fail(f"Anel step runner failed:\n{result.stderr}")

        for step in range(1, _N_STEPS + 1):
            for name in ["s", "ds", "p", "dp", "w", "dw", "ddw", "z", "dz"]:
                key = f"{name}_step{step}"
                arr = np.load(f"{tmpdir}/{key}.npy")
                _test_data[key] = torch.from_numpy(arr.copy()).to(CDTYPE)


# Relative tolerance per field.
# After switching anelastic WP solver from precomputed inverse to
# torch.linalg.lu_solve (cond(wpMat) ~1e10 was amplifying inverse errors),
# all fields now match Fortran to near machine precision.
# Measured worst-case relative errors across 3 steps:
#   s: 7e-13   ds: 8e-12   p: 7e-11   dp: 3e-11
#   w: 2e-11   dw: 2e-10   ddw: 3e-9
#   z: 5e-12   dz: 4e-10
# Tolerances set to ~10x measured worst-case.
_RTOL = {
    "s": 1e-11, "ds": 1e-10, "p": 1e-9, "dp": 1e-9,
    "w": 1e-9, "dw": 1e-8, "ddw": 1e-7,
    "z": 1e-10, "dz": 1e-8,
}


@pytest.mark.parametrize("step", range(1, _N_STEPS + 1))
@pytest.mark.parametrize("field", ["s", "ds", "p", "dp", "w", "dw", "ddw", "z", "dz"])
def test_anel_step(field, step):
    """Compare anelastic field after step against Fortran reference."""
    key = f"{field}_step{step}"
    ref_path = ANEL_REF / f"{key}.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference: {key}")

    test = _test_data[key]
    ref = _load_anel_ref(key)

    ref_max = ref.abs().max().item()
    if ref_max == 0.0 and test.abs().max().item() == 0.0:
        return  # Both zero

    abs_err = (test - ref).abs().max().item()
    rel_err = abs_err / ref_max if ref_max > 0 else abs_err

    rtol = _RTOL[field]
    assert rel_err < rtol, (
        f"{key}: rel={rel_err:.3e} (tol {rtol:.0e}), "
        f"abs={abs_err:.3e}, ref_max={ref_max:.3e}"
    )
