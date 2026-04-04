"""condIC + condICrotIC: Multi-step comparison against Fortran (10 steps).

Fresh start (no checkpoint), CNAB2 time scheme, conducting IC.
Two configs:
  - condIC: nRotIC=0 (14 OC + 6 IC fields = 20 fields × 10 steps = 200 tests)
  - condICrotIC: nRotIC=1 (20 fields + omega_ic = 21 quantities × 10 steps = 210 tests)

Runs each config in a subprocess for module-level isolation.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_condic, load_ref_condicrotic

N_STEPS = 10

_RUNNER = Path(__file__).parent / "_condic_multistep_runner.py"

# ==================== condIC (nRotIC=0) ====================

_results_condic = {}
_ran_condic = False


def _run_condic():
    global _ran_condic
    if _ran_condic:
        return
    _ran_condic = True

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="condic_ms_")

    env = os.environ.copy()
    env.update({
        "MAGIC_SIGMA_RATIO": "1.0",
        "MAGIC_KBOTB": "3",
        "MAGIC_NROTIC": "0",
        "MAGIC_DEVICE": "cpu",
    })

    result = subprocess.run(
        [sys.executable, str(_RUNNER), "0", out_dir, str(N_STEPS)],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"condIC multistep runner failed: {result.returncode}")

    for step in range(1, N_STEPS + 1):
        for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                     "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                     "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc",
                     "b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"):
            _results_condic[(name, step)] = np.load(
                os.path.join(out_dir, f"{name}_step{step}.npy"))


@pytest.fixture(scope="module")
def condic_results():
    _run_condic()
    return _results_condic


# OC fields: (ref_prefix, py_attr, atol)
# Error sources: solve condition number + Chebyshev matmul accumulation
_OC_SPECS = {
    "w": ("w_LMloc", 1e-10),
    "dw": ("dw_LMloc", 1e-10),
    "ddw": ("ddw_LMloc", 1e-9),
    "z": ("z_LMloc", 1e-10),
    "dz": ("dz_LMloc", 1e-10),
    "s": ("s_LMloc", 1e-10),
    "ds": ("ds_LMloc", 1e-10),
    "p": ("p_LMloc", 1e-10),
    "dp": ("dp_LMloc", 5e-10),  # FP accumulation over steps; measured 1.5e-10 at step 6
    "b": ("b_LMloc", 1e-10),
    "db": ("db_LMloc", 1e-10),
    "ddb": ("ddb_LMloc", 1e-9),
    "aj": ("aj_LMloc", 1e-10),
    "dj": ("dj_LMloc", 1e-10),
}

_IC_SPECS = {
    "b_ic": ("b_ic", 1e-10),
    "db_ic": ("db_ic", 1e-9),
    "ddb_ic": ("ddb_ic", 1e-7),
    "aj_ic": ("aj_ic", 1e-10),
    "dj_ic": ("dj_ic", 1e-10),
    "ddj_ic": ("ddj_ic", 1e-9),
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_OC_SPECS.items()))
def test_condic_oc_field(condic_results, ref_prefix, spec, step):
    """condIC OC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_condic(f"{ref_prefix}_step{step}").cpu().numpy()
    py = condic_results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_IC_SPECS.items()))
def test_condic_ic_field(condic_results, ref_prefix, spec, step):
    """condIC IC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_condic(f"{ref_prefix}_step{step}").cpu().numpy()
    py = condic_results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


# ==================== condICrotIC (nRotIC=1) ====================

_results_rotic = {}
_ran_rotic = False


def _run_rotic():
    global _ran_rotic
    if _ran_rotic:
        return
    _ran_rotic = True

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="rotic_ms_")

    env = os.environ.copy()
    env.update({
        "MAGIC_SIGMA_RATIO": "1.0",
        "MAGIC_KBOTB": "3",
        "MAGIC_NROTIC": "1",
        "MAGIC_DEVICE": "cpu",
    })

    result = subprocess.run(
        [sys.executable, str(_RUNNER), "1", out_dir, str(N_STEPS)],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"condICrotIC multistep runner failed: {result.returncode}")

    for step in range(1, N_STEPS + 1):
        for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                     "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                     "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc",
                     "b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"):
            _results_rotic[(name, step)] = np.load(
                os.path.join(out_dir, f"{name}_step{step}.npy"))
        _results_rotic[("omega_ic", step)] = np.load(
            os.path.join(out_dir, f"omega_ic_step{step}.npy"))


@pytest.fixture(scope="module")
def rotic_results():
    _run_rotic()
    return _results_rotic


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_OC_SPECS.items()))
def test_rotic_oc_field(rotic_results, ref_prefix, spec, step):
    """condICrotIC OC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_condicrotic(f"{ref_prefix}_step{step}").cpu().numpy()
    py = rotic_results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_IC_SPECS.items()))
def test_rotic_ic_field(rotic_results, ref_prefix, spec, step):
    """condICrotIC IC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_condicrotic(f"{ref_prefix}_step{step}").cpu().numpy()
    py = rotic_results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
def test_rotic_omega_ic(rotic_results, step):
    """omega_ic at each step must match Fortran reference."""
    ref = load_ref_condicrotic(f"omega_ic_step{step}").cpu().item()
    py = float(rotic_results[("omega_ic", step)])
    if ref == 0:
        assert py == 0, f"omega_ic step{step}: py={py}, expected 0"
    else:
        rel_err = abs(py - ref) / abs(ref)
        assert rel_err < 1e-10, (
            f"omega_ic step{step}: py={py:.12e}, ref={ref:.12e}, rel_err={rel_err:.2e}")
