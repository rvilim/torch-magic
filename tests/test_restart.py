"""Test checkpoint restart produces identical results to continuous run.

Strategy:
  1. Run 5 steps continuous (save all snapshots + checkpoint at step 3)
  2. Load checkpoint from step 3, run 2 more steps
  3. Compare steps 4-5 between continuous and restarted runs
  Must match to machine precision.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

RUNNER = os.path.join(os.path.dirname(__file__), "_restart_runner.py")

FIELD_NAMES = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "p_LMloc", "dp_LMloc",
    "s_LMloc", "ds_LMloc",
    "b_LMloc", "db_LMloc", "ddb_LMloc",
    "aj_LMloc", "dj_LMloc", "ddj_LMloc",
]

DT_NAMES = ["dwdt", "dzdt", "dpdt", "dsdt", "dbdt", "djdt"]

SCALAR_DT_NAMES = ["domega_ic_dt", "domega_ma_dt"]


@pytest.fixture(scope="module")
def run_data(tmp_path_factory):
    """Run continuous + restart and return output directories."""
    base = tmp_path_factory.mktemp("restart_test")
    cont_dir = str(base / "continuous")
    ckpt_dir = str(base / "checkpoints")
    rest_dir = str(base / "restarted")

    env = os.environ.copy()
    env["MAGIC_DEVICE"] = "cpu"

    # Run 5 steps continuous, checkpoint at step 3
    result = subprocess.run(
        [sys.executable, RUNNER, "continuous", "5", cont_dir, ckpt_dir, "3"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Continuous run failed:\n{result.stderr}"

    # Restart from step 3 checkpoint, run 2 more steps
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_000003.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    result = subprocess.run(
        [sys.executable, RUNNER, "restart", ckpt_path, "2", rest_dir],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Restart run failed:\n{result.stderr}"

    return cont_dir, rest_dir


_FIELD_PARAMS = []
for step in [4, 5]:
    for field in FIELD_NAMES:
        _FIELD_PARAMS.append((field, step))


@pytest.mark.parametrize("field,step", _FIELD_PARAMS,
                         ids=[f"{f}-{s}" for f, s in _FIELD_PARAMS])
def test_restart_field(run_data, field, step):
    """Fields after restart must match continuous run exactly."""
    cont_dir, rest_dir = run_data
    cont_path = os.path.join(cont_dir, f"{field}_step{step}.npy")
    rest_path = os.path.join(rest_dir, f"{field}_step{step}.npy")

    if not os.path.exists(cont_path):
        pytest.skip(f"Field {field} not in continuous output (l_mag=False?)")
    if not os.path.exists(rest_path):
        pytest.skip(f"Field {field} not in restart output")

    cont = np.load(cont_path)
    rest = np.load(rest_path)

    diff = np.abs(cont - rest)
    scale = max(np.abs(cont).max(), 1e-30)
    rel_err = diff.max() / scale

    assert rel_err < 1e-13, (
        f"{field} step {step}: max rel error {rel_err:.3e} "
        f"(max abs diff {diff.max():.3e}, scale {scale:.3e})"
    )


_DT_PARAMS = []
for step in [4, 5]:
    for dtname in DT_NAMES:
        for slot in ["expl", "impl", "old"]:
            _DT_PARAMS.append((dtname, slot, step))


@pytest.mark.parametrize("dtname,slot,step", _DT_PARAMS,
                         ids=[f"{d}_{s}-{st}" for d, s, st in _DT_PARAMS])
def test_restart_derivatives(run_data, dtname, slot, step):
    """Derivative arrays after restart must match continuous run."""
    cont_dir, rest_dir = run_data
    fname = f"{dtname}_{slot}_step{step}.npy"
    cont_path = os.path.join(cont_dir, fname)
    rest_path = os.path.join(rest_dir, fname)

    if not os.path.exists(cont_path) or not os.path.exists(rest_path):
        pytest.skip(f"{fname} not available")

    cont = np.load(cont_path)
    rest = np.load(rest_path)

    diff = np.abs(cont - rest)
    scale = max(np.abs(cont).max(), 1e-30)
    rel_err = diff.max() / scale

    assert rel_err < 1e-13, (
        f"{dtname}.{slot} step {step}: max rel error {rel_err:.3e}"
    )


def test_restart_omega_ic(run_data):
    """omega_ic after restart must match continuous run."""
    cont_dir, rest_dir = run_data
    for step in [4, 5]:
        cont = np.load(os.path.join(cont_dir, f"omega_ic_step{step}.npy"))
        rest = np.load(os.path.join(rest_dir, f"omega_ic_step{step}.npy"))
        assert cont == rest, f"omega_ic step {step}: cont={cont} rest={rest}"


_SCALAR_DT_PARAMS = []
for step in [4, 5]:
    for sname in SCALAR_DT_NAMES:
        for slot in ["expl", "impl", "old"]:
            _SCALAR_DT_PARAMS.append((sname, slot, step))


@pytest.mark.parametrize("sname,slot,step", _SCALAR_DT_PARAMS,
                         ids=[f"{s}_{sl}-{st}" for s, sl, st in _SCALAR_DT_PARAMS])
def test_restart_scalar_derivatives(run_data, sname, slot, step):
    """Scalar derivative arrays after restart must match continuous run."""
    cont_dir, rest_dir = run_data
    fname = f"{sname}_{slot}_step{step}.npy"
    cont_path = os.path.join(cont_dir, fname)
    rest_path = os.path.join(rest_dir, fname)

    if not os.path.exists(cont_path) or not os.path.exists(rest_path):
        pytest.skip(f"{fname} not available")

    cont = np.load(cont_path)
    rest = np.load(rest_path)

    np.testing.assert_allclose(rest, cont, atol=1e-30, rtol=1e-13,
                               err_msg=f"{sname}.{slot} step {step}")


def test_derivatives_nonzero_after_restart(run_data):
    """Sanity check: derivative arrays should be non-zero after restart."""
    _, rest_dir = run_data
    # Check at step 4 (first step after restart from step 3)
    for dtname in ["dwdt", "dzdt"]:
        for slot in ["expl"]:
            path = os.path.join(rest_dir, f"{dtname}_{slot}_step4.npy")
            if os.path.exists(path):
                arr = np.load(path)
                assert np.abs(arr).max() > 0, (
                    f"{dtname}.{slot} is all-zero after restart (derivatives not restored?)")
