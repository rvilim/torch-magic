"""Test Couette flow (mode=7, l_SRIC) against Fortran reference.

Runs Python couetteAxi in a subprocess (env vars set before import),
compares velocity fields × 3 steps against Fortran reference data.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RUNNER = os.path.join(os.path.dirname(__file__), "_couette_runner.py")
REF_DIR = Path(__file__).parent.parent.parent / "samples" / "couetteAxi_fresh" / "fortran_ref"

# Snake→standard LM reorder (same as conftest.py for l_max=16, minc=1)
def _compute_snake_to_standard_perm(l_max=16, minc=1):
    import torch
    l_list = list(range(l_max, -1, -1))
    idx_0 = l_list.index(0)
    l_list[0], l_list[idx_0] = l_list[idx_0], l_list[0]
    snake_lm2l, snake_lm2m = [], []
    for l in l_list:
        for m in range(0, min(l_max, l) + 1, minc):
            snake_lm2l.append(l)
            snake_lm2m.append(m)
    lm_max = len(snake_lm2l)
    st_lm2 = {}
    st_idx = 0
    for m in range(0, l_max + 1, minc):
        for l in range(m, l_max + 1):
            st_lm2[(l, m)] = st_idx
            st_idx += 1
    perm = torch.zeros(lm_max, dtype=torch.long)
    for si in range(lm_max):
        perm[si] = st_lm2[(snake_lm2l[si], snake_lm2m[si])]
    return perm


_SNAKE2ST = _compute_snake_to_standard_perm()

# Field name mapping: Python attr → Fortran dump base name
_FIELD_MAP = {
    "w_LMloc": "w",
    "dw_LMloc": "dw",
    "ddw_LMloc": "ddw",
    "z_LMloc": "z",
    "dz_LMloc": "dz",
    "p_LMloc": "p",
    "dp_LMloc": "dp",
    "s_LMloc": "s",
    "ds_LMloc": "ds",
}

N_STEPS = 3


@pytest.fixture(scope="module")
def run_data(tmp_path_factory):
    """Run Python couetteAxi and return output directory."""
    out_dir = str(tmp_path_factory.mktemp("couette_test"))
    env = os.environ.copy()
    env["MAGIC_DEVICE"] = "cpu"
    result = subprocess.run(
        [sys.executable, RUNNER, str(N_STEPS), out_dir],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Couette run failed:\n{result.stderr}"
    return out_dir


_PARAMS = []
for step in range(1, N_STEPS + 1):
    for py_name, fort_name in _FIELD_MAP.items():
        _PARAMS.append((py_name, fort_name, step))


@pytest.mark.parametrize("py_name,fort_name,step", _PARAMS,
                         ids=[f"{fn}-step{s}" for _, fn, s in _PARAMS])
def test_couette_field(run_data, py_name, fort_name, step):
    """Velocity field after step must match Fortran reference."""
    out_dir = run_data
    py_path = os.path.join(out_dir, f"{py_name}_step{step}.npy")
    ref_path = REF_DIR / f"{fort_name}_step{step}.npy"

    assert ref_path.exists(), f"Fortran ref not found: {ref_path}"
    assert os.path.exists(py_path), f"Python output not found: {py_path}"

    py_arr = np.load(py_path)
    ref_arr = np.load(str(ref_path))

    # Reorder Fortran snake→standard: result[st_idx, :] = source[snake_idx, :]
    import torch
    ref_t = torch.from_numpy(ref_arr)
    ref_reord_t = torch.zeros_like(ref_t)
    ref_reord_t[_SNAKE2ST] = ref_t
    ref_reord = ref_reord_t.numpy()

    diff = np.abs(py_arr - ref_reord)
    scale = max(np.abs(ref_reord).max(), 1e-30)
    rel_err = diff.max() / scale

    assert rel_err < 1e-10, (
        f"{fort_name} step {step}: max rel error {rel_err:.3e} "
        f"(max abs diff {diff.max():.3e}, scale {scale:.3e})"
    )


def test_couette_omega_ic(run_data):
    """omega_ic must stay at prescribed value (-4000) for all steps."""
    out_dir = run_data
    for step in range(1, N_STEPS + 1):
        py_val = np.load(os.path.join(out_dir, f"omega_ic_step{step}.npy"))
        ref_val = np.load(str(REF_DIR / f"omega_ic_step{step}.npy"))
        assert py_val == ref_val, (
            f"omega_ic step {step}: py={py_val} ref={ref_val}"
        )
