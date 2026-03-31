"""Compare BPR353 (4-stage SDIRK) 1-step output against Fortran reference.

This test runs in a subprocess with MAGIC_TIME_SCHEME=BPR353 to ensure
module-level singletons (tscheme, dt_fields) use the DIRK scheme.

Each of 14 field arrays after 1 BPR353 time step is compared against
Fortran dumps from the dynamo_benchmark_bpr353 sample.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
import torch
from pathlib import Path

# BPR353 Fortran reference data
BPR353_REF = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark_bpr353" / "fortran_ref"

# Snake→standard reordering (same l_max=16, minc=1 as CNAB2)
def _compute_snake_to_standard_perm(l_max=16, minc=1):
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
    perm = np.zeros(lm_max, dtype=np.int64)
    for snake_idx in range(lm_max):
        perm[snake_idx] = st_lm2[(snake_lm2l[snake_idx], snake_lm2m[snake_idx])]
    return perm

_SNAKE2ST = _compute_snake_to_standard_perm()

_LM_FIELD_NAMES = {
    's_step1', 'b_step1', 'db_step1', 'ddb_step1', 'aj_step1', 'dj_step1',
    'w_step1', 'dw_step1', 'ddw_step1', 'z_step1', 'dz_step1', 'p_step1',
    'dp_step1', 'ds_step1',
}


def load_bpr353_ref(name):
    """Load BPR353 Fortran reference, reordering snake→standard for LM fields."""
    arr = np.load(BPR353_REF / f"{name}.npy")
    if name in _LM_FIELD_NAMES and arr.ndim == 2 and arr.shape[0] == len(_SNAKE2ST):
        result = np.zeros_like(arr)
        result[_SNAKE2ST] = arr
        return result
    return arr


# The actual computation runs in a subprocess to get clean module state
_RUNNER_SCRIPT = Path(__file__).parent / "_bpr353_runner.py"


def _ensure_runner():
    """Create the runner script if it doesn't exist."""
    if _RUNNER_SCRIPT.exists():
        return
    _RUNNER_SCRIPT.write_text('''\
"""BPR353 runner: set env vars, import, run 1 step, dump results."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.params import dtmax
from magic_torch import fields

# Initialize and run 1 step
initialize_fields()
setup_initial_state()
initialize_dt(dtmax)
one_step(n_time_step=1, dt=dtmax)

# Save all 14 fields
out_dir = sys.argv[1]
field_map = {
    "w_step1": fields.w_LMloc,
    "dw_step1": fields.dw_LMloc,
    "ddw_step1": fields.ddw_LMloc,
    "z_step1": fields.z_LMloc,
    "dz_step1": fields.dz_LMloc,
    "s_step1": fields.s_LMloc,
    "ds_step1": fields.ds_LMloc,
    "p_step1": fields.p_LMloc,
    "dp_step1": fields.dp_LMloc,
    "b_step1": fields.b_LMloc,
    "db_step1": fields.db_LMloc,
    "ddb_step1": fields.ddb_LMloc,
    "aj_step1": fields.aj_LMloc,
    "dj_step1": fields.dj_LMloc,
}

for name, tensor in field_map.items():
    arr = tensor.cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

print("BPR353 runner completed successfully")
''')


# Run the subprocess once and cache results
_results = {}
_ran = False


def _run_bpr353():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="bpr353_")

    env = os.environ.copy()
    env["MAGIC_TIME_SCHEME"] = "BPR353"
    env["MAGIC_DEVICE"] = "cpu"  # float64 for precision

    result = subprocess.run(
        [sys.executable, str(_RUNNER_SCRIPT), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"BPR353 runner failed with code {result.returncode}")

    # Load results
    for name in _LM_FIELD_NAMES:
        fpath = os.path.join(out_dir, f"{name}.npy")
        _results[name] = np.load(fpath)


# Field specs: (ref_name, python_name, atol, rtol)
_FIELD_SPECS = {
    "w": ("w_step1", 1e-11, 1e-12),
    "z": ("z_step1", 1e-11, 1e-12),
    "s": ("s_step1", 1e-11, 1e-12),
    "p": ("p_step1", 1e-7, 1e-12),
    "b": ("b_step1", 1e-11, 1e-12),
    "aj": ("aj_step1", 1e-11, 1e-12),
    "dw": ("dw_step1", 1e-11, 1e-10),
    "dz": ("dz_step1", 1e-9, 1e-10),
    "ds": ("ds_step1", 1e-11, 1e-10),
    "dp": ("dp_step1", 1e-4, 1e-10),
    "db": ("db_step1", 1e-11, 1e-10),
    "dj": ("dj_step1", 1e-11, 1e-10),
    "ddw": ("ddw_step1", 1e-8, 1e-10),
    "ddb": ("ddb_step1", 1e-8, 1e-10),
}


@pytest.mark.parametrize("field_name", list(_FIELD_SPECS.keys()))
def test_bpr353_step1(field_name):
    """Compare BPR353 Python output against Fortran reference after 1 step."""
    _run_bpr353()
    ref_name, atol, rtol = _FIELD_SPECS[field_name]
    ref = load_bpr353_ref(ref_name)
    actual = _results[ref_name]
    np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol,
                               err_msg=f"BPR353 {field_name} mismatch")
