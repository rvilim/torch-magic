"""Variable-dt integration test: CFL triggers dt change at step 1.

Uses dtMax=5e-4 (with intfac=10 to disable Coriolis clamp), which exceeds
dtrkc_min~3.66e-4 at step 1. CFL fires, reducing dt to ~2.74e-4. This exercises:
  - dt decrease (branch 2 of dt_courant)
  - dt history rolling (dt[0]=new, dt[1]=old)
  - IMEX weight recomputation with dt[0] != dt[1]
  - Full matrix rebuild

Runs in subprocess (like test_bpr353.py) to avoid module-level state pollution
from other tests that import with dtMax=1e-4.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

N_STEPS = 5

VARDT_REF = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark_vardt" / "fortran_ref"


# Snake -> standard reordering (l_max=16, minc=1)
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
    "w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp",
    "b", "db", "ddb", "aj", "dj",
}


def load_vardt_ref(name):
    arr = np.load(VARDT_REF / f"{name}.npy")
    # Reorder snake -> standard for LM fields
    base = name.split("_step")[0]  # e.g. "w" from "w_step1"
    if base in _LM_FIELD_NAMES and arr.ndim == 2 and arr.shape[0] == len(_SNAKE2ST):
        result = np.zeros_like(arr)
        result[_SNAKE2ST] = arr
        return result
    return arr


# --- Runner script (written to temp file) ---
_RUNNER_SCRIPT = Path(__file__).parent / "_vardt_runner.py"


def _ensure_runner():
    _RUNNER_SCRIPT.write_text('''\
"""Variable-dt runner: set env vars, import, run N steps, dump results."""
import os
os.environ["MAGIC_DTMAX"] = "5e-4"
os.environ["MAGIC_INTFAC"] = "10.0"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt, tscheme
from magic_torch.params import dtmax
from magic_torch import fields

N_STEPS = int(sys.argv[2])
out_dir = sys.argv[1]

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)

field_attrs = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "s_LMloc", "ds_LMloc",
    "p_LMloc", "dp_LMloc",
    "b_LMloc", "db_LMloc", "ddb_LMloc",
    "aj_LMloc", "dj_LMloc",
]
field_ref_names = [
    "w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp",
    "b", "db", "ddb", "aj", "dj",
]

dt = dtmax
for step in range(1, N_STEPS + 1):
    dt_new = one_step(n_time_step=step, dt=dt)

    # Save dt values
    np.save(os.path.join(out_dir, f"dt_new_step{step}.npy"),
            np.array(dt_new, dtype=np.float64))
    np.save(os.path.join(out_dir, f"dt1_step{step}.npy"),
            np.array(tscheme.dt[0].item(), dtype=np.float64))
    np.save(os.path.join(out_dir, f"dt2_step{step}.npy"),
            np.array(tscheme.dt[1].item(), dtype=np.float64))

    # Save fields
    for attr, ref_name in zip(field_attrs, field_ref_names):
        arr = getattr(fields, attr).cpu().numpy()
        np.save(os.path.join(out_dir, f"{ref_name}_step{step}.npy"), arr)

    dt = dt_new

print("Variable-dt runner completed successfully")
''')


# Cached results
_results = {}
_dt_results = {}
_ran = False


def _run_vardt():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="vardt_")

    env = os.environ.copy()
    env["MAGIC_DTMAX"] = "5e-4"
    env["MAGIC_INTFAC"] = "10.0"
    env["MAGIC_DEVICE"] = "cpu"

    result = subprocess.run(
        [sys.executable, str(_RUNNER_SCRIPT), out_dir, str(N_STEPS)],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Variable-dt runner failed with code {result.returncode}")

    # Load dt results
    for step in range(1, N_STEPS + 1):
        _dt_results[step] = {
            "dt_new": np.load(os.path.join(out_dir, f"dt_new_step{step}.npy")).item(),
            "dt1": np.load(os.path.join(out_dir, f"dt1_step{step}.npy")).item(),
            "dt2": np.load(os.path.join(out_dir, f"dt2_step{step}.npy")).item(),
        }

    # Load field results
    for step in range(1, N_STEPS + 1):
        _results[step] = {}
        for ref_name in ["w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp",
                         "b", "db", "ddb", "aj", "dj"]:
            _results[step][ref_name] = np.load(
                os.path.join(out_dir, f"{ref_name}_step{step}.npy"))


# --- Tests ---

def test_cfl_fires_step1():
    """CFL must fire at step 1: dt decreases from 5e-4."""
    _run_vardt()
    dt_new = _dt_results[1]["dt_new"]
    assert dt_new < 5e-4, f"CFL did not fire: dt_new={dt_new}"
    # dt2 should be the old dt (5e-4)
    dt2 = _dt_results[1]["dt2"]
    assert dt2 == 5e-4, f"dt2 should be old dtMax=5e-4, got {dt2}"
    # dt1 != dt2 (asymmetric history)
    dt1 = _dt_results[1]["dt1"]
    assert dt1 != dt2, f"dt1 should differ from dt2 after CFL fires"


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
def test_dt_values(step):
    """dt_new, dt1, dt2 must match Fortran scalars within ULP-level tolerance."""
    _run_vardt()
    for key in ["dt_new", "dt1", "dt2"]:
        ref = load_vardt_ref(f"{key}_step{step}")
        actual = _dt_results[step][key]
        np.testing.assert_allclose(actual, ref.item(), rtol=1e-12, atol=0,
                                   err_msg=f"step {step} {key} mismatch")


# Per-field tolerances (same as test_multistep.py, may need loosening for variable dt)
_FIELD_SPECS = {
    "w": 1e-11,
    "z": 1e-11,
    "s": 1e-11,
    "p": 1e-7,
    "b": 1e-11,
    "aj": 1e-11,
    "dw": 1e-11,
    "dz": 1e-9,
    "ds": 1e-11,
    "dp": 1e-4,
    "db": 1e-11,
    "dj": 1e-11,
    "ddw": 1e-8,
    "ddb": 1e-8,
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("field_name", list(_FIELD_SPECS.keys()))
def test_field_step(field_name, step):
    """Compare all 14 fields at each step against Fortran reference."""
    _run_vardt()
    atol = _FIELD_SPECS[field_name]
    ref = load_vardt_ref(f"{field_name}_step{step}")
    actual = _results[step][field_name]
    np.testing.assert_allclose(
        actual, ref, atol=atol, rtol=1e-10,
        err_msg=f"vardt {field_name} step {step} mismatch",
    )
