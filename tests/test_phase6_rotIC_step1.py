"""Phase 6: Coupled OC+IC magnetic solve with rotating IC — compare step 1 fields.

Tests that after 1 time step with l_cond_ic=True AND l_rot_ic=True
(sigma_ratio=1.0, kbotb=3, nRotIC=1), all fields match Fortran reference.

Runs in a subprocess with env vars to activate conducting+rotating IC.
Compares against Fortran reference in samples/dynamo_benchmark_condICrotIC/fortran_ref/.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

ROTRIC_REF = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark_condICrotIC" / "fortran_ref"

_RUNNER = Path(__file__).parent / "_rotIC_step1_runner.py"


def _ensure_runner():
    if _RUNNER.exists():
        _RUNNER.unlink()
    _RUNNER.write_text('''\
"""RotIC step1 runner: set env vars, import, run 1 step, dump fields."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_NROTIC"] = "1"

import sys
import numpy as np

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.params import dtmax
from magic_torch import fields

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)
one_step(1, dtmax)

out_dir = sys.argv[1]

# IC fields after step 1
np.save(os.path.join(out_dir, "b_ic_step1.npy"), fields.b_ic.cpu().numpy())
np.save(os.path.join(out_dir, "db_ic_step1.npy"), fields.db_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_ic_step1.npy"), fields.ddb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "aj_ic_step1.npy"), fields.aj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dj_ic_step1.npy"), fields.dj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddj_ic_step1.npy"), fields.ddj_ic.cpu().numpy())

# OC fields after step 1
np.save(os.path.join(out_dir, "b_step1.npy"), fields.b_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "db_step1.npy"), fields.db_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_step1.npy"), fields.ddb_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "aj_step1.npy"), fields.aj_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "dj_step1.npy"), fields.dj_LMloc.cpu().numpy())

# Non-magnetic OC fields
np.save(os.path.join(out_dir, "w_step1.npy"), fields.w_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "z_step1.npy"), fields.z_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "s_step1.npy"), fields.s_LMloc.cpu().numpy())

# IC rotation rate
np.save(os.path.join(out_dir, "omega_ic_step1.npy"), np.array(fields.omega_ic))

print("RotIC step1 runner completed")
''')


_results = {}
_ran = False


def _run_rotIC_step1():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="rotIC_step1_")

    env = os.environ.copy()
    env["MAGIC_SIGMA_RATIO"] = "1.0"
    env["MAGIC_KBOTB"] = "3"
    env["MAGIC_NROTIC"] = "1"
    env["MAGIC_DEVICE"] = "cpu"

    result = subprocess.run(
        [sys.executable, str(_RUNNER), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"RotIC step1 runner failed: {result.returncode}")

    for name in ["b_ic_step1", "db_ic_step1", "ddb_ic_step1",
                  "aj_ic_step1", "dj_ic_step1", "ddj_ic_step1",
                  "b_step1", "db_step1", "ddb_step1", "aj_step1", "dj_step1",
                  "w_step1", "z_step1", "s_step1",
                  "omega_ic_step1"]:
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


def _load_ref(name):
    """Load Fortran reference, reordering from snake to standard LM if needed."""
    arr = np.load(ROTRIC_REF / f"{name}.npy")
    # Scalar (e.g., omega_ic) — no reorder needed
    if arr.ndim == 0:
        return arr
    # Check if this is an LM field (shape (153, ...))
    if arr.shape[0] == 153:
        # Snake ordering for n_procs=1, l_max=16
        snake_l_order = [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 16]
        snake_lm = []
        for l_val in snake_l_order:
            for m_val in range(0, l_val + 1):
                snake_lm.append((l_val, m_val))
        # Standard ordering
        std_lm = []
        for m_val in range(0, 17):
            for l_val in range(m_val, 17):
                std_lm.append((l_val, m_val))
        snake_idx = {lm: i for i, lm in enumerate(snake_lm)}
        reorder = [snake_idx[lm] for lm in std_lm]
        arr = arr[reorder]
    return arr


# Tolerances for step 1 fields with rotating IC
_TOLERANCES = {
    # IC magnetic fields after step 1
    "b_ic_step1":   (1e-10, 1e-10),
    "db_ic_step1":  (1e-9, 1e-9),
    "ddb_ic_step1": (1e-9, 1e-9),
    "aj_ic_step1":  (1e-10, 1e-10),
    "dj_ic_step1":  (1e-9, 1e-9),
    "ddj_ic_step1": (1e-9, 1e-9),
    # OC magnetic fields after step 1
    "b_step1":      (1e-11, 1e-11),
    "db_step1":     (1e-11, 1e-11),
    "ddb_step1":    (1e-9, 1e-9),
    "aj_step1":     (1e-11, 1e-11),
    "dj_step1":     (1e-11, 1e-11),
    # Non-magnetic OC fields
    "w_step1":      (1e-11, 1e-11),
    "z_step1":      (1e-11, 1e-11),
    "s_step1":      (1e-11, 1e-11),
    # IC rotation rate (scalar)
    "omega_ic_step1": (1e-10, 1e-10),
}


@pytest.mark.parametrize("name", [
    "b_ic_step1", "aj_ic_step1",
    "db_ic_step1", "ddb_ic_step1",
    "dj_ic_step1", "ddj_ic_step1",
    "b_step1", "db_step1", "ddb_step1", "aj_step1", "dj_step1",
    "w_step1", "z_step1", "s_step1",
    "omega_ic_step1",
])
def test_rotIC_step1(name):
    """Compare step 1 field against Fortran reference with rotating IC."""
    _run_rotIC_step1()
    ref = _load_ref(name)
    actual = _results[name]
    atol, rtol = _TOLERANCES[name]
    np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol,
                               err_msg=f"Step 1 rotIC {name} mismatch")
