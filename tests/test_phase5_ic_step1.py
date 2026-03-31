"""Phase 5: Coupled OC+IC magnetic solve — compare step 1 fields against Fortran.

Tests that after 1 time step with l_cond_ic=True (sigma_ratio=1.0, kbotb=3),
both OC and IC magnetic fields match the Fortran reference.

Runs in a subprocess with MAGIC_SIGMA_RATIO=1.0, MAGIC_KBOTB=3 to activate l_cond_ic.
Compares against Fortran reference in samples/dynamo_benchmark_condIC/fortran_ref/.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

CONDIC_REF = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark_condIC" / "fortran_ref"

_RUNNER = Path(__file__).parent / "_ic_step1_runner.py"


def _ensure_runner():
    if _RUNNER.exists():
        _RUNNER.unlink()
    _RUNNER.write_text('''\
"""IC step1 runner: set env vars, import, run 1 step, dump fields."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"

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

# OC fields after step 1 (different from insulating case)
np.save(os.path.join(out_dir, "b_step1.npy"), fields.b_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "db_step1.npy"), fields.db_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_step1.npy"), fields.ddb_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "aj_step1.npy"), fields.aj_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "dj_step1.npy"), fields.dj_LMloc.cpu().numpy())

# Non-magnetic OC fields (should also match)
np.save(os.path.join(out_dir, "w_step1.npy"), fields.w_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "z_step1.npy"), fields.z_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "s_step1.npy"), fields.s_LMloc.cpu().numpy())

print("IC step1 runner completed")
''')


_results = {}
_ran = False


def _run_ic_step1():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="ic_step1_")

    env = os.environ.copy()
    env["MAGIC_SIGMA_RATIO"] = "1.0"
    env["MAGIC_KBOTB"] = "3"
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
        raise RuntimeError(f"IC step1 runner failed: {result.returncode}")

    for name in ["b_ic_step1", "db_ic_step1", "ddb_ic_step1",
                  "aj_ic_step1", "dj_ic_step1", "ddj_ic_step1",
                  "b_step1", "db_step1", "ddb_step1", "aj_step1", "dj_step1",
                  "w_step1", "z_step1", "s_step1"]:
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


def _load_ref(name):
    """Load Fortran reference, reordering from snake to standard LM if needed."""
    arr = np.load(CONDIC_REF / f"{name}.npy")
    # Check if this is an LM field (shape (153, ...))
    if arr.shape[0] == 153:
        # Snake ordering for n_procs=1, l_max=16
        snake_l_order = [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 16]
        import itertools
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


# Tolerances for step 1 fields
# IC fields: coupled solve + even-Chebyshev costf + matrix derivative
# OC fields: coupled solve changes ICB BC, affecting poloidal/toroidal magnetic
_TOLERANCES = {
    # IC magnetic fields after step 1
    "b_ic_step1":   (1e-10, 1e-10),
    "db_ic_step1":  (1e-9, 1e-9),    # IC derivative accumulation
    "ddb_ic_step1": (1e-9, 1e-9),
    "aj_ic_step1":  (1e-10, 1e-10),
    "dj_ic_step1":  (1e-9, 1e-9),
    "ddj_ic_step1": (1e-9, 1e-9),
    # OC magnetic fields after step 1 (different from insulating)
    "b_step1":      (1e-11, 1e-11),
    "db_step1":     (1e-11, 1e-11),
    "ddb_step1":    (1e-9, 1e-9),    # 50x50 coupled matrix → more FP accumulation in d2
    "aj_step1":     (1e-11, 1e-11),
    "dj_step1":     (1e-11, 1e-11),
    # Non-magnetic OC fields (should match insulating case)
    "w_step1":      (1e-11, 1e-11),
    "z_step1":      (1e-11, 1e-11),
    "s_step1":      (1e-11, 1e-11),
}


@pytest.mark.parametrize("name", [
    "b_ic_step1", "aj_ic_step1",
    "db_ic_step1", "ddb_ic_step1",
    "dj_ic_step1", "ddj_ic_step1",
    "b_step1", "db_step1", "ddb_step1", "aj_step1", "dj_step1",
    "w_step1", "z_step1", "s_step1",
])
def test_ic_step1(name):
    """Compare step 1 field against Fortran reference."""
    _run_ic_step1()
    ref = _load_ref(name)
    actual = _results[name]
    atol, rtol = _TOLERANCES[name]
    np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol,
                               err_msg=f"Step 1 condIC {name} mismatch")
