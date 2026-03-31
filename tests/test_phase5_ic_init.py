"""Phase 5: IC field initialization — compare against Fortran reference.

Tests the inner core magnetic field initialization and IC derivatives:
- b_ic_init, aj_ic_init (field profiles after initB)
- db_ic_init, ddb_ic_init, dj_ic_init, ddj_ic_init (after get_mag_ic_rhs_imp)

Also tests OC fields b_init, db_init, aj_init which change with l_cond_ic.

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

_RUNNER = Path(__file__).parent / "_ic_init_runner.py"


def _ensure_runner():
    if _RUNNER.exists():
        return
    _RUNNER.write_text('''\
"""IC init runner: set env vars, import, run init, dump IC fields."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"

import sys
import numpy as np

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state
from magic_torch import fields

initialize_fields()
setup_initial_state()

out_dir = sys.argv[1]

# IC fields after init + get_mag_ic_rhs_imp
np.save(os.path.join(out_dir, "b_ic_init.npy"), fields.b_ic.cpu().numpy())
np.save(os.path.join(out_dir, "db_ic_init.npy"), fields.db_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_ic_init.npy"), fields.ddb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "aj_ic_init.npy"), fields.aj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dj_ic_init.npy"), fields.dj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddj_ic_init.npy"), fields.ddj_ic.cpu().numpy())

# OC fields (different from dynamo_benchmark when l_cond_ic)
np.save(os.path.join(out_dir, "b_init.npy"), fields.b_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "db_init.npy"), fields.db_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "aj_init.npy"), fields.aj_LMloc.cpu().numpy())

print("IC init runner completed")
''')


_results = {}
_ran = False


def _run_ic_init():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="ic_init_")

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
        timeout=60,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"IC init runner failed: {result.returncode}")

    for name in ["b_ic_init", "db_ic_init", "ddb_ic_init",
                  "aj_ic_init", "dj_ic_init", "ddj_ic_init",
                  "b_init", "db_init", "aj_init"]:
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


# Fortran field dumps are in snake LM ordering; must reorder to standard.
# conftest.py has this logic but we do it inline since we run in a separate test.
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
        # Build reorder index: for each standard lm, find its position in snake
        snake_idx = {lm: i for i, lm in enumerate(snake_lm)}
        reorder = [snake_idx[lm] for lm in std_lm]
        arr = arr[reorder]
    return arr


# Tolerances: IC derivatives have FMA-related FP differences from Chebyshev recursion
# (see test_phase5_ic_grid.py). Derivative arrays accumulate this through get_ddr_even.
_TOLERANCES = {
    "b_ic_init":   (1e-14, 1e-14),
    "aj_ic_init":  (1e-14, 1e-14),
    "db_ic_init":  (1e-10, 1e-10),   # matrix-based IC derivative vs Fortran costf-based
    "ddb_ic_init": (1e-10, 1e-10),
    "dj_ic_init":  (1e-10, 1e-10),
    "ddj_ic_init": (1e-10, 1e-10),
    "b_init":      (1e-14, 1e-14),
    "db_init":     (1e-11, 1e-11),   # OC derivative FP accumulation
    "aj_init":     (1e-14, 1e-14),
}


@pytest.mark.parametrize("name", [
    "b_ic_init", "aj_ic_init",
    "db_ic_init", "ddb_ic_init",
    "dj_ic_init", "ddj_ic_init",
    "b_init", "db_init", "aj_init",
])
def test_ic_init(name):
    """Compare IC init field against Fortran reference."""
    _run_ic_init()
    ref = _load_ref(name)
    actual = _results[name]
    atol, rtol = _TOLERANCES[name]
    np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol,
                               err_msg=f"IC init {name} mismatch")
