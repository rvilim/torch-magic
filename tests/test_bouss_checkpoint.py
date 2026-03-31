"""boussBenchSat: Checkpoint load verification.

Loads the Fortran checkpoint and compares all initial fields against
Fortran reference init dumps. If the checkpoint load is wrong,
everything downstream is wrong.

Runs in a subprocess with boussBenchSat env vars.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_bouss

_RUNNER = Path(__file__).parent / "_bouss_checkpoint_runner.py"

_RUNNER_CODE = '''\
"""Checkpoint load runner for boussBenchSat."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_NCHEBICMAX"] = "15"
os.environ["MAGIC_RA"] = "1.1e5"
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_NROTIC"] = "1"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt
from magic_torch import fields

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

# Dump all OC fields
for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

# Dump IC fields
for name in ("b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

# omega_ic
np.save(os.path.join(out_dir, "omega_ic.npy"), np.array(fields.omega_ic))

print(f"Checkpoint runner completed, sim_time={sim_time}")
'''

_results = {}
_ran = False

BOUSS_DIR = Path(__file__).parent.parent.parent / "samples" / "boussBenchSat"
CKPT = BOUSS_DIR / "checkpoint_end.start"


def _run():
    global _ran
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="bouss_ckpt_")

    env = os.environ.copy()
    env.update({
        "MAGIC_TIME_SCHEME": "BPR353",
        "MAGIC_LMAX": "64",
        "MAGIC_NR": "33",
        "MAGIC_MINC": "4",
        "MAGIC_NCHEBMAX": "31",
        "MAGIC_NCHEBICMAX": "15",
        "MAGIC_RA": "1.1e5",
        "MAGIC_SIGMA_RATIO": "1.0",
        "MAGIC_KBOTB": "3",
        "MAGIC_NROTIC": "1",
        "MAGIC_DEVICE": "cpu",
    })

    result = subprocess.run(
        [sys.executable, str(_RUNNER), str(CKPT), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Checkpoint runner failed: {result.returncode}")

    for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                 "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                 "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc",
                 "b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic",
                 "omega_ic"):
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


# Mapping from field attribute name to (Fortran reference name, tolerance).
# Primary fields loaded directly from checkpoint: exact match (1e-13).
# Derivative fields computed via matrix multiply (Python) vs costf+recurrence
# (Fortran): equivalent algorithms differ at FP rounding level.
# First derivatives: ~1e-13, second: ~1e-11, third: ~1e-10.
# IC even-Chebyshev derivatives have larger condition numbers (17-point grid).
_OC_FIELDS = {
    "w_LMloc": ("w_init", 1e-13),
    "dw_LMloc": ("dw_init", 1e-12),
    "ddw_LMloc": ("ddw_init", 1e-11),
    "z_LMloc": ("z_init", 1e-13),
    "dz_LMloc": ("dz_init", 1e-12),
    "s_LMloc": ("s_init", 1e-13),
    "ds_LMloc": ("ds_init", 1e-12),
    "p_LMloc": ("p_init", 1e-13),
    "dp_LMloc": ("dp_init", 1e-12),
    "b_LMloc": ("b_init", 1e-13),
    "db_LMloc": ("db_init", 1e-11),
    "ddb_LMloc": ("ddb_init", 1e-10),
    "aj_LMloc": ("aj_init", 1e-13),
    "dj_LMloc": ("dj_init", 1e-13),
}

_IC_FIELDS = {
    "b_ic": ("b_ic_init", 1e-13),
    "db_ic": ("db_ic_init", 1e-10),
    "ddb_ic": ("ddb_ic_init", 1e-8),
    "aj_ic": ("aj_ic_init", 1e-13),
    "dj_ic": ("dj_ic_init", 1e-12),
    "ddj_ic": ("ddj_ic_init", 1e-10),
}


@pytest.mark.parametrize("attr,ref_name_tol", list(_OC_FIELDS.items()))
def test_oc_field(results, attr, ref_name_tol):
    """Each OC field from checkpoint must match Fortran init dump."""
    ref_name, tol = ref_name_tol
    ref = load_ref_bouss(ref_name).cpu().numpy()
    py = results[attr]
    assert py.shape == ref.shape, f"{attr}: shape mismatch {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref, err_msg=f"{attr} should be zero")
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, f"{attr}: max rel err = {rel_err:.2e} (tol={tol:.0e})"


@pytest.mark.parametrize("attr,ref_name_tol", list(_IC_FIELDS.items()))
def test_ic_field(results, attr, ref_name_tol):
    """Each IC field from checkpoint must match Fortran init dump."""
    ref_name, tol = ref_name_tol
    ref = load_ref_bouss(ref_name).cpu().numpy()
    py = results[attr]
    assert py.shape == ref.shape, f"{attr}: shape mismatch {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref, err_msg=f"{attr} should be zero")
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, f"{attr}: max rel err = {rel_err:.2e} (tol={tol:.0e})"


def test_omega_ic(results):
    """omega_ic from checkpoint must match Fortran init dump."""
    ref = load_ref_bouss("omega_ic_init").cpu().item()
    py = float(results["omega_ic"])
    rel_err = abs(py - ref) / abs(ref) if ref != 0 else abs(py)
    assert rel_err < 1e-13, f"omega_ic: py={py}, ref={ref}, rel_err={rel_err:.2e}"
