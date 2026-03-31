"""doubleDiffusion: One-step comparison against Fortran.

Loads checkpoint, takes ONE BPR353 time step (mode=1, no magnetic),
compares all fields (including xi/dxi) against Fortran reference.

Runs in a subprocess with doubleDiffusion env vars.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_dd

_RUNNER = Path(__file__).parent / "_dd_step1_runner.py"

_RUNNER_CODE = '''\
"""Step1 runner for doubleDiffusion."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_RA"] = "4.8e4"
os.environ["MAGIC_RAXI"] = "1.2e5"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_PR"] = "0.3"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Take one time step
one_step(1, 3.0e-4)

# Dump all OC fields (no magnetic for mode=1)
for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "xi_LMloc", "dxi_LMloc"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

print(f"DD Step1 runner completed, sim_time={sim_time}")
'''

_results = {}
_ran = False

DD_DIR = Path(__file__).parent.parent.parent / "samples" / "doubleDiffusion"
CKPT = DD_DIR / "checkpoint_end.start"


def _run():
    global _ran
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="dd_step1_")

    env = os.environ.copy()
    env.update({
        "MAGIC_TIME_SCHEME": "BPR353",
        "MAGIC_LMAX": "64",
        "MAGIC_NR": "33",
        "MAGIC_MINC": "4",
        "MAGIC_NCHEBMAX": "31",
        "MAGIC_MODE": "1",
        "MAGIC_RA": "4.8e4",
        "MAGIC_RAXI": "1.2e5",
        "MAGIC_SC": "3.0",
        "MAGIC_PR": "0.3",
        "MAGIC_EK": "1e-3",
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
        raise RuntimeError(f"DD Step1 runner failed: {result.returncode}")

    for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                 "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                 "xi_LMloc", "dxi_LMloc"):
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


# Field -> (ref_name, tolerance)
_FIELDS = {
    "w_LMloc": ("w_step1", 1e-10),
    "dw_LMloc": ("dw_step1", 1e-10),
    "ddw_LMloc": ("ddw_step1", 1e-9),
    "z_LMloc": ("z_step1", 1e-10),
    "dz_LMloc": ("dz_step1", 1e-10),
    "s_LMloc": ("s_step1", 1e-10),
    "ds_LMloc": ("ds_step1", 1e-10),
    "p_LMloc": ("p_step1", 1e-9),
    "dp_LMloc": ("dp_step1", 1e-10),
    "xi_LMloc": ("xi_step1", 1e-10),
    "dxi_LMloc": ("dxi_step1", 1e-10),
}


@pytest.mark.parametrize("attr,ref_name_tol", list(_FIELDS.items()))
def test_field(results, attr, ref_name_tol):
    """Each field after step 1 must match Fortran reference."""
    ref_name, tol = ref_name_tol
    ref = load_ref_dd(ref_name).cpu().numpy()
    py = results[attr]
    assert py.shape == ref.shape, f"{attr}: shape mismatch {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref, err_msg=f"{attr} should be zero")
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, f"{attr}: max rel err = {rel_err:.2e} (tol={tol:.0e})"
