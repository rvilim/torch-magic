"""doubleDiffusion: Multi-step comparison against Fortran (10 steps).

Loads checkpoint, takes 10 BPR353 time steps (mode=1, no magnetic),
compares all 11 OC fields (including xi/dxi) at each step against
Fortran reference dumps.

This mirrors test_bouss_multistep.py but for the doubleDiffusion config:
BPR353 DIRK, mode=1, l_max=64, minc=4, composition field active.

Runs in a subprocess to isolate the BPR353/DD module-level config.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_dd

N_STEPS = 10

_RUNNER = Path(__file__).parent / "_dd_multistep_runner.py"

_RUNNER_CODE = '''\
"""Multi-step runner for doubleDiffusion."""
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
n_steps = int(sys.argv[3])

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

OC_FIELDS = ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "xi_LMloc", "dxi_LMloc")

for step in range(1, n_steps + 1):
    one_step(step, 3.0e-4)
    for name in OC_FIELDS:
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)

print(f"Completed {n_steps} steps")
'''

DD_DIR = Path(__file__).parent.parent.parent / "samples" / "doubleDiffusion"
CKPT = DD_DIR / "checkpoint_end.start"

_results = {}
_ran = False


def _run():
    global _ran
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="dd_multistep_")

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
        [sys.executable, str(_RUNNER), str(CKPT), out_dir, str(N_STEPS)],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"DD multistep runner failed: {result.returncode}")

    for step in range(1, N_STEPS + 1):
        for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                     "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                     "xi_LMloc", "dxi_LMloc"):
            _results[(name, step)] = np.load(
                os.path.join(out_dir, f"{name}_step{step}.npy"))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


# Field specs: (ref_prefix, py_attr, tolerance)
# Tolerances match test_dd_step1.py for step 1.
_FIELD_SPECS = {
    "w": ("w_LMloc", 1e-10),
    "dw": ("dw_LMloc", 1e-10),
    "ddw": ("ddw_LMloc", 1e-9),
    "z": ("z_LMloc", 1e-10),
    "dz": ("dz_LMloc", 1e-10),
    "s": ("s_LMloc", 1e-10),
    "ds": ("ds_LMloc", 1e-10),
    "p": ("p_LMloc", 1e-9),
    "dp": ("dp_LMloc", 1e-10),
    "xi": ("xi_LMloc", 1e-10),
    "dxi": ("dxi_LMloc", 1e-10),
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_FIELD_SPECS.items()))
def test_field(results, ref_prefix, spec, step):
    """Each field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_dd(f"{ref_prefix}_step{step}").cpu().numpy()
    py = results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref,
                                       err_msg=f"{attr} step{step} should be zero")
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")
