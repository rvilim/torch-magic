"""boussBenchSat: Multi-step comparison against Fortran (10 steps).

Loads checkpoint, takes 10 BPR353 time steps, compares all OC + IC fields
and omega_ic at each step against Fortran reference dumps.

This mirrors test_multistep.py (dynamo_benchmark CNAB2) but for the full
boussBenchSat config: BPR353 DIRK, conducting+rotating IC, minc=4.

Runs in a subprocess to isolate the BPR353 module-level config.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_bouss

N_STEPS = 10

_RUNNER = Path(__file__).parent / "_bouss_multistep_runner.py"

_RUNNER_CODE = '''\
"""Multi-step runner for boussBenchSat."""
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
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]
n_steps = int(sys.argv[3])

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

OC_FIELDS = ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc")
IC_FIELDS = ("b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic")

for step in range(1, n_steps + 1):
    one_step(step, 2.0e-4)
    for name in OC_FIELDS + IC_FIELDS:
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    np.save(os.path.join(out_dir, f"omega_ic_step{step}.npy"),
            np.array(fields.omega_ic))

print(f"Completed {n_steps} steps")
'''

BOUSS_DIR = Path(__file__).parent.parent.parent / "samples" / "boussBenchSat"
CKPT = BOUSS_DIR / "checkpoint_end.start"

_results = {}
_ran = False


def _run():
    global _ran
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="bouss_multistep_")

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
        [sys.executable, str(_RUNNER), str(CKPT), out_dir, str(N_STEPS)],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Multistep runner failed: {result.returncode}")

    for step in range(1, N_STEPS + 1):
        for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
                     "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
                     "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc",
                     "b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"):
            _results[(name, step)] = np.load(
                os.path.join(out_dir, f"{name}_step{step}.npy"))
        _results[("omega_ic", step)] = np.load(
            os.path.join(out_dir, f"omega_ic_step{step}.npy"))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


# OC fields: (ref_prefix, py_attr, atol)
# Tolerances match test_bouss_step1.py for step 1; may grow over steps.
_OC_SPECS = {
    "w": ("w_LMloc", 1e-10),
    "dw": ("dw_LMloc", 1e-10),
    "ddw": ("ddw_LMloc", 1e-9),
    "z": ("z_LMloc", 1e-10),
    "dz": ("dz_LMloc", 1e-10),
    "s": ("s_LMloc", 1e-10),
    "ds": ("ds_LMloc", 1e-10),
    "p": ("p_LMloc", 1e-10),
    "dp": ("dp_LMloc", 1e-10),
    "b": ("b_LMloc", 1e-10),
    "db": ("db_LMloc", 1e-10),
    "ddb": ("ddb_LMloc", 1e-9),
    "aj": ("aj_LMloc", 1e-10),
    "dj": ("dj_LMloc", 1e-10),
}

_IC_SPECS = {
    "b_ic": ("b_ic", 1e-10),
    "db_ic": ("db_ic", 1e-9),
    "ddb_ic": ("ddb_ic", 1e-7),
    "aj_ic": ("aj_ic", 1e-10),
    "dj_ic": ("dj_ic", 1e-10),
    "ddj_ic": ("ddj_ic", 1e-9),
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_OC_SPECS.items()))
def test_oc_field(results, ref_prefix, spec, step):
    """OC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_bouss(f"{ref_prefix}_step{step}").cpu().numpy()
    py = results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("ref_prefix,spec", list(_IC_SPECS.items()))
def test_ic_field(results, ref_prefix, spec, step):
    """IC field at each step must match Fortran reference."""
    attr, tol = spec
    ref = load_ref_bouss(f"{ref_prefix}_step{step}").cpu().numpy()
    py = results[(attr, step)]
    assert py.shape == ref.shape, f"{attr} step{step}: shape {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    if max_abs == 0:
        np.testing.assert_array_equal(py, ref)
    else:
        rel_err = np.abs(py - ref).max() / max_abs
        assert rel_err < tol, (
            f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
def test_omega_ic(results, step):
    """omega_ic at each step must match Fortran reference."""
    ref = load_ref_bouss(f"omega_ic_step{step}").cpu().item()
    py = float(results[("omega_ic", step)])
    if ref == 0:
        assert py == 0, f"omega_ic step{step}: py={py}, expected 0"
    else:
        rel_err = abs(py - ref) / abs(ref)
        assert rel_err < 1e-10, (
            f"omega_ic step{step}: py={py:.12e}, ref={ref:.12e}, rel_err={rel_err:.2e}")
