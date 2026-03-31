"""doubleDiffusion: Init field verification against Fortran reference.

Tests that checkpoint loading produces xi/dxi fields matching Fortran,
verifies composition boundary values, and confirms magnetic fields are
zero for mode=1.

Runs in a subprocess to isolate DD env vars.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path
import math

from conftest import load_ref_dd

_RUNNER = Path(__file__).parent / "_dd_init_runner.py"

_RUNNER_CODE = '''\
"""DD init runner: load checkpoint, dump initial fields."""
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
from magic_torch.step_time import setup_initial_state, initialize_dt
from magic_torch import fields
from magic_torch.params import l_mag

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Dump xi/dxi init fields (dxi computed by setup_initial_state via get_ddr)
for name in ("xi_LMloc", "dxi_LMloc", "b_LMloc", "aj_LMloc",
             "p_LMloc", "s_LMloc", "w_LMloc"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

# Dump l_mag flag
np.save(os.path.join(out_dir, "l_mag.npy"), np.array(l_mag))

print(f"DD init runner completed, sim_time={sim_time}")
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
    out_dir = tempfile.mkdtemp(prefix="dd_init_")

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
        raise RuntimeError(f"DD init runner failed: {result.returncode}")

    for name in ("xi_LMloc", "dxi_LMloc", "b_LMloc", "aj_LMloc",
                 "p_LMloc", "s_LMloc", "w_LMloc"):
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))
    _results["l_mag"] = bool(np.load(os.path.join(out_dir, "l_mag.npy")))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


def test_xi_init_matches_fortran(results):
    """xi_LMloc after checkpoint load must match Fortran xi_init."""
    ref = load_ref_dd("xi_init").cpu().numpy()
    py = results["xi_LMloc"]
    assert py.shape == ref.shape, f"shape: {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    rel_err = np.abs(py - ref).max() / max_abs
    assert rel_err < 1e-14, f"xi_init rel err = {rel_err:.2e}"


def test_dxi_init_matches_fortran(results):
    """dxi_LMloc after checkpoint load must match Fortran dxi_init."""
    ref = load_ref_dd("dxi_init").cpu().numpy()
    py = results["dxi_LMloc"]
    assert py.shape == ref.shape, f"shape: {py.shape} vs {ref.shape}"
    max_abs = np.abs(ref).max()
    rel_err = np.abs(py - ref).max() / max_abs
    # dxi is computed via get_ddr (matrix multiply), so FP ordering
    # differs from Fortran's costf-based derivative — 1e-12 is expected.
    assert rel_err < 1e-12, f"dxi_init rel err = {rel_err:.2e}"


def test_xi_boundary_values(results):
    """xi(l=0,m=0) at CMB should be ~0, at ICB should be ~sqrt(4pi)."""
    xi = results["xi_LMloc"]
    sq4pi = math.sqrt(4.0 * math.pi)

    # lm00 = 0 in standard ordering
    xi_lm00_cmb = xi[0, 0]   # CMB is nR=0
    xi_lm00_icb = xi[0, -1]  # ICB is nR=N-1

    # CMB: fixed at 0
    assert abs(xi_lm00_cmb) < 1e-10, f"xi(lm00, CMB) = {xi_lm00_cmb}"
    # ICB: fixed at sqrt(4*pi)
    assert abs(xi_lm00_icb.real - sq4pi) < 1e-10, (
        f"xi(lm00, ICB) = {xi_lm00_icb.real} (expect {sq4pi})")
    assert abs(xi_lm00_icb.imag) < 1e-14, (
        f"xi(lm00, ICB) imag = {xi_lm00_icb.imag}")


def test_mag_fields_zero_mode1(results):
    """In mode=1 (convection only), magnetic fields must be zero."""
    assert results["l_mag"] is False
    for name in ("b_LMloc", "aj_LMloc"):
        arr = results[name]
        assert np.all(arr == 0), f"{name} should be zero for mode=1, max={np.abs(arr).max()}"


def test_other_fields_match_fortran(results):
    """s, w, p after checkpoint load must match Fortran init values."""
    for field_name, ref_name in [("s_LMloc", "s_init"), ("w_LMloc", "w_init"),
                                  ("p_LMloc", "p_init")]:
        ref = load_ref_dd(ref_name).cpu().numpy()
        py = results[field_name]
        assert py.shape == ref.shape, f"{field_name}: shape {py.shape} vs {ref.shape}"
        max_abs = np.abs(ref).max()
        if max_abs == 0:
            np.testing.assert_array_equal(py, ref, err_msg=f"{field_name} should be zero")
        else:
            rel_err = np.abs(py - ref).max() / max_abs
            assert rel_err < 1e-14, f"{field_name} rel err = {rel_err:.2e}"
