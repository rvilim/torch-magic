"""boussBenchSat Phase 0: Blocking arrays at l_max=64, minc=4.

Verifies st_lm2l, st_lm2m, lm2lmA, lm2lmS match Fortran reference.
These are fundamental index mappings — if wrong, everything downstream fails.

Runs in a subprocess with MAGIC_LMAX=64, MAGIC_MINC=4 env vars.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

from conftest import load_ref_bouss

_RUNNER = Path(__file__).parent / "_bouss_blocking_runner.py"

_RUNNER_CODE = '''\
"""Blocking runner for boussBenchSat config."""
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
from magic_torch.blocking import st_lm2l, st_lm2m, st_lm2lmA, st_lm2lmS
from magic_torch.params import lm_max, l_max, m_max, minc

out_dir = sys.argv[1]
np.save(os.path.join(out_dir, "lm2l.npy"), st_lm2l.cpu().numpy())
np.save(os.path.join(out_dir, "lm2m.npy"), st_lm2m.cpu().numpy())
np.save(os.path.join(out_dir, "lm2lmA.npy"), st_lm2lmA.cpu().numpy())
np.save(os.path.join(out_dir, "lm2lmS.npy"), st_lm2lmS.cpu().numpy())
np.save(os.path.join(out_dir, "lm_max.npy"), np.array(lm_max))
np.save(os.path.join(out_dir, "l_max.npy"), np.array(l_max))
np.save(os.path.join(out_dir, "minc.npy"), np.array(minc))
print("Blocking runner completed")
'''

_results = {}
_ran = False


def _run():
    global _ran
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="bouss_blocking_")

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
        [sys.executable, str(_RUNNER), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Blocking runner failed: {result.returncode}")

    for name in ("lm2l", "lm2m", "lm2lmA", "lm2lmS", "lm_max", "l_max", "minc"):
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


def test_lm_max(results):
    """lm_max should be 561 for l_max=64, minc=4."""
    assert results["lm_max"] == 561


def test_l_max(results):
    assert results["l_max"] == 64


def test_minc(results):
    assert results["minc"] == 4


def test_lm2l(results):
    """st_lm2l must match Fortran reference exactly."""
    ref = load_ref_bouss("lm2l", reorder_lm=False)
    py = results["lm2l"]
    # Fortran lm2l stores 0-based l degree values (0..64)
    np.testing.assert_array_equal(py, ref.cpu().numpy(),
                                  err_msg="st_lm2l mismatch vs Fortran")


def test_lm2m(results):
    """st_lm2m must match Fortran reference exactly."""
    ref = load_ref_bouss("lm2m", reorder_lm=False)
    py = results["lm2m"]
    # Fortran lm2m is absolute m values (0, 4, 8, ...), Python same
    np.testing.assert_array_equal(py, ref.cpu().numpy(),
                                  err_msg="st_lm2m mismatch vs Fortran")


def test_lm2lmA(results):
    """st_lm2lmA (l+1 neighbor) must match Fortran reference."""
    ref = load_ref_bouss("lm2lmA", reorder_lm=False)
    py = results["lm2lmA"]
    # Fortran is 1-based (>0 = valid), -1 = no neighbor
    # Python is 0-based, -1 = no neighbor
    ref_np = ref.cpu().numpy()
    ref_0based = np.where(ref_np >= 1, ref_np - 1, -1)
    np.testing.assert_array_equal(py, ref_0based,
                                  err_msg="st_lm2lmA mismatch vs Fortran")


def test_lm2lmS(results):
    """st_lm2lmS (l-1 neighbor) must match Fortran reference."""
    ref = load_ref_bouss("lm2lmS", reorder_lm=False)
    py = results["lm2lmS"]
    ref_np = ref.cpu().numpy()
    ref_0based = np.where(ref_np >= 1, ref_np - 1, -1)
    np.testing.assert_array_equal(py, ref_0based,
                                  err_msg="st_lm2lmS mismatch vs Fortran")


def test_m_values_only_multiples_of_minc(results):
    """All m values should be multiples of minc=4."""
    m_vals = np.unique(results["lm2m"])
    for m in m_vals:
        assert m % 4 == 0, f"m={m} is not a multiple of minc=4"
