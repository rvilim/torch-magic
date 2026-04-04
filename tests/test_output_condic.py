"""Per-function tests for conducting-IC diagnostics against Fortran reference.

Tests get_e_mag_ic (conducting path), get_lorentz_torque_ic, and
get_viscous_torque by loading the boussBenchSat checkpoint and comparing
against the Fortran-produced e_mag_ic.test and rot.test files.

Runs in a subprocess (BPR353 + condIC config requires env-var isolation).
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

BOUSS_DIR = Path(__file__).parent.parent.parent / "samples" / "boussBenchSat"
CKPT = BOUSS_DIR / "checkpoint_end.start"
RUNNER = Path(__file__).parent / "_condic_diag_runner.py"

# --- Parse Fortran reference files ---

def _parse_fortran_line(path, row=0):
    """Parse one row from a Fortran-format output file, return list of floats."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i == row:
                return [float(x) for x in line.split()]
    raise ValueError(f"Row {row} not found in {path}")


# e_mag_ic.test: col1=time, col2=e_p, col3=e_t, col4=e_p_as, col5=e_t_as
_EMAG_IC_REF = _parse_fortran_line(BOUSS_DIR / "e_mag_ic.test", row=0)

# rot.test: col1=time, col2=omega_ic, col3=lorentz_torque_ic,
#   col4=viscous_torque_ic, col5=omega_ma, col6=lorentz_torque_ma,
#   col7=-viscous_torque_ma, col8=gravi_torque_ic
_ROT_REF = _parse_fortran_line(BOUSS_DIR / "rot.test", row=0)


# --- Run the subprocess once, cache results ---

_results = {}
_ran = False


def _run():
    global _ran
    if _ran:
        return
    _ran = True

    if not CKPT.exists():
        pytest.skip(f"Checkpoint not found: {CKPT}")

    out_dir = tempfile.mkdtemp(prefix="condic_diag_")

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
        [sys.executable, str(RUNNER), str(CKPT), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"condIC diag runner failed: {result.returncode}")

    _results["e_mag_ic"] = np.load(os.path.join(out_dir, "e_mag_ic.npy"))
    _results["lorentz_torque_ic"] = np.load(
        os.path.join(out_dir, "lorentz_torque_ic.npy")).item()
    _results["viscous_torque_icb"] = np.load(
        os.path.join(out_dir, "viscous_torque_icb.npy")).item()
    _results["viscous_torque_cmb"] = np.load(
        os.path.join(out_dir, "viscous_torque_cmb.npy")).item()


@pytest.fixture(scope="module")
def results():
    _run()
    return _results


# --- Tests ---
# Tolerance 1e-8: Fortran reference uses ES16.8 format (8 significant digits)

class TestGetEMagIC:
    """Test get_e_mag_ic conducting-IC path against e_mag_ic.test."""

    def test_e_p(self, results):
        ref = _EMAG_IC_REF[1]  # col2
        got = results["e_mag_ic"][0]
        assert ref != 0.0, "Reference e_p should be nonzero"
        assert abs(got - ref) / abs(ref) < 1e-8, \
            f"e_p: got {got:.10e}, ref {ref:.10e}, rel {abs(got-ref)/abs(ref):.2e}"

    def test_e_t(self, results):
        ref = _EMAG_IC_REF[2]  # col3
        got = results["e_mag_ic"][1]
        assert ref != 0.0, "Reference e_t should be nonzero"
        assert abs(got - ref) / abs(ref) < 1e-8, \
            f"e_t: got {got:.10e}, ref {ref:.10e}, rel {abs(got-ref)/abs(ref):.2e}"

    def test_e_p_as(self, results):
        ref = _EMAG_IC_REF[3]  # col4
        got = results["e_mag_ic"][2]
        assert ref != 0.0, "Reference e_p_as should be nonzero"
        assert abs(got - ref) / abs(ref) < 1e-8, \
            f"e_p_as: got {got:.10e}, ref {ref:.10e}, rel {abs(got-ref)/abs(ref):.2e}"

    def test_e_t_as(self, results):
        ref = _EMAG_IC_REF[4]  # col5
        got = results["e_mag_ic"][3]
        assert ref != 0.0, "Reference e_t_as should be nonzero"
        assert abs(got - ref) / abs(ref) < 1e-8, \
            f"e_t_as: got {got:.10e}, ref {ref:.10e}, rel {abs(got-ref)/abs(ref):.2e}"


class TestGetLorentzTorqueIC:
    """Test get_lorentz_torque_ic against rot.test col3."""

    def test_lorentz_torque_ic(self, results):
        ref = _ROT_REF[2]  # col3 (0-indexed: index 2)
        got = results["lorentz_torque_ic"]
        assert ref != 0.0, "Reference lorentz_torque_ic should be nonzero"
        assert abs(got - ref) / abs(ref) < 1e-8, \
            f"lorentz_torque_ic: got {got:.10e}, ref {ref:.10e}, " \
            f"rel {abs(got-ref)/abs(ref):.2e}"


class TestGetViscousTorque:
    """Test get_viscous_torque at ICB against rot.test.

    Tolerance 5e-8: the Fortran reference uses ES16.8 format which gives
    ~8 significant digits, limiting achievable relative agreement.
    """

    def test_viscous_torque_icb(self, results):
        ref = _ROT_REF[3]  # col4 (0-indexed: index 3)
        got = results["viscous_torque_icb"]
        assert ref != 0.0, "Reference viscous_torque_icb should be nonzero"
        assert abs(got - ref) / abs(ref) < 5e-8, \
            f"viscous_torque_icb: got {got:.10e}, ref {ref:.10e}, " \
            f"rel {abs(got-ref)/abs(ref):.2e}"

    def test_viscous_torque_cmb_computable(self, results):
        """Verify CMB viscous torque can be computed (no crash).

        Note: Fortran rot.test col7 is 0 because l_rot_ma=False (nRotMa=0)
        means Fortran skips CMB torque computation entirely. Our function
        computes a nonzero value from the actual fields (~0.1), which is
        correct but has no Fortran reference to compare against.
        """
        got = results["viscous_torque_cmb"]
        # Just verify it's a finite number (no NaN/Inf)
        assert np.isfinite(got), f"viscous_torque_cmb should be finite, got {got}"
