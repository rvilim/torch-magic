"""Tests for CFL Courant condition (courant.py, radial_functions delxr2/delxh2).

Tests cover:
1. Grid spacing arrays (delxr2, delxh2) vs Fortran reference
2. CFL arrays (dtrkc, dthkc) at step 1 vs Fortran reference (magnetic path)
3. dt_courant decision logic (all 4 branches)
4. CFL arrays at step 1 for doubleDiffusion (non-magnetic path, subprocess)
"""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from magic_torch.radial_functions import delxr2, delxh2
from magic_torch.courant import dt_courant


def _load_fortran_1d(path):
    """Load a 1D real dump from Fortran (big-endian, with ndim+n header)."""
    with open(path, "rb") as f:
        ndim = struct.unpack(">i", f.read(4))[0]
        n = struct.unpack(">i", f.read(4))[0]
        return np.frombuffer(f.read(n * 8), dtype=">f8").copy()


DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "dynamo_benchmark", "fortran_dumps")


class TestGridSpacing:
    """delxr2, delxh2 vs Fortran preCalculations.f90."""

    def test_delxr2(self):
        ref = _load_fortran_1d(f"{DUMP_DIR}/delxr2.dat")
        py = delxr2.cpu().numpy()
        assert py.shape == ref.shape
        np.testing.assert_allclose(py, ref, atol=1e-17, rtol=2e-13)

    def test_delxh2(self):
        ref = _load_fortran_1d(f"{DUMP_DIR}/delxh2.dat")
        py = delxh2.cpu().numpy()
        assert py.shape == ref.shape
        np.testing.assert_allclose(py, ref, atol=1e-17, rtol=1e-15)


@pytest.mark.skipif(
    int(os.environ.get("MAGIC_RADIAL_CHUNK", "0")) > 0,
    reason="Per-level CFL arrays not available in chunked mode"
)
class TestCFLStep1:
    """dtrkc, dthkc at step 1 vs Fortran courant.f90."""

    @pytest.fixture(scope="class")
    def cfl_arrays(self):
        """Run one radial loop and return CFL arrays."""
        from magic_torch.init_fields import initialize_fields
        from magic_torch.step_time import (
            setup_initial_state, initialize_dt, _radial_loop_dispatch,
            _vrc, _vtc, _vpc,
        )
        from magic_torch.courant import courant_check
        from magic_torch.time_scheme import tscheme

        initialize_fields()
        setup_initial_state()
        initialize_dt(1e-4)
        # Run the radial loop (fills velocity buffers in non-chunked mode)
        _radial_loop_dispatch()
        # Get per-level CFL arrays by calling courant_check directly
        from magic_torch.step_time import last_brc, last_btc, last_bpc
        from magic_torch.params import l_mag
        from magic_torch import fields
        # In non-chunked mode, velocity buffers are filled; use them
        dtrkc, dthkc = courant_check(
            _vrc, _vtc, _vpc,
            last_brc if l_mag else None,
            last_btc if l_mag else None,
            last_bpc if l_mag else None,
            courfac=tscheme.courfac,
            alffac=tscheme.alffac,
        )
        return dtrkc.cpu().numpy(), dthkc.cpu().numpy()

    def test_dtrkc(self, cfl_arrays):
        py_dtrkc, _ = cfl_arrays
        ref = _load_fortran_1d(f"{DUMP_DIR}/dtrkc_step1.dat")
        assert py_dtrkc.shape == ref.shape
        np.testing.assert_allclose(py_dtrkc, ref, atol=1e-16, rtol=1e-13)

    def test_dthkc(self, cfl_arrays):
        _, py_dthkc = cfl_arrays
        ref = _load_fortran_1d(f"{DUMP_DIR}/dthkc_step1.dat")
        assert py_dthkc.shape == ref.shape
        np.testing.assert_allclose(py_dthkc, ref, atol=1e-15, rtol=1e-12)

    def test_cfl_does_not_change_dt(self, cfl_arrays):
        """With dtMax=1e-4, CFL should NOT trigger a dt change."""
        py_dtrkc, py_dthkc = cfl_arrays
        l_new_dt, dt_new = dt_courant(1e-4, 1e-4,
                                       float(py_dtrkc.min()),
                                       float(py_dthkc.min()))
        assert not l_new_dt
        assert dt_new == 1e-4


class TestDtCourant:
    """Unit tests for dt_courant decision logic (all 4 branches)."""

    def test_branch1_dt_exceeds_dtmax(self):
        """dt > dtMax: clamp to dtMax."""
        l_new, dt_new = dt_courant(0.5, 0.1, 1.0, 1.0)
        assert l_new
        assert dt_new == 0.1

    def test_branch2_dt_exceeds_cfl_min(self):
        """dt > dtMin: decrease to dt_2."""
        l_new, dt_new = dt_courant(0.1, 0.2, 0.05, 1.0)
        assert l_new
        dt_2 = min(0.5 * (1.0 / 2.0 + 1.0) * 0.05, 0.2)  # = 0.0375
        assert abs(dt_new - dt_2) < 1e-15

    def test_branch3_dt_too_small(self):
        """dt_fac * dt < dtMin and dt < dtMax: increase to dt_2."""
        l_new, dt_new = dt_courant(0.1, 0.5, 1.0, 1.0)
        assert l_new
        dt_2 = min(0.5 * (1.0 / 2.0 + 1.0) * 1.0, 0.5)  # = 0.5
        assert abs(dt_new - dt_2) < 1e-15

    def test_branch4_dt_in_range(self):
        """dt in comfortable range: no change."""
        l_new, dt_new = dt_courant(0.1, 0.2, 0.15, 1.0)
        # dt=0.1 < dtMin=0.15, dt_fac*dt=0.2 >= dtMin=0.15 → no increase
        assert not l_new
        assert dt_new == 0.1

    def test_branch4_dt_equals_dtmax(self):
        """dt == dtMax and CFL is satisfied: no change (like dynamo_benchmark)."""
        l_new, dt_new = dt_courant(1e-4, 1e-4, 3.66e-4, 5.14e-4)
        # dt_fac*dt=2e-4 < dtMin=3.66e-4 but dt=1e-4 is NOT < dtMax=1e-4
        assert not l_new
        assert dt_new == 1e-4


# ---------------------------------------------------------------------------
# Non-magnetic CFL tests (doubleDiffusion, subprocess)
# ---------------------------------------------------------------------------

_NOMAG_RUNNER_CODE = '''\
"""CFL runner for doubleDiffusion (nomag path)."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_RADIAL_CHUNK"] = "0"
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
from magic_torch.step_time import setup_initial_state, initialize_dt, _radial_loop_dispatch

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

# Load checkpoint (DD starts from restart, not conduction state)
sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Call radial loop once — matches Fortran stage 1 of step 1
_radial_loop_dispatch()
# Get per-level CFL arrays for comparison with Fortran reference
from magic_torch.courant import courant_check
from magic_torch.step_time import _vrc, _vtc, _vpc
from magic_torch.time_scheme import tscheme
dtrkc, dthkc = courant_check(_vrc, _vtc, _vpc,
                              courfac=tscheme.courfac, alffac=tscheme.alffac)
np.save(os.path.join(out_dir, "dtrkc.npy"), dtrkc.cpu().numpy())
np.save(os.path.join(out_dir, "dthkc.npy"), dthkc.cpu().numpy())
print("nomag CFL runner completed")
'''

DD_DIR = Path(__file__).parent.parent.parent / "samples" / "doubleDiffusion"
DD_CKPT = DD_DIR / "checkpoint_end.start"
DD_REF = DD_DIR / "fortran_ref"

_nomag_results = {}
_nomag_ran = False


def _run_nomag():
    global _nomag_ran
    if _nomag_ran:
        return
    _nomag_ran = True

    runner_path = Path(__file__).parent / "_courant_nomag_runner.py"
    runner_path.write_text(_NOMAG_RUNNER_CODE)

    out_dir = tempfile.mkdtemp(prefix="courant_nomag_")

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
        [sys.executable, str(runner_path), str(DD_CKPT), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"nomag CFL runner failed: {result.returncode}")

    _nomag_results["dtrkc"] = np.load(os.path.join(out_dir, "dtrkc.npy"))
    _nomag_results["dthkc"] = np.load(os.path.join(out_dir, "dthkc.npy"))


@pytest.fixture(scope="module")
def nomag_cfl():
    _run_nomag()
    return _nomag_results


class TestCFLStep1NoMag:
    """dtrkc, dthkc at step 1 for doubleDiffusion (non-magnetic _courant_nomag path)."""

    def test_nomag_dtrkc(self, nomag_cfl):
        """dtrkc matches Fortran reference (nomag, BPR353 courfac=0.8)."""
        py = nomag_cfl["dtrkc"]
        ref = np.load(DD_REF / "dtrkc_step1.npy")
        assert py.shape == ref.shape
        # Compare interior levels only — boundaries differ because Python
        # now computes velocity at all levels (needed for CFL with IC rotation)
        np.testing.assert_allclose(py[1:-1], ref[1:-1], atol=1e-16, rtol=1e-13)

    def test_nomag_dthkc(self, nomag_cfl):
        """dthkc matches Fortran reference (nomag, BPR353 courfac=0.8)."""
        py = nomag_cfl["dthkc"]
        ref = np.load(DD_REF / "dthkc_step1.npy")
        assert py.shape == ref.shape
        # Compare interior levels only — boundaries differ because Python
        # now computes velocity at all levels (needed for CFL with IC rotation)
        np.testing.assert_allclose(py[1:-1], ref[1:-1], atol=1e-16, rtol=5e-13)

    def test_nomag_cfl_no_change(self, nomag_cfl):
        """With dtMax=3e-4, CFL should NOT trigger a dt change for DD."""
        dtrkc_min = float(nomag_cfl["dtrkc"].min())
        dthkc_min = float(nomag_cfl["dthkc"].min())

        l_new_dt, dt_new = dt_courant(3e-4, 3e-4, dtrkc_min, dthkc_min)
        assert not l_new_dt
        assert dt_new == 3e-4


class TestCFLZeroVelocity:
    """Edge case: zero velocity → dtrkc/dthkc should be the large sentinel (1e10)."""

    def test_zero_velocity_returns_large(self):
        """courant_check with zero velocity gives dtrkc=dthkc=1e10 everywhere."""
        import torch
        from magic_torch.precision import DTYPE, DEVICE
        from magic_torch.params import n_r_max, n_theta_max, n_phi_max
        from magic_torch.courant import courant_check

        zero = torch.zeros(n_r_max, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
        dtrkc, dthkc = courant_check(zero, zero, zero)

        assert (dtrkc == 1e10).all()
        assert (dthkc == 1e10).all()
