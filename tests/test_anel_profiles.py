"""Tests for anelastic reference state profiles against Fortran dumps.

Runs in a subprocess to set anelastic env vars before module import.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Fortran reference data
ANEL_REF = Path(__file__).parent.parent.parent / "samples" / "hydro_bench_anel_lowres" / "fortran_ref"

_RUNNER = Path(__file__).parent / "_anel_profiles_runner.py"

# Run subprocess once and cache
_results = {}
_ran = False


def _ensure_run():
    global _results, _ran
    if _ran:
        return
    _ran = True

    assert ANEL_REF.exists(), f"Anelastic reference data not found at {ANEL_REF}"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, str(_RUNNER), tmpdir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.fail(f"Anel profiles runner failed:\n{result.stderr}\n{result.stdout}")

        for f in Path(tmpdir).glob("*.npy"):
            _results[f.stem] = np.load(f)


def load_ref(name):
    return np.load(ANEL_REF / f"anel_{name}.npy")


class TestAnelScalars:

    def setup_method(self):
        _ensure_run()

    @pytest.mark.parametrize("name", [
        "DissNb", "GrunNb", "ThExpNb", "ViscHeatFac", "OhmLossFac",
    ])
    def test_scalar(self, name):
        ref = load_ref(name).item()
        py = _results[name].item()
        assert py == pytest.approx(ref, abs=1e-16, rel=1e-15), \
            f"{name}: py={py} ref={ref}"


class TestAnelProfiles:

    def setup_method(self):
        _ensure_run()

    @pytest.mark.parametrize("name", [
        "temp0", "rho0", "beta", "dbeta", "ddbeta", "rgrav",
        "orho1", "alpha0", "ogrun", "otemp1", "dLtemp0", "ddLtemp0",
    ])
    def test_profile(self, name):
        ref = load_ref(name)
        py = _results[name]
        max_abs = np.max(np.abs(py - ref))
        max_ref = np.max(np.abs(ref))
        rel = max_abs / max_ref if max_ref > 0 else max_abs
        assert rel < 1e-13 or max_abs < 1e-15, \
            f"{name}: max_abs_err={max_abs:.2e} rel_err={rel:.2e}"

    @pytest.mark.parametrize("name", [
        "visc", "dLvisc", "ddLvisc", "kappa", "dLkappa",
    ])
    def test_transport(self, name):
        ref = load_ref(name)
        py = _results[name]
        max_abs = np.max(np.abs(py - ref))
        assert max_abs < 1e-15, \
            f"{name}: max_abs_err={max_abs:.2e}"
