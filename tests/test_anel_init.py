"""Test anelastic field initialization against Fortran reference.

Tests ps_cond (conduction state) and initS (entropy + perturbation) for the
hydro_bench_anel_lowres configuration (strat=5.0, polind=2.0, g2=1, mode=1).

Runs in a subprocess to set anelastic env vars before module import.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Fortran reference data
ANEL_REF = Path(__file__).parent.parent.parent / "samples" / "hydro_bench_anel_lowres" / "fortran_ref"

if not ANEL_REF.exists():
    pytest.skip("Anelastic reference data not found", allow_module_level=True)


# Snake→standard LM reordering (l_max=16, minc=1)
def _compute_snake_to_standard_perm(l_max=16, minc=1):
    l_list = list(range(l_max, -1, -1))
    idx_0 = l_list.index(0)
    l_list[0], l_list[idx_0] = l_list[idx_0], l_list[0]
    snake_lm2l, snake_lm2m = [], []
    for l in l_list:
        for m in range(0, min(l_max, l) + 1, minc):
            snake_lm2l.append(l)
            snake_lm2m.append(m)
    lm_max = len(snake_lm2l)
    st_lm2 = {}
    st_idx = 0
    for m in range(0, l_max + 1, minc):
        for l in range(m, l_max + 1):
            st_lm2[(l, m)] = st_idx
            st_idx += 1
    perm = np.zeros(lm_max, dtype=np.int64)
    for snake_idx in range(lm_max):
        perm[snake_idx] = st_lm2[(snake_lm2l[snake_idx], snake_lm2m[snake_idx])]
    return perm


_SNAKE2ST = _compute_snake_to_standard_perm()

_LM_FIELD_NAMES = {
    's_init', 'p_init', 'w_init', 'z_init', 'dw_init', 'dz_init',
    'aj_init', 'b_init', 'db_init',
}


def load_ref(name):
    """Load Fortran reference, reordering snake→standard for LM fields."""
    arr = np.load(ANEL_REF / f"{name}.npy")
    if name in _LM_FIELD_NAMES and arr.ndim == 2 and arr.shape[0] == len(_SNAKE2ST):
        result = np.zeros_like(arr)
        result[_SNAKE2ST] = arr
        return result
    return arr


# Run subprocess once and cache
_results = {}
_ran = False


def _ensure_run():
    global _results, _ran
    if _ran:
        return
    _ran = True

    runner = Path(__file__).parent / "_anel_init_runner.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, str(runner), tmpdir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            pytest.fail(f"Anel init runner failed:\n{result.stderr}\n{result.stdout}")

        for f in Path(tmpdir).glob("*.npy"):
            _results[f.stem] = np.load(f)


class TestAnelConduction:
    """Test conduction state (lm=0 mode) from ps_cond."""

    def setup_method(self):
        _ensure_run()

    def test_s0_cond(self):
        """Entropy conduction state matches Fortran."""
        py = _results["s0_cond"]
        ref = load_ref("s_init")[0, :]  # lm=0 in standard ordering
        max_abs = np.max(np.abs(py - ref))
        max_ref = np.max(np.abs(ref))
        rel = max_abs / max_ref if max_ref > 0 else max_abs
        assert rel < 1e-12, f"s0_cond: abs={max_abs:.2e} rel={rel:.2e}"

    def test_p0_cond(self):
        """Pressure conduction state matches Fortran."""
        py = _results["p0_cond"]
        ref = load_ref("p_init")[0, :]  # lm=0 in standard ordering
        max_abs = np.max(np.abs(py - ref))
        max_ref = np.max(np.abs(ref))
        rel = max_abs / max_ref if max_ref > 0 else max_abs
        assert rel < 1e-12, f"p0_cond: abs={max_abs:.2e} rel={rel:.2e}"


class TestAnelInitFields:
    """Test full field initialization including perturbation."""

    def setup_method(self):
        _ensure_run()

    @pytest.mark.parametrize("name", [
        "s_init", "p_init", "w_init", "z_init",
    ])
    def test_field(self, name):
        """Field matches Fortran reference after initialization."""
        py = _results[name]
        ref = load_ref(name)
        max_abs = np.max(np.abs(py - ref))
        max_ref = np.max(np.abs(ref))
        rel = max_abs / max_ref if max_ref > 0 else max_abs
        # float64 on CPU → machine precision
        assert rel < 1e-12 or max_abs < 1e-15, \
            f"{name}: abs={max_abs:.2e} rel={rel:.2e}"
