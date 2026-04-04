"""FD + anelastic: smoke test verifying the code runs and produces reasonable output.

Runs 10 steps with FD radial scheme + anelastic (strat=5, mode=1).
Verifies:
1. No crashes (code runs end-to-end)
2. Entropy field has reasonable magnitude (conduction state + perturbation)
3. Velocity fields develop (nonzero after a few steps)
4. No NaN/Inf in any field

Note: Exact Fortran-matching reference data is not available because
Fortran's non-parallel p0Mat path has a bug for FD+anelastic.
When this is fixed upstream, this test should be upgraded to field-level
comparison against Fortran reference data.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_FD_ANEL_ENV = {
    "MAGIC_RADIAL_SCHEME": "FD",
    "MAGIC_LMAX": "16", "MAGIC_NR": "33", "MAGIC_MINC": "1",
    "MAGIC_RADRATIO": "0.35",
    "MAGIC_FD_ORDER": "2", "MAGIC_FD_ORDER_BOUND": "2",
    "MAGIC_FD_STRETCH": "0.3", "MAGIC_FD_RATIO": "0.1",
    "MAGIC_STRAT": "0.1", "MAGIC_POLIND": "2.0",
    "MAGIC_G0": "0", "MAGIC_G1": "0", "MAGIC_G2": "1",
    "MAGIC_MODE": "1", "MAGIC_KTOPV": "1", "MAGIC_KBOTV": "1",
    "MAGIC_RA": "1.48638035e5", "MAGIC_EK": "1e-3", "MAGIC_PR": "1.0",
    "MAGIC_DTMAX": "1e-4", "MAGIC_ALPHA": "0.6",
    "MAGIC_INIT_S1": "1010", "MAGIC_AMP_S1": "0.01", "MAGIC_INIT_V1": "0",
    "MAGIC_L_CORRECT_AMZ": "false", "MAGIC_L_CORRECT_AME": "false",
    "MAGIC_DEVICE": "cpu",
}

N_STEPS = 10

_results = {}
_ran = False


@pytest.fixture(scope="module")
def results():
    global _results, _ran
    if _ran:
        return _results
    _ran = True

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_fd_anel_runner.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {**os.environ, **_FD_ANEL_ENV}
        result = subprocess.run(
            [sys.executable, runner, tmpdir, str(N_STEPS)],
            env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            pytest.fail(f"FD anel runner failed:\n{result.stderr[-2000:]}")

        for step in range(1, N_STEPS + 1):
            for attr in ["s_LMloc", "ds_LMloc", "w_LMloc", "dw_LMloc", "ddw_LMloc",
                         "z_LMloc", "dz_LMloc", "p_LMloc", "dp_LMloc"]:
                path = os.path.join(tmpdir, f"{attr}_step{step}.npy")
                if os.path.exists(path):
                    _results[(attr, step)] = np.load(path)

    return _results


def test_no_nan(results):
    """No NaN or Inf in any field at any step."""
    for (attr, step), arr in results.items():
        assert np.all(np.isfinite(arr)), f"{attr} step{step} has NaN/Inf"


def test_entropy_magnitude(results):
    """Entropy should be O(1) — conduction state is ~sq4pi ≈ 3.54."""
    s = results[("s_LMloc", 1)]
    assert np.max(np.abs(s)) > 1.0, f"s too small: max={np.max(np.abs(s)):.3e}"
    assert np.max(np.abs(s)) < 100.0, f"s too large: max={np.max(np.abs(s)):.3e}"


def test_velocity_develops(results):
    """Velocity should develop from the entropy perturbation after several steps."""
    # At step 1, w/z might still be ~0 (convection onset takes time).
    # By step 10, there should be nonzero velocity.
    w10 = results[("w_LMloc", N_STEPS)]
    z10 = results[("z_LMloc", N_STEPS)]
    total_vel = np.max(np.abs(w10)) + np.max(np.abs(z10))
    assert total_vel > 1e-20, f"No velocity after {N_STEPS} steps"


def test_pressure_nonzero(results):
    """Pressure should be nonzero (hydrostatic balance with entropy)."""
    p = results[("p_LMloc", 1)]
    assert np.max(np.abs(p)) > 1.0, f"p too small: max={np.max(np.abs(p)):.3e}"


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
def test_entropy_bounded(results, step):
    """Entropy should not blow up over 10 steps."""
    s = results[("s_LMloc", step)]
    assert np.max(np.abs(s)) < 1000.0, (
        f"s blew up at step {step}: max={np.max(np.abs(s)):.3e}")


# --- Field-level comparison against Fortran FD reference ---

import torch

_REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "samples", "hydro_bench_anel_fd", "fortran_ref",
)

def _compute_snake_perm(l_max=16, minc=1):
    l_list = list(range(l_max, -1, -1))
    idx_0 = l_list.index(0)
    l_list[0], l_list[idx_0] = l_list[idx_0], l_list[0]
    snake_lm = []
    for l_val in l_list:
        for m_val in range(0, min(l_val, l_max) + 1, minc):
            snake_lm.append((l_val, m_val))
    std_lm = []
    for m_val in range(0, l_max + 1, minc):
        for l_val in range(m_val, l_max + 1):
            std_lm.append((l_val, m_val))
    std_map = {lm: i for i, lm in enumerate(std_lm)}
    return torch.tensor([std_map[lm] for lm in snake_lm], dtype=torch.long)

_PERM = _compute_snake_perm()
_LM_FIELDS = {"s", "ds", "w", "dw", "ddw", "z", "dz", "p", "dp"}
_REF_NAME_MAP = {
    "s_LMloc": "s", "ds_LMloc": "ds",
    "w_LMloc": "w", "dw_LMloc": "dw", "ddw_LMloc": "ddw",
    "z_LMloc": "z", "dz_LMloc": "dz",
    "p_LMloc": "p", "dp_LMloc": "dp",
}

def _load_ref(name):
    t = torch.from_numpy(np.load(os.path.join(_REF_DIR, f"{name}.npy")))
    if t.is_complex():
        t = t.to(torch.complex128)
    base = name.split("_step")[0]
    if base in _LM_FIELDS and t.dim() == 2 and t.shape[0] == len(_PERM):
        result = torch.empty_like(t)
        result[_PERM] = t
        t = result
    return t

_FIELD_TOLS = {
    "s_LMloc": 1e-11, "ds_LMloc": 1e-10,
    "w_LMloc": 1e-8, "dw_LMloc": 1e-8, "ddw_LMloc": 1e-8,
    "z_LMloc": 1e-12, "dz_LMloc": 1e-12,
    "p_LMloc": 1e-8, "dp_LMloc": 1e-12,
}

@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("attr,tol", list(_FIELD_TOLS.items()))
def test_field_vs_fortran(results, attr, tol, step):
    """FD anelastic field at each step must match Fortran."""
    ref_name = _REF_NAME_MAP[attr]
    ref_path = os.path.join(_REF_DIR, f"{ref_name}_step{step}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No reference for {ref_name}_step{step}")
    ref = _load_ref(f"{ref_name}_step{step}").numpy()
    py = results[(attr, step)]
    max_abs = np.abs(ref).max()
    if max_abs < 1e-15:
        abs_err = np.abs(py - ref).max()
        assert abs_err < 1e-10, (
            f"{attr} step{step}: abs err = {abs_err:.2e} (expected ~zero)")
        return
    rel_err = np.abs(py - ref).max() / max_abs
    assert rel_err < tol, (
        f"{attr} step{step}: max rel err = {rel_err:.2e} (tol={tol:.0e})")
