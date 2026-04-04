"""Anelastic + rotating IC: field comparison against Fortran after 1 step.

Tests z10Mat with anelastic profiles (beta in ICB row), domega_ic_dt.impl
with (2*or1+beta) coefficient, and IC rotation constants with rho0(ICB).
Config: Chebyshev, strat=5.0, mode=0 (MHD), nRotIC=1, kbotv=2 (no-slip),
sigma_ratio=0 (insulating IC, viscous-drag-only IC rotation).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

_REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "samples", "dynamo_benchmark_anel_rotIC", "fortran_ref",
)

# Snake-to-standard permutation for l_max=16, minc=1
def _compute_snake_perm():
    l_max, minc = 16, 1
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
_LM_FIELDS = {"w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp", "b", "db", "ddb", "aj", "dj"}


def _load_ref(name):
    t = torch.from_numpy(np.load(os.path.join(_REF_DIR, f"{name}.npy")))
    if t.is_complex():
        t = t.to(torch.complex128)
    base = name.replace("_init", "").replace("_step1", "")
    if base in _LM_FIELDS and t.dim() == 2 and t.shape[0] == len(_PERM):
        result = torch.empty_like(t)
        result[_PERM] = t
        t = result
    return t


_results = {}


@pytest.fixture(scope="module", autouse=True)
def run_anel_rotic():
    global _results
    if _results:
        return
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_anel_rotic_runner.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {**os.environ, "MAGIC_DEVICE": "cpu"}
        result = subprocess.run(
            [sys.executable, runner, tmpdir],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            pytest.fail(f"Anel+rotIC runner failed:\n{result.stderr[-2000:]}")
        for npy in os.listdir(tmpdir):
            if npy.endswith(".npy"):
                name = npy[:-4]
                _results[name] = torch.from_numpy(np.load(os.path.join(tmpdir, npy)))
                if _results[name].is_complex():
                    _results[name] = _results[name].to(torch.complex128)


# Tolerances with documented precision limits.
#
# Pressure (p, dp): p0Mat condition number ~1e5 for anelastic strat=5.
#   Expected: κ(p0Mat) × eps ≈ 1e5 × 2.2e-16 ≈ 2e-11 relative.
#   |p| ~ 1e8, so abs errors look large but relative is ~7e-11.
#
# Derivatives at ICB (ddb, dj): Chebyshev spectral differentiation amplifies
#   errors by ~N^(2k) at boundary points (k=derivative order).
#   For N=33, d2: ~N^4 × eps ≈ 1.2e6 × 2.2e-16 ≈ 3e-10.
#
# Init p: ps0Mat (2N×2N coupled entropy-pressure) has κ ~1e8.
#   Expected: κ × eps × ||p|| ≈ 1e8 × 2.2e-16 × 1e8 ≈ 2e-6 absolute.
_STEP1_FIELDS = [
    ("w_step1", 1e-9, 1e-12),
    ("dw_step1", 1e-8, 1e-12),
    ("ddw_step1", 1e-7, 1e-12),
    ("z_step1", 1e-10, 1e-14),
    ("dz_step1", 1e-8, 1e-14),
    ("s_step1", 1e-11, 1e-13),
    ("ds_step1", 1e-9, 1e-11),
    ("p_step1", 1e-2, 1e-9),   # rel=6.7e-11; atol irrelevant (rtol × |p|~1e8 dominates)
    ("dp_step1", 1e-1, 1e-9),  # rel=3.0e-11; atol irrelevant (rtol × |dp|~1e9 dominates)
    ("b_step1", 1e-13, 1e-13),
    ("db_step1", 1e-11, 1e-12),
    ("ddb_step1", 1e-9, 1e-10),  # Chebyshev d2 amplification at ICB
    ("aj_step1", 1e-14, 1e-14),
    ("dj_step1", 1e-11, 1e-12),
]

_INIT_FIELDS = [
    ("w_init", 1e-14, 0),
    ("dw_init", 1e-14, 0),
    ("z_init", 1e-14, 0),
    ("dz_init", 1e-14, 0),
    ("s_init", 1e-5, 1e-11),
    ("p_init", 1e-5, 1e-11),   # ps0Mat κ~1e8 → abs~1e-6 at |p|~1e8
    ("b_init", 1e-14, 1e-14),
    ("db_init", 1e-12, 1e-12),
    ("ddb_init", 1e-8, 1e-9),   # Chebyshev d2 amplification: N^4 × eps ≈ 3e-10
    ("aj_init", 1e-14, 1e-14),
    ("dj_init", 1e-11, 1e-12),  # Chebyshev d1 at ICB
]


@pytest.mark.parametrize("name,atol,rtol", _INIT_FIELDS, ids=[f[0] for f in _INIT_FIELDS])
def test_init_field(name, atol, rtol):
    ref_path = os.path.join(_REF_DIR, f"{name}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No reference: {name}")
    torch.testing.assert_close(_results[name], _load_ref(name), atol=atol, rtol=rtol)


@pytest.mark.parametrize("name,atol,rtol", _STEP1_FIELDS, ids=[f[0] for f in _STEP1_FIELDS])
def test_step1_field(name, atol, rtol):
    ref_path = os.path.join(_REF_DIR, f"{name}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No reference: {name}")
    torch.testing.assert_close(_results[name], _load_ref(name), atol=atol, rtol=rtol)


def test_omega_ic_init():
    py = _results["omega_ic_init"].item()
    assert abs(py) < 1e-14, f"omega_ic_init should be 0, got {py}"


def test_omega_ic_step1():
    ref_path = os.path.join(_REF_DIR, "omega_ic_step1.npy")
    if not os.path.exists(ref_path):
        pytest.skip("No omega_ic_step1 reference")
    py = _results["omega_ic_step1"].item()
    ref = np.load(ref_path).item()
    rel_err = abs(py - ref) / abs(ref) if abs(ref) > 1e-30 else abs(py - ref)
    assert rel_err < 1e-10, f"omega_ic_step1: py={py:.10e}, ref={ref:.10e}, rel={rel_err:.2e}"
