"""Anelastic + composition: field-level comparison against Fortran.

Tests the combination of anelastic (strat=5.0, l_anel=True) with composition
(raxi=1e5, l_chemical_conv=True). This exercises:
- xiMat with beta in diffusion operator
- finish_exp_comp with orho1 multiplication
- updateXi impl with beta+2*or1
- setup_initial_state xi impl with beta (bug fix)
- updateWP ChemFac*rgrav*xi buoyancy with anelastic rho0*rgrav
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
    "..", "samples", "anel_comp_lowres", "fortran_ref",
)

_N_STEPS = 3

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

_LM_FIELDS = {"s", "ds", "p", "dp", "w", "dw", "ddw", "z", "dz", "xi", "dxi"}


def _load_ref(name):
    """Load Fortran reference with snake→standard reorder."""
    t = torch.from_numpy(np.load(os.path.join(_REF_DIR, f"{name}.npy")))
    if t.is_complex():
        t = t.to(torch.complex128)
    base = name.replace("_init", "").replace("_step1", "").replace("_step2", "").replace("_step3", "")
    if base in _LM_FIELDS and t.dim() == 2 and t.shape[0] == len(_PERM):
        result = torch.empty_like(t)
        result[_PERM] = t
        t = result
    return t


_results = {}


@pytest.fixture(scope="module", autouse=True)
def run_anel_comp():
    global _results
    if _results:
        return
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_anel_comp_runner.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, runner, tmpdir, str(_N_STEPS)],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            pytest.fail(f"Anel+comp runner failed:\n{result.stderr[-2000:]}")
        for npy in os.listdir(tmpdir):
            if npy.endswith(".npy"):
                name = npy[:-4]
                _results[name] = torch.from_numpy(np.load(os.path.join(tmpdir, npy)))
                if _results[name].is_complex():
                    _results[name] = _results[name].to(torch.complex128)


# Tolerances: same as test_anel_step.py (post-LU-fix) + xi/dxi
_RTOL = {
    "s": 1e-11, "ds": 1e-10, "p": 1e-9, "dp": 1e-9,
    "w": 1e-9, "dw": 1e-8, "ddw": 1e-7,
    "z": 1e-10, "dz": 1e-8,
    "xi": 1e-10, "dxi": 1e-9,
}

_FIELDS = list(_RTOL.keys())


@pytest.mark.parametrize("step", range(1, _N_STEPS + 1))
@pytest.mark.parametrize("field", _FIELDS)
def test_anel_comp_step(field, step):
    """Compare anelastic+composition field against Fortran reference."""
    key = f"{field}_step{step}"
    ref_path = os.path.join(_REF_DIR, f"{key}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No reference: {key}")
    test = _results[key]
    ref = _load_ref(key)
    ref_max = ref.abs().max().item()
    if ref_max == 0.0 and test.abs().max().item() == 0.0:
        return
    abs_err = (test - ref).abs().max().item()
    rel_err = abs_err / ref_max if ref_max > 0 else abs_err
    rtol = _RTOL[field]
    assert rel_err < rtol, (
        f"{key}: rel={rel_err:.3e} (tol {rtol:.0e}), "
        f"abs={abs_err:.3e}, ref_max={ref_max:.3e}"
    )


@pytest.mark.parametrize("field", _FIELDS)
def test_anel_comp_init(field):
    """Compare init field against Fortran reference."""
    key = f"{field}_init"
    ref_path = os.path.join(_REF_DIR, f"{key}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No reference: {key}")
    test = _results[key]
    ref = _load_ref(key)
    # p/dp have ~1e-6 absolute from ps_cond_anel conditioning;
    # ds has ~3e-5 absolute from Chebyshev derivative FP ordering
    torch.testing.assert_close(test, ref, atol=1e-5, rtol=1e-11)
