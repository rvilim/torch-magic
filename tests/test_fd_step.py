"""FD radial scheme: field-level comparison against Fortran after 1 step.

Runs in a subprocess with MAGIC_RADIAL_SCHEME=FD. Compares all 14 fields
against Fortran FD reference data in samples/dynamo_benchmark_fd/fortran_ref/.

The Fortran reference was generated with l_parallel_solve=.false. (forced via
l_single_matrix=.true. in input.nml) to match the non-parallel per-l solve
path used by the Python port.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

# Fortran reference directory
_REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "samples", "dynamo_benchmark_fd", "fortran_ref",
)

# FD environment variables
_FD_ENV = {
    "MAGIC_RADIAL_SCHEME": "FD",
    "MAGIC_LMAX": "16",
    "MAGIC_NR": "33",
    "MAGIC_MINC": "1",
    "MAGIC_RADRATIO": "0.35",
    "MAGIC_FD_ORDER": "2",
    "MAGIC_FD_ORDER_BOUND": "2",
    "MAGIC_FD_STRETCH": "0.3",
    "MAGIC_FD_RATIO": "0.1",
    "MAGIC_RA": "1e5",
    "MAGIC_EK": "1e-3",
    "MAGIC_PR": "1.0",
    "MAGIC_PRMAG": "5.0",
    "MAGIC_DTMAX": "1e-4",
    "MAGIC_ALPHA": "0.6",
    "MAGIC_INIT_S1": "404",
    "MAGIC_AMP_S1": "0.1",
    "MAGIC_INIT_V1": "0",
    "MAGIC_KTOPV": "2",
    "MAGIC_KBOTV": "2",
    "MAGIC_DEVICE": "cpu",
}

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
    """Load Fortran reference array and reorder from snake to standard LM."""
    t = torch.from_numpy(np.load(os.path.join(_REF_DIR, f"{name}.npy")))
    if t.is_complex():
        t = t.to(torch.complex128)
    base = name.replace("_init", "").replace("_step1", "")
    if base in _LM_FIELDS and t.dim() == 2 and t.shape[0] == len(_PERM):
        result = torch.empty_like(t)
        result[_PERM] = t
        t = result
    return t


# Module-scoped fixture: run ONE subprocess, save all fields
_results = {}


@pytest.fixture(scope="module", autouse=True)
def run_fd_step():
    """Run the FD step runner once and cache all field arrays."""
    global _results
    if _results:
        return

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fd_step_runner.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {**os.environ, **_FD_ENV}
        result = subprocess.run(
            [sys.executable, runner, tmpdir],
            env=env, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            pytest.fail(f"FD step runner failed:\n{result.stderr[-2000:]}")

        for npy in os.listdir(tmpdir):
            if npy.endswith(".npy"):
                name = npy[:-4]
                _results[name] = torch.from_numpy(np.load(os.path.join(tmpdir, npy)))
                if _results[name].is_complex():
                    _results[name] = _results[name].to(torch.complex128)


# Field specs: (name, atol, rtol)
_INIT_FIELDS = [
    ("w_init", 1e-14, 0),
    ("dw_init", 1e-14, 0),
    ("z_init", 1e-14, 0),
    ("dz_init", 1e-14, 0),
    ("s_init", 1e-13, 1e-13),
    ("p_init", 1e-8, 1e-13),
    ("b_init", 1e-14, 1e-14),
    ("aj_init", 1e-14, 1e-14),
]

_STEP1_FIELDS = [
    ("w_step1", 1e-11, 1e-12),
    ("dw_step1", 1e-11, 1e-12),
    ("ddw_step1", 1e-9, 1e-12),
    ("z_step1", 1e-14, 1e-14),
    ("dz_step1", 1e-12, 1e-14),
    ("s_step1", 1e-12, 1e-13),
    ("ds_step1", 1e-11, 1e-12),
    ("b_step1", 1e-14, 1e-14),
    ("db_step1", 1e-12, 1e-13),
    ("ddb_step1", 1e-10, 1e-11),
    ("aj_step1", 1e-14, 1e-14),
    ("dj_step1", 1e-13, 1e-14),
    ("p_step1", 1e-7, 1e-12),
    ("dp_step1", 1e-14, 1e-14),
]

@pytest.mark.parametrize("name,atol,rtol", _INIT_FIELDS,
                         ids=[f[0] for f in _INIT_FIELDS])
def test_init_field(name, atol, rtol, run_fd_step):
    """Init fields should match Fortran exactly."""
    ref_path = os.path.join(_REF_DIR, f"{name}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No Fortran reference for {name}")
    ref = _load_ref(name)
    py = _results[name]
    torch.testing.assert_close(py, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("name,atol,rtol", _STEP1_FIELDS,
                         ids=[f[0] for f in _STEP1_FIELDS])
def test_step1_field(name, atol, rtol, run_fd_step):
    """Step 1 fields should match Fortran FD to machine precision."""
    ref_path = os.path.join(_REF_DIR, f"{name}.npy")
    if not os.path.exists(ref_path):
        pytest.skip(f"No Fortran reference for {name}")
    ref = _load_ref(name)
    py = _results[name]
    torch.testing.assert_close(py, ref, atol=atol, rtol=rtol)
