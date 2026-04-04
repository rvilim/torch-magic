"""FD + anelastic + MHD: multi-step comparison against Fortran.

Tests the combination of FD radial scheme + anelastic (strat=0.1) + magnetic
fields (mode=0). This exercises the Ohmic heating code path in get_nl_anel.py
which only fires when both l_anel and l_mag are true.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

N_STEPS = 10

_REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "samples", "dynamo_benchmark_fd_anel_mhd", "fortran_ref",
)

_ENV = {
    "MAGIC_RADIAL_SCHEME": "FD",
    "MAGIC_LMAX": "16", "MAGIC_NR": "33", "MAGIC_MINC": "1",
    "MAGIC_RADRATIO": "0.35",
    "MAGIC_FD_ORDER": "2", "MAGIC_FD_ORDER_BOUND": "2",
    "MAGIC_FD_STRETCH": "0.3", "MAGIC_FD_RATIO": "0.1",
    "MAGIC_STRAT": "0.1", "MAGIC_POLIND": "2.0",
    "MAGIC_G0": "0", "MAGIC_G1": "0", "MAGIC_G2": "1",
    "MAGIC_MODE": "0", "MAGIC_KTOPV": "2", "MAGIC_KBOTV": "2",
    "MAGIC_RA": "1e5", "MAGIC_EK": "1e-3", "MAGIC_PR": "1.0", "MAGIC_PRMAG": "5.0",
    "MAGIC_DTMAX": "1e-4", "MAGIC_ALPHA": "0.6",
    "MAGIC_INIT_S1": "404", "MAGIC_AMP_S1": "0.1", "MAGIC_INIT_V1": "0",
    "MAGIC_DEVICE": "cpu",
}


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
_LM_FIELDS = {"w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp", "b", "db", "ddb", "aj", "dj"}
_REF_NAME_MAP = {
    "w_LMloc": "w", "dw_LMloc": "dw", "ddw_LMloc": "ddw",
    "z_LMloc": "z", "dz_LMloc": "dz",
    "s_LMloc": "s", "ds_LMloc": "ds",
    "p_LMloc": "p", "dp_LMloc": "dp",
    "b_LMloc": "b", "db_LMloc": "db", "ddb_LMloc": "ddb",
    "aj_LMloc": "aj", "dj_LMloc": "dj",
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


_results = {}
_ran = False


@pytest.fixture(scope="module")
def results():
    global _results, _ran
    if _ran:
        return _results
    _ran = True

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_fd_multistep_runner.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {**os.environ, **_ENV}
        result = subprocess.run(
            [sys.executable, runner, tmpdir, str(N_STEPS)],
            env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            pytest.fail(f"FD anel MHD runner failed:\n{result.stderr[-2000:]}")

        OC = ["w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
              "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
              "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc"]
        for step in range(1, N_STEPS + 1):
            for name in OC:
                path = os.path.join(tmpdir, f"{name}_step{step}.npy")
                if os.path.exists(path):
                    _results[(name, step)] = np.load(path)

    return _results


_FIELD_TOLS = {
    "w_LMloc": 1e-12, "dw_LMloc": 1e-11, "ddw_LMloc": 1e-10,
    "z_LMloc": 1e-12, "dz_LMloc": 1e-12,
    "s_LMloc": 1e-12, "ds_LMloc": 1e-10,
    "p_LMloc": 1e-12, "dp_LMloc": 1e-12,
    "b_LMloc": 1e-13, "db_LMloc": 1e-12, "ddb_LMloc": 1e-10,
    "aj_LMloc": 1e-13, "dj_LMloc": 1e-13,
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("attr,tol", list(_FIELD_TOLS.items()))
def test_field(results, attr, tol, step):
    """FD+anel+MHD field at each step must match Fortran."""
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
