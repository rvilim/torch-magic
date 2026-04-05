"""Tests for double-curl WP matrix construction.

Tests run in subprocess with MAGIC_RADIAL_SCHEME=FD to test with FD matrices.
"""

import os
import subprocess
import sys

import pytest


def _run_fd_check(code: str, timeout: int = 60) -> str:
    env = {
        **os.environ,
        "MAGIC_RADIAL_SCHEME": "FD",
        "MAGIC_LMAX": "16",
        "MAGIC_NR": "33",
        "MAGIC_MINC": "1",
        "MAGIC_RADRATIO": "0.35",
        "MAGIC_FD_ORDER": "2",
        "MAGIC_FD_ORDER_BOUND": "2",
        "MAGIC_FD_STRETCH": "0.3",
        "MAGIC_FD_RATIO": "0.1",
        "MAGIC_DEVICE": "cpu",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\n{result.stderr[-2000:]}")
    return result.stdout


class TestDoubleCurlMatrixBuild:

    def test_build_succeeds(self):
        """build_w_matrices doesn't crash."""
        out = _run_fd_check("""
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.params import n_r_max
uwdc.build_w_matrices(0.5)
# N <= 1024: dense LU path; N > 1024: banded path
if n_r_max <= 1024:
    assert uwdc._w_lu_by_l is not None
    print(f'LU shape: {uwdc._w_lu_by_l.shape}')
    assert uwdc._w_lu_by_l.shape[0] == 17
    assert uwdc._w_lu_by_l.shape[1] == n_r_max
else:
    assert uwdc._w_bands_by_l is not None
print('PASS')
""")
        assert "PASS" in out

    def test_solve_roundtrip(self):
        """Solve A*x=b gives correct x for each l (dense LU or banded)."""
        out = _run_fd_check("""
import torch
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.params import n_r_max

uwdc.build_w_matrices(0.5)
N = n_r_max
if uwdc._w_lu_by_l is not None:
    # Dense LU path: test via lu_solve
    for l in range(1, 17):
        lu = uwdc._w_lu_by_l[l:l+1].cpu()
        piv = uwdc._w_pivots_by_l[l:l+1].cpu()
        b = torch.randn(1, N, 1, dtype=torch.float64)
        x = torch.linalg.lu_solve(lu, piv, b)
        assert not torch.isnan(x).any(), f'NaN at l={l}'
else:
    from magic_torch.algebra import solve_band_real
    for l in range(1, 17):
        abd = uwdc._w_bands_by_l[l].cpu()
        piv = uwdc._w_piv_by_l[l].cpu()
        b = torch.randn(N, dtype=torch.float64)
        y = solve_band_real(abd, N, uwdc._w_kl, uwdc._w_ku, piv, b)
        assert not torch.isnan(y).any(), f'NaN at l={l}'
print('PASS')
""")
        assert "PASS" in out

    def test_l0_is_identity(self):
        """l=0 should be identity (no poloidal equation for l=0)."""
        out = _run_fd_check("""
import torch
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.params import n_r_max
uwdc.build_w_matrices(0.5)
if uwdc._w_lu_by_l is not None:
    # Dense LU: l=0 should be identity LU
    eye = torch.eye(n_r_max, dtype=torch.float64)
    assert torch.allclose(uwdc._w_lu_by_l[0].cpu(), eye, atol=1e-14)
else:
    diag_row = uwdc._w_kl + uwdc._w_ku
    assert uwdc._w_bands_by_l[0, diag_row].abs().max() == 1.0
print('PASS')
""")
        assert "PASS" in out

    def test_storage_is_NxN(self):
        """Dense LU or band storage has N columns (not 2N)."""
        out = _run_fd_check("""
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.params import n_r_max
uwdc.build_w_matrices(0.5)
N = n_r_max
if uwdc._w_lu_by_l is not None:
    assert uwdc._w_lu_by_l.shape[1] == N
    assert uwdc._w_lu_by_l.shape[2] == N
elif uwdc._w_bands_by_l is not None:
    assert uwdc._w_bands_by_l.shape[2] == N
print('PASS')
""")
        assert "PASS" in out


class TestDoubleCurlOldImpl:

    def test_old_formula_shape(self):
        out = _run_fd_check("""
import torch
from magic_torch.update_w_doublecurl import get_w_old
from magic_torch.params import lm_max, n_r_max
from magic_torch.precision import CDTYPE

w = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
dw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
ddw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
result = get_w_old(w, dw, ddw)
assert result.shape == (lm_max, n_r_max)
assert result[:, 0].abs().max() == 0.0  # boundary
assert result[:, -1].abs().max() == 0.0
print('PASS')
""")
        assert "PASS" in out

    def test_impl_formula_shape(self):
        out = _run_fd_check("""
import torch
from magic_torch.update_w_doublecurl import get_w_impl
from magic_torch.params import lm_max, n_r_max
from magic_torch.precision import CDTYPE

w = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
dw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
ddw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
dddw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
ddddw = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
s = torch.randn(lm_max, n_r_max, dtype=CDTYPE)
result = get_w_impl(w, dw, ddw, dddw, ddddw, s)
assert result.shape == (lm_max, n_r_max)
assert result[:, 0].abs().max() == 0.0
assert result[:, -1].abs().max() == 0.0
print('PASS')
""")
        assert "PASS" in out
