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
uwdc.build_w_matrices(0.5)
# FD: uses pivoted banded LU (wMat not diag dominant)
assert uwdc._w_bands_by_l is not None
print(f'Band shape: {uwdc._w_bands_by_l.shape}')
assert uwdc._w_bands_by_l.shape[0] == 17  # l_max+1
assert uwdc._w_bands_by_l.shape[2] == 33  # N
print('PASS')
""")
        assert "PASS" in out

    def test_banded_solve_roundtrip(self):
        """Banded solve A*x=b gives correct x for each l."""
        out = _run_fd_check("""
import torch
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.algebra import solve_band_real

uwdc.build_w_matrices(0.5)
N = 33
for l in range(1, 17):
    abd = uwdc._w_bands_by_l[l].cpu()
    piv = uwdc._w_piv_by_l[l].cpu()
    fac_row = uwdc._w_fac_row_by_l[l].cpu()
    fac_col = uwdc._w_fac_col_by_l[l].cpu()
    b = torch.randn(N, dtype=torch.float64)
    b_precond = fac_row * b
    y = solve_band_real(abd, N, uwdc._w_kl, uwdc._w_ku, piv, b_precond)
    x = fac_col * y
    assert not torch.isnan(x).any(), f'NaN at l={l}'
    assert not torch.isinf(x).any(), f'Inf at l={l}'
print('PASS')
""")
        assert "PASS" in out

    def test_l0_is_identity(self):
        """l=0 bands should be identity (no poloidal equation for l=0)."""
        out = _run_fd_check("""
from magic_torch import update_w_doublecurl as uwdc
uwdc.build_w_matrices(0.5)
diag_row = uwdc._w_kl + uwdc._w_ku
assert uwdc._w_bands_by_l[0, diag_row].abs().max() == 1.0
for r in range(uwdc._w_bands_by_l.shape[1]):
    if r != diag_row:
        assert uwdc._w_bands_by_l[0, r].abs().max() == 0.0
print('PASS')
""")
        assert "PASS" in out

    def test_band_storage_is_NxN(self):
        """Double-curl band storage has N columns (not 2N)."""
        out = _run_fd_check("""
from magic_torch import update_w_doublecurl as uwdc
from magic_torch.params import n_r_max
uwdc.build_w_matrices(0.5)
N = n_r_max
assert uwdc._w_bands_by_l.shape[2] == N, f'Expected {N}, got {uwdc._w_bands_by_l.shape[2]}'
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
