"""Tests for finite difference grid and derivative matrices.

Compares against Fortran reference dumps from samples/dynamo_benchmark_fd/.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

# These tests run in a subprocess with MAGIC_RADIAL_SCHEME=FD to avoid
# polluting the module-level state of the main test process.

REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "samples", "dynamo_benchmark_fd", "fortran_ref",
)


def _run_fd_check(code: str, timeout: int = 30) -> str:
    """Run Python code in a subprocess with FD environment."""
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


class TestFDGrid:
    """Test FD grid matches Fortran."""

    def test_grid_matches_fortran(self):
        ref = np.load(os.path.join(REF_DIR, "r.npy"))
        out = _run_fd_check(f"""
import numpy as np, torch
from magic_torch.finite_differences import r
r_np = r.cpu().numpy()
ref = np.load('{os.path.join(REF_DIR, "r.npy")}')
diff = np.abs(r_np - ref).max()
print(f'grid diff: {{diff:.3e}}')
assert diff < 1e-14, f'Grid mismatch: {{diff}}'
print('PASS')
""")
        assert "PASS" in out

    def test_grid_endpoints(self):
        out = _run_fd_check("""
from magic_torch.finite_differences import r, r_cmb, r_icb
assert abs(r[0].item() - r_cmb) < 1e-14, f'r[0]={r[0]} != r_cmb={r_cmb}'
assert abs(r[-1].item() - r_icb) < 1e-14, f'r[-1]={r[-1]} != r_icb={r_icb}'
print('PASS')
""")
        assert "PASS" in out

    def test_grid_decreasing(self):
        out = _run_fd_check("""
from magic_torch.finite_differences import r
assert (r[:-1] > r[1:]).all(), 'Grid should be strictly decreasing'
print('PASS')
""")
        assert "PASS" in out


class TestFDDerivativeMatrices:
    """Test FD derivative matrices match Fortran dumps."""

    def test_rMat_is_identity(self):
        out = _run_fd_check(f"""
import numpy as np, torch
from magic_torch.finite_differences import rMat
ref = np.load('{os.path.join(REF_DIR, "rMat.npy")}')
diff = (rMat.cpu().numpy() - ref).max()
print(f'rMat diff: {{diff:.3e}}')
assert abs(diff) < 1e-15, f'rMat mismatch: {{diff}}'
I = torch.eye(rMat.shape[0], dtype=torch.float64)
assert (rMat.cpu() - I).abs().max() < 1e-15, 'rMat should be identity'
print('PASS')
""")
        assert "PASS" in out

    def test_drMat_matches_fortran(self):
        out = _run_fd_check(f"""
import numpy as np
from magic_torch.finite_differences import drMat
ref = np.load('{os.path.join(REF_DIR, "drMat.npy")}')
diff = np.abs(drMat.cpu().numpy() - ref).max()
rel = diff / np.abs(ref).max()
print(f'drMat: abs={{diff:.3e}} rel={{rel:.3e}}')
assert rel < 1e-13, f'drMat mismatch: rel={{rel}}'
print('PASS')
""")
        assert "PASS" in out

    def test_d2rMat_matches_fortran(self):
        out = _run_fd_check(f"""
import numpy as np
from magic_torch.finite_differences import d2rMat
ref = np.load('{os.path.join(REF_DIR, "d2rMat.npy")}')
diff = np.abs(d2rMat.cpu().numpy() - ref).max()
rel = diff / np.abs(ref).max()
print(f'd2rMat: abs={{diff:.3e}} rel={{rel:.3e}}')
assert rel < 1e-13, f'd2rMat mismatch: rel={{rel}}'
print('PASS')
""")
        assert "PASS" in out

    def test_d3rMat_matches_fortran(self):
        out = _run_fd_check(f"""
import numpy as np
from magic_torch.finite_differences import d3rMat
ref = np.load('{os.path.join(REF_DIR, "d3rMat.npy")}')
diff = np.abs(d3rMat.cpu().numpy() - ref).max()
rel = diff / np.abs(ref).max()
print(f'd3rMat: abs={{diff:.3e}} rel={{rel:.3e}}')
assert rel < 1e-13, f'd3rMat mismatch: rel={{rel}}'
print('PASS')
""")
        assert "PASS" in out


class TestFDConstants:
    """Test FD-specific constants."""

    def test_rnorm_is_one(self):
        out = _run_fd_check("""
from magic_torch.finite_differences import rnorm
assert rnorm == 1.0, f'rnorm should be 1.0, got {rnorm}'
print('PASS')
""")
        assert "PASS" in out

    def test_boundary_fac_is_one(self):
        out = _run_fd_check("""
from magic_torch.finite_differences import boundary_fac
assert boundary_fac == 1.0, f'boundary_fac should be 1.0, got {boundary_fac}'
print('PASS')
""")
        assert "PASS" in out

    def test_d4rMat_exists(self):
        out = _run_fd_check("""
from magic_torch.finite_differences import d4rMat
assert d4rMat is not None, 'd4rMat should exist for FD'
assert d4rMat.shape == (33, 33), f'd4rMat shape={d4rMat.shape}'
print('PASS')
""")
        assert "PASS" in out


class TestFornbergWeights:
    """Test the Fornberg FD weight algorithm independently."""

    def test_2nd_order_uniform_first_deriv(self):
        """Standard 3-point centered difference on uniform grid."""
        out = _run_fd_check("""
import torch
from magic_torch.finite_differences import populate_fd_weights
h = 0.1
x = torch.tensor([-h, 0.0, h], dtype=torch.float64)
c = populate_fd_weights(0.0, x, 2, 1)
# d/dx weights: [-1/(2h), 0, 1/(2h)]
assert abs(c[0, 1].item() - (-1.0/(2*h))) < 1e-14
assert abs(c[1, 1].item()) < 1e-14
assert abs(c[2, 1].item() - (1.0/(2*h))) < 1e-14
print('PASS')
""")
        assert "PASS" in out

    def test_2nd_order_uniform_second_deriv(self):
        """Standard 3-point centered difference for d²/dx²."""
        out = _run_fd_check("""
import torch
from magic_torch.finite_differences import populate_fd_weights
h = 0.1
x = torch.tensor([-h, 0.0, h], dtype=torch.float64)
c = populate_fd_weights(0.0, x, 2, 2)
# d²/dx² weights: [1/h², -2/h², 1/h²]
assert abs(c[0, 2].item() - 1.0/h**2) < 1e-12
assert abs(c[1, 2].item() - (-2.0/h**2)) < 1e-12
assert abs(c[2, 2].item() - 1.0/h**2) < 1e-12
print('PASS')
""")
        assert "PASS" in out

    def test_interpolation_weights_sum_to_one(self):
        """Zeroth derivative weights should sum to 1 (interpolation)."""
        out = _run_fd_check("""
import torch
from magic_torch.finite_differences import populate_fd_weights
x = torch.tensor([-0.3, 0.0, 0.2, 0.7], dtype=torch.float64)
c = populate_fd_weights(0.0, x, 3, 2)
assert abs(c[:, 0].sum().item() - 1.0) < 1e-14
print('PASS')
""")
        assert "PASS" in out
