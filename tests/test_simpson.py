"""Tests for Simpson integration (FD radial scheme).

Verifies:
1. Exact integration of polynomials up to degree 3
2. Consistency with analytical integrals on the FD grid
3. Batched operation
4. Both odd and even N
5. Chebyshev rInt_R is unchanged (regression)
"""

import os
import subprocess
import sys
import math

import torch
import pytest

from magic_torch.integration import simps, rInt_R

DTYPE = torch.float64


class TestSimpsStandalone:
    """Test simps() directly (no subprocess needed — it's grid-independent)."""

    def test_constant_uniform(self):
        """Integral of f=1 over [0,1] = 1."""
        N = 33
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)  # decreasing
        f = torch.ones(N, dtype=DTYPE)
        # simps negates because r decreases, so integral of 1 over [0,1] = 1
        result = simps(f, r)
        torch.testing.assert_close(result, torch.tensor(1.0, dtype=DTYPE), atol=1e-14, rtol=0)

    def test_linear_uniform(self):
        """Integral of f=r over [0,1] = 0.5."""
        N = 33
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)
        f = r.clone()
        result = simps(f, r)
        torch.testing.assert_close(result, torch.tensor(0.5, dtype=DTYPE), atol=1e-14, rtol=0)

    def test_quadratic_uniform(self):
        """Integral of f=r^2 over [0,1] = 1/3."""
        N = 33
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)
        f = r ** 2
        result = simps(f, r)
        torch.testing.assert_close(result, torch.tensor(1.0 / 3.0, dtype=DTYPE), atol=1e-13, rtol=0)

    def test_cubic_uniform(self):
        """Integral of f=r^3 over [0,1] = 0.25. Simpson exact for degree 3."""
        N = 33
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)
        f = r ** 3
        result = simps(f, r)
        torch.testing.assert_close(result, torch.tensor(0.25, dtype=DTYPE), atol=1e-13, rtol=0)

    def test_even_N(self):
        """Even number of points uses double-Simpson + trapz."""
        N = 32
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)
        f = r ** 2
        result = simps(f, r)
        # Even-N Simpson is less accurate (trapezoidal correction on endpoints)
        torch.testing.assert_close(result, torch.tensor(1.0 / 3.0, dtype=DTYPE), atol=1e-5, rtol=0)

    def test_nonuniform_quadratic(self):
        """Simpson on non-uniform grid, integrating r^2."""
        # Use 7 points (odd) for exact Simpson on quadratic
        r = torch.tensor([1.0, 0.9, 0.75, 0.55, 0.3, 0.1, 0.0], dtype=DTYPE)
        f = r ** 2
        result = simps(f, r)
        torch.testing.assert_close(result, torch.tensor(1.0 / 3.0, dtype=DTYPE), atol=1e-13, rtol=0)

    def test_batched(self):
        """Batched integration over multiple functions."""
        N = 33
        r = torch.linspace(1.0, 0.0, N, dtype=DTYPE)
        # Stack f=1, f=r, f=r^2
        f = torch.stack([torch.ones(N, dtype=DTYPE), r, r ** 2])  # (3, N)
        result = simps(f, r)
        expected = torch.tensor([1.0, 0.5, 1.0 / 3.0], dtype=DTYPE)
        torch.testing.assert_close(result, expected, atol=1e-13, rtol=0)

    def test_physical_shell(self):
        """Integration over the physical shell [r_icb, r_cmb]."""
        # r_cmb = 1/(1-0.35) ≈ 1.5385, r_icb ≈ 0.5385
        r_cmb = 1.0 / (1.0 - 0.35)
        r_icb = r_cmb - 1.0
        N = 33
        r = torch.linspace(r_cmb, r_icb, N, dtype=DTYPE)
        f = r ** 2
        result = simps(f, r)
        # Analytical: ∫_{r_icb}^{r_cmb} r^2 dr = (r_cmb^3 - r_icb^3) / 3
        expected = (r_cmb ** 3 - r_icb ** 3) / 3.0
        torch.testing.assert_close(result, torch.tensor(expected, dtype=DTYPE), atol=1e-12, rtol=0)


class TestChebyshevRIntRRegression:
    """Verify Chebyshev rInt_R is still correct after the dispatch change."""

    def test_constant(self):
        """Integral of f=1 over [r_icb, r_cmb] = r_cmb - r_icb = 1."""
        from magic_torch.chebyshev import r as r_grid
        f = torch.ones(r_grid.shape[0], dtype=DTYPE, device=r_grid.device)
        result = rInt_R(f)
        torch.testing.assert_close(result, torch.tensor(1.0, dtype=DTYPE, device=r_grid.device),
                                   atol=1e-13, rtol=0)

    def test_r_squared(self):
        """Integral of r^2 over [r_icb, r_cmb]."""
        from magic_torch.chebyshev import r as r_grid, r_cmb, r_icb
        f = r_grid ** 2
        result = rInt_R(f)
        expected = (r_cmb ** 3 - r_icb ** 3) / 3.0
        torch.testing.assert_close(result, torch.tensor(expected, dtype=DTYPE, device=r_grid.device),
                                   atol=1e-12, rtol=0)


class TestFDRIntR:
    """Test rInt_R dispatch to Simpson for FD scheme (subprocess)."""

    def test_r_squared(self):
        """rInt_R(r^2) matches analytical for FD grid."""
        result = subprocess.run(
            [sys.executable, "-c", """
import torch
import os
os.environ['MAGIC_RADIAL_SCHEME'] = 'FD'
os.environ['MAGIC_LMAX'] = '16'
os.environ['MAGIC_NR'] = '33'
os.environ['MAGIC_RADRATIO'] = '0.35'
os.environ['MAGIC_DEVICE'] = 'cpu'

from magic_torch.integration import rInt_R
from magic_torch.radial_scheme import r, r_cmb, r_icb

f = r ** 2
result = rInt_R(f).item()
expected = (r_cmb ** 3 - r_icb ** 3) / 3.0
rel = abs(result - expected) / abs(expected)
print(f'result={result:.15e} expected={expected:.15e} rel={rel:.3e}')
assert rel < 1e-10, f'rInt_R(r^2) rel error: {rel}'
print('PASS')
"""],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            pytest.fail(f"Subprocess failed:\n{result.stderr[-2000:]}")
        assert "PASS" in result.stdout

    def test_constant(self):
        """rInt_R(1) = r_cmb - r_icb = 1 for FD."""
        result = subprocess.run(
            [sys.executable, "-c", """
import torch, os
os.environ['MAGIC_RADIAL_SCHEME'] = 'FD'
os.environ['MAGIC_LMAX'] = '16'
os.environ['MAGIC_NR'] = '33'
os.environ['MAGIC_RADRATIO'] = '0.35'
os.environ['MAGIC_DEVICE'] = 'cpu'

from magic_torch.integration import rInt_R
from magic_torch.radial_scheme import r

f = torch.ones_like(r)
result = rInt_R(f).item()
rel = abs(result - 1.0)
print(f'result={result:.15e} rel={rel:.3e}')
assert rel < 1e-10, f'rInt_R(1) error: {rel}'
print('PASS')
"""],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            pytest.fail(f"Subprocess failed:\n{result.stderr[-2000:]}")
        assert "PASS" in result.stdout
