"""Tests for the radial_scheme dispatch layer.

Verifies that:
1. Chebyshev backend exports match chebyshev.py directly
2. to_physical() is costf for Chebyshev
3. FD selection raises NotImplementedError (until implemented)
4. All expected names are exported
"""

import subprocess
import sys

import pytest
import torch

from magic_torch import radial_scheme as rs
from magic_torch import chebyshev
from magic_torch.cosine_transform import costf
from magic_torch.integration import rInt_R


class TestChebyshevBackend:
    """Verify radial_scheme exports match chebyshev.py for the default backend."""

    def test_grid_points_are_same_object(self):
        assert rs.r is chebyshev.r
        assert rs.r_cmb is chebyshev.r_cmb
        assert rs.r_icb is chebyshev.r_icb

    def test_derivative_matrices_are_same_object(self):
        assert rs.rMat is chebyshev.rMat
        assert rs.drMat is chebyshev.drMat
        assert rs.d2rMat is chebyshev.d2rMat
        assert rs.d3rMat is chebyshev.d3rMat

    def test_normalization_constants(self):
        assert rs.rnorm == chebyshev.rnorm
        assert rs.boundary_fac == chebyshev.boundary_fac

    def test_mapping_derivatives_are_same_object(self):
        assert rs.drx is chebyshev.drx
        assert rs.ddrx is chebyshev.ddrx
        assert rs.dddrx is chebyshev.dddrx

    def test_boundary_vectors_are_same_object(self):
        assert rs.dr_top is chebyshev.dr_top
        assert rs.dr_bot is chebyshev.dr_bot

    def test_d4rMat_is_none(self):
        assert rs.d4rMat is None

    def test_costf_is_same_function(self):
        assert rs.costf is costf

    def test_rInt_R_is_same_function(self):
        assert rs.rInt_R is rInt_R

    def test_to_physical_is_costf(self):
        f = torch.randn(10, chebyshev.r.shape[0], dtype=torch.float64)
        result = rs.to_physical(f)
        expected = costf(f)
        torch.testing.assert_close(result, expected)


class TestExportCompleteness:
    """Verify all expected names are exported."""

    EXPECTED_NAMES = [
        "r", "r_cmb", "r_icb", "x_cheb",
        "rMat", "drMat", "d2rMat", "d3rMat", "d4rMat",
        "drx", "ddrx", "dddrx",
        "rnorm", "boundary_fac",
        "dr_top", "dr_bot",
        "costf", "rInt_R", "to_physical",
    ]

    @pytest.mark.parametrize("name", EXPECTED_NAMES)
    def test_name_exported(self, name):
        assert hasattr(rs, name), f"radial_scheme missing export: {name}"


class TestFDImports:
    """Verify FD selection imports successfully."""

    def test_fd_imports_without_error(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import os; os.environ['MAGIC_RADIAL_SCHEME'] = 'FD'; "
             "os.environ['MAGIC_LMAX'] = '16'; os.environ['MAGIC_NR'] = '33'; "
             "os.environ['MAGIC_RADRATIO'] = '0.35'; os.environ['MAGIC_DEVICE'] = 'cpu'; "
             "from magic_torch import radial_scheme; "
             "print('OK:', radial_scheme.r.shape, radial_scheme.d4rMat.shape)"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"FD import failed:\n{result.stderr[-1000:]}"
        assert "OK:" in result.stdout
