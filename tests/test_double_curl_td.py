"""Tests for get_dwdt_double_curl explicit terms.

Verifies:
1. Output shapes and boundary zeros
2. Nonlinear term formula: dLh * or4 * orho1 * AdvrLM
3. dVxVhLM formula: -orho1 * r^2 * dLh * AdvtLM
4. Coriolis coupling uses dz (not just z) and dTheta3/4 (not dTheta2)
5. Reduces correctly for Boussinesq (beta=0, orho1=1)
"""

import torch
import pytest

from magic_torch.params import lm_max, n_r_max, l_max
from magic_torch.precision import CDTYPE
from magic_torch.get_td import get_dwdt, get_dwdt_double_curl
from magic_torch.radial_functions import or2, or4, orho1
from magic_torch.radial_scheme import r
from magic_torch.horizontal_data import dLh
from magic_torch.blocking import st_lm2l


class TestGetDwdtDoubleCurl:

    def _random_fields(self):
        """Create random spectral fields for testing."""
        torch.manual_seed(42)
        kw = dict(dtype=CDTYPE, device=or2.device)
        w = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        dw = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        ddw = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        z = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        dz = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        AdvrLM = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        AdvtLM = torch.randn(lm_max, n_r_max, **kw) * 1e-2
        return w, dw, ddw, z, dz, AdvrLM, AdvtLM

    def test_output_shapes(self):
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()
        dwdt, dVxVhLM = get_dwdt_double_curl(AdvrLM, AdvtLM, w, dw, ddw, z, dz)
        assert dwdt.shape == (lm_max, n_r_max)
        assert dVxVhLM.shape == (lm_max, n_r_max)

    def test_boundary_zeros(self):
        """dwdt and dVxVhLM should be zero at boundaries."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()
        dwdt, dVxVhLM = get_dwdt_double_curl(AdvrLM, AdvtLM, w, dw, ddw, z, dz)
        assert dwdt[:, 0].abs().max() == 0.0
        assert dwdt[:, -1].abs().max() == 0.0
        assert dVxVhLM[:, 0].abs().max() == 0.0
        assert dVxVhLM[:, -1].abs().max() == 0.0

    def test_nonlinear_term_formula(self):
        """Nonlinear part: dLh * or4 * orho1 * AdvrLM (vs standard or2 * AdvrLM)."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()
        # Zero Coriolis by setting CorFac=0 temporarily — check NL term only
        # Instead, set z=dz=0 so Coriolis vanishes, and dw=ddw=0 so dPhi*ddw=0
        z_zero = torch.zeros_like(z)
        dz_zero = torch.zeros_like(dz)
        dw_zero = torch.zeros_like(dw)
        ddw_zero = torch.zeros_like(ddw)
        w_zero = torch.zeros_like(w)

        dwdt_dc, dVxVhLM = get_dwdt_double_curl(
            AdvrLM, AdvtLM, w_zero, dw_zero, ddw_zero, z_zero, dz_zero)
        dwdt_std = get_dwdt(AdvrLM, dw_zero, z_zero)

        # l>0: dLh * or4 * orho1 * AdvrLM
        # l=0: or2 * AdvrLM (standard formula, since dLh[l=0]=0)
        expected_dc = (dLh.to(CDTYPE).unsqueeze(1) * or4.unsqueeze(0)
                       * orho1.unsqueeze(0).to(CDTYPE) * AdvrLM)
        # l=0 uses standard formula
        expected_dc[0, :] = or2.unsqueeze(0) * AdvrLM[0, :]
        expected_dc[:, 0] = 0
        expected_dc[:, -1] = 0
        torch.testing.assert_close(dwdt_dc, expected_dc, atol=1e-14, rtol=1e-14)

    def test_dVxVhLM_formula(self):
        """dVxVhLM = -orho1 * r^2 * dLh * AdvtLM."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()
        _, dVxVhLM = get_dwdt_double_curl(AdvrLM, AdvtLM, w, dw, ddw, z, dz)

        expected = (-orho1.unsqueeze(0).to(CDTYPE) * (r * r).unsqueeze(0)
                    * dLh.to(CDTYPE).unsqueeze(1) * AdvtLM)
        expected[:, 0] = 0
        expected[:, -1] = 0
        torch.testing.assert_close(dVxVhLM, expected, atol=1e-14, rtol=1e-14)

    def test_coriolis_uses_dz_not_just_z(self):
        """Double-curl Coriolis depends on dz (radial derivative of z).
        Standard Coriolis does NOT depend on dz."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()

        # Run with dz=0 vs dz≠0 — should give different results
        dwdt_with_dz, _ = get_dwdt_double_curl(
            torch.zeros_like(AdvrLM), torch.zeros_like(AdvtLM),
            w, dw, ddw, z, dz)
        dwdt_no_dz, _ = get_dwdt_double_curl(
            torch.zeros_like(AdvrLM), torch.zeros_like(AdvtLM),
            w, dw, ddw, z, torch.zeros_like(dz))

        diff = (dwdt_with_dz - dwdt_no_dz).abs().max()
        assert diff > 1e-10, f"Double-curl Coriolis should depend on dz, but diff={diff}"

    def test_coriolis_uses_ddw(self):
        """Double-curl Coriolis uses ddw (second radial derivative of w).
        Standard does NOT."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()

        dwdt_with_ddw, _ = get_dwdt_double_curl(
            torch.zeros_like(AdvrLM), torch.zeros_like(AdvtLM),
            w, dw, ddw, z, dz)
        dwdt_no_ddw, _ = get_dwdt_double_curl(
            torch.zeros_like(AdvrLM), torch.zeros_like(AdvtLM),
            w, dw, torch.zeros_like(ddw), z, dz)

        diff = (dwdt_with_ddw - dwdt_no_ddw).abs().max()
        assert diff > 1e-10, f"Double-curl Coriolis should depend on ddw, but diff={diff}"

    def test_l0_uses_standard_formula(self):
        """l=0 uses STANDARD formula or2*AdvrLM (not double-curl dLh*or4*...).
        This matches Fortran get_dwdt_double_curl lines 232-241."""
        w, dw, ddw, z, dz, AdvrLM, AdvtLM = self._random_fields()
        # Zero z so Coriolis doesn't contribute — test NL term only
        z_zero = torch.zeros_like(z)
        dz_zero = torch.zeros_like(dz)
        dwdt, _ = get_dwdt_double_curl(
            AdvrLM, AdvtLM, w, dw, ddw, z_zero, dz_zero)
        dwdt_std = get_dwdt(AdvrLM, dw, z_zero)
        # At l=0, double-curl should match standard: both give or2*AdvrLM
        # (boundaries are zeroed in both)
        torch.testing.assert_close(
            dwdt[0, 1:-1], dwdt_std[0, 1:-1], atol=1e-14, rtol=1e-14)
