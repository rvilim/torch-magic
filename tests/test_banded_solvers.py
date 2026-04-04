"""Tests for banded matrix solvers (tridiagonal and pentadiagonal).

All solvers are pure PyTorch, GPU-compatible, and vectorized over batch.
Tests verify:
1. Roundtrip: A * solve(A, b) == b to machine precision
2. Consistency with torch.linalg.solve (dense)
3. Complex RHS
4. Batched operation (multiple systems simultaneously)
5. Pentadiagonal: consistency with tridiagonal for tridiag input
"""

import torch
import pytest

from magic_torch.algebra import (
    batched_tridiag_solve,
    batched_pentadiag_solve,
    prepare_tridiag, solve_tridiag_real, solve_tridiag_complex,
    prepare_band, solve_band_real,
    prepare_mat, solve_mat_real,
    prepare_bordered, solve_bordered,
)

DTYPE = torch.float64
CDTYPE = torch.complex128


def _random_tridiag(n, batch=1, seed=42):
    """Create random diagonally-dominant tridiagonal bands."""
    torch.manual_seed(seed)
    dl = torch.randn(batch, n - 1, dtype=DTYPE)
    du = torch.randn(batch, n - 1, dtype=DTYPE)
    d = torch.randn(batch, n, dtype=DTYPE).abs() + 3.0  # diag dominant
    return dl, d, du


def _tridiag_to_dense(dl, d, du):
    """Convert batched tridiagonal bands to dense matrices. (B, N) → (B, N, N)"""
    B, N = d.shape
    A = torch.zeros(B, N, N, dtype=DTYPE)
    for b in range(B):
        A[b, range(N), range(N)] = d[b]
        A[b, range(1, N), range(N - 1)] = dl[b]
        A[b, range(N - 1), range(1, N)] = du[b]
    return A


def _random_pentadiag(n, batch=1, seed=42):
    """Create random diagonally-dominant pentadiagonal bands."""
    torch.manual_seed(seed)
    dl2 = torch.randn(batch, n - 2, dtype=DTYPE) * 0.5
    dl1 = torch.randn(batch, n - 1, dtype=DTYPE)
    du1 = torch.randn(batch, n - 1, dtype=DTYPE)
    du2 = torch.randn(batch, n - 2, dtype=DTYPE) * 0.5
    d = torch.randn(batch, n, dtype=DTYPE).abs() + 5.0  # diag dominant
    return dl2, dl1, d, du1, du2


def _pentadiag_to_dense(dl2, dl1, d, du1, du2):
    """Convert batched pentadiagonal bands to dense. (B, N) → (B, N, N)"""
    B, N = d.shape
    A = torch.zeros(B, N, N, dtype=DTYPE)
    for b in range(B):
        A[b, range(N), range(N)] = d[b]
        A[b, range(1, N), range(N - 1)] = dl1[b]
        A[b, range(N - 1), range(1, N)] = du1[b]
        A[b, range(2, N), range(N - 2)] = dl2[b]
        A[b, range(N - 2), range(2, N)] = du2[b]
    return A


# ===== Batched Thomas (tridiagonal) =====

class TestBatchedTridiag:
    @pytest.mark.parametrize("n", [5, 10, 33, 73])
    def test_roundtrip_real(self, n):
        dl, d, du = _random_tridiag(n)
        A = _tridiag_to_dense(dl, d, du)  # (1, N, N)
        x_known = torch.randn(1, n, dtype=DTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)  # (1, N)

        x = batched_tridiag_solve(dl, d, du, b)
        torch.testing.assert_close(x, x_known, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("n", [5, 33])
    def test_roundtrip_complex(self, n):
        dl, d, du = _random_tridiag(n)
        A = _tridiag_to_dense(dl, d, du).to(CDTYPE)
        x_known = torch.randn(1, n, dtype=CDTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)

        x = batched_tridiag_solve(dl, d, du, b)
        torch.testing.assert_close(x, x_known, atol=1e-12, rtol=1e-12)

    def test_multi_batch(self):
        """Multiple independent systems solved simultaneously."""
        B, n = 16, 33
        dl, d, du = _random_tridiag(n, batch=B, seed=123)
        A = _tridiag_to_dense(dl, d, du)  # (B, N, N)
        x_known = torch.randn(B, n, dtype=DTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)  # (B, N)

        x = batched_tridiag_solve(dl, d, du, b)
        torch.testing.assert_close(x, x_known, atol=1e-11, rtol=1e-11)

    def test_matches_dense_solve(self):
        """Matches torch.linalg.solve on dense equivalent."""
        n = 20
        dl, d, du = _random_tridiag(n)
        A = _tridiag_to_dense(dl, d, du).squeeze(0)  # (N, N)
        b = torch.randn(n, dtype=DTYPE)

        x_dense = torch.linalg.solve(A, b)
        x_thomas = batched_tridiag_solve(dl, d, du, b.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(x_thomas, x_dense, atol=1e-12, rtol=1e-12)

    def test_matches_scalar_tridiag(self):
        """Matches the scalar prepare_tridiag/solve_tridiag_real."""
        n = 33
        dl_raw, d_raw, du_raw = _random_tridiag(n)
        b = torch.randn(n, dtype=DTYPE)

        # Scalar (existing) solver
        dl_f, d_f, du_f, du2, pivot, info = prepare_tridiag(
            dl_raw.squeeze(0), d_raw.squeeze(0), du_raw.squeeze(0))
        assert info == 0
        x_scalar = solve_tridiag_real(dl_f, d_f, du_f, du2, pivot, b)

        # Batched solver
        x_batched = batched_tridiag_solve(dl_raw, d_raw, du_raw, b.unsqueeze(0)).squeeze(0)

        # Both should be correct; may differ slightly due to pivoting vs no-pivoting
        A = _tridiag_to_dense(dl_raw, d_raw, du_raw).squeeze(0)
        torch.testing.assert_close(A @ x_scalar, b, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(A @ x_batched, b, atol=1e-12, rtol=1e-12)

    def test_unbatched_bands(self):
        """1D band tensors (no batch dim) work correctly."""
        n = 10
        dl, d, du = _random_tridiag(n)
        A = _tridiag_to_dense(dl, d, du)
        b = torch.randn(1, n, dtype=DTYPE)
        rhs_b = (A @ b.unsqueeze(-1)).squeeze(-1)

        # Pass 1D bands
        x = batched_tridiag_solve(dl.squeeze(0), d.squeeze(0), du.squeeze(0), rhs_b)
        torch.testing.assert_close(x, b, atol=1e-12, rtol=1e-12)


# ===== Batched Pentadiagonal =====

class TestBatchedPentadiag:
    @pytest.mark.parametrize("n", [7, 10, 33, 73])
    def test_roundtrip_real(self, n):
        dl2, dl1, d, du1, du2 = _random_pentadiag(n)
        A = _pentadiag_to_dense(dl2, dl1, d, du1, du2)
        x_known = torch.randn(1, n, dtype=DTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)

        x = batched_pentadiag_solve(dl2, dl1, d, du1, du2, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("n", [7, 33])
    def test_roundtrip_complex(self, n):
        dl2, dl1, d, du1, du2 = _random_pentadiag(n)
        A = _pentadiag_to_dense(dl2, dl1, d, du1, du2).to(CDTYPE)
        x_known = torch.randn(1, n, dtype=CDTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)

        x = batched_pentadiag_solve(dl2, dl1, d, du1, du2, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    def test_multi_batch(self):
        B, n = 16, 33
        dl2, dl1, d, du1, du2 = _random_pentadiag(n, batch=B, seed=77)
        A = _pentadiag_to_dense(dl2, dl1, d, du1, du2)
        x_known = torch.randn(B, n, dtype=DTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)

        x = batched_pentadiag_solve(dl2, dl1, d, du1, du2, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    def test_matches_dense_solve(self):
        n = 20
        dl2, dl1, d, du1, du2 = _random_pentadiag(n)
        A = _pentadiag_to_dense(dl2, dl1, d, du1, du2).squeeze(0)
        b = torch.randn(n, dtype=DTYPE)

        x_dense = torch.linalg.solve(A, b)
        x_penta = batched_pentadiag_solve(dl2, dl1, d, du1, du2, b.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(x_penta, x_dense, atol=1e-10, rtol=1e-10)

    def test_reduces_to_tridiag(self):
        """Pentadiag with dl2=du2=0 matches tridiag solver."""
        n = 33
        dl, d, du = _random_tridiag(n)
        b = torch.randn(1, n, dtype=DTYPE)
        A = _tridiag_to_dense(dl, d, du)
        rhs = (A @ b.unsqueeze(-1)).squeeze(-1)

        x_tri = batched_tridiag_solve(dl, d, du, rhs)
        dl2 = torch.zeros(1, n - 2, dtype=DTYPE)
        du2 = torch.zeros(1, n - 2, dtype=DTYPE)
        x_penta = batched_pentadiag_solve(dl2, dl, d, du, du2, rhs)

        torch.testing.assert_close(x_tri, x_penta, atol=1e-12, rtol=1e-12)

    def test_unbatched_bands(self):
        """1D band tensors (no batch dim) work correctly."""
        n = 10
        dl2, dl1, d, du1, du2 = _random_pentadiag(n)
        A = _pentadiag_to_dense(dl2, dl1, d, du1, du2)
        x_known = torch.randn(1, n, dtype=DTYPE)
        b = (A @ x_known.unsqueeze(-1)).squeeze(-1)

        x = batched_pentadiag_solve(
            dl2.squeeze(0), dl1.squeeze(0), d.squeeze(0),
            du1.squeeze(0), du2.squeeze(0), b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)


# ===== Scalar tridiag (existing, with pivoting) =====

class TestScalarTridiag:
    """Keep existing scalar tridiag solver tests."""

    @pytest.mark.parametrize("n", [5, 33, 73])
    def test_roundtrip(self, n):
        dl_raw, d_raw, du_raw = _random_tridiag(n)
        dl = dl_raw.squeeze(0)
        d = d_raw.squeeze(0)
        du = du_raw.squeeze(0)
        A = _tridiag_to_dense(dl_raw, d_raw, du_raw).squeeze(0)
        x_known = torch.randn(n, dtype=DTYPE)
        b = A @ x_known

        dl_f, d_f, du_f, du2, pivot, info = prepare_tridiag(dl, d, du)
        assert info == 0
        x = solve_tridiag_real(dl_f, d_f, du_f, du2, pivot, b)
        torch.testing.assert_close(x, x_known, atol=1e-12, rtol=1e-12)

    def test_complex(self):
        n = 33
        dl_raw, d_raw, du_raw = _random_tridiag(n)
        dl = dl_raw.squeeze(0)
        d = d_raw.squeeze(0)
        du = du_raw.squeeze(0)
        A = _tridiag_to_dense(dl_raw, d_raw, du_raw).squeeze(0).to(CDTYPE)
        x_known = torch.randn(n, dtype=CDTYPE)
        b = A @ x_known

        dl_f, d_f, du_f, du2, pivot, info = prepare_tridiag(dl, d, du)
        assert info == 0
        x = solve_tridiag_complex(dl_f, d_f, du_f, du2, pivot, b)
        torch.testing.assert_close(x, x_known, atol=1e-12, rtol=1e-12)


# ===== Pivoted banded LU (matching Fortran prepare_band/solve_band) =====

from magic_torch.algebra import prepare_band, solve_band_real


def _random_banded_dense(n, kl, ku, seed=42):
    """Create a random diagonally-dominant banded matrix as dense."""
    torch.manual_seed(seed)
    A = torch.zeros(n, n, dtype=DTYPE)
    for j in range(n):
        for i in range(max(0, j - ku), min(n, j + kl + 1)):
            A[i, j] = torch.randn(1, dtype=DTYPE).item()
    for i in range(n):
        A[i, i] = A[i].abs().sum() + 1.0
    return A


def _dense_to_band_storage(A_dense, kl, ku):
    """Convert dense matrix to LAPACK-style band storage for prepare_band.

    Band storage: abd[kl+ku+i-j, j] = A[i, j] for max(0,j-ku) <= i <= min(n-1,j+kl).
    First kl rows are workspace for fill-in (initially zero).
    Total rows: 2*kl+ku+1.
    """
    n = A_dense.shape[0]
    abd = torch.zeros(2 * kl + ku + 1, n, dtype=A_dense.dtype)
    for j in range(n):
        for i in range(max(0, j - ku), min(n, j + kl + 1)):
            abd[kl + ku + i - j, j] = A_dense[i, j]
    return abd


class TestPivotedBandLU:
    """Test pivoted banded LU matching Fortran prepare_band/solve_band."""

    @pytest.mark.parametrize("n,kl,ku", [
        (5, 1, 1), (10, 1, 1), (10, 2, 2), (33, 2, 2), (73, 2, 2),
    ])
    def test_roundtrip_real(self, n, kl, ku):
        A = _random_banded_dense(n, kl, ku)
        x_known = torch.randn(n, dtype=DTYPE)
        b = A @ x_known

        abd = _dense_to_band_storage(A, kl, ku)
        abd_f, pivot, info = prepare_band(abd, n, kl, ku)
        assert info == 0
        x = solve_band_real(abd_f, n, kl, ku, pivot, b)
        torch.testing.assert_close(x, x_known, atol=1e-11, rtol=1e-11)

    @pytest.mark.parametrize("n,kl,ku", [
        (10, 2, 2), (33, 2, 2),
    ])
    def test_roundtrip_complex(self, n, kl, ku):
        A = _random_banded_dense(n, kl, ku)
        x_known = torch.randn(n, dtype=CDTYPE)
        b = A.to(CDTYPE) @ x_known

        abd = _dense_to_band_storage(A, kl, ku)
        abd_f, pivot, info = prepare_band(abd, n, kl, ku)
        assert info == 0
        x = solve_band_real(abd_f, n, kl, ku, pivot, b)
        torch.testing.assert_close(x, x_known, atol=1e-11, rtol=1e-11)

    @pytest.mark.parametrize("n,kl,ku", [(10, 1, 1), (33, 2, 2)])
    def test_matches_dense_solver(self, n, kl, ku):
        A = _random_banded_dense(n, kl, ku)
        b = torch.randn(n, dtype=DTYPE)

        # Dense solve
        lu, ip, info_d = prepare_mat(A)
        assert info_d == 0
        x_dense = solve_mat_real(lu, ip, b)

        # Banded solve
        abd = _dense_to_band_storage(A, kl, ku)
        abd_f, pivot, info_b = prepare_band(abd, n, kl, ku)
        assert info_b == 0
        x_band = solve_band_real(abd_f, n, kl, ku, pivot, b)

        torch.testing.assert_close(x_band, x_dense, atol=1e-12, rtol=1e-12)

    def test_multi_rhs(self):
        n, kl, ku = 20, 2, 2
        A = _random_banded_dense(n, kl, ku)
        B = torch.randn(n, 5, dtype=DTYPE)

        abd = _dense_to_band_storage(A, kl, ku)
        abd_f, pivot, info = prepare_band(abd, n, kl, ku)
        assert info == 0
        X = solve_band_real(abd_f, n, kl, ku, pivot, B)
        torch.testing.assert_close(A @ X, B, atol=1e-11, rtol=1e-11)

    def test_pivoting_needed(self):
        """Matrix where pivoting is required."""
        n, kl, ku = 10, 1, 1
        A = torch.zeros(n, n, dtype=DTYPE)
        # Tiny diagonal, large off-diag → pivoting needed
        for i in range(n):
            A[i, i] = 0.01
        for i in range(n - 1):
            A[i + 1, i] = 5.0
            A[i, i + 1] = 0.1
        A[0, 0] = 10.0
        A[n - 1, n - 1] = 10.0
        # Make it non-singular
        A = A + 2.0 * torch.eye(n, dtype=DTYPE)

        b = torch.randn(n, dtype=DTYPE)
        abd = _dense_to_band_storage(A, kl, ku)
        abd_f, pivot, info = prepare_band(abd, n, kl, ku)
        assert info == 0
        x = solve_band_real(abd_f, n, kl, ku, pivot, b)
        torch.testing.assert_close(A @ x, b, atol=1e-10, rtol=1e-10)


# ===== Bordered-band solver =====

class TestBorderedSolver:
    """Test bordered-band solver (OC banded + IC dense)."""

    def _make_bordered_system(self, ncol=20, nfull=10, kl=2, ku=2, seed=42):
        """Create a test bordered system with banded A1 + dense IC block."""
        torch.manual_seed(seed)
        NT = ncol + nfull
        # Build full matrix: A1 banded, A2 sparse, A3 single row, A4 dense
        A = torch.zeros(NT, NT, dtype=DTYPE)
        # A1: banded (ncol x ncol), diag dominant
        for j in range(ncol):
            for i in range(max(0, j - ku), min(ncol, j + kl + 1)):
                A[i, j] = torch.randn(1, dtype=DTYPE).item()
        for i in range(ncol):
            A[i, i] = A[i, :ncol].abs().sum() + 2.0
        # A2: sparse coupling (only last OC row has entries)
        A[ncol - 1, ncol:] = torch.randn(nfull, dtype=DTYPE) * 0.5
        # A3: border row
        A[ncol, :ncol] = torch.randn(ncol, dtype=DTYPE) * 0.3
        # A4: dense IC block, diag dominant
        A[ncol:, ncol:] = torch.randn(nfull, nfull, dtype=DTYPE)
        for i in range(nfull):
            A[ncol + i, ncol + i] = A[ncol + i, ncol:].abs().sum() + 2.0
        return A

    def test_roundtrip_real(self):
        ncol, nfull, kl, ku = 20, 10, 2, 2
        A = self._make_bordered_system(ncol, nfull, kl, ku)
        x_known = torch.randn(ncol + nfull, dtype=DTYPE)
        b = A @ x_known

        factored = prepare_bordered(A, ncol, nfull, kl, ku)
        x = solve_bordered(factored, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    def test_roundtrip_complex(self):
        ncol, nfull, kl, ku = 20, 10, 2, 2
        A = self._make_bordered_system(ncol, nfull, kl, ku)
        x_known = torch.randn(ncol + nfull, dtype=CDTYPE)
        b = A.to(CDTYPE) @ x_known

        factored = prepare_bordered(A, ncol, nfull, kl, ku)
        x = solve_bordered(factored, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    def test_matches_dense_solve(self):
        ncol, nfull, kl, ku = 20, 10, 2, 2
        A = self._make_bordered_system(ncol, nfull, kl, ku)
        b = torch.randn(ncol + nfull, dtype=DTYPE)

        # Dense solve
        lu, ip, info = prepare_mat(A)
        assert info == 0
        x_dense = solve_mat_real(lu, ip, b)

        # Bordered solve
        factored = prepare_bordered(A, ncol, nfull, kl, ku)
        x_bordered = solve_bordered(factored, b)

        torch.testing.assert_close(x_bordered, x_dense, atol=1e-11, rtol=1e-11)

    def test_realistic_sizes(self):
        """Test with sizes matching the dynamo benchmark (ncol=33, nfull=17)."""
        ncol, nfull, kl, ku = 33, 17, 2, 2
        A = self._make_bordered_system(ncol, nfull, kl, ku, seed=99)
        x_known = torch.randn(ncol + nfull, dtype=CDTYPE)
        b = A.to(CDTYPE) @ x_known

        factored = prepare_bordered(A, ncol, nfull, kl, ku)
        x = solve_bordered(factored, b)
        torch.testing.assert_close(x, x_known, atol=1e-10, rtol=1e-10)

    def test_multi_rhs(self):
        ncol, nfull, kl, ku = 20, 10, 2, 2
        A = self._make_bordered_system(ncol, nfull, kl, ku)
        B = torch.randn(ncol + nfull, 5, dtype=DTYPE)

        factored = prepare_bordered(A, ncol, nfull, kl, ku)
        X = solve_bordered(factored, B)
        torch.testing.assert_close(A @ X, B, atol=1e-10, rtol=1e-10)
