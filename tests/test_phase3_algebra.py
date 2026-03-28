"""Phase 3: Verify LU algebra against Fortran-derived implicit matrices.

Tests prepare_mat + solve_mat by:
1. Building the sMat for l=5 (same matrix used by the implicit entropy solver)
2. Factorizing it and solving A*x = b for a known RHS
3. Verifying A*x_solved = b to machine precision
4. Cross-checking: the w_imex_rhs Fortran data was solved by the same LU code
   in phase 7/8 — if algebra were wrong, those phases would fail.

Also tests real and complex solvers independently, and batched solve.
"""

import torch
from magic_torch.algebra import prepare_mat, solve_mat, solve_mat_complex, solve_mat_real
from magic_torch.precision import DTYPE, CDTYPE, DEVICE
from magic_torch.params import n_r_max


def _build_s_matrix_for_l(l_val: int, wimp_lin0: float) -> torch.Tensor:
    """Build the entropy implicit matrix for a given l value.

    Reproduces the matrix construction from update_s.py build_s_matrices.
    """
    from magic_torch.chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
    from magic_torch.radial_functions import or1, or2
    from magic_torch.horizontal_data import dLh, hdif_S
    from magic_torch.pre_calculations import opr

    N = n_r_max
    dLh_l = dLh[0].item()  # placeholder, we need l-specific
    # Actually compute dLh for this l: dLh = l*(l+1)
    dLh_l = l_val * (l_val + 1.0)
    hdif_l = hdif_S[l_val].item()

    sMat = torch.zeros(N, N, dtype=DTYPE, device=DEVICE)
    for i in range(N):
        for j in range(N):
            sMat[i, j] = rMat[i, j] - wimp_lin0 * opr * hdif_l * (
                d2rMat[i, j] + 2.0 * or1[i].item() * drMat[i, j]
                - dLh_l * or2[i].item() * rMat[i, j]
            )

    # Boundary conditions: top (nR=0) and bottom (nR=N-1)
    # Fixed entropy: row 0 and row N-1 set to boundary_fac * rMat row
    sMat[0, :] = boundary_fac * rMat[0, :] * rnorm
    sMat[N - 1, :] = boundary_fac * rMat[N - 1, :] * rnorm

    return sMat


def test_lu_solve_consistency_real():
    """prepare_mat + solve_mat_real satisfies A*x = b."""
    from magic_torch.params import alpha, dtmax

    wimp_lin0 = alpha * dtmax
    A_orig = _build_s_matrix_for_l(5, wimp_lin0)

    # Known RHS
    b = torch.randn(n_r_max, dtype=DTYPE, device=DEVICE)
    b[0] = 0.0  # boundary
    b[-1] = 0.0

    A_lu, ip, info = prepare_mat(A_orig)
    assert info == 0, f"LU factorization failed with info={info}"

    x = solve_mat_real(A_lu, ip, b)

    # Verify: A_orig @ x == b
    residual = A_orig @ x - b
    assert residual.abs().max().item() < 1e-12, f"max residual = {residual.abs().max().item()}"


def test_lu_solve_consistency_complex():
    """prepare_mat + solve_mat_complex satisfies A*x = b."""
    from magic_torch.params import alpha, dtmax

    wimp_lin0 = alpha * dtmax
    A_orig = _build_s_matrix_for_l(5, wimp_lin0)

    # Complex RHS
    b = torch.randn(n_r_max, dtype=DTYPE, device=DEVICE) + \
        1j * torch.randn(n_r_max, dtype=DTYPE, device=DEVICE)
    b = b.to(CDTYPE)
    b[0] = 0.0
    b[-1] = 0.0

    A_lu, ip, info = prepare_mat(A_orig)
    assert info == 0

    x = solve_mat_complex(A_lu, ip, b)

    residual = (A_orig.to(CDTYPE) @ x) - b
    assert residual.abs().max().item() < 1e-12, f"max residual = {residual.abs().max().item()}"


def test_lu_solve_batched():
    """Batched solve: A*X = B where B has multiple columns."""
    from magic_torch.params import alpha, dtmax

    wimp_lin0 = alpha * dtmax
    A_orig = _build_s_matrix_for_l(3, wimp_lin0)

    nbatch = 10
    B = torch.randn(n_r_max, nbatch, dtype=DTYPE, device=DEVICE) + \
        1j * torch.randn(n_r_max, nbatch, dtype=DTYPE, device=DEVICE)
    B = B.to(CDTYPE)
    B[0, :] = 0.0
    B[-1, :] = 0.0

    A_lu, ip, info = prepare_mat(A_orig)
    assert info == 0

    X = solve_mat_complex(A_lu, ip, B)

    residual = (A_orig.to(CDTYPE) @ X) - B
    assert residual.abs().max().item() < 1e-12, f"max residual = {residual.abs().max().item()}"


def test_lu_different_l_values():
    """LU solve is consistent for multiple l values used in the benchmark."""
    from magic_torch.params import alpha, dtmax, l_max

    wimp_lin0 = alpha * dtmax

    for l_val in [1, 5, 10, l_max]:
        A_orig = _build_s_matrix_for_l(l_val, wimp_lin0)
        b = torch.ones(n_r_max, dtype=CDTYPE, device=DEVICE)
        b[0] = 0.0
        b[-1] = 0.0

        A_lu, ip, info = prepare_mat(A_orig)
        assert info == 0, f"LU failed for l={l_val}"

        x = solve_mat(A_lu, ip, b)
        residual = (A_orig.to(CDTYPE) @ x) - b
        assert residual.abs().max().item() < 1e-12, \
            f"l={l_val}: max residual = {residual.abs().max().item()}"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Phase 3: algebra tests done")
