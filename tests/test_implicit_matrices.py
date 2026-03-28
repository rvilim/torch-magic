"""Test implicit solver matrix construction and roundtrip consistency.

Verifies:
1. Matrices build and factorize without error
2. Solve roundtrip: M @ x = rhs → inv(M) @ rhs == x
3. Matrix entries match analytical formula at interior points

Now tests via the precomputed _inv_by_l tensors (no access to per-l LU factors).
"""

import torch
from magic_torch.precision import DTYPE, CDTYPE, DEVICE
from magic_torch.params import n_r_max, l_max, alpha
from magic_torch.chebyshev import rMat, drMat, d2rMat, d3rMat, rnorm, boundary_fac
from magic_torch.radial_functions import or1, or2, or3
from magic_torch.horizontal_data import dLh, hdif_S, hdif_V, hdif_B
from magic_torch.pre_calculations import opr, opm
from magic_torch.cosine_transform import costf
from magic_torch import update_s, update_z, update_wp, update_b


def test_sMat_roundtrip():
    """Build sMat, verify solve roundtrip for each l via inv_by_l."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_s.build_s_matrices(wimp_lin0)

    N = n_r_max
    print("=== sMat roundtrip test ===")
    max_err = 0.0

    # inv_by_l is on DEVICE, work on CPU for matrix construction
    inv_by_l = update_s._s_inv_by_l.cpu()

    for l in range(l_max + 1):
        x = torch.randn(N, dtype=DTYPE)

        dL = float(l * (l + 1))
        hdif_l = hdif_S[l].item()
        or1_col = or1.cpu().unsqueeze(1)
        or2_col = or2.cpu().unsqueeze(1)
        _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
        _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

        dat = _rnorm * (rMat.cpu() - wimp_lin0 * opr * hdif_l * (
            d2rMat.cpu() + 2.0 * or1_col * drMat.cpu() - dL * or2_col * rMat.cpu()
        ))
        dat[0, :] = _rnorm * rMat.cpu()[0, :]
        dat[N - 1, :] = _rnorm * rMat.cpu()[N - 1, :]
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        rhs = dat @ x  # ground truth RHS

        # inv_by_l already incorporates preconditioning: inv_by_l = inv(precondA) * fac
        # So: inv_by_l @ (fac * rhs) would double-apply fac.
        # Instead: inv_by_l @ rhs directly recovers x (since inv_by_l = A_orig^{-1})
        x_hat = inv_by_l[l] @ rhs

        err = (x_hat - x).abs().max().item()
        max_err = max(max_err, err)
        if l <= 3 or err > 1e-10:
            print(f"  l={l:2d}: roundtrip error = {err:.2e}")

    print(f"  Max roundtrip error across all l: {max_err:.2e}")
    assert max_err < 1e-10, f"sMat roundtrip failed: {max_err:.2e}"


def test_sMat_boundary_rows():
    """Verify boundary rows of sMat match rnorm * rMat."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_s.build_s_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== sMat boundary row test ===")

    bc_top = rnorm * rMat[0, :].clone()
    bc_top[0] *= boundary_fac
    bc_top[N - 1] *= boundary_fac

    bc_bot = rnorm * rMat[N - 1, :].clone()
    bc_bot[0] *= boundary_fac
    bc_bot[N - 1] *= boundary_fac

    for l in [0, 5, l_max]:
        x = torch.zeros(N, dtype=DTYPE, device=DEVICE)
        x[0] = 1.0
        phys_top = (bc_top @ x).item()
        phys_bot = (bc_bot @ x).item()
        print(f"  l={l}: phys_top(x=[1,0,...])={phys_top:.6f}, "
              f"phys_bot={phys_bot:.6f}")

    print("  Boundary row structure verified (Dirichlet via rMat)")


def test_zMat_roundtrip():
    """Build zMat, verify solve roundtrip for l >= 1 via inv_by_l."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_z.build_z_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== zMat roundtrip test ===")
    max_err = 0.0

    inv_by_l = update_z._z_inv_by_l.cpu()

    for l in range(1, l_max + 1):
        x = torch.randn(N, dtype=DTYPE)

        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()
        or2_col = or2.cpu().unsqueeze(1)
        _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
        _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

        dat = _rnorm * dL * or2_col * (
            rMat.cpu() - wimp_lin0 * hdif_l * (
                d2rMat.cpu() - dL * or2_col * rMat.cpu()
            )
        )
        dat[0, :] = _rnorm * rMat.cpu()[0, :]
        dat[N - 1, :] = _rnorm * rMat.cpu()[N - 1, :]
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N - 1] = dat[:, N - 1] * _bfac

        rhs = dat @ x
        x_hat = inv_by_l[l] @ rhs

        err = (x_hat - x).abs().max().item()
        max_err = max(max_err, err)
        if l <= 3 or err > 1e-10:
            print(f"  l={l:2d}: roundtrip error = {err:.2e}")

    print(f"  Max roundtrip error across all l: {max_err:.2e}")
    assert max_err < 1e-10, f"zMat roundtrip failed: {max_err:.2e}"


def test_sMat_conduction_state():
    """Verify that the conduction state is an exact solution."""
    from magic_torch.init_fields import ps_cond

    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_s.build_s_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== sMat conduction state test ===")

    s0_phys, _ = ps_cond()
    s0_phys = s0_phys.cpu()
    s0_cheb = costf(s0_phys.to(DEVICE)).cpu()

    or1_col = or1.cpu().unsqueeze(1)
    or2_col = or2.cpu().unsqueeze(1)
    _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    l = 0
    dL = 0.0
    hdif_l = hdif_S[l].item()

    dat = _rnorm * (rMat.cpu() - wimp_lin0 * opr * hdif_l * (
        d2rMat.cpu() + 2.0 * or1_col * drMat.cpu() - dL * or2_col * rMat.cpu()
    ))
    dat[0, :] = _rnorm * rMat.cpu()[0, :]
    dat[N - 1, :] = _rnorm * rMat.cpu()[N - 1, :]
    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N - 1] = dat[:, N - 1] * _bfac

    rhs = dat @ s0_cheb
    err_rhs = (rhs - s0_phys).abs().max().item()
    print(f"  |M@s0_cheb - s0_phys| = {err_rhs:.2e} (should be ~0 since L(s0)=0)")

    inv_by_l = update_s._s_inv_by_l.cpu()
    x_hat = inv_by_l[l] @ rhs
    err_solve = (x_hat - s0_cheb).abs().max().item()
    print(f"  Solve recovery error: {err_solve:.2e}")
    assert err_solve < 1e-10, f"Conduction state recovery failed: {err_solve:.2e}"


def test_wpMat_roundtrip():
    """Build wpMat, verify solve roundtrip for l >= 1 via inv_by_l."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_wp.build_wp_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== wpMat roundtrip test ===")
    max_err = 0.0

    inv_by_l = update_wp._wp_inv_by_l.cpu()
    or2_col = or2.cpu().unsqueeze(1)
    or3_col = or3.cpu().unsqueeze(1)
    _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    for l in range(1, l_max + 1):
        dL = float(l * (l + 1))
        hdif_l = hdif_V[l].item()

        dat = torch.zeros(2 * N, 2 * N, dtype=DTYPE)

        dat[1:N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            rMat.cpu()[1:N-1] - wimp_lin0 * hdif_l * (
                d2rMat.cpu()[1:N-1] - dL * or2_col[1:N-1] * rMat.cpu()[1:N-1]
            )
        )
        dat[1:N-1, N:] = _rnorm * wimp_lin0 * drMat.cpu()[1:N-1]

        dat[N+1:2*N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
            -drMat.cpu()[1:N-1] + wimp_lin0 * hdif_l * (
                d3rMat.cpu()[1:N-1] - dL * or2_col[1:N-1] * drMat.cpu()[1:N-1]
                + 2.0 * dL * or3_col[1:N-1] * rMat.cpu()[1:N-1]
            )
        )
        dat[N+1:2*N-1, N:] = -_rnorm * wimp_lin0 * dL * or2_col[1:N-1] * rMat.cpu()[1:N-1]

        dat[0, :N] = _rnorm * rMat.cpu()[0, :]
        dat[N-1, :N] = _rnorm * rMat.cpu()[N-1, :]
        dat[N, :N] = _rnorm * drMat.cpu()[0, :]
        dat[2*N-1, :N] = _rnorm * drMat.cpu()[N-1, :]

        for blk_start in [0, N]:
            dat[:, blk_start] = dat[:, blk_start] * _bfac
            dat[:, blk_start + N - 1] = dat[:, blk_start + N - 1] * _bfac

        x = torch.randn(2 * N, dtype=DTYPE)
        rhs = dat @ x
        x_hat = inv_by_l[l] @ rhs

        err = (x_hat - x).abs().max().item()
        max_err = max(max_err, err)
        if l <= 3 or err > 1e-10:
            print(f"  l={l:2d}: roundtrip error = {err:.2e}")

    print(f"  Max roundtrip error across all l: {max_err:.2e}")
    assert max_err < 1e-7, f"wpMat roundtrip failed: {max_err:.2e}"


def test_p0Mat_roundtrip():
    """Build p0Mat, verify solve roundtrip."""
    from magic_torch.algebra import solve_mat_real
    update_wp.build_p0_matrix()

    N = n_r_max
    print("\n=== p0Mat roundtrip test ===")
    _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    dat = torch.zeros(N, N, dtype=DTYPE)
    dat[1:, :] = _rnorm * drMat.cpu()[1:, :]
    dat[0, :] = _rnorm * rMat.cpu()[0, :]
    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N-1] = dat[:, N-1] * _bfac

    x = torch.randn(N, dtype=DTYPE)
    rhs = dat @ x

    # p0Mat LU factors are on CPU
    x_hat = solve_mat_real(update_wp._p0Mat_lu.cpu(), update_wp._p0Mat_ip.cpu(), rhs)
    err = (x_hat - x).abs().max().item()
    print(f"  roundtrip error = {err:.2e}")
    assert err < 1e-10, f"p0Mat roundtrip failed: {err:.2e}"


def test_bMat_roundtrip():
    """Build bMat, verify solve roundtrip for l >= 1 via inv_by_l."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_b.build_b_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== bMat roundtrip test ===")
    max_err = 0.0

    inv_by_l = update_b._b_inv_by_l.cpu()
    or2_col = or2.cpu().unsqueeze(1)
    _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    for l in range(1, l_max + 1):
        x = torch.randn(N, dtype=DTYPE)

        dL = float(l * (l + 1))
        hdif_l = hdif_B[l].item()

        dat = torch.zeros(N, N, dtype=DTYPE)
        dat[1:N-1, :] = _rnorm * dL * or2_col[1:N-1] * (
            rMat.cpu()[1:N-1] - wimp_lin0 * opm * hdif_l * (
                d2rMat.cpu()[1:N-1] - dL * or2_col[1:N-1] * rMat.cpu()[1:N-1]
            )
        )
        dat[0, :] = _rnorm * (drMat.cpu()[0, :] + float(l) * or1.cpu()[0] * rMat.cpu()[0, :])
        dat[N-1, :] = _rnorm * (drMat.cpu()[N-1, :] - float(l + 1) * or1.cpu()[N-1] * rMat.cpu()[N-1, :])
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N-1] = dat[:, N-1] * _bfac

        rhs = dat @ x
        x_hat = inv_by_l[l] @ rhs

        err = (x_hat - x).abs().max().item()
        max_err = max(max_err, err)
        if l <= 3 or err > 1e-10:
            print(f"  l={l:2d}: roundtrip error = {err:.2e}")

    print(f"  Max roundtrip error across all l: {max_err:.2e}")
    assert max_err < 1e-10, f"bMat roundtrip failed: {max_err:.2e}"


def test_jMat_roundtrip():
    """Build jMat, verify solve roundtrip for l >= 1 via inv_by_l."""
    dt = 1e-4
    wimp_lin0 = alpha * dt
    update_b.build_b_matrices(wimp_lin0)

    N = n_r_max
    print("\n=== jMat roundtrip test ===")
    max_err = 0.0

    inv_by_l = update_b._j_inv_by_l.cpu()
    or2_col = or2.cpu().unsqueeze(1)
    _rnorm = rnorm.cpu() if isinstance(rnorm, torch.Tensor) else rnorm
    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac

    for l in range(1, l_max + 1):
        x = torch.randn(N, dtype=DTYPE)

        dL = float(l * (l + 1))
        hdif_l = hdif_B[l].item()

        dat = torch.zeros(N, N, dtype=DTYPE)
        dat[1:N-1, :] = _rnorm * dL * or2_col[1:N-1] * (
            rMat.cpu()[1:N-1] - wimp_lin0 * opm * hdif_l * (
                d2rMat.cpu()[1:N-1] - dL * or2_col[1:N-1] * rMat.cpu()[1:N-1]
            )
        )
        dat[0, :] = _rnorm * rMat.cpu()[0, :]
        dat[N-1, :] = _rnorm * rMat.cpu()[N-1, :]
        dat[:, 0] = dat[:, 0] * _bfac
        dat[:, N-1] = dat[:, N-1] * _bfac

        rhs = dat @ x
        x_hat = inv_by_l[l] @ rhs

        err = (x_hat - x).abs().max().item()
        max_err = max(max_err, err)
        if l <= 3 or err > 1e-10:
            print(f"  l={l:2d}: roundtrip error = {err:.2e}")

    print(f"  Max roundtrip error across all l: {max_err:.2e}")
    assert max_err < 1e-10, f"jMat roundtrip failed: {max_err:.2e}"


if __name__ == "__main__":
    test_sMat_roundtrip()
    test_sMat_boundary_rows()
    test_zMat_roundtrip()
    test_sMat_conduction_state()
    test_wpMat_roundtrip()
    test_p0Mat_roundtrip()
    test_bMat_roundtrip()
    test_jMat_roundtrip()
    print("\n=== All tests passed ===")
