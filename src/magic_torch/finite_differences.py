"""Finite difference radial scheme matching Fortran finite_differences.f90.

Provides:
- FD grid generation (uniform or stretched)
- Fornberg weights for arbitrary-grid FD stencils
- FD derivative matrices (drMat, d2rMat, d3rMat, d4rMat)
- rMat = identity, rnorm = 1.0, boundary_fac = 1.0

All matrices are stored as dense (n_r_max, n_r_max) for now.
Banded storage optimization is deferred to when N > ~256.
"""

import math
import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, radratio


def populate_fd_weights(z: float, x: torch.Tensor, nd: int, m: int) -> torch.Tensor:
    """Fornberg algorithm for FD weights on arbitrary grids.

    Matches finite_differences.f90 populate_fd_weights exactly.
    Bengt Fornberg, Mathematics of Computation, 51, 184, 1988, 699-706.

    Args:
        z: point where approximation is evaluated
        x: (nd+1,) grid point locations
        nd: number of grid points minus 1
        m: highest derivative order for which weights are computed

    Returns:
        c: (nd+1, m+1) weights. c[j, k] = weight for derivative k at point x[j]
    """
    c = torch.zeros(nd + 1, m + 1, dtype=DTYPE, device="cpu")
    c1 = 1.0
    c4 = x[0].item() - z
    c[0, 0] = 1.0

    for i in range(1, nd + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i].item() - z
        for j in range(i):
            c3 = x[i].item() - x[j].item()
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2

    return c


def _get_fd_grid(n: int, ricb: float, rcmb: float, ratio1: float, ratio2: float):
    """Build the FD radial grid. Matches get_FD_grid in finite_differences.f90.

    Args:
        n: n_r_max
        ricb: inner core boundary radius
        rcmb: core-mantle boundary radius
        ratio1: fd_stretch (boundary/bulk ratio)
        ratio2: fd_ratio (drMin/drMax). 1.0 = uniform grid.

    Returns:
        r: (n,) radial grid, r[0]=rcmb (outer), r[n-1]=ricb (inner)
    """
    r = torch.zeros(n, dtype=DTYPE, device="cpu")
    r[0] = rcmb

    if ratio2 == 1.0:
        # Uniform grid
        dr = 1.0 / (n - 1)
        for i in range(1, n):
            r[i] = r[i - 1] - dr
    else:
        # Stretched grid
        n_boundary_points = int((n - 1) / (2.0 * (1.0 + ratio1)))
        ratio1 = (n - 1) / (2.0 * n_boundary_points) - 1.0

        if abs(ricb) <= 10.0 * torch.finfo(DTYPE).eps:
            # Full sphere
            n_bulk_points = n - 1 - n_boundary_points
            dr_after = math.exp(math.log(ratio2) / n_boundary_points)
            dr_before = 1.0
            for _ in range(n_boundary_points):
                dr_before *= dr_after
            dr_before = 1.0 / (n_bulk_points + dr_after * (1.0 - dr_before) / (1.0 - dr_after))
        else:
            n_bulk_points = n - 1 - 2 * n_boundary_points
            dr_after = math.exp(math.log(ratio2) / n_boundary_points)
            dr_before = 1.0
            for _ in range(n_boundary_points):
                dr_before *= dr_after
            dr_before = 1.0 / (n_bulk_points + 2.0 * dr_after * (1.0 - dr_before) / (1.0 - dr_after))

        # Refine dr_before from drMax to drMin (Fortran lines 161-163)
        for _ in range(n_boundary_points):
            dr_before *= dr_after  # dr_before is now drMin

        # Outer boundary region: start with drMin (fine), grow by dividing by dr_after
        # (Fortran lines 168-171)
        for i in range(1, n_boundary_points + 1):
            r[i] = r[i - 1] - dr_before
            dr_before /= dr_after  # spacing grows each step

        # Bulk region: uniform spacing at drMax (= current dr_before)
        # (Fortran lines 173-175)
        for i in range(n_bulk_points):
            r[n_boundary_points + 1 + i] = r[n_boundary_points + i] - dr_before

        # Inner boundary region: spacing shrinks by multiplying by dr_after
        # (Fortran lines 178-183)
        if abs(ricb) > 10.0 * torch.finfo(DTYPE).eps:
            for i in range(n_boundary_points):
                dr_before *= dr_after  # spacing shrinks
                r[n_boundary_points + n_bulk_points + 1 + i] = (
                    r[n_boundary_points + n_bulk_points + i] - dr_before)

    # Fix last point to be exactly ricb
    if r[n - 1] != ricb:
        r[n - 1] = ricb

    return r


def _get_fd_coeffs(r: torch.Tensor, order: int, order_boundary: int):
    """Compute FD stencil coefficients. Matches get_FD_coeffs in finite_differences.f90.

    Returns dict of stencil arrays matching Fortran naming.
    """
    n = r.shape[0]
    half_order = order // 2

    # Step 1: 1st and 2nd derivatives in the bulk
    dr = torch.zeros(n, order + 1, dtype=DTYPE, device="cpu")
    ddr = torch.zeros(n, order + 1, dtype=DTYPE, device="cpu")

    if order == 2:
        for n_r in range(n):  # 0-based
            if n_r == 0:
                drl = r[1] - r[0]
                dr[n_r, 0] = -0.5 / drl
                dr[n_r, 1] = 0.0
                dr[n_r, 2] = 0.5 / drl
                ddr[n_r, 0] = 1.0 / drl / drl
                ddr[n_r, 1] = -2.0 / drl / drl
                ddr[n_r, 2] = 1.0 / drl / drl
            elif n_r == n - 1:
                drl = r[n - 1] - r[n - 2]
                dr[n_r, 0] = -0.5 / drl
                dr[n_r, 1] = 0.0
                dr[n_r, 2] = 0.5 / drl
                ddr[n_r, 0] = 1.0 / drl / drl
                ddr[n_r, 1] = -2.0 / drl / drl
                ddr[n_r, 2] = 1.0 / drl / drl
            else:
                spacing = torch.tensor([r[n_r - 1] - r[n_r], 0.0, r[n_r + 1] - r[n_r]], dtype=DTYPE)
                c = populate_fd_weights(0.0, spacing, 2, 2)
                dr[n_r, :] = c[:, 1]
                ddr[n_r, :] = c[:, 2]
    else:
        for n_r in range(half_order, n - half_order):
            spacing = torch.tensor(
                [r[n_r - half_order + od] - r[n_r] for od in range(order + 1)], dtype=DTYPE)
            c = populate_fd_weights(0.0, spacing, order, order)
            dr[n_r, :] = c[:, 1]
            ddr[n_r, :] = c[:, 2]

    # Step 2: 1st derivative boundary stencils (top)
    dr_top = torch.zeros(half_order, order_boundary + 1, dtype=DTYPE, device="cpu")
    for n_r in range(half_order):  # 0-based: rows 0..half_order-1
        spacing = torch.tensor(
            [r[od] - r[n_r] for od in range(order_boundary + 1)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary, order_boundary)
        dr_top[n_r, :] = c[:, 1]

    # Step 3: 1st derivative boundary stencils (bottom)
    dr_bot = torch.zeros(half_order, order_boundary + 1, dtype=DTYPE, device="cpu")
    for n_r in range(half_order):
        spacing = torch.tensor(
            [r[n - 1 - od] - r[n - 1 - n_r] for od in range(order_boundary + 1)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary, order_boundary)
        dr_bot[n_r, :] = c[:, 1]

    # Step 4: 2nd derivative boundary stencils (top)
    ddr_top = torch.zeros(half_order, order_boundary + 2, dtype=DTYPE, device="cpu")
    for n_r in range(half_order):
        spacing = torch.tensor(
            [r[od] - r[n_r] for od in range(order_boundary + 2)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 1, order_boundary + 1)
        ddr_top[n_r, :] = c[:, 2]

    # Step 5: 2nd derivative boundary stencils (bottom)
    ddr_bot = torch.zeros(half_order, order_boundary + 2, dtype=DTYPE, device="cpu")
    for n_r in range(half_order):
        spacing = torch.tensor(
            [r[n - 1 - od] - r[n - 1 - n_r] for od in range(order_boundary + 2)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 1, order_boundary + 1)
        ddr_bot[n_r, :] = c[:, 2]

    # Step 6: 3rd and 4th derivatives in the bulk
    dddr = torch.zeros(n, order + 3, dtype=DTYPE, device="cpu")
    ddddr = torch.zeros(n, order + 3, dtype=DTYPE, device="cpu")

    if order == 2:
        for n_r in range(1, n - 1):  # 0-based: rows 1..n-2 (Fortran 2..n_max-1)
            spacing = torch.zeros(5, dtype=DTYPE)
            for od in range(5):
                if n_r == 1 and od == 0:
                    spacing[od] = 2.0 * (r[0] - r[1])  # symmetric ghost
                elif n_r == n - 2 and od == 4:
                    spacing[od] = 2.0 * (r[n - 1] - r[n - 2])  # symmetric ghost
                else:
                    spacing[od] = r[n_r - 2 + od] - r[n_r]
            c = populate_fd_weights(0.0, spacing, 4, 4)
            dddr[n_r, :] = c[:, 3]
            ddddr[n_r, :] = c[:, 4]
    else:
        for n_r in range(1 + half_order, n - half_order - 1):
            spacing = torch.tensor(
                [r[n_r - half_order - 1 + od] - r[n_r] for od in range(order + 3)], dtype=DTYPE)
            c = populate_fd_weights(0.0, spacing, order + 2, order + 2)
            dddr[n_r, :] = c[:, 3]
            ddddr[n_r, :] = c[:, 4]

    # Steps 7-10: 3rd and 4th derivative boundary stencils
    dddr_top = torch.zeros(half_order + 1, order_boundary + 3, dtype=DTYPE, device="cpu")
    dddr_bot = torch.zeros(half_order + 1, order_boundary + 3, dtype=DTYPE, device="cpu")
    ddddr_top = torch.zeros(half_order + 1, order_boundary + 4, dtype=DTYPE, device="cpu")
    ddddr_bot = torch.zeros(half_order + 1, order_boundary + 4, dtype=DTYPE, device="cpu")

    for n_r in range(half_order + 1):
        # Step 7: 3rd deriv top
        spacing = torch.tensor(
            [r[od] - r[n_r] for od in range(order_boundary + 3)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 2, order_boundary + 2)
        dddr_top[n_r, :] = c[:, 3]

        # Step 8: 3rd deriv bottom
        spacing = torch.tensor(
            [r[n - 1 - od] - r[n - 1 - n_r] for od in range(order_boundary + 3)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 2, order_boundary + 2)
        dddr_bot[n_r, :] = c[:, 3]

        # Step 9: 4th deriv top
        spacing = torch.tensor(
            [r[od] - r[n_r] for od in range(order_boundary + 4)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 3, 4)
        ddddr_top[n_r, :] = c[:, 4]

        # Step 10: 4th deriv bottom
        spacing = torch.tensor(
            [r[n - 1 - od] - r[n - 1 - n_r] for od in range(order_boundary + 4)], dtype=DTYPE)
        c = populate_fd_weights(0.0, spacing, order_boundary + 3, 4)
        ddddr_bot[n_r, :] = c[:, 4]

    return {
        "dr": dr, "ddr": ddr, "dddr": dddr, "ddddr": ddddr,
        "dr_top": dr_top, "dr_bot": dr_bot,
        "ddr_top": ddr_top, "ddr_bot": ddr_bot,
        "dddr_top": dddr_top, "dddr_bot": dddr_bot,
        "ddddr_top": ddddr_top, "ddddr_bot": ddddr_bot,
    }


def _build_fd_der_mats(r: torch.Tensor, coeffs: dict, order: int, order_boundary: int):
    """Build dense derivative matrices from FD stencils.

    Matches get_der_mat in finite_differences.f90.

    Returns (rMat, drMat, d2rMat, d3rMat, d4rMat), each (n, n).
    """
    n = r.shape[0]
    half_order = order // 2

    rMat = torch.eye(n, dtype=DTYPE, device="cpu")
    drMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    d2rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    d3rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")
    d4rMat = torch.zeros(n, n, dtype=DTYPE, device="cpu")

    dr = coeffs["dr"]
    ddr = coeffs["ddr"]
    dddr = coeffs["dddr"]
    ddddr = coeffs["ddddr"]

    # Bulk points for 1st and 2nd derivatives
    for n_r in range(half_order, n - half_order):
        drMat[n_r, n_r - half_order:n_r + half_order + 1] = dr[n_r]
        d2rMat[n_r, n_r - half_order:n_r + half_order + 1] = ddr[n_r]

    # Bulk points for 3rd and 4th derivatives
    for n_r in range(1 + half_order, n - half_order - 1):
        d3rMat[n_r, n_r - half_order - 1:n_r + half_order + 2] = dddr[n_r]
        d4rMat[n_r, n_r - half_order - 1:n_r + half_order + 2] = ddddr[n_r]

    # Boundary points for 1st derivative
    # Fortran bottom stencils store coefficients in reversed column order:
    # drMat(n_max-n_r+1, n_max:n_max-order_boundary:-1) = dr_bot(n_r,:)
    # In 0-based: drMat[n-1-n_r, n-1], drMat[n-1-n_r, n-2], ... = dr_bot[n_r, 0], dr_bot[n_r, 1], ...
    for n_r in range(half_order):
        drMat[n_r, :order_boundary + 1] = coeffs["dr_top"][n_r]
        bot_cols = order_boundary + 1
        drMat[n - 1 - n_r, n - bot_cols:n] = coeffs["dr_bot"][n_r].flip(0)

    # Boundary points for 2nd derivative
    for n_r in range(half_order):
        d2rMat[n_r, :order_boundary + 2] = coeffs["ddr_top"][n_r]
        bot_cols = order_boundary + 2
        d2rMat[n - 1 - n_r, n - bot_cols:n] = coeffs["ddr_bot"][n_r].flip(0)

    # Boundary points for 3rd derivative
    for n_r in range(half_order + 1):
        d3rMat[n_r, :order_boundary + 3] = coeffs["dddr_top"][n_r]
        bot_cols = order_boundary + 3
        d3rMat[n - 1 - n_r, n - bot_cols:n] = coeffs["dddr_bot"][n_r].flip(0)

    # Boundary points for 4th derivative
    for n_r in range(half_order + 1):
        d4rMat[n_r, :order_boundary + 4] = coeffs["ddddr_top"][n_r]
        bot_cols = order_boundary + 4
        d4rMat[n - 1 - n_r, n - bot_cols:n] = coeffs["ddddr_bot"][n_r].flip(0)

    return rMat, drMat, d2rMat, d3rMat, d4rMat


# ---- Module-level initialization ----

# Import FD parameters
from .params import (
    fd_order as _fd_order, fd_order_bound as _fd_order_bound,
    fd_stretch as _fd_stretch, fd_ratio as _fd_ratio,
)

# Boundary radii (same as Chebyshev)
one = 1.0
r_cmb = one / (one - radratio)
r_icb = r_cmb - one

# FD-specific constants
rnorm = 1.0
boundary_fac = 1.0

# Build grid
r = _get_fd_grid(n_r_max, r_icb, r_cmb, _fd_stretch, _fd_ratio)

# Build stencils and derivative matrices
_coeffs = _get_fd_coeffs(r, _fd_order, _fd_order_bound)
rMat, drMat, d2rMat, d3rMat, d4rMat = _build_fd_der_mats(
    r, _coeffs, _fd_order, _fd_order_bound)

# FD doesn't use mapping derivatives — set to 1 and 0
drx = torch.ones(n_r_max, dtype=DTYPE, device="cpu")
ddrx = torch.zeros(n_r_max, dtype=DTYPE, device="cpu")
dddrx = torch.zeros(n_r_max, dtype=DTYPE, device="cpu")

# Boundary derivative vectors (for robin_bc compatibility)
# For FD, these come from dr_top/dr_bot stencils
dr_top_vec = torch.zeros(n_r_max, dtype=DTYPE, device="cpu")
dr_bot_vec = torch.zeros(n_r_max, dtype=DTYPE, device="cpu")
dr_top_vec[:_fd_order_bound + 1] = _coeffs["dr_top"][0]
bot_cols = _fd_order_bound + 1
dr_bot_vec[n_r_max - bot_cols:n_r_max] = _coeffs["dr_bot"][0].flip(0)

# Transfer to device
r = r.to(DEVICE)
rMat = rMat.to(DEVICE)
drMat = drMat.to(DEVICE)
d2rMat = d2rMat.to(DEVICE)
d3rMat = d3rMat.to(DEVICE)
d4rMat = d4rMat.to(DEVICE)
drx = drx.to(DEVICE)
ddrx = ddrx.to(DEVICE)
dddrx = dddrx.to(DEVICE)
dr_top = dr_top_vec.to(DEVICE)
dr_bot = dr_bot_vec.to(DEVICE)
