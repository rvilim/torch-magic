"""Linear algebra routines matching algebra.f90 (hand-rolled LU).

Only ports routines used by the Chebyshev benchmark path:
- prepare_mat: Dense LU factorization with partial pivoting
- solve_mat: Forward/backward substitution (real or complex RHS)

The factorization stores 1/a[i,i] on the diagonal (non-standard).
Pivot ordering matches Fortran exactly for bit-reproducibility.
"""

import torch

from .precision import DTYPE, DEVICE

ZERO_TOLERANCE = 1.0e-15


def prepare_mat(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """LU decomposition of dense real matrix via Gaussian elimination.

    Matches algebra.f90 prepare_mat exactly. Stores 1/a[i,i] on diagonal.

    Args:
        a: (n, n) real matrix (will be copied, not modified in-place)

    Returns:
        (a_lu, ip, info): LU-decomposed matrix, pivot array (0-based), error flag
    """
    a = a.clone()
    n = a.shape[0]
    ip = torch.zeros(n, dtype=torch.long, device=a.device)
    info = 0

    for k in range(n - 1):
        # Find pivot: max |a[k:n, k]|
        col_below = a[k:n, k].abs()
        l_rel = col_below.argmax().item()
        l = k + l_rel  # absolute row index (0-based)

        ip[k] = l

        if a[l, k].abs().item() > ZERO_TOLERANCE:
            # Swap rows k and l (full row swap)
            if l != k:
                a[[k, l], :] = a[[l, k], :]

            # Compute multipliers
            a[k + 1:n, k] = a[k + 1:n, k] / a[k, k]

            # Eliminate: a[i,j] -= a[k,j] * a[i,k] for i,j in [k+1, n)
            # Outer product update: A[k+1:,k+1:] -= A[k+1:,k] * A[k,k+1:]
            a[k + 1:n, k + 1:n] -= a[k + 1:n, k:k + 1] @ a[k:k + 1, k + 1:n]
        else:
            info = k + 1  # 1-based error index like Fortran

    ip[n - 1] = n - 1

    if a[n - 1, n - 1].abs().item() <= ZERO_TOLERANCE:
        info = n

    if info > 0:
        return a, ip, info

    # Store 1/diagonal
    diag_idx = torch.arange(n, device=a.device)
    a[diag_idx, diag_idx] = 1.0 / a[diag_idx, diag_idx]

    return a, ip, info


def solve_mat_complex(a: torch.Tensor, ip: torch.Tensor,
                      bc1: torch.Tensor) -> torch.Tensor:
    """Backward substitution for complex RHS into LU-decomposed real matrix.

    Matches solve_mat_complex_rhs in algebra.f90 exactly.

    Args:
        a: (n, n) LU-decomposed real matrix from prepare_mat
        ip: (n,) pivot array (0-based)
        bc1: (n,) or (n, nbatch) complex RHS vector(s)

    Returns:
        Solution vector(s), same shape as bc1
    """
    batched = bc1.dim() == 2
    if not batched:
        bc1 = bc1.unsqueeze(1)

    bc1 = bc1.clone()
    n = a.shape[0]
    nm1 = n - 1
    nodd = n % 2

    # Permute RHS
    for k in range(nm1):
        m = ip[k].item()
        if m != k:
            bc1[[k, m]] = bc1[[m, k]]

    # Forward substitution: solve L * y = b
    # Process in pairs of columns (k, k+1) matching Fortran loop stride 2
    for k in range(0, n - 2, 2):
        k1 = k + 1
        bc1[k1] = bc1[k1] - bc1[k] * a[k1, k]
        # bc1[i] -= bc1[k]*a[i,k] + bc1[k1]*a[i,k1] for i in [k+2, n)
        if k + 2 < n:
            bc1[k + 2:n] -= bc1[k] * a[k + 2:n, k:k + 1] + bc1[k1] * a[k + 2:n, k1:k1 + 1]

    if nodd == 0:
        bc1[n - 1] = bc1[n - 1] - bc1[nm1 - 1] * a[n - 1, nm1 - 1]

    # Back substitution: solve U * x = y
    # Process in pairs (k, k-1) matching Fortran loop
    for k in range(n - 1, 1, -2):
        k1 = k - 1
        bc1[k] = bc1[k] * a[k, k]
        bc1[k1] = (bc1[k1] - bc1[k] * a[k1, k]) * a[k1, k1]
        if k1 > 0:
            bc1[:k1] -= bc1[k] * a[:k1, k:k + 1] + bc1[k1] * a[:k1, k1:k1 + 1]

    if nodd == 0:
        bc1[1] = bc1[1] * a[1, 1]
        bc1[0] = (bc1[0] - bc1[1] * a[0, 1]) * a[0, 0]
    else:
        bc1[0] = bc1[0] * a[0, 0]

    if not batched:
        bc1 = bc1.squeeze(1)

    return bc1


def solve_mat_real(a: torch.Tensor, ip: torch.Tensor,
                   b: torch.Tensor) -> torch.Tensor:
    """Backward substitution for real RHS into LU-decomposed real matrix.

    Matches solve_mat_real_rhs in algebra.f90 exactly.

    Args:
        a: (n, n) LU-decomposed real matrix from prepare_mat
        ip: (n,) pivot array (0-based)
        b: (n,) or (n, nbatch) real RHS vector(s)

    Returns:
        Solution vector(s), same shape as b
    """
    batched = b.dim() == 2
    if not batched:
        b = b.unsqueeze(1)

    b = b.clone()
    n = a.shape[0]
    nm1 = n - 1
    nodd = n % 2

    # Permute RHS
    for k in range(nm1):
        m = ip[k].item()
        if m != k:
            b[[k, m]] = b[[m, k]]

    # Forward substitution: solve L * y = b
    for k in range(0, n - 2, 2):
        k1 = k + 1
        b[k1] = b[k1] - b[k] * a[k1, k]
        if k + 2 < n:
            b[k + 2:n] -= b[k] * a[k + 2:n, k:k + 1] + b[k1] * a[k + 2:n, k1:k1 + 1]

    if nodd == 0:
        b[n - 1] = b[n - 1] - b[nm1 - 1] * a[n - 1, nm1 - 1]

    # Back substitution: solve U * x = y
    for k in range(n - 1, 1, -2):
        k1 = k - 1
        b[k] = b[k] * a[k, k]
        b[k1] = (b[k1] - b[k] * a[k1, k]) * a[k1, k1]
        if k1 > 0:
            b[:k1] -= b[k] * a[:k1, k:k + 1] + b[k1] * a[:k1, k1:k1 + 1]

    if nodd == 0:
        b[1] = b[1] * a[1, 1]
        b[0] = (b[0] - b[1] * a[0, 1]) * a[0, 0]
    else:
        b[0] = b[0] * a[0, 0]

    if not batched:
        b = b.squeeze(1)

    return b


def solve_mat(a: torch.Tensor, ip: torch.Tensor,
              rhs: torch.Tensor) -> torch.Tensor:
    """Solve A*x = rhs using LU-decomposed matrix. Auto-dispatches real/complex."""
    if rhs.is_complex():
        return solve_mat_complex(a, ip, rhs)
    else:
        return solve_mat_real(a, ip, rhs)


def _build_pack_indices(l_index, device=None):
    """Precompute packing/unpacking indices for packed bmm solver.

    Instead of expanding (l_max+1, N, N) → (lm_max, N, N) via l_index,
    we pack the RHS into (l_max+1, N, max_modes*2) and do a single bmm
    of shape (l_max+1, N, N) @ (l_max+1, N, max_modes*2).

    Index tensors are moved to `device` to avoid per-call CPU→GPU transfers.
    """
    l_cpu = l_index.cpu()
    l_max_plus_1 = l_cpu.max().item() + 1
    lm_max = l_cpu.shape[0]

    counts = torch.bincount(l_cpu, minlength=l_max_plus_1)
    max_modes = counts.max().item()

    # Sort lm indices by l value for contiguous packing
    sorted_idx = l_cpu.argsort(stable=True)
    offsets = torch.cumsum(counts, 0) - counts

    # Build flat pack index: pack_idx[i] = l_sorted[i] * max_modes + pos_in_group[i]
    # pos_in_group = arange within each l-group
    l_sorted = l_cpu[sorted_idx]
    pos_in_group = torch.zeros(lm_max, dtype=torch.long)
    for l in range(l_max_plus_1):
        o = offsets[l].item()
        c = counts[l].item()
        if c > 0:
            pos_in_group[o:o + c] = torch.arange(c)
    pack_idx = l_sorted * max_modes + pos_in_group  # (lm_max,)

    # Combined scatter/gather index: maps lm-order directly to packed position
    # Eliminates separate sort + scatter steps (2 ops → 1 op each direction)
    combined_idx = torch.empty(lm_max, dtype=torch.long)
    combined_idx[sorted_idx] = pack_idx  # combined_idx[lm_mode] = packed_position

    # Move index tensors to target device (avoids CPU→GPU transfer per call)
    if device is not None:
        combined_idx = combined_idx.to(device)

    return {
        "combined_idx": combined_idx,
        "max_modes": max_modes,
        "l_max_plus_1": l_max_plus_1,
        "lm_max": lm_max,
    }


# Cache for solver state: pack indices + pre-allocated buffers
# Keyed by (l_index data_ptr, N_rhs_cols) to support different RHS widths
_solver_cache: dict[tuple[int, int], dict] = {}


def _get_solver_state(inv_by_l, l_index, N_cols):
    """Get or build cached solver state (indices + buffers) for given configuration."""
    key = (l_index.data_ptr(), N_cols)
    if key not in _solver_cache:
        device = inv_by_l.device
        pi = _build_pack_indices(l_index, device=device)
        L = pi["l_max_plus_1"]
        max_m = pi["max_modes"]
        lm = pi["lm_max"]
        dtype = inv_by_l.dtype
        # Pre-allocate input packing buffer on device (output not pre-allocated:
        # empty_like is cheaper than clone since it skips zeroing)
        pi["rhs_packed_flat"] = torch.zeros(L * max_m, N_cols, 2, dtype=dtype, device=device)
        _solver_cache[key] = pi
    return _solver_cache[key]


def chunked_solve_complex(inv_by_l, l_index, rhs_complex, chunk_size=512):
    """Batched solve using packed bmm for GPU efficiency.

    Packs RHS by l-degree into (l_max+1, N, max_modes*2) and does a single
    bmm of (l_max+1, N, N) @ (l_max+1, N, max_modes*2), avoiding the
    expensive expansion of inv_by_l from (l_max+1) to (lm_max) matrices.

    Uses view_as_real/view_as_complex for float64 bmm (no complex gemm).
    Pre-allocated buffers and device-resident combined index minimize
    MPS dispatch overhead (single scatter + single gather, no sort step).

    Args:
        inv_by_l: (l_max+1, N, N) float64 — unique inverse per l degree
        l_index: (lm_max,) long — maps each lm mode to its l value
        rhs_complex: (lm_max, N) complex — right-hand side

    Returns:
        (lm_max, N) complex — solution
    """
    rhs_real = torch.view_as_real(rhs_complex)  # (lm_max, N, 2) — zero-copy
    N = rhs_real.shape[1]

    ss = _get_solver_state(inv_by_l, l_index, N)
    L = ss["l_max_plus_1"]
    max_m = ss["max_modes"]
    cidx = ss["combined_idx"]  # device tensor: lm-order → packed position
    rhs_packed_flat = ss["rhs_packed_flat"]  # pre-allocated (L*max_m, N, 2)

    # Pack: scatter directly from lm-order to packed layout (1 op, not 2)
    rhs_packed_flat[cidx] = rhs_real
    rhs_packed = rhs_packed_flat.reshape(L, max_m, N, 2).permute(0, 2, 1, 3).reshape(L, N, max_m * 2)

    # Single bmm: (L, N, N) @ (L, N, max_m*2) → (L, N, max_m*2)
    out_packed = torch.bmm(inv_by_l, rhs_packed)

    # Unpack: gather directly from packed layout to lm-order (1 op, not 2)
    out_flat = out_packed.reshape(L, N, max_m, 2).permute(0, 2, 1, 3).reshape(L * max_m, N, 2)
    out_real = out_flat[cidx]  # new tensor (no clone needed)

    return torch.view_as_complex(out_real)
