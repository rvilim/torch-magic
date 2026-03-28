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


def chunked_solve_complex(inv_by_l, l_index, rhs_complex, chunk_size=512):
    """Batched solve using unique per-l real inverses, chunked to bound memory.

    Instead of precomputing (lm_max, N, N) expanded inverses, stores only
    (l_max+1, N, N) unique inverses and expands in chunks at solve time.

    Uses view_as_real/view_as_complex for float64 bmm (no complex gemm).

    Args:
        inv_by_l: (l_max+1, N, N) float64 — unique inverse per l degree
        l_index: (lm_max,) long — maps each lm mode to its l value
        rhs_complex: (lm_max, N) complex — right-hand side
        chunk_size: max lm modes per chunk (bounds peak memory)

    Returns:
        (lm_max, N) complex — solution
    """
    rhs_real = torch.view_as_real(rhs_complex)  # (lm_max, N, 2) — zero-copy
    lm_max = rhs_real.shape[0]

    if lm_max <= chunk_size:
        # Single pass — no loop, same as before
        out_real = torch.bmm(inv_by_l[l_index], rhs_real)
        return torch.view_as_complex(out_real)

    out_real = torch.empty_like(rhs_real)
    for start in range(0, lm_max, chunk_size):
        end = min(start + chunk_size, lm_max)
        out_real[start:end] = torch.bmm(
            inv_by_l[l_index[start:end]], rhs_real[start:end]
        )
    return torch.view_as_complex(out_real)
