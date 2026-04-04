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


def chunked_lu_solve_complex(lu_by_l, pivots_by_l, fac_row_by_l, fac_col_by_l,
                             l_index, rhs_complex):
    """Batched LU solve with preconditioning, using packed layout.

    Same packing as chunked_solve_complex but uses torch.linalg.lu_solve
    instead of bmm with precomputed inverse. Gives LU-level accuracy
    (backward error O(eps)) for ill-conditioned systems where the
    precomputed-inverse approach loses digits.

    The view_as_real trick works because the WP matrix is real: for real A,
    lu_solve(A, [br, bi]) = [lu_solve(A, br), lu_solve(A, bi)].

    Args:
        lu_by_l: (l_max+1, N, N) float64 — LU factors from torch.linalg.lu_factor
        pivots_by_l: (l_max+1, N) int32 — pivot indices
        fac_row_by_l: (l_max+1, N) float64 — row preconditioning factors
        fac_col_by_l: (l_max+1, N) float64 — column preconditioning factors
        l_index: (lm_max,) long — maps each lm mode to its l value
        rhs_complex: (lm_max, N) complex — right-hand side

    Returns:
        (lm_max, N) complex — solution
    """
    rhs_real = torch.view_as_real(rhs_complex)  # (lm_max, N, 2) — zero-copy
    N = rhs_real.shape[1]

    ss = _get_solver_state(lu_by_l, l_index, N)
    L = ss["l_max_plus_1"]
    max_m = ss["max_modes"]
    cidx = ss["combined_idx"]
    rhs_packed_flat = ss["rhs_packed_flat"]  # pre-allocated (L*max_m, N, 2)

    # Apply row preconditioning: rhs_precond[lm, :] = fac_row[l[lm], :] * rhs[lm, :]
    fac_row_lm = fac_row_by_l[l_index]  # (lm_max, N)
    rhs_precond = rhs_real * fac_row_lm.unsqueeze(2)  # (lm_max, N, 2)

    # Pack: scatter directly from lm-order to packed layout
    rhs_packed_flat[cidx] = rhs_precond
    rhs_packed = rhs_packed_flat.reshape(L, max_m, N, 2).permute(0, 2, 1, 3).reshape(L, N, max_m * 2)

    # Batched LU solve: (L, N, N) LU factors, (L, N, max_m*2) RHS
    out_packed = torch.linalg.lu_solve(lu_by_l, pivots_by_l, rhs_packed)

    # Unpack: gather directly from packed layout to lm-order
    out_flat = out_packed.reshape(L, N, max_m, 2).permute(0, 2, 1, 3).reshape(L * max_m, N, 2)
    out_real = out_flat[cidx]  # (lm_max, N, 2)

    # Apply column preconditioning: x[lm, :] = fac_col[l[lm], :] * x_precond[lm, :]
    fac_col_lm = fac_col_by_l[l_index]  # (lm_max, N)
    out_real = out_real * fac_col_lm.unsqueeze(2)

    return torch.view_as_complex(out_real.contiguous())


# ===========================================================================
# Pivoted banded LU (matching Fortran algebra.f90 prepare_band/solve_band)
# ===========================================================================

def prepare_band(abd: torch.Tensor, n: int, kl: int, ku: int):
    """Pivoted banded LU factorization matching Fortran LINPACK dgbfa.

    All indexing follows Fortran's 1-based convention internally,
    translated to 0-based for Python tensor access.

    Args:
        abd: (2*kl+ku+1, n) band matrix in LAPACK storage (MODIFIED in-place).
             Row kl+ku (0-based) is the diagonal. First kl rows are fill-in workspace.
        n: matrix dimension
        kl: number of lower diagonals
        ku: number of upper diagonals

    Returns:
        abd: factored (in-place)
        pivot: (n,) int pivot indices (1-based, matching Fortran)
        info: 0 if success, k (1-based) if singular
    """
    abd = abd.clone()
    m = kl + ku + 1  # 1-based diagonal row = m; 0-based = m-1
    pivot = torch.zeros(n, dtype=torch.long, device=abd.device)
    info = 0

    # Zero fill-in region (Fortran lines 505-514, 1-based j0=ku+2, j1=min(n,m)-1)
    j0 = ku + 2  # 1-based
    j1 = min(n, m) - 1  # 1-based
    if j1 >= j0:
        for jz in range(j0, j1 + 1):  # 1-based jz = j0..j1
            i0 = m + 1 - jz  # 1-based
            for i in range(i0, kl + 1):  # 1-based i = i0..kl
                abd[i - 1, jz - 1] = 0.0  # 0-based

    jz = j1  # 1-based
    ju = 0  # 1-based

    # Gaussian elimination
    nm1 = n - 1
    if nm1 >= 1:
        for k in range(1, nm1 + 1):  # 1-based k = 1..nm1
            kp1 = k + 1

            jz = jz + 1
            if jz <= n and kl >= 1:
                for i in range(1, kl + 1):  # 1-based i = 1..kl
                    abd[i - 1, jz - 1] = 0.0

            lm = min(kl, n - k)
            # Find pivot: max |abd(m:m+lm, k)| (1-based row indices)
            # 0-based: abd[m-1:m-1+lm+1, k-1]
            col_slice = abd[m - 1:m + lm, k - 1]
            l_rel = col_slice.abs().argmax().item()
            l = l_rel + m  # 1-based row index in abd

            pivot[k - 1] = l + k - m  # 1-based pivot (Fortran convention)

            if abd[l - 1, k - 1].abs().item() > ZERO_TOLERANCE:
                if l != m:
                    # Swap abd[l,k] and abd[m,k] (1-based)
                    t = abd[l - 1, k - 1].clone()
                    abd[l - 1, k - 1] = abd[m - 1, k - 1].clone()
                    abd[m - 1, k - 1] = t

                # Compute multipliers: abd(m+1:m+lm, k) *= -1/abd(m,k)
                t = -1.0 / abd[m - 1, k - 1]
                abd[m:m + lm, k - 1] *= t  # 0-based: rows m to m+lm-1

                # Row elimination
                ju = min(max(ju, ku + pivot[k - 1].item()), n)  # 1-based
                mm = m  # 1-based
                if ju >= kp1:
                    for j in range(kp1, ju + 1):  # 1-based j = kp1..ju
                        l = l - 1
                        mm = mm - 1
                        t = abd[l - 1, j - 1].clone()
                        if l != mm:
                            abd[l - 1, j - 1] = abd[mm - 1, j - 1].clone()
                            abd[mm - 1, j - 1] = t
                        # abd(mm+1:mm+lm, j) += t * abd(m+1:m+lm, k) (1-based)
                        abd[mm:mm + lm, j - 1] += t * abd[m:m + lm, k - 1]
            else:
                info = k  # 1-based

    pivot[n - 1] = n  # 1-based
    if abd[m - 1, n - 1].abs().item() <= ZERO_TOLERANCE:
        info = n

    return abd, pivot, info


def solve_band_real(abd: torch.Tensor, n: int, kl: int, ku: int,
                    pivot: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve banded system using pivoted LU from prepare_band.

    Matches Fortran solve_band_real_rhs exactly. Handles real or complex RHS.

    Args:
        abd: (2*kl+ku+1, n) LU-factored band from prepare_band
        n: matrix dimension
        kl: number of lower diagonals
        ku: number of upper diagonals
        pivot: (n,) pivot indices (1-based)
        rhs: (n,) or (n, nrhs) RHS (real or complex)

    Returns:
        Solution, same shape as rhs
    """
    batched = rhs.dim() == 2
    if not batched:
        rhs = rhs.unsqueeze(1)
    rhs = rhs.clone()
    m = ku + kl + 1  # 1-based diagonal row

    # Forward solve: Ly = rhs (1-based k = 1..n-1)
    if kl != 0:
        for k in range(1, n):  # 1-based k = 1..nm1
            lm = min(kl, n - k)
            l = pivot[k - 1].item()  # 1-based
            if l != k:
                rhs[[k - 1, l - 1]] = rhs[[l - 1, k - 1]]
            # rhs(k+1:k+lm) += rhs(k) * abd(m+1:m+lm, k) (1-based)
            rhs[k:k + lm] += rhs[k - 1:k] * abd[m:m + lm, k - 1:k]

    # Backward solve: Ux = y (1-based k = n..1)
    for k in range(n, 0, -1):
        rhs[k - 1] = rhs[k - 1] / abd[m - 1, k - 1]
        lm = min(k, m) - 1
        la = m - lm  # 1-based
        lb = k - lm  # 1-based
        if lm > 0:
            t = -rhs[k - 1:k]
            # rhs(lb:lb+lm-1) += t * abd(la:la+lm-1, k) (1-based)
            rhs[lb - 1:lb - 1 + lm] += t * abd[la - 1:la - 1 + lm, k - 1:k]

    if not batched:
        rhs = rhs.squeeze(1)
    return rhs


# ===========================================================================
# Bordered-band solver (OC banded + IC dense, matching Fortran type_bordmat)
# ===========================================================================

def prepare_bordered(dat_full, ncol, nfull, kl, ku):
    """Prepare a bordered-band system for solving.

    Matches Fortran prepare_bordered in algebra.f90. The system is:
        [ A1  A2 ] [ x1 ]   [ b1 ]
        [ A3  A4 ] [ x2 ] = [ b2 ]
    where A1 is banded (ncol × ncol), A2 is dense (ncol × nfull),
    A3 is a single row vector (ncol,), A4 is dense (nfull × nfull).

    Args:
        dat_full: (ncol+nfull, ncol+nfull) full preconditioned dense matrix
        ncol: size of the banded block (= n_r_max)
        nfull: size of the dense block (= n_r_ic_max)
        kl: lower bandwidth
        ku: upper bandwidth

    Returns:
        dict with factored components:
            A1_abd: (2*kl+ku+1, ncol) factored band
            A1_piv: (ncol,) pivots
            A2_V: (ncol, nfull) = A1^{-1} @ A2 (V overwrites A2)
            A3: (ncol,) border row vector
            A4_lu: (nfull, nfull) factored Schur complement
            A4_ip: (nfull,) pivots
    """
    # Extract blocks from the full matrix
    A1_dense = dat_full[:ncol, :ncol]
    A2 = dat_full[:ncol, ncol:].clone()
    A3 = dat_full[ncol, :ncol].clone()  # single row
    A4 = dat_full[ncol:, ncol:].clone()

    # Step 1: LU-factorize A1 as banded
    A1_abd = dense_to_band_storage(A1_dense, kl, ku)
    A1_abd, A1_piv, info = prepare_band(A1_abd, ncol, kl, ku)
    assert info == 0, f"Singular A1 in bordered matrix, info={info}"

    # Step 2: Solve A1 * V = A2, overwriting A2 with V
    for j in range(nfull):
        A2[:, j] = solve_band_real(A1_abd, ncol, kl, ku, A1_piv, A2[:, j])

    # Step 3: Schur complement: A4[0,:] -= A3 @ V
    A4[0, :] -= A3 @ A2

    # Step 4: LU-factorize A4 as dense
    A4_lu, A4_ip, info = prepare_mat(A4)
    assert info == 0, f"Singular Schur complement in bordered matrix, info={info}"

    return {
        "A1_abd": A1_abd, "A1_piv": A1_piv,
        "V": A2,  # A2 now stores V = A1^{-1} @ A2
        "A3": A3,
        "A4_lu": A4_lu, "A4_ip": A4_ip,
        "ncol": ncol, "nfull": nfull,
        "kl": kl, "ku": ku,
    }


def solve_bordered(factored, rhs):
    """Solve a bordered-band system using factored components.

    Matches Fortran solve_bordered in algebra.f90. Handles real or complex RHS.

    Args:
        factored: dict from prepare_bordered
        rhs: (ncol+nfull,) or (ncol+nfull, nrhs) RHS

    Returns:
        solution, same shape as rhs
    """
    ncol = factored["ncol"]
    nfull = factored["nfull"]
    kl = factored["kl"]
    ku = factored["ku"]

    batched = rhs.dim() == 2
    if not batched:
        rhs = rhs.unsqueeze(1)
    rhs = rhs.clone()

    # Step 1: Band solve: rhs[:ncol] = A1^{-1} @ rhs[:ncol]
    rhs[:ncol] = solve_band_real(
        factored["A1_abd"], ncol, kl, ku, factored["A1_piv"], rhs[:ncol])

    # Step 2: Border update: rhs[ncol] -= A3 · rhs[:ncol]
    # A3 is real (ncol,), rhs[:ncol] may be complex (ncol, nrhs)
    A3 = factored["A3"].to(rhs.dtype)
    rhs[ncol:ncol + 1] -= (A3 @ rhs[:ncol]).unsqueeze(0)

    # Step 3: Dense solve: rhs[ncol:] = A4^{-1} @ rhs[ncol:]
    for j in range(rhs.shape[1]):
        col = rhs[ncol:, j]
        if col.is_complex():
            rhs[ncol:, j] = solve_mat_complex(
                factored["A4_lu"], factored["A4_ip"], col)
        else:
            rhs[ncol:, j] = solve_mat_real(
                factored["A4_lu"], factored["A4_ip"], col)

    # Step 4: Back-substitute: rhs[:ncol] -= V @ rhs[ncol:]
    V = factored["V"].to(rhs.dtype)
    rhs[:ncol] -= V @ rhs[ncol:]

    if not batched:
        rhs = rhs.squeeze(1)
    return rhs


# ===========================================================================
# Per-l banded solve dispatch (for FD radial scheme)
# ===========================================================================

def extract_tridiag(dat: torch.Tensor):
    """Extract tridiagonal bands from a dense matrix.

    Args:
        dat: (N, N) dense matrix

    Returns:
        dl: (N-1,) lower diagonal
        d:  (N,) main diagonal
        du: (N-1,) upper diagonal
    """
    N = dat.shape[0]
    d = dat[range(N), range(N)].clone()
    dl = dat[range(1, N), range(N - 1)].clone()
    du = dat[range(N - 1), range(1, N)].clone()
    return dl, d, du


def extract_pentadiag(dat: torch.Tensor):
    """Extract pentadiagonal bands from a dense matrix.

    Args:
        dat: (N, N) dense matrix

    Returns:
        dl2: (N-2,) second lower diagonal
        dl1: (N-1,) first lower diagonal
        d:   (N,) main diagonal
        du1: (N-1,) first upper diagonal
        du2: (N-2,) second upper diagonal
    """
    N = dat.shape[0]
    d = dat[range(N), range(N)].clone()
    dl1 = dat[range(1, N), range(N - 1)].clone()
    du1 = dat[range(N - 1), range(1, N)].clone()
    dl2 = dat[range(2, N), range(N - 2)].clone()
    du2 = dat[range(N - 2), range(2, N)].clone()
    return dl2, dl1, d, du1, du2


def dense_to_band_storage(A_dense: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    """Convert dense matrix to LAPACK-style band storage for prepare_band.

    Band storage: abd[kl+ku+i-j, j] = A[i, j]
    First kl rows are fill-in workspace (initially zero).
    Total rows: 2*kl+ku+1.

    Args:
        A_dense: (N, N) dense matrix
        kl: number of lower diagonals
        ku: number of upper diagonals

    Returns:
        abd: (2*kl+ku+1, N) band storage
    """
    N = A_dense.shape[0]
    # Validate: no nonzero entries outside the band
    for j in range(N):
        for i in range(N):
            if (i < j - ku or i > j + kl) and abs(A_dense[i, j].item()) > 1e-14:
                raise ValueError(
                    f"dense_to_band_storage: nonzero entry A[{i},{j}]="
                    f"{A_dense[i,j].item():.3e} outside band kl={kl}, ku={ku}")
    abd = torch.zeros(2 * kl + ku + 1, N, dtype=A_dense.dtype, device=A_dense.device)
    for j in range(N):
        for i in range(max(0, j - ku), min(N, j + kl + 1)):
            abd[kl + ku + i - j, j] = A_dense[i, j]
    return abd


def banded_solve_by_l(abd_by_l, pivot_by_l, l_index, rhs_complex,
                      n, kl, ku, fac_row_by_l=None, fac_col_by_l=None):
    """Solve banded systems grouped by l-degree using pivoted banded LU.

    For each l value, all lm modes with that l share the same banded LU factors.
    Groups RHS by l-degree, applies preconditioning, solves, and scatters back.

    Preconditioning is applied INSIDE the per-l loop to avoid O(lm_max * N)
    intermediate tensors.

    Args:
        abd_by_l: (l_max+1, 2*kl+ku+1, N) — pre-factored band matrices per l
        pivot_by_l: (l_max+1, N) — pivot indices per l (1-based)
        l_index: (lm_max,) long — l value per lm mode
        rhs_complex: (lm_max, N) complex RHS
        n: matrix dimension
        kl: number of lower diagonals
        ku: number of upper diagonals
        fac_row_by_l: optional (l_max+1, N) — row preconditioning per l
        fac_col_by_l: optional (l_max+1, N) — column preconditioning per l

    Returns:
        (lm_max, N) complex solution
    """
    sol = torch.zeros_like(rhs_complex)
    l_max_p1 = l_index.max().item() + 1

    for l in range(l_max_p1):
        mask = (l_index == l)
        count = mask.sum().item()
        if count == 0:
            continue

        rhs_l = rhs_complex[mask].clone()  # (count, N)

        # Apply row preconditioning: rhs = fac_row * rhs
        if fac_row_by_l is not None:
            rhs_l = fac_row_by_l[l].unsqueeze(0) * rhs_l

        # Solve each RHS column with the banded LU
        # solve_band_real handles (N,) or (N, nrhs) — transpose to (N, count)
        rhs_T = rhs_l.T  # (N, count) — complex
        sol_T = solve_band_real(abd_by_l[l], n, kl, ku, pivot_by_l[l], rhs_T)
        sol_l = sol_T.T  # (count, N)

        # Apply column post-scaling: sol = fac_col * sol
        if fac_col_by_l is not None:
            sol_l = fac_col_by_l[l].unsqueeze(0) * sol_l

        sol[mask] = sol_l

    return sol


# ===========================================================================
# Banded matrix solvers (for finite-difference radial scheme)
# ===========================================================================

def prepare_tridiag(dl: torch.Tensor, d: torch.Tensor, du: torch.Tensor):
    """LU factorization of tridiagonal matrix with partial pivoting.

    Matches algebra.f90 prepare_tridiag exactly. Modifies dl, d, du in-place
    and returns du2, pivot, info.

    Args:
        dl: (n-1,) lower diagonal (modified in-place to store L factors)
        d:  (n,) main diagonal (modified in-place to store U diagonal)
        du: (n-1,) upper diagonal (modified in-place to store U upper)

    Returns:
        du2: (n-2,) second upper diagonal (fill-in from pivoting)
        pivot: (n,) pivot indices (1-based, matching Fortran)
        info: 0 if success, k if d[k-1] is zero
    """
    dl = dl.clone()
    d = d.clone()
    du = du.clone()
    n = d.shape[0]
    du2 = torch.zeros(max(n - 2, 0), dtype=d.dtype, device=d.device)
    pivot = torch.arange(1, n + 1, dtype=torch.long, device=d.device)  # 1-based
    info = 0

    for i in range(n - 2):
        if abs(d[i].item()) >= abs(dl[i].item()):
            # No row interchange
            if abs(d[i].item()) > ZERO_TOLERANCE:
                fact = dl[i] / d[i]
                dl[i] = fact
                d[i + 1] = d[i + 1] - fact * du[i]
        else:
            # Interchange rows i and i+1
            fact = d[i] / dl[i]
            d[i] = dl[i].clone()
            dl[i] = fact
            temp = du[i].clone()
            du[i] = d[i + 1].clone()
            d[i + 1] = temp - fact * d[i + 1]
            du2[i] = du[i + 1].clone()
            du[i + 1] = -fact * du[i + 1]
            pivot[i] = i + 2  # 1-based: i+1 in Fortran → i+2 here

    # Last pair (i = n-2)
    i = n - 2
    if i >= 0:
        if abs(d[i].item()) >= abs(dl[i].item()):
            if abs(d[i].item()) > ZERO_TOLERANCE:
                fact = dl[i] / d[i]
                dl[i] = fact
                d[i + 1] = d[i + 1] - fact * du[i]
        else:
            fact = d[i] / dl[i]
            d[i] = dl[i].clone()
            dl[i] = fact
            temp = du[i].clone()
            du[i] = d[i + 1].clone()
            d[i + 1] = temp - fact * d[i + 1]
            pivot[i] = i + 2

    # Check for zero diagonal
    for i in range(n):
        if abs(d[i].item()) <= ZERO_TOLERANCE:
            info = i + 1  # 1-based
            break

    return dl, d, du, du2, pivot, info


def solve_tridiag_real(dl, d, du, du2, pivot, rhs):
    """Solve tridiagonal system using LU factors from prepare_tridiag.

    Matches algebra.f90 solve_tridiag_real_rhs exactly.

    Args:
        dl: (n-1,) L factors
        d:  (n,) U diagonal
        du: (n-1,) U upper diagonal
        du2: (n-2,) U second upper diagonal (fill-in)
        pivot: (n,) pivot indices (1-based)
        rhs: (n,) or (n, nrhs) right-hand side

    Returns:
        Solution, same shape as rhs
    """
    batched = rhs.dim() == 2
    if not batched:
        rhs = rhs.unsqueeze(1)
    rhs = rhs.clone()
    n = d.shape[0]

    # Forward sweep: solve L*y = rhs
    for i in range(n - 1):
        ip = pivot[i].item()  # 1-based
        if ip != i + 1:
            # Swap rows i and ip-1 (0-based)
            rhs[[i, ip - 1]] = rhs[[ip - 1, i]]
        rhs[i + 1] = rhs[i + 1] - dl[i] * rhs[i]

    # Backward sweep: solve U*x = y
    rhs[n - 1] = rhs[n - 1] / d[n - 1]
    if n >= 2:
        rhs[n - 2] = (rhs[n - 2] - du[n - 2] * rhs[n - 1]) / d[n - 2]
    for i in range(n - 3, -1, -1):
        rhs[i] = (rhs[i] - du[i] * rhs[i + 1] - du2[i] * rhs[i + 2]) / d[i]

    if not batched:
        rhs = rhs.squeeze(1)
    return rhs


def solve_tridiag_complex(dl, d, du, du2, pivot, rhs):
    """Solve tridiagonal system with complex RHS using real LU factors.

    Args:
        dl, d, du, du2, pivot: real LU factors from prepare_tridiag
        rhs: (n,) or (n, nrhs) complex RHS

    Returns:
        Complex solution, same shape as rhs
    """
    batched = rhs.dim() == 2
    if not batched:
        rhs = rhs.unsqueeze(1)
    rhs = rhs.clone()
    n = d.shape[0]

    # Forward sweep
    for i in range(n - 1):
        ip = pivot[i].item()
        if ip != i + 1:
            rhs[[i, ip - 1]] = rhs[[ip - 1, i]]
        rhs[i + 1] = rhs[i + 1] - dl[i] * rhs[i]

    # Backward sweep
    rhs[n - 1] = rhs[n - 1] / d[n - 1]
    if n >= 2:
        rhs[n - 2] = (rhs[n - 2] - du[n - 2] * rhs[n - 1]) / d[n - 2]
    for i in range(n - 3, -1, -1):
        rhs[i] = (rhs[i] - du[i] * rhs[i + 1] - du2[i] * rhs[i + 2]) / d[i]

    if not batched:
        rhs = rhs.squeeze(1)
    return rhs


def batched_tridiag_solve(dl: torch.Tensor, d: torch.Tensor, du: torch.Tensor,
                          rhs: torch.Tensor) -> torch.Tensor:
    """Batched tridiagonal solve without pivoting (Thomas algorithm).

    GPU-vectorized: sequential over N (the tridiagonal dimension),
    parallel over the batch dimension. No Python data-dependent branches.

    Uses the standard Thomas algorithm (no pivoting). Safe for diagonally
    dominant systems, which all FD implicit matrices are.

    Args:
        dl: (N-1,) or (B, N-1) lower diagonal (real)
        d:  (N,) or (B, N) main diagonal (real)
        du: (N-1,) or (B, N-1) upper diagonal (real)
        rhs: (B, N) real or complex right-hand side

    Returns:
        (B, N) solution, same dtype as rhs
    """
    # Ensure batch dimension
    if dl.dim() == 1:
        dl = dl.unsqueeze(0)
        d = d.unsqueeze(0)
        du = du.unsqueeze(0)

    rhs = rhs.clone()
    d = d.clone()
    N = d.shape[-1]

    # Forward sweep: eliminate lower diagonal
    for i in range(1, N):
        w = dl[:, i - 1] / d[:, i - 1]  # (B,)
        d[:, i] = d[:, i] - w * du[:, i - 1]
        rhs[:, i] = rhs[:, i] - w.unsqueeze(-1) * rhs[:, i - 1] if rhs.dim() == 3 else rhs[:, i] - w * rhs[:, i - 1]

    # Backward sweep: solve upper triangular
    rhs[:, N - 1] = rhs[:, N - 1] / d[:, N - 1]
    for i in range(N - 2, -1, -1):
        rhs[:, i] = (rhs[:, i] - du[:, i] * rhs[:, i + 1]) / d[:, i]

    return rhs


def batched_pentadiag_solve(dl2: torch.Tensor, dl1: torch.Tensor,
                            d: torch.Tensor, du1: torch.Tensor,
                            du2: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched pentadiagonal solve without pivoting.

    GPU-vectorized: sequential over N, parallel over the batch dimension.
    Uses forward Gaussian elimination (no pivoting) to reduce to upper
    triangular, then back-substitution. Safe for diagonally dominant systems.

    The elimination proceeds column by column: for column k, eliminate
    entries at rows k+1 and k+2 (if they exist).

    Args:
        dl2: (N-2,) or (B, N-2) second lower diagonal (A[i, i-2])
        dl1: (N-1,) or (B, N-1) first lower diagonal (A[i, i-1])
        d:   (N,) or (B, N) main diagonal
        du1: (N-1,) or (B, N-1) first upper diagonal (A[i, i+1])
        du2: (N-2,) or (B, N-2) second upper diagonal (A[i, i+2])
        rhs: (B, N) right-hand side

    Returns:
        (B, N) solution, same dtype as rhs
    """
    if dl2.dim() == 1:
        dl2 = dl2.unsqueeze(0)
        dl1 = dl1.unsqueeze(0)
        d = d.unsqueeze(0)
        du1 = du1.unsqueeze(0)
        du2 = du2.unsqueeze(0)

    rhs = rhs.clone()
    # Store bands in a (B, 5, N) tensor for easy column access
    # Rows: [dl2, dl1, d, du1, du2] at positions relative to diagonal
    B = d.shape[0]
    N = d.shape[-1]

    # Work with padded arrays so indexing is uniform
    # a[b, i, j] = A[i,j] only for |i-j| <= 2
    # Store as 5 diagonals: bands[b, k, i] where k=0..4 maps to offsets -2,-1,0,+1,+2
    # bands[b, 0, i] = A[i, i-2]  (dl2, valid for i>=2)
    # bands[b, 1, i] = A[i, i-1]  (dl1, valid for i>=1)
    # bands[b, 2, i] = A[i, i]    (d)
    # bands[b, 3, i] = A[i, i+1]  (du1, valid for i<N-1)
    # bands[b, 4, i] = A[i, i+2]  (du2, valid for i<N-2)
    bands = torch.zeros(B, 5, N, dtype=d.dtype, device=d.device)
    bands[:, 2, :] = d
    bands[:, 1, 1:] = dl1
    bands[:, 3, :N - 1] = du1
    bands[:, 0, 2:] = dl2
    bands[:, 4, :N - 2] = du2

    # Forward elimination: for each column k, zero out entries below diagonal
    for k in range(N - 1):
        # Eliminate entry at (k+1, k) using row k
        if k + 1 < N:
            w1 = bands[:, 1, k + 1] / bands[:, 2, k]  # (B,)
            # Row k+1 -= w1 * row k
            # bands[:, 1, k+1] becomes 0 (eliminated)
            bands[:, 2, k + 1] -= w1 * bands[:, 3, k]      # d[k+1] -= w1 * du1[k]
            if k + 2 < N:
                bands[:, 3, k + 1] -= w1 * bands[:, 4, k]  # du1[k+1] -= w1 * du2[k]
            bands[:, 1, k + 1] = 0.0
            rhs[:, k + 1] -= w1 * rhs[:, k]

        # Eliminate entry at (k+2, k) using row k
        if k + 2 < N:
            w2 = bands[:, 0, k + 2] / bands[:, 2, k]  # (B,)
            # Row k+2 -= w2 * row k
            bands[:, 1, k + 2] -= w2 * bands[:, 3, k]      # dl1[k+2] -= w2 * du1[k]
            bands[:, 2, k + 2] -= w2 * bands[:, 4, k]      # d[k+2] -= w2 * du2[k]
            bands[:, 0, k + 2] = 0.0
            rhs[:, k + 2] -= w2 * rhs[:, k]

    # Backward substitution with upper triangular (d, du1, du2)
    rhs[:, N - 1] = rhs[:, N - 1] / bands[:, 2, N - 1]
    if N >= 2:
        rhs[:, N - 2] = (rhs[:, N - 2] - bands[:, 3, N - 2] * rhs[:, N - 1]) / bands[:, 2, N - 2]
    for i in range(N - 3, -1, -1):
        rhs[:, i] = (rhs[:, i] - bands[:, 3, i] * rhs[:, i + 1]
                     - bands[:, 4, i] * rhs[:, i + 2]) / bands[:, 2, i]

    return rhs
