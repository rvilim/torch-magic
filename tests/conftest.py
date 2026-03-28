"""Shared fixtures for Fortran reference comparison tests."""

from pathlib import Path
import numpy as np
import pytest
import torch

from magic_torch.precision import DEVICE, DTYPE, CDTYPE

# Path to Fortran reference data
FORTRAN_REF = Path(__file__).parent.parent / "fortran_ref"


def _compute_snake_to_standard_perm(l_max: int, minc: int = 1) -> torch.Tensor:
    """Compute permutation from Fortran snake LM ordering to standard ordering.

    For n_procs=1 with l_max modes, the Fortran snake ordering assigns LM indices
    in l-order: [0, l_max-1, l_max-2, ..., 1, l_max], with m running 0..l for
    each l. The standard ordering uses m-major: for m=0..l_max, for l=m..l_max.

    Returns:
        snake2st: int tensor of shape (lm_max,) where snake2st[snake_idx] = st_idx
    """
    # Reproduce the snake l-ordering for n_procs=1
    # Snake loop: l from l_max down to 0, all assigned to proc 0
    l_list = list(range(l_max, -1, -1))  # [l_max, l_max-1, ..., 0]
    # Move l=0 to front (swap with first element)
    idx_0 = l_list.index(0)
    l_list[0], l_list[idx_0] = l_list[idx_0], l_list[0]
    # l_list is now: [0, l_max-1, l_max-2, ..., 1, l_max]

    # Build snake lm2l and lm2m (1-based in Fortran, 0-based here)
    snake_lm2l = []
    snake_lm2m = []
    for l in l_list:
        for m in range(0, min(l_max, l) + 1, minc):
            snake_lm2l.append(l)
            snake_lm2m.append(m)

    # Build standard lm2 mapping: st_lm2[l, m] -> index
    lm_max = len(snake_lm2l)
    st_lm2 = {}
    st_idx = 0
    for m in range(0, l_max + 1, minc):
        for l in range(m, l_max + 1):
            st_lm2[(l, m)] = st_idx
            st_idx += 1

    # Build permutation: snake_idx -> st_idx
    perm = torch.zeros(lm_max, dtype=torch.long)
    for snake_idx in range(lm_max):
        l = snake_lm2l[snake_idx]
        m = snake_lm2m[snake_idx]
        perm[snake_idx] = st_lm2[(l, m)]

    return perm


# Precompute the permutation (l_max=16 for the benchmark)
_SNAKE2ST = _compute_snake_to_standard_perm(l_max=16, minc=1).to(DEVICE)

# Names of field arrays that use LM ordering (first dimension is lm_max)
_LM_FIELD_NAMES = {
    's_init', 'b_init', 'db_init', 'aj_init', 'w_init', 'dw_init', 'z_init',
    'p_init', 'dp_init',
    's_step1', 'b_step1', 'db_step1', 'ddb_step1', 'aj_step1', 'dj_step1',
    'w_step1', 'dw_step1', 'ddw_step1', 'z_step1', 'dz_step1', 'p_step1',
    'dp_step1', 'ds_step1',
    # step1-step11 fields (generated for N-step comparison)
    *[f'{f}_step{n}' for n in range(1, 102)
      for f in ('s', 'b', 'db', 'ddb', 'aj', 'dj', 'w', 'dw', 'ddw', 'z', 'dz', 'p', 'dp', 'ds')],
    # Phase 6: spectral nonlinear terms at nR=17
    'AdvrLM_nR17', 'AdvtLM_nR17', 'AdvpLM_nR17',
    # Phase 7: dt_field components and IMEX RHS (lm_max × n_r_max)
    'dwdt_expl', 'dwdt_impl', 'dwdt_old',
    'dzdt_expl', 'dzdt_impl', 'dzdt_old',
    'z_imex_rhs', 'w_imex_rhs', 'p_imex_rhs',
}


def load_ref(name: str, reorder_lm: bool = None) -> torch.Tensor:
    """Load a Fortran reference array as a PyTorch tensor.

    For field arrays (LM-distributed), reorders from Fortran snake ordering
    to standard ordering used by PyTorch.
    """
    arr = np.load(FORTRAN_REF / f"{name}.npy")
    if arr.ndim == 0:
        return torch.tensor(arr.item(), device=DEVICE, dtype=DTYPE)
    if np.issubdtype(arr.dtype, np.integer):
        t = torch.from_numpy(arr.copy()).long().to(DEVICE)
    elif np.issubdtype(arr.dtype, np.complexfloating):
        t = torch.from_numpy(arr.copy()).to(CDTYPE).to(DEVICE)
    else:
        t = torch.from_numpy(arr.copy()).to(DTYPE).to(DEVICE)

    # Reorder LM dimension if this is a field array
    should_reorder = reorder_lm if reorder_lm is not None else (name in _LM_FIELD_NAMES)
    if should_reorder and t.shape[0] == len(_SNAKE2ST):
        if t.ndim == 2:
            # t[snake_idx, :] -> result[st_idx, :] = t[snake_idx, :]
            result = torch.zeros_like(t)
            result[_SNAKE2ST] = t
            return result
        elif t.ndim == 1:
            # t[snake_idx] -> result[st_idx] = t[snake_idx]
            result = torch.zeros_like(t)
            result[_SNAKE2ST] = t
            return result

    return t
