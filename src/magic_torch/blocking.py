"""LM index mappings matching Fortran's blocking.f90 + LMmapping.f90.

For single-process (n_procs=1), the standard mapping is used where
lm2 points to st_map (standard ordering: m-major, l-minor).

The "lo_map" (lorder) would be used for n_procs>1, but since we're
single-thread only, the standard mapping is used for lm2/lm2l/lm2m.

However, the lo_map is still needed for the LM loop ordering.
For n_procs=1, n_procs <= l_max/2 is false, so get_lorder_lm_blocking is used.
"""

import torch

from .precision import DEVICE
from .params import l_max, m_max, m_min, minc, lm_max


def _build_standard_mapping():
    """Build standard (m-major) LM mapping matching get_standard_lm_blocking."""
    # Build on CPU to avoid GPU sync overhead from scalar loops
    cpu = "cpu"
    lm2 = torch.full((l_max + 1, l_max + 1), -1, dtype=torch.long, device=cpu)
    lm2l = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2m = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2lmS = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2lmA = torch.zeros(lm_max, dtype=torch.long, device=cpu)

    lm = 0
    for m in range(m_min, m_max + 1, minc):
        for l in range(m, l_max + 1):
            lm2l[lm] = l
            lm2m[lm] = m
            lm2[l, m] = lm
            lm += 1

    assert lm == lm_max, f"Wrong lm count: {lm} != {lm_max}"

    for lm in range(lm_max):
        l = lm2l[lm].item()
        m = lm2m[lm].item()
        if l > 0 and l > m:
            lm2lmS[lm] = lm2[l - 1, m]
        else:
            lm2lmS[lm] = lm  # dummy, will be multiplied by zero
        if l < l_max:
            lm2lmA[lm] = lm2[l + 1, m]
        else:
            lm2lmA[lm] = -1

    return (lm2.to(DEVICE), lm2l.to(DEVICE), lm2m.to(DEVICE),
            lm2lmS.to(DEVICE), lm2lmA.to(DEVICE))


def _build_lorder_mapping():
    """Build l-order LM mapping matching get_lorder_lm_blocking."""
    cpu = "cpu"
    lm2 = torch.full((l_max + 1, l_max + 1), -1, dtype=torch.long, device=cpu)
    lm2l = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2m = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2lmS = torch.zeros(lm_max, dtype=torch.long, device=cpu)
    lm2lmA = torch.zeros(lm_max, dtype=torch.long, device=cpu)

    lm = 0
    for l in range(m_min, l_max + 1):
        for m in range(m_min, min(m_max, l) + 1, minc):
            lm2l[lm] = l
            lm2m[lm] = m
            lm2[l, m] = lm
            lm += 1

    assert lm == lm_max, f"Wrong lm count: {lm} != {lm_max}"

    for lm in range(lm_max):
        l = lm2l[lm].item()
        m = lm2m[lm].item()
        if l > 0 and l > m:
            lm2lmS[lm] = lm2[l - 1, m]
        else:
            lm2lmS[lm] = -1
        if l < l_max:
            lm2lmA[lm] = lm2[l + 1, m]
        else:
            lm2lmA[lm] = -1

    return (lm2.to(DEVICE), lm2l.to(DEVICE), lm2m.to(DEVICE),
            lm2lmS.to(DEVICE), lm2lmA.to(DEVICE))


# Standard mapping (st_map) — used as default lm2/lm2l/lm2m pointers
st_lm2, st_lm2l, st_lm2m, st_lm2lmS, st_lm2lmA = _build_standard_mapping()

# L-order mapping (lo_map) — used for LM loop ordering (n_procs=1 path)
lo_lm2, lo_lm2l, lo_lm2m, lo_lm2lmS, lo_lm2lmA = _build_lorder_mapping()

# Default pointers (Fortran sets these to st_map in initialize_blocking)
lm2 = st_lm2
lm2l = st_lm2l
lm2m = st_lm2m
lm2lmS = st_lm2lmS
lm2lmA = st_lm2lmA
