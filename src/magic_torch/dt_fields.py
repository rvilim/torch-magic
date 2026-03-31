"""Time derivative arrays matching dt_fieldsLast.f90 + time_array.f90.

Stores multi-level history for Adams-Bashforth explicit terms,
Crank-Nicolson implicit terms, and old state values.

The 5 main spectral dfdts (S, [Xi], Z, W, P) use contiguous mega-tensor
storage so that RHS assembly can operate on a single pre-stacked tensor
without torch.cat — critical for reducing MPS dispatch overhead.
"""

import torch

from .precision import CDTYPE, DTYPE, DEVICE
from .params import lm_max, n_r_max, n_r_ic_max, l_cond_ic, l_chemical_conv
from .time_scheme import tscheme


class TimeArray:
    """Multi-level time array for spectral fields.

    Matches type_tarray from time_array.f90.

    Attributes:
        impl: (lm_max, nr, nimp) - implicit terms
        expl: (lm_max, nr, nexp) - explicit terms
        old:  (lm_max, nr, nold) - old state values
    """

    def __init__(self, nold: int, nexp: int, nimp: int, nr: int = 0):
        if nr == 0:
            nr = n_r_max
        self.impl = torch.zeros(lm_max, nr, nimp, dtype=CDTYPE, device=DEVICE)
        self.expl = torch.zeros(lm_max, nr, nexp, dtype=CDTYPE, device=DEVICE)
        self.old = torch.zeros(lm_max, nr, nold, dtype=CDTYPE, device=DEVICE)


class TimeScalar:
    """Multi-level time array for scalar quantities.

    Matches type_tscalar from time_array.f90.
    """

    def __init__(self, nold: int, nexp: int, nimp: int):
        self.impl = torch.zeros(nimp, dtype=DTYPE, device=DEVICE)
        self.expl = torch.zeros(nexp, dtype=DTYPE, device=DEVICE)
        self.old = torch.zeros(nold, dtype=DTYPE, device=DEVICE)


# Time derivative arrays for each field equation
# CNAB2: nold=1, nexp=2, nimp=1
_no = tscheme.nold
_ne = tscheme.nexp
_ni = tscheme.nimp

dsdt = TimeArray(_no, _ne, _ni)     # entropy
dxidt = TimeArray(_no, _ne, _ni)    # composition
dwdt = TimeArray(_no, _ne, _ni)     # poloidal velocity (w)
dzdt = TimeArray(_no, _ne, _ni)     # toroidal velocity (z)
dpdt = TimeArray(_no, _ne, _ni)     # pressure
dbdt = TimeArray(_no, _ne, _ni)     # poloidal magnetic (b)
djdt = TimeArray(_no, _ne, _ni)     # toroidal magnetic (j)

# Scalar time derivatives for rotation rates
domega_ma_dt = TimeScalar(_no, _ne, _ni)
domega_ic_dt = TimeScalar(_no, _ne, _ni)

# IC time derivative arrays (conducting inner core)
if l_cond_ic:
    dbdt_ic = TimeArray(_no, _ne, _ni, nr=n_r_ic_max)
    djdt_ic = TimeArray(_no, _ne, _ni, nr=n_r_ic_max)
else:
    dbdt_ic = None
    djdt_ic = None


# === Contiguous mega-tensor storage for cat-free RHS assembly ===
# All 5 main spectral dfdts (S, [Xi], Z, W, P) share ONE contiguous tensor.
# Layout: [S, (Xi), Z, W, P] — scalar block first, then WP block.
# Enables single _assemble_rhs_mega call for all 5 fields at once.

_nS = 3 if l_chemical_conv else 2  # scalar block: S [+Xi] + Z
_nAll = _nS + 2  # total: scalar + W + P
LM = lm_max

# Unified mega-tensors for all 5 dfdts
_all_old = torch.zeros(_nAll * LM, n_r_max, _no, dtype=CDTYPE, device=DEVICE)
_all_expl = torch.zeros(_nAll * LM, n_r_max, _ne, dtype=CDTYPE, device=DEVICE)
_all_impl = torch.zeros(_nAll * LM, n_r_max, _ni, dtype=CDTYPE, device=DEVICE)

# Convenience views for scalar block [0:nS*LM] and WP block [nS*LM:]
_scalar_old = _all_old[:_nS * LM]
_scalar_expl = _all_expl[:_nS * LM]
_scalar_impl = _all_impl[:_nS * LM]
_wp_old = _all_old[_nS * LM:]
_wp_expl = _all_expl[_nS * LM:]
_wp_impl = _all_impl[_nS * LM:]

# Assign views to individual dfdts: [S, (Xi), Z, W, P]
_off = 0
dsdt.old = _all_old[_off:_off + LM]
dsdt.expl = _all_expl[_off:_off + LM]
dsdt.impl = _all_impl[_off:_off + LM]
_off += LM

if l_chemical_conv:
    dxidt.old = _all_old[_off:_off + LM]
    dxidt.expl = _all_expl[_off:_off + LM]
    dxidt.impl = _all_impl[_off:_off + LM]
    _off += LM

dzdt.old = _all_old[_off:_off + LM]
dzdt.expl = _all_expl[_off:_off + LM]
dzdt.impl = _all_impl[_off:_off + LM]
_off += LM

dwdt.old = _all_old[_off:_off + LM]
dwdt.expl = _all_expl[_off:_off + LM]
dwdt.impl = _all_impl[_off:_off + LM]
_off += LM

dpdt.old = _all_old[_off:_off + LM]
dpdt.expl = _all_expl[_off:_off + LM]
dpdt.impl = _all_impl[_off:_off + LM]
