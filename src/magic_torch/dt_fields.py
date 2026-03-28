"""Time derivative arrays matching dt_fieldsLast.f90 + time_array.f90.

Stores multi-level history for Adams-Bashforth explicit terms,
Crank-Nicolson implicit terms, and old state values.
"""

import torch

from .precision import CDTYPE, DEVICE
from .params import lm_max, n_r_max
from .time_scheme import tscheme


class TimeArray:
    """Multi-level time array for spectral fields.

    Matches type_tarray from time_array.f90.

    Attributes:
        impl: (lm_max, n_r_max, nimp) - implicit terms
        expl: (lm_max, n_r_max, nexp) - explicit terms
        old:  (lm_max, n_r_max, nold) - old state values
    """

    def __init__(self, nold: int, nexp: int, nimp: int):
        self.impl = torch.zeros(lm_max, n_r_max, nimp, dtype=CDTYPE, device=DEVICE)
        self.expl = torch.zeros(lm_max, n_r_max, nexp, dtype=CDTYPE, device=DEVICE)
        self.old = torch.zeros(lm_max, n_r_max, nold, dtype=CDTYPE, device=DEVICE)


class TimeScalar:
    """Multi-level time array for scalar quantities.

    Matches type_tscalar from time_array.f90.
    """

    def __init__(self, nold: int, nexp: int, nimp: int):
        self.impl = torch.zeros(nimp, dtype=torch.float64, device=DEVICE)
        self.expl = torch.zeros(nexp, dtype=torch.float64, device=DEVICE)
        self.old = torch.zeros(nold, dtype=torch.float64, device=DEVICE)


# Time derivative arrays for each field equation
# CNAB2: nold=1, nexp=2, nimp=1
_no = tscheme.nold
_ne = tscheme.nexp
_ni = tscheme.nimp

dsdt = TimeArray(_no, _ne, _ni)     # entropy
dwdt = TimeArray(_no, _ne, _ni)     # poloidal velocity (w)
dzdt = TimeArray(_no, _ne, _ni)     # toroidal velocity (z)
dpdt = TimeArray(_no, _ne, _ni)     # pressure
dbdt = TimeArray(_no, _ne, _ni)     # poloidal magnetic (b)
djdt = TimeArray(_no, _ne, _ni)     # toroidal magnetic (j)

# Scalar time derivatives for rotation rates
domega_ma_dt = TimeScalar(_no, _ne, _ni)
domega_ic_dt = TimeScalar(_no, _ne, _ni)
