"""Spectral field arrays matching fields.f90.

All fields are complex128, shape (lm_max, n_r_max).
Only allocates fields used by the dynamo benchmark path.
"""

import torch

from .precision import CDTYPE, DEVICE
from .constants import zero
from .params import lm_max, n_r_max, l_mag, l_heat


def _zeros():
    """Allocate a zero-filled spectral field."""
    return torch.zeros(lm_max, n_r_max, dtype=CDTYPE, device=DEVICE)


# Velocity potentials
w_LMloc = _zeros()     # poloidal
dw_LMloc = _zeros()    # d/dr of w
ddw_LMloc = _zeros()   # d²/dr² of w
z_LMloc = _zeros()     # toroidal
dz_LMloc = _zeros()    # d/dr of z

# Pressure
p_LMloc = _zeros()
dp_LMloc = _zeros()

# Entropy / Temperature
s_LMloc = _zeros()
ds_LMloc = _zeros()

# Magnetic field potentials (only if l_mag)
b_LMloc = _zeros()
db_LMloc = _zeros()
ddb_LMloc = _zeros()
aj_LMloc = _zeros()    # toroidal magnetic (j = curl B)
dj_LMloc = _zeros()
ddj_LMloc = _zeros()

# Rotation rates
omega_ic = 0.0
omega_ma = 0.0

# Work array used in update routines
work_LMloc = _zeros()
