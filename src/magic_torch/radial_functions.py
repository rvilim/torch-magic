"""Radial grid and background profiles for the Boussinesq benchmark.

Matches radial.f90 for the non-anelastic, non-FD, no-mapping case with:
- radratio = 0.35
- No variable viscosity/diffusivity/conductivity
- Gravity: rgrav = r / r_cmb (default g0=0, g1=1, g2=0)
"""

import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, radratio
from .chebyshev import r, r_cmb, r_icb

# Inverse radial functions
or1 = 1.0 / r          # 1/r
or2 = or1 * or1         # 1/r^2
or3 = or2 * or1         # 1/r^3
or4 = or2 * or2         # 1/r^4

# Gravity (default: g0=0, g1=1, g2=0 → rgrav = r/r_cmb)
rgrav = r / r_cmb

# Boussinesq: all background profiles are trivial
rho0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
temp0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
orho1 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
orho2 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)

alpha0 = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
beta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
dbeta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddbeta = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

dLtemp0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLtemp0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
dLalpha0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLalpha0 = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

ogrun = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

# Transport properties (constant for Boussinesq)
visc = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLvisc = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)
ddLvisc = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

kappa = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLkappa = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

lambda_ = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)
dLlambda = torch.zeros(n_r_max, dtype=DTYPE, device=DEVICE)

sigma = torch.ones(n_r_max, dtype=DTYPE, device=DEVICE)

# Heating factors (zero for Boussinesq)
ViscHeatFac = 0.0
OhmLossFac = 0.0

# Volume of outer core
vol_oc = (4.0 / 3.0) * torch.pi * (r_cmb**3 - r_icb**3)
surf_cmb = 4.0 * torch.pi * r_cmb**2
