"""Derived physical parameters matching preCalculations.f90.

Computes dimensionless parameters from the input physics (Ra, Ek, Pr, Pm).
Specialized for the Boussinesq benchmark (n_tScale=0, n_lScale=0).
"""

from .params import (ra, ek, pr, prmag, dtmax, l_mag, l_heat)
from .constants import one

# Scale factors (Boussinesq: tScale=1, lScale=1)
tScale = 1.0
lScale = 1.0
vScale = lScale / tScale
ekScaled = ek * lScale ** 2
raScaled = ra / lScale ** 3

# Inverse Prandtl numbers
opr = one / pr       # 1/Pr = 1.0
opm = one / prmag    # 1/Pm = 0.2

# Coriolis factor = 1/Ek
CorFac = one / ekScaled   # 1000.0
oek = CorFac

# Lorentz force factor = 1/(Ek*Pm)
LFfac = one / (ekScaled * prmag)  # 200.0

# Buoyancy factor = Ra/Pr
BuoFac = raScaled / pr  # 100000.0

# Time step limits
dtMax = dtmax / tScale
dtMin = dtMax / 1.0e6

# Internal heating (zero for benchmark)
epsc = 0.0
