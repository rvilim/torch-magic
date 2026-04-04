"""Derived physical parameters matching preCalculations.f90.

Computes dimensionless parameters from the input physics (Ra, Ek, Pr, Pm).
Specialized for the Boussinesq benchmark (n_tScale=0, n_lScale=0).
"""

import math
from .params import (ra, ek, pr, prmag, dtmax, l_mag, l_heat, sigma_ratio, l_cond_ic,
                     l_rot_ic, kbotv, radratio, raxi, sc, l_chemical_conv,
                     l_correct_AMz, l_correct_AMe)
from .constants import one, third

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

# Composition parameters
osc = one / sc if l_chemical_conv else 1.0  # 1/Schmidt number
ChemFac = raxi / sc if l_chemical_conv else 0.0  # composition buoyancy factor
epscXi = 0.0  # internal composition source (zero for benchmark)

# Energy scale factor (always 1.0 for dimensional benchmarks)
eScale = 1.0

# Mass (preCalculations.f90:302, Boussinesq: mass = 1.0)
mass = 1.0

# Internal heating (zero for benchmark)
epsc = 0.0

# Conducting inner core: O_sr = 1/sigma_ratio
O_sr = one / sigma_ratio if l_cond_ic else 0.0

# --- Rotating inner core constants (preCalculations.f90 lines 314-344) ---
# Boussinesq: rho0=1, rho_ratio_ic=1
_r_icb = radratio / (one - radratio)   # same as r_icb from chebyshev.py
_or2_icb = (one - radratio) ** 2 / radratio ** 2  # 1/r_icb^2

# Y10 normalization: Y_1^0 = y10_norm * cos(theta)
y10_norm = 0.5 * math.sqrt(3.0 / math.pi)

# z10 -> omega_ic conversion (preCalculations.f90 line 319)
# c_z10_omega_ic = y10_norm * or2(n_r_max) / rho0(n_r_max)
# Boussinesq: rho0=1
c_z10_omega_ic = y10_norm * _or2_icb

# IC moment-of-inertia mass term for z10 (preCalculations.f90 line 337)
# c_dt_z10_ic = 0.2 * r_icb * rho_ratio_ic * rho0(n_r_max)
# Boussinesq: rho_ratio_ic=1, rho0=1
c_dt_z10_ic = 0.2 * _r_icb

# IC moment of inertia (preCalculations.f90 line 326)
# c_moi_ic = 8*pi/15 * r_icb^5 * rho_ratio_ic * rho0(n_r_max)
c_moi_ic = 8.0 * math.pi / 15.0 * _r_icb ** 5

# Lorentz torque coefficient (preCalculations.f90 line 344)
# c_lorentz_ic = 0.25 * sqrt(3/pi) * or2(n_r_max)
c_lorentz_ic = 0.25 * math.sqrt(3.0 / math.pi) * _or2_icb

# l_z10mat: use special matrix for z(l=1,m=0) when IC rotates with no-slip
l_z10mat = l_rot_ic and kbotv == 2

# gammatau_gravi (gravitational IC-mantle coupling) — default 0
gammatau_gravi = 0.0

# --- Angular momentum correction constants (preCalculations.f90 lines 314-331) ---
# Y11 normalization: Y_1^1 = y11_norm * sin(theta) * e^{i*phi}
y11_norm = 0.5 * math.sqrt(1.5 / math.pi)

# Outer core moment of inertia: c_moi_oc = 8/3 * pi * int(r^4 * rho0 dr)
# Always compute (preCalculations.f90:334-335), needed for log.TAG output
from .radial_scheme import r as _r_grid
from .radial_functions import rho0 as _rho0
from .integration import rInt_R as _rInt_R
_mom = _r_grid ** 4 * _rho0
c_moi_oc = 8.0 * third * math.pi * _rInt_R(_mom).item()

# Mantle moment of inertia (preCalculations.f90:338)
_r_surface = 2.8209  # Namelists.f90:1587 default
_r_cmb = 1.0 / (1.0 - radratio)
c_moi_ma = 8.0 * math.pi / 15.0 * (_r_surface**5 - _r_cmb**5)

# AMstart: target angular momentum (zero for fresh start)
AMstart = 0.0
