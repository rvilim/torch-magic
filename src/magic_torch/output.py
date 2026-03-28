"""Energy diagnostics matching kinetic_energy.f90 + magnetic_energy.f90.

Computes volume-averaged kinetic and magnetic energies for validation
against reference.out and referenceMag.out.
"""

import torch

from .precision import CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .chebyshev import r
from .radial_functions import or2
from .horizontal_data import dLh
from .blocking import st_lm2l, st_lm2m
from .integration import rInt_R
from .pre_calculations import LFfac


def _cc2real(c, m):
    """Convert complex spectral coefficient to real contribution.

    For m=0: |c|² (only real part contributes)
    For m>0: 2*|c|² (accounts for conjugate mode)
    """
    fac = torch.where(m == 0, 1.0, 2.0)
    return fac * (c.real ** 2 + c.imag ** 2)


def get_e_kin(w, dw, ddw, z, dz):
    """Compute kinetic energy (poloidal + toroidal).

    E_kin_pol = 0.5 * sum_{l,m} int dr r² [ l(l+1)/r² * (l(l+1)/r² * |w|² + |dw/dr|²) ]
    E_kin_tor = 0.5 * sum_{l,m} int dr r² * l(l+1)/r² * |z|²

    Args:
        w, dw, ddw, z, dz: (lm_max, n_r_max) complex spectral fields

    Returns:
        e_kin_pol, e_kin_tor: scalar kinetic energies
    """
    l_arr = st_lm2l.to(CDTYPE)
    m_arr = st_lm2m
    dLh_lm = dLh.unsqueeze(1)  # (lm_max, 1)
    r2 = (r * r).unsqueeze(0)  # (1, n_r_max)
    or2_r = or2.unsqueeze(0)   # (1, n_r_max)

    # Poloidal: l(l+1) [l(l+1)/r² * |w|² + |dw|²]
    e_pol_lm = 0.5 * dLh_lm * (dLh_lm * or2_r * _cc2real(w, m_arr.unsqueeze(1))
                                + _cc2real(dw, m_arr.unsqueeze(1)))

    # Toroidal: l(l+1) * |z|²
    e_tor_lm = 0.5 * dLh_lm * _cc2real(z, m_arr.unsqueeze(1))

    # Radial integration (sum over lm, integrate over r)
    e_kin_pol = rInt_R(e_pol_lm.sum(0).real).item()
    e_kin_tor = rInt_R(e_tor_lm.sum(0).real).item()

    return e_kin_pol, e_kin_tor


def get_e_mag(b, db, ddb, aj, dj):
    """Compute magnetic energy (poloidal + toroidal).

    E_mag_pol = 0.5 * sum_{l,m} int dr [ l(l+1)/r² * (l(l+1)/r² * |b|² + |db/dr|²) ]
    E_mag_tor = 0.5 * sum_{l,m} int dr * l(l+1)/r² * |aj|²

    Note: for magnetic energy, the integrand uses r² from the volume element
    but the field itself has a 1/r factor, giving a net factor of 1 (not r²).

    Args:
        b, db, ddb, aj, dj: (lm_max, n_r_max) complex spectral fields

    Returns:
        e_mag_pol, e_mag_tor: scalar magnetic energies
    """
    m_arr = st_lm2m
    dLh_lm = dLh.unsqueeze(1)
    or2_r = or2.unsqueeze(0)

    # Poloidal: fac = 0.5 * LFfac (LFfac = 1/(Ek*Pm) = 200 for benchmark)
    fac = 0.5 * LFfac
    e_pol_lm = fac * dLh_lm * (dLh_lm * or2_r * _cc2real(b, m_arr.unsqueeze(1))
                                + _cc2real(db, m_arr.unsqueeze(1)))

    # Toroidal
    e_tor_lm = fac * dLh_lm * _cc2real(aj, m_arr.unsqueeze(1))

    e_mag_pol = rInt_R(e_pol_lm.sum(0).real).item()
    e_mag_tor = rInt_R(e_tor_lm.sum(0).real).item()

    return e_mag_pol, e_mag_tor
