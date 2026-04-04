"""Energy diagnostics matching kinetic_energy.f90 + magnetic_energy.f90.

Computes volume-averaged kinetic and magnetic energies with full
decompositions (total, axisymmetric, equatorially symmetric, EAS)
and writes Fortran-format energy files (e_kin.TAG, e_mag_oc.TAG, e_mag_ic.TAG,
eKinR.TAG, eMagR.TAG, dipole.TAG, radius.TAG, signal.TAG, timestep.TAG).
"""

import math
from collections import namedtuple

import torch

from .precision import CDTYPE, DEVICE
from .params import (n_r_max, lm_max, l_max, l_mag, l_cond_ic, l_heat, l_chemical_conv,
                     minc, kbotb, ktopb, m_max, prmag, ek, n_theta_max, n_phi_max)
from .radial_scheme import r, r_cmb, r_icb
from .radial_functions import (or1, or2, or4, orho1, orho2, beta as r_beta,
                               sigma, lambda_ as r_lambda, visc as r_visc,
                               rgrav, vol_oc, surf_cmb)
from .horizontal_data import dLh
from .blocking import st_lm2l, st_lm2m
from .integration import rInt_R
from .pre_calculations import (LFfac, ekScaled, mass, eScale, vScale, opm,
                               BuoFac, ChemFac)
from .constants import pi

# --- Precomputed masks (1D, lm_max) ---
_m0 = (st_lm2m == 0)  # axisymmetric modes
_lpm_even = ((st_lm2l + st_lm2m) % 2 == 0)  # (l+m) even
_lpm_odd = ~_lpm_even  # (l+m) odd
_l_even = (st_lm2l % 2 == 0)
_l_odd = ~_l_even

# Kinetic ES: pol uses (l+m) even, tor uses (l+m) odd
# Kinetic EAS: pol uses m=0 & l even, tor uses m=0 & l odd
_eas_p_kin = _m0 & _l_even
_eas_t_kin = _m0 & _l_odd

# Magnetic ES: pol uses (l+m) odd, tor uses (l+m) even (OPPOSITE of kinetic)
# Magnetic EAS: pol uses m=0 & l odd, tor uses m=0 & l even
_eas_p_mag = _m0 & _l_odd
_eas_t_mag = _m0 & _l_even

# Float masks for matmul-based masked sums (avoids copies from boolean indexing)
_m0_f = _m0.to(dtype=torch.float64)
_lpm_even_f = _lpm_even.to(dtype=torch.float64)
_lpm_odd_f = _lpm_odd.to(dtype=torch.float64)
_eas_p_kin_f = _eas_p_kin.to(dtype=torch.float64)
_eas_t_kin_f = _eas_t_kin.to(dtype=torch.float64)
_eas_p_mag_f = _eas_p_mag.to(dtype=torch.float64)
_eas_t_mag_f = _eas_t_mag.to(dtype=torch.float64)

# l-array as float for outside-shell / IC energy
_l_f = st_lm2l.to(dtype=torch.float64)
# m-array expanded for cc2real
_m_arr = st_lm2m.unsqueeze(1)  # (lm_max, 1) for broadcasting

# --- Additional masks for radial profiles and dipole ---
_l1_f = (st_lm2l == 1).to(dtype=torch.float64)  # l=1 modes
_l1m0_f = ((st_lm2l == 1) & (st_lm2m == 0)).to(dtype=torch.float64)

# Geostrophic: l <= l_geo = 11 (Namelists.f90:1572)
_L_GEO = 11
_l_le_lgeo_f = (st_lm2l <= _L_GEO).to(dtype=torch.float64)
_l_le_lgeo_es_f = ((st_lm2l <= _L_GEO) & _lpm_odd).to(dtype=torch.float64)
_l_le_lgeo_as_f = ((st_lm2l <= _L_GEO) & _m0).to(dtype=torch.float64)

# lm indices for dipole tilt angles
_lm_l1m0 = ((st_lm2l == 1) & (st_lm2m == 0)).nonzero(as_tuple=True)[0].item()
_lm_l1m1 = (((st_lm2l == 1) & (st_lm2m == 1)).nonzero(as_tuple=True)[0].item()
             if minc <= 1 else None)


# --- Named tuples for structured returns ---
EKin = namedtuple('EKin', 'e_p e_t e_p_as e_t_as e_p_es e_t_es e_p_eas e_t_eas')
EMagOC = namedtuple('EMagOC',
                    'e_p e_t e_p_as e_t_as e_p_os e_p_as_os '
                    'e_p_es e_t_es e_p_eas e_t_eas e_p_e e_p_as_e')
EMagIC = namedtuple('EMagIC', 'e_p e_t e_p_as e_t_as')


def _cc2real(c, m):
    """Convert complex spectral coefficient to real contribution.

    For m=0: real(c)^2 only (Fortran cc2real convention)
    For m>0: 2*(real^2 + imag^2) (accounts for conjugate mode)
    """
    return torch.where(m == 0, c.real ** 2, 2.0 * (c.real ** 2 + c.imag ** 2))


def _cc22real(a, b, m):
    """Cross-spectrum: Re(a*conj(b)) for m=0, 2*Re(a*conj(b)) for m>0.

    Fortran cc22real convention: Re(a)*Re(b) for m=0, 2*(Re(a)*Re(b)+Im(a)*Im(b)) for m>0.
    """
    cross = a.real * b.real + a.imag * b.imag  # Re(a*conj(b))
    return torch.where(m == 0, cross, 2.0 * cross)


def get_e_kin(w, dw, ddw, z, dz):
    """Compute kinetic energy with full decompositions.

    Backward compatible: still accepts (w, dw, ddw, z, dz) and returns
    (e_kin_pol, e_kin_tor) as before. Use get_e_kin_full() for all 8 components.
    """
    ek = get_e_kin_full(w, dw, z)
    return ek.e_p, ek.e_t


def get_e_kin_full(w, dw, z):
    """Compute kinetic energy with full decompositions.

    Returns EKin namedtuple with 8 components matching Fortran e_kin.TAG.
    """
    dLh_lm = dLh.unsqueeze(1)  # (lm_max, 1)
    or2_r = or2.unsqueeze(0)   # (1, n_r_max)
    orho1_r = orho1.unsqueeze(0)  # (1, n_r_max)

    # Per-mode radial profiles (lm_max, n_r_max) — real-valued
    e_pol_r = 0.5 * orho1_r * dLh_lm * (
        dLh_lm * or2_r * _cc2real(w, _m_arr) + _cc2real(dw, _m_arr))
    e_tor_r = 0.5 * orho1_r * dLh_lm * _cc2real(z, _m_arr)

    # Use matmul for masked sums: mask (lm_max,) @ e (lm_max, n_r_max) -> (n_r_max,)
    ones = torch.ones(lm_max, dtype=torch.float64, device=e_pol_r.device)

    e_p = rInt_R(ones @ e_pol_r).item()
    e_t = rInt_R(ones @ e_tor_r).item()
    e_p_as = rInt_R(_m0_f @ e_pol_r).item()
    e_t_as = rInt_R(_m0_f @ e_tor_r).item()
    # ES: poloidal uses (l+m) even, toroidal uses (l+m) odd
    e_p_es = rInt_R(_lpm_even_f @ e_pol_r).item()
    e_t_es = rInt_R(_lpm_odd_f @ e_tor_r).item()
    # EAS: poloidal uses m=0 & l even, toroidal uses m=0 & l odd
    e_p_eas = rInt_R(_eas_p_kin_f @ e_pol_r).item()
    e_t_eas = rInt_R(_eas_t_kin_f @ e_tor_r).item()

    return EKin(e_p, e_t, e_p_as, e_t_as, e_p_es, e_t_es, e_p_eas, e_t_eas)


def get_e_mag(b, db, ddb, aj, dj):
    """Compute magnetic energy (poloidal + toroidal).

    Backward compatible: returns (e_mag_pol, e_mag_tor).
    Use get_e_mag_oc_full() for all 12 OC components.
    """
    em = get_e_mag_oc_full(b, db, aj)
    return em.e_p, em.e_t


def get_e_mag_oc_full(b, db, aj):
    """Compute outer-core magnetic energy with full decompositions.

    Returns EMagOC namedtuple with 12 components matching Fortran e_mag_oc.TAG.
    """
    dLh_lm = dLh.unsqueeze(1)
    or2_r = or2.unsqueeze(0)
    fac = 0.5 * LFfac  # eScale=1.0 for benchmark

    # Per-mode radial profiles (lm_max, n_r_max)
    e_pol_r = fac * dLh_lm * (
        dLh_lm * or2_r * _cc2real(b, _m_arr) + _cc2real(db, _m_arr))
    e_tor_r = fac * dLh_lm * _cc2real(aj, _m_arr)

    ones = torch.ones(lm_max, dtype=torch.float64, device=e_pol_r.device)

    e_p = rInt_R(ones @ e_pol_r).item()
    e_t = rInt_R(ones @ e_tor_r).item()
    e_p_as = rInt_R(_m0_f @ e_pol_r).item()
    e_t_as = rInt_R(_m0_f @ e_tor_r).item()
    # ES: magnetic uses OPPOSITE parity from kinetic
    # poloidal uses (l+m) odd, toroidal uses (l+m) even
    e_p_es = rInt_R(_lpm_odd_f @ e_pol_r).item()
    e_t_es = rInt_R(_lpm_even_f @ e_tor_r).item()
    # EAS: poloidal uses m=0 & l odd, toroidal uses m=0 & l even
    e_p_eas = rInt_R(_eas_p_mag_f @ e_pol_r).item()
    e_t_eas = rInt_R(_eas_t_mag_f @ e_tor_r).item()

    # Outside-shell energy at CMB (nR=0 in Python): l^2*(l+1) weighting
    fac_os = 0.5 * LFfac / r_cmb  # eScale=1.0
    b_cmb = b[:, 0]  # (lm_max,)
    cc_cmb = _cc2real(b_cmb, st_lm2m)  # (lm_max,)
    l2lp1 = _l_f * _l_f * (_l_f + 1.0)
    e_p_os = (fac_os * l2lp1 * cc_cmb).sum().item()
    e_p_as_os = (fac_os * l2lp1 * cc_cmb * _m0_f).sum().item()

    # External field (n_imp != 1 for benchmarks, always 0)
    e_p_e = 0.0
    e_p_as_e = 0.0

    return EMagOC(e_p, e_t, e_p_as, e_t_as, e_p_os, e_p_as_os,
                  e_p_es, e_t_es, e_p_eas, e_t_eas, e_p_e, e_p_as_e)


def _cc22real(c1, c2, m):
    """Cross-term: real(c1)*real(c2) for m=0, 2*(re*re+im*im) for m>0."""
    return torch.where(m == 0, c1.real * c2.real,
                       2.0 * (c1.real * c2.real + c1.imag * c2.imag))


def _rIntIC(f_r):
    """Chebyshev integration on IC even grid (integration.f90 rIntIC).

    f_r: shape (n_r_ic_max,) — radial profile on IC Chebyshev grid.
    Returns: scalar integral value.
    """
    from .radial_functions import dr_fac_ic
    from .params import n_r_ic_max
    from .radial_scheme import costf

    # Apply costf (same DCT-I as OC)
    a = costf(f_r.unsqueeze(0)).squeeze(0)
    # Half endpoints
    a = a.clone()
    a[0] = 0.5 * a[0]
    a[-1] = 0.5 * a[-1]
    # Vectorized IC Chebyshev weights: nChebInt = 2*(k+1)-1 for k=1..N-1
    k = torch.arange(1, n_r_ic_max, dtype=torch.float64, device=a.device)
    nChebInt = 2.0 * (k + 1.0) - 1.0
    weights = -1.0 / (nChebInt * (nChebInt - 2.0))
    result = (a[0] + (weights * a[1:]).sum()).item()
    # Normalize
    result *= math.sqrt(2.0 / (n_r_ic_max - 1)) / dr_fac_ic
    return result


def get_e_mag_ic(b, b_ic=None, db_ic=None, aj_ic=None):
    """Compute inner-core magnetic energy.

    For insulating IC (l_cond_ic=False): uses b at ICB with l*(l+1)^2 weighting.
    For conducting IC (l_cond_ic=True): integrates over IC grid using b_ic/db_ic/aj_ic.
    Returns EMagIC namedtuple with 4 components matching Fortran e_mag_ic.TAG.
    """
    if l_cond_ic and b_ic is not None and db_ic is not None and aj_ic is not None:
        # Conducting IC: magnetic_energy.f90 lines 485-541
        from .radial_functions import r_ic
        from .params import n_r_ic_max

        O_r_icb_E_2 = 1.0 / (r_icb * r_icb)
        ll = _l_f  # (lm_max,)
        llp1 = ll * (ll + 1.0)
        m_ic = st_lm2m.unsqueeze(1)  # (lm_max, 1) for broadcasting

        # r_ratio = r_ic[nR] / r_ic[0], shape (n_r_ic_max,)
        r_ratio = (r_ic / r_ic[0]).unsqueeze(0)  # (1, n_r_ic_max)

        # r_dr_b = r_ic * db_ic, shape (lm_max, n_r_ic_max)
        r_dr_b = r_ic.unsqueeze(0) * db_ic

        # Poloidal: l(l+1) * O_r_icb^2 * r_ratio^(2l) * [...]
        # Need per-lm r_ratio^(2l): (lm_max, n_r_ic_max)
        rr_2l = r_ratio ** (2.0 * ll.unsqueeze(1))

        cc_b = _cc2real(b_ic, m_ic)       # (lm_max, n_r_ic_max)
        cc_rdrb = _cc2real(r_dr_b, m_ic)  # (lm_max, n_r_ic_max)
        cc_cross = _cc22real(b_ic, r_dr_b, m_ic)  # (lm_max, n_r_ic_max)

        e_p_r_lm = llp1.unsqueeze(1) * O_r_icb_E_2 * rr_2l * (
            ((ll + 1.0) * (2.0 * ll + 1.0)).unsqueeze(1) * cc_b +
            (2.0 * (ll + 1.0)).unsqueeze(1) * cc_cross +
            cc_rdrb
        )
        # Toroidal: l(l+1) * r_ratio^(2l+2) * cc2real(aj_ic)
        rr_2lp2 = r_ratio ** (2.0 * ll.unsqueeze(1) + 2.0)
        cc_aj = _cc2real(aj_ic, m_ic)
        e_t_r_lm = llp1.unsqueeze(1) * rr_2lp2 * cc_aj

        # Skip l=0 (lm=0) — Fortran starts from lm=2 (1-based), i.e. lm>=1 (0-based)
        e_p_r_lm[0, :] = 0.0
        e_t_r_lm[0, :] = 0.0

        # Total and axisymmetric radial profiles
        e_p_r = e_p_r_lm.sum(dim=0)  # (n_r_ic_max,)
        e_t_r = e_t_r_lm.sum(dim=0)
        e_p_as_r = (e_p_r_lm * _m0_f.unsqueeze(1)).sum(dim=0)
        e_t_as_r = (e_t_r_lm * _m0_f.unsqueeze(1)).sum(dim=0)

        # Integrate over IC grid
        fac = 0.5 * LFfac  # eScale=1.0
        e_p = fac * _rIntIC(e_p_r)
        e_t = fac * _rIntIC(e_t_r)
        e_p_as = fac * _rIntIC(e_p_as_r)
        e_t_as = fac * _rIntIC(e_t_as_r)

        return EMagIC(e_p, e_t, e_p_as, e_t_as)

    # Insulating IC: energy from potential field extrapolation at ICB
    fac_ic = 0.5 * LFfac / r_icb  # eScale=1.0
    b_icb = b[:, -1]  # (lm_max,) — ICB is last radial point
    cc_icb = _cc2real(b_icb, st_lm2m)  # (lm_max,)
    llp1sq = _l_f * (_l_f + 1.0) ** 2  # l*(l+1)^2

    e_p = (fac_ic * llp1sq * cc_icb).sum().item()
    e_t = 0.0  # insulating IC has no toroidal field
    e_p_as = (fac_ic * llp1sq * cc_icb * _m0_f).sum().item()
    e_t_as = 0.0

    return EMagIC(e_p, e_t, e_p_as, e_t_as)


# --- File writers matching Fortran format ---

def write_e_kin_line(f, time, ek: EKin):
    """Write one line to e_kin.TAG matching Fortran ES20.12,8ES16.8."""
    f.write(f"{time:20.12E}{ek.e_p:16.8E}{ek.e_t:16.8E}"
            f"{ek.e_p_as:16.8E}{ek.e_t_as:16.8E}"
            f"{ek.e_p_es:16.8E}{ek.e_t_es:16.8E}"
            f"{ek.e_p_eas:16.8E}{ek.e_t_eas:16.8E}\n")


def write_e_mag_oc_line(f, time, em: EMagOC):
    """Write one line to e_mag_oc.TAG matching Fortran ES20.12,12ES16.8."""
    f.write(f"{time:20.12E}{em.e_p:16.8E}{em.e_t:16.8E}"
            f"{em.e_p_as:16.8E}{em.e_t_as:16.8E}"
            f"{em.e_p_os:16.8E}{em.e_p_as_os:16.8E}"
            f"{em.e_p_es:16.8E}{em.e_t_es:16.8E}"
            f"{em.e_p_eas:16.8E}{em.e_t_eas:16.8E}"
            f"{em.e_p_e:16.8E}{em.e_p_as_e:16.8E}\n")


def write_e_mag_ic_line(f, time, eic: EMagIC):
    """Write one line to e_mag_ic.TAG matching Fortran ES20.12,4ES16.8."""
    f.write(f"{time:20.12E}{eic.e_p:16.8E}{eic.e_t:16.8E}"
            f"{eic.e_p_as:16.8E}{eic.e_t_as:16.8E}\n")


# --- Radial energy profiles for eKinR / eMagR ---

def get_e_kin_radial(w, dw, z):
    """Per-radius kinetic energy profiles (unfactored: multiply by fac=0.5 at output).

    Returns: (e_p_r, e_p_as_r, e_t_r, e_t_as_r) each (n_r_max,)
    """
    dLh_lm = dLh.unsqueeze(1)  # (lm_max, 1)
    or2_r = or2.unsqueeze(0)   # (1, n_r_max)
    orho1_r = orho1.unsqueeze(0)

    e_pol_r = orho1_r * dLh_lm * (
        dLh_lm * or2_r * _cc2real(w, _m_arr) + _cc2real(dw, _m_arr))
    e_tor_r = orho1_r * dLh_lm * _cc2real(z, _m_arr)

    ones = torch.ones(lm_max, dtype=torch.float64, device=e_pol_r.device)
    return (ones @ e_pol_r, _m0_f @ e_pol_r,
            ones @ e_tor_r, _m0_f @ e_tor_r)


def get_e_mag_radial(b, db, aj):
    """Per-radius magnetic energy profiles (unfactored: multiply by fac=0.5*LFfac).

    Returns: (e_p_r, e_p_as_r, e_t_r, e_t_as_r, e_dip_r) each (n_r_max,)
    """
    dLh_lm = dLh.unsqueeze(1)
    or2_r = or2.unsqueeze(0)

    e_pol_r = dLh_lm * (
        dLh_lm * or2_r * _cc2real(b, _m_arr) + _cc2real(db, _m_arr))
    e_tor_r = dLh_lm * _cc2real(aj, _m_arr)

    ones = torch.ones(lm_max, dtype=torch.float64, device=e_pol_r.device)
    return (ones @ e_pol_r, _m0_f @ e_pol_r,
            ones @ e_tor_r, _m0_f @ e_tor_r,
            _l1_f @ e_pol_r)


# --- Dipole diagnostics ---

def get_dipole(b, db, aj, e_p_vol, e_t_vol):
    """Compute all 19 dipole.TAG value columns (cols 2-20, caller prepends time).

    Args:
        b, db, aj: magnetic field arrays (lm_max, n_r_max)
        e_p_vol, e_t_vol: volume-integrated OC magnetic energies (already fac-scaled)

    Returns: list of 19 floats matching Fortran dipole.TAG columns 2-20
    """
    fac = 0.5 * LFfac

    # --- Tilt angles from b at CMB (index 0) ---
    b10 = b[_lm_l1m0, 0]
    if _lm_l1m1 is not None:
        b11 = b[_lm_l1m1, 0]
    else:
        b11 = torch.tensor(0.0 + 0.0j, dtype=b.dtype, device=b.device)

    rad = 180.0 / math.pi
    theta_dip = rad * math.atan2(math.sqrt(2.0) * abs(b11).item(), b10.real.item())
    if theta_dip < 0.0:
        theta_dip = 180.0 + theta_dip
    if abs(b11).item() < 1e-20:
        phi_dip = 0.0
    else:
        phi_dip = -rad * math.atan2(b11.imag.item(), b11.real.item())

    # --- Per-mode energies at CMB (unfactored) ---
    b_cmb = b[:, 0]
    db_cmb = db[:, 0]
    aj_cmb = aj[:, 0]
    or2_cmb = or2[0]
    dLh_1d = dLh

    cc_b = _cc2real(b_cmb, st_lm2m)
    cc_db = _cc2real(db_cmb, st_lm2m)
    cc_aj = _cc2real(aj_cmb, st_lm2m)

    e_p_lm = dLh_1d * (dLh_1d * or2_cmb * cc_b + cc_db)  # (lm_max,)
    e_t_lm = dLh_1d * cc_aj

    e_cmb = fac * (e_p_lm.sum() + e_t_lm.sum()).item()
    e_dip_cmb = fac * (e_p_lm * _l1_f).sum().item()
    e_dipole_ax_cmb = fac * (e_p_lm * _l1m0_f).sum().item()
    e_es_cmb = fac * (e_p_lm * _lpm_odd_f).sum().item()  # mag ES: (l+m) odd
    e_as_cmb = fac * (e_p_lm * _m0_f).sum().item()

    # Geostrophic (l <= 11, poloidal only at CMB)
    e_geo = fac * (e_p_lm * _l_le_lgeo_f).sum().item()
    e_es_geo = fac * (e_p_lm * _l_le_lgeo_es_f).sum().item()
    e_as_geo = fac * (e_p_lm * _l_le_lgeo_as_f).sum().item()

    # --- Volume-integrated dipole energies ---
    dLh_lm2 = dLh.unsqueeze(1)
    or2_r = or2.unsqueeze(0)
    e_pol_r = dLh_lm2 * (
        dLh_lm2 * or2_r * _cc2real(b, _m_arr) + _cc2real(db, _m_arr))

    e_dipole = fac * rInt_R(_l1_f @ e_pol_r).item()
    e_dipole_ax = fac * rInt_R(_l1m0_f @ e_pol_r).item()

    # --- Ratios ---
    e_tot = e_p_vol + e_t_vol

    return [
        theta_dip,                                # col 2
        phi_dip,                                  # col 3
        e_dipole_ax / e_tot,                      # col 4
        e_dipole_ax_cmb / e_cmb,                  # col 5
        e_dipole_ax_cmb / e_geo,                  # col 6
        e_dipole / e_tot,                         # col 7
        e_dip_cmb / e_cmb,                        # col 8
        e_dip_cmb / e_geo,                        # col 9
        e_dip_cmb,                                # col 10
        e_dipole_ax_cmb,                          # col 11
        e_dipole,                                 # col 12
        e_dipole_ax,                              # col 13
        e_cmb,                                    # col 14
        e_geo,                                    # col 15
        0.0,                                      # col 16: e_p_e_ratio (no imposed field)
        (e_cmb - e_es_cmb) / e_cmb,              # col 17
        (e_cmb - e_as_cmb) / e_cmb,              # col 18
        (e_geo - e_es_geo) / e_geo,              # col 19
        (e_geo - e_as_geo) / e_geo,              # col 20
    ]


# --- Trapezoidal time-averaging accumulator ---

class RadialAccumulator:
    """Trapezoidal time-integration of radial profiles, matching Fortran."""

    def __init__(self, n_profiles, n_r):
        self.profiles = [torch.zeros(n_r, dtype=torch.float64) for _ in range(n_profiles)]
        self.n_e_sets = 0
        self.time_tot = 0.0
        self.time_last = 0.0

    def accumulate(self, time, *new_profiles):
        self.n_e_sets += 1
        dt = time - self.time_last
        if self.n_e_sets == 1:
            for i, p in enumerate(new_profiles):
                self.profiles[i] = p.clone().to(dtype=torch.float64, device='cpu')
            self.time_tot = 1.0  # placeholder
        elif self.n_e_sets == 2:
            for i, p in enumerate(new_profiles):
                self.profiles[i] = dt * (self.profiles[i] + p.to(dtype=torch.float64, device='cpu'))
            self.time_tot = 2.0 * dt
        else:
            for i, p in enumerate(new_profiles):
                self.profiles[i] += dt * p.to(dtype=torch.float64, device='cpu')
            self.time_tot += dt
        self.time_last = time


# --- File writers for new output files ---

def write_radius_file(path):
    """Write radius.TAG matching Fortran '(I4, ES16.8)'."""
    with open(path, 'w') as f:
        for i in range(n_r_max):
            f.write(f"{i + 1:4d}{r[i].item():16.8E}\n")


def write_signal_file(path):
    """Write signal.TAG (just 'NOT')."""
    with open(path, 'w') as f:
        f.write("NOT\n")


def write_timestep_line(f, time, dt):
    """Write one line to timestep.TAG matching Fortran '(1P, ES20.12, ES16.8)'."""
    f.write(f"{time:20.12E}{dt:16.8E}\n")


def write_eKinR_file(path, accum):
    """Write eKinR.TAG matching Fortran '(ES20.10, 8ES15.7)'.

    accum.profiles: [e_p_r, e_p_as_r, e_t_r, e_t_as_r]
    """
    fac = 0.5  # eScale = 1.0
    tt = accum.time_tot
    with open(path, 'w') as f:
        for nR in range(n_r_max):
            e_p = fac * accum.profiles[0][nR].item() / tt
            e_p_as = fac * accum.profiles[1][nR].item() / tt
            e_t = fac * accum.profiles[2][nR].item() / tt
            e_t_as = fac * accum.profiles[3][nR].item() / tt
            osurf = 0.25 / math.pi * or2[nR].item()
            f.write(f"{r[nR].item():20.10E}"
                    f"{e_p:15.7E}{e_p_as:15.7E}{e_t:15.7E}{e_t_as:15.7E}"
                    f"{e_p * osurf:15.7E}{e_p_as * osurf:15.7E}"
                    f"{e_t * osurf:15.7E}{e_t_as * osurf:15.7E}\n")


def write_eMagR_file(path, accum):
    """Write eMagR.TAG matching Fortran '(ES20.10, 9ES15.7)'.

    accum.profiles: [e_p_r, e_p_as_r, e_t_r, e_t_as_r, e_dip_r]
    """
    fac = 0.5 * LFfac  # eScale = 1.0
    tt = accum.time_tot
    with open(path, 'w') as f:
        for nR in range(n_r_max):
            e_p = fac * accum.profiles[0][nR].item() / tt
            e_p_as = fac * accum.profiles[1][nR].item() / tt
            e_t = fac * accum.profiles[2][nR].item() / tt
            e_t_as = fac * accum.profiles[3][nR].item() / tt
            e_dip = fac * accum.profiles[4][nR].item() / tt
            osurf = 0.25 / math.pi * or2[nR].item()
            eTot = e_p + e_t
            eDR = e_dip / eTot if e_dip >= 1e-6 * eTot else 0.0
            f.write(f"{r[nR].item():20.10E}"
                    f"{e_p:15.7E}{e_p_as:15.7E}{e_t:15.7E}{e_t_as:15.7E}"
                    f"{e_p * osurf:15.7E}{e_p_as * osurf:15.7E}"
                    f"{e_t * osurf:15.7E}{e_t_as * osurf:15.7E}"
                    f"{eDR:15.7E}\n")


def write_dipole_line(f, time, cols):
    """Write one line to dipole.TAG matching Fortran '(1P, ES20.12, 19ES14.6)'."""
    f.write(f"{time:20.12E}")
    for v in cols:
        f.write(f"{v:14.6E}")
    f.write("\n")


# --- Heat diagnostics (outMisc.f90) ---

def _round_off(val, ref):
    """Zero near-zero values relative to reference, matching useful.f90:round_off."""
    eps = 2.220446049250313e-16  # float64 machine epsilon
    if abs(ref) > 0 and abs(val) < 1e3 * eps * abs(ref):
        return 0.0
    return val


class MeanSD:
    """Welford online weighted mean + variance, matching mean_sd.f90."""

    def __init__(self, n_r):
        self.mean = torch.zeros(n_r, dtype=torch.float64)
        self.SD = torch.zeros(n_r, dtype=torch.float64)
        self.n_calls = 0

    def compute(self, input_data, dt, total_time):
        """Update mean and variance. dt=timePassed, total_time=timeNorm."""
        self.n_calls += 1
        data = input_data.to(dtype=torch.float64) if input_data.dtype != torch.float64 else input_data
        if self.n_calls == 1:
            self.mean = data.clone()
            self.SD = torch.zeros_like(self.mean)
        else:
            delta = data - self.mean
            self.mean = self.mean + delta * (dt / total_time)
            self.SD = self.SD + dt * delta * (data - self.mean)  # uses UPDATED mean

    def finalize(self, total_time):
        self.SD = torch.sqrt(self.SD / total_time)


def get_heat_data(s00, ds00, p00):
    """Compute 16 heat.TAG value columns from l=0,m=0 spectral coefficients.

    Args: s00, ds00, p00 are (n_r_max,) real tensors = field[0,:].real
    Returns: list of 16 floats (cols 2-17)

    Implements outMisc.f90:543-656, Boussinesq entropy diffusion path.
    """
    from .radial_functions import temp0, rho0, alpha0, orho1, kappa, ViscHeatFac, ThExpNb, ogrun
    from .radial_scheme import r_cmb, r_icb
    from .init_fields import topcond, botcond, deltacond
    from .pre_calculations import lScale
    from .params import l_chemical_conv

    _osq4pi = 1.0 / math.sqrt(4.0 * math.pi)
    _sq4pi = math.sqrt(4.0 * math.pi)

    # Boundary values (outMisc.f90:545-546, 474-475)
    botentropy = _osq4pi * s00[-1].item()
    topentropy = _osq4pi * s00[0].item()
    toppres = _osq4pi * p00[0].item()
    botpres = _osq4pi * p00[-1].item()

    # Temperature (outMisc.f90:548-553, Boussinesq: ViscHeatFac=0)
    bottemp = (temp0[-1].item() * botentropy +
               ViscHeatFac * ThExpNb * orho1[-1].item() * temp0[-1].item() *
               alpha0[-1].item() * botpres)
    toptemp = (temp0[0].item() * topentropy +
               ViscHeatFac * ThExpNb * orho1[0].item() * temp0[0].item() *
               alpha0[0].item() * toppres)

    # Nusselt (outMisc.f90:589-597, entropy diffusion)
    if abs(botcond) >= 1e-10:
        botnuss = -_osq4pi / botcond * ds00[-1].item() / lScale
    else:
        botnuss = 1.0
    if abs(topcond) >= 1e-10:
        topnuss = -_osq4pi / topcond * ds00[0].item() / lScale
    else:
        topnuss = 1.0

    # Heat fluxes (outMisc.f90:599-602, entropy diffusion)
    # NOTE: NO osq4pi on ds00 — Fortran uses raw real(ds(1,...))
    botflux = (-rho0[-1].item() * temp0[-1].item() * ds00[-1].item() *
               float(r_icb) ** 2 * _sq4pi * kappa[-1].item() / lScale)
    topflux = (-rho0[0].item() * temp0[0].item() * ds00[0].item() / lScale *
               float(r_cmb) ** 2 * _sq4pi * kappa[0].item())

    # Delta Nusselt (outMisc.f90:603-606)
    if botentropy != topentropy:
        deltanuss = deltacond / (botentropy - topentropy)
    else:
        deltanuss = 1.0

    # Mass (outMisc.f90:468-470, 639-640)
    tmp = (_osq4pi * ThExpNb * alpha0 *
           (-rho0 * temp0 * s00 + ViscHeatFac * ogrun * p00))
    mass = (4.0 * math.pi * rInt_R(tmp * r * r)).item()

    # Flush small values (outMisc.f90:648-650)
    if abs(toppres) <= 1e-11:
        toppres = 0.0
    if abs(mass) <= 1e-11:
        mass = 0.0

    # Chemical convection (outMisc.f90:631-637)
    if l_chemical_conv:
        topxi = 0.0
        botxi = 0.0
        topsherwood = 1.0
        botsherwood = 1.0
        deltasherwood = 1.0
    else:
        topxi = 0.0
        botxi = 0.0
        topsherwood = 1.0
        botsherwood = 1.0
        deltasherwood = 1.0

    return [botnuss, topnuss, deltanuss,
            bottemp, toptemp, botentropy, topentropy,
            botflux, topflux, toppres, mass,
            topsherwood, botsherwood, deltasherwood, botxi, topxi]


def update_heat_means(smean, tmean, pmean, rhomean, ximean,
                      s00, p00, dt, total_time):
    """Update all 5 heatR accumulators in correct order (outMisc.f90:458-470).

    Args:
        smean, tmean, pmean, rhomean, ximean: MeanSD accumulators
        s00, p00: (n_r_max,) real tensors = field[0,:].real (raw spectral)
        dt: timePassed (time since last call)
        total_time: timeNorm (cumulative time)
    """
    from .radial_functions import temp0, rho0, alpha0, orho1, ogrun, ViscHeatFac, ThExpNb
    from .params import l_chemical_conv

    _osq4pi = 1.0 / math.sqrt(4.0 * math.pi)

    # 1. Entropy mean (outMisc.f90:459)
    smean.compute(_osq4pi * s00, dt, total_time)

    # 2. Temperature mean — uses UPDATED smean.mean, OLD pmean.mean (outMisc.f90:460-462)
    tmp_t = (temp0 * smean.mean +
             ViscHeatFac * ThExpNb * alpha0 * temp0 * orho1 * pmean.mean)
    tmean.compute(tmp_t, dt, total_time)

    # 3. Composition (outMisc.f90:464-465)
    if l_chemical_conv:
        pass  # ximean.compute(...)

    # 4. Pressure mean — NOW updated (outMisc.f90:467)
    pmean.compute(_osq4pi * p00, dt, total_time)

    # 5. Density mean — uses raw spectral values (outMisc.f90:468-470)
    tmp_rho = (_osq4pi * ThExpNb * alpha0 *
               (-rho0 * temp0 * s00 + ViscHeatFac * ogrun * p00))
    rhomean.compute(tmp_rho, dt, total_time)


def write_heat_line(f, time, cols):
    """Write one line to heat.TAG matching Fortran '(1P,ES20.12,16ES16.8)'."""
    f.write(f"{time:20.12E}")
    for v in cols:
        f.write(f"{v:16.8E}")
    f.write("\n")


def write_heatR_file(path, smean, tmean, pmean, rhomean, ximean):
    """Write heatR.TAG matching Fortran '(ES20.10,5ES15.7,5ES13.5)'."""
    # Precompute round_off references (Fortran maxval = algebraic max)
    refs = [smean.mean.max(), tmean.mean.max(), pmean.mean.max(),
            rhomean.mean.max(), ximean.mean.max(),
            smean.SD.max(), tmean.SD.max(), pmean.SD.max(),
            rhomean.SD.max(), ximean.SD.max()]
    refs = [x.item() for x in refs]

    with open(path, 'w') as f:
        for nR in range(n_r_max):
            vals = [smean.mean[nR], tmean.mean[nR], pmean.mean[nR],
                    rhomean.mean[nR], ximean.mean[nR]]
            sds = [smean.SD[nR], tmean.SD[nR], pmean.SD[nR],
                   rhomean.SD[nR], ximean.SD[nR]]
            f.write(f"{r[nR].item():20.10E}")
            for i, v in enumerate(vals):
                f.write(f"{_round_off(v.item(), refs[i]):15.7E}")
            for i, v in enumerate(sds):
                f.write(f"{_round_off(v.item(), refs[5 + i]):13.5E}")
            f.write("\n")


# --- Par diagnostics (getDlm.f90, outPar.f90, output.f90) ---

# Precomputed index arrays for get_dlm scatter_add
_l_expanded = st_lm2l.unsqueeze(1).expand(-1, n_r_max)  # (lm_max, n_r_max)
_m_expanded = st_lm2m.unsqueeze(1).expand(-1, n_r_max)  # (lm_max, n_r_max)
_eps = 2.220446049250313e-16  # float64 machine epsilon


def get_dlm(w, dw, z, switch='V'):
    """Compute integral lengthscales matching getDlm.f90.

    Returns: (dl, dlR, dm, dlc, dlPolPeak, dlRc, dlPolPeakR)
      dl, dlc, dm, dlPolPeak: scalars (floats)
      dlR, dlRc, dlPolPeakR: (n_r_max,) tensors
    """
    dLh_lm = dLh.unsqueeze(1)   # (lm_max, 1)
    or2_r = or2.unsqueeze(0)     # (1, n_r_max)

    # Per-lm energy contributions (lm_max, n_r_max)
    e_p_lm = dLh_lm * (dLh_lm * or2_r * _cc2real(w, _m_arr) + _cc2real(dw, _m_arr))
    e_t_lm = dLh_lm * _cc2real(z, _m_arr)

    if switch == 'V':
        orho1_r = orho1.unsqueeze(0)
        e_p_lm = orho1_r * e_p_lm
        e_t_lm = orho1_r * e_t_lm
        lFirst = 1
    else:  # 'B'
        lFirst = 2

    e_total = e_p_lm + e_t_lm  # (lm_max, n_r_max)

    # Aggregate per-l: scatter_add over lm -> l
    e_lr = torch.zeros(l_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
    e_lr.scatter_add_(0, _l_expanded, e_total)

    e_pol_lr = torch.zeros(l_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
    e_pol_lr.scatter_add_(0, _l_expanded, e_p_lm)

    # Per-m aggregation
    e_mr = torch.zeros(m_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
    e_mr.scatter_add_(0, _m_expanded[:, :n_r_max], e_total)

    # Convective (m != 0) for 'V' only
    if switch == 'V':
        m_ne0 = (st_lm2m != 0).unsqueeze(1)  # (lm_max, 1)
        e_conv = e_total * m_ne0
        e_lr_c = torch.zeros(l_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
        e_lr_c.scatter_add_(0, _l_expanded, e_conv)

    # --- Global scalars ---
    fac = 0.5  # half * eScale, eScale=1.0

    # Per-l integrated energy
    e_l_integrated = torch.zeros(l_max + 1, dtype=torch.float64, device=e_total.device)
    e_pol_l_integrated = torch.zeros(l_max + 1, dtype=torch.float64, device=e_total.device)
    for l in range(lFirst, l_max + 1):
        e_l_integrated[l] = fac * rInt_R(e_lr[l, :])
        e_pol_l_integrated[l] = fac * rInt_R(e_pol_lr[l, :])

    E_slice = e_l_integrated[lFirst:]
    E = E_slice.sum().item()
    l_range = torch.arange(lFirst, l_max + 1, dtype=torch.float64, device=e_total.device)
    EL = (l_range * E_slice).sum().item()

    dl = math.pi * E / EL if EL != 0.0 else 0.0

    # dlPolPeak: Fortran uses e_pol_l(1:l_max) where indices 1..l_max correspond to l=1..l_max
    # maxloc returns 1-based index into that subarray
    e_pol_l = e_pol_l_integrated[1:l_max + 1]  # l=1 to l=l_max (size l_max)
    lpeak = torch.argmax(e_pol_l).item() + 1  # +1 because l starts at 1
    dlPolPeak = math.pi / lpeak

    if switch == 'V':
        e_l_c_integrated = torch.zeros(l_max + 1, dtype=torch.float64, device=e_total.device)
        for l in range(lFirst, l_max + 1):
            e_l_c_integrated[l] = fac * rInt_R(e_lr_c[l, :])
        Ec = e_l_c_integrated[lFirst:].sum().item()
        ELc = (l_range * e_l_c_integrated[lFirst:]).sum().item()
        dlc = math.pi * Ec / ELc if ELc != 0.0 else 0.0
    else:
        dlc = 0.0

    # dm: sum over m = minc, 2*minc, ..., m_max
    E_m = 0.0
    EM_m = 0.0
    for m in range(minc, m_max + 1, minc):
        e_m = fac * rInt_R(e_mr[m, :]).item()
        E_m += e_m
        EM_m += m * e_m
    dm = math.pi * E_m / EM_m if EM_m != 0.0 else 0.0

    # --- Radial profiles ---
    if switch == 'V':
        # dlR, dlRc, dlPolPeakR per radius
        ER = fac * e_lr[lFirst:].sum(dim=0)   # (n_r_max,)
        ELR = fac * (l_range.unsqueeze(1) * e_lr[lFirst:]).sum(dim=0)
        dlR = torch.where(ELR > 10.0 * _eps, math.pi * ER / ELR,
                          torch.zeros_like(ELR))

        ERc = fac * e_lr_c[lFirst:].sum(dim=0)
        ELRc = fac * (l_range.unsqueeze(1) * e_lr_c[lFirst:]).sum(dim=0)
        dlRc = torch.where(ELRc > 10.0 * _eps, math.pi * ERc / ELRc,
                           torch.zeros_like(ELRc))

        # Per-radius peak of poloidal energy
        # Fortran: e_pol_l(:) = e_pol_lr_global(nR, 1:l_max)
        e_pol_per_r = e_pol_lr[1:l_max + 1, :]  # (l_max, n_r_max), l=1..l_max
        max_vals, max_idx = e_pol_per_r.max(dim=0)  # (n_r_max,)
        lpeak_r = max_idx + 1  # l starts at 1
        dlPolPeakR = torch.where(max_vals > 10.0 * _eps,
                                 math.pi / lpeak_r.to(dtype=torch.float64),
                                 torch.zeros(n_r_max, dtype=torch.float64,
                                             device=e_total.device))
    else:  # 'B'
        dlR = torch.zeros(n_r_max, dtype=torch.float64, device=e_total.device)
        dlRc = torch.zeros(n_r_max, dtype=torch.float64, device=e_total.device)
        dlPolPeakR = torch.zeros(n_r_max, dtype=torch.float64, device=e_total.device)

    return (dl, dlR, dm, dlc, dlPolPeak, dlRc, dlPolPeakR)


USquare = namedtuple('USquare', 'e_p e_t e_p_as e_t_as Ro Rm Rol dl RolC dlc')


def get_u_square(w, dw, z):
    """Compute squared velocity u² diagnostics (kinetic_energy.f90 get_u_square).

    Like get_e_kin_full but weighted by orho2 (1/rho²) instead of orho1 (1/rho),
    plus length scale and Rossby/Reynolds number diagnostics.
    Only meaningful for anelastic (l_anel=True).

    Returns USquare namedtuple with 10 components matching u_square.TAG cols 2-11.
    """
    dLh_lm = dLh.unsqueeze(1)   # (lm_max, 1)
    or2_r = or2.unsqueeze(0)     # (1, n_r_max)
    orho2_r = orho2.unsqueeze(0) # (1, n_r_max)

    # Per-mode radial profiles (lm_max, n_r_max) — no 0.5 factor yet
    e_p_lm = orho2_r * dLh_lm * (dLh_lm * or2_r * _cc2real(w, _m_arr) + _cc2real(dw, _m_arr))
    e_t_lm = orho2_r * dLh_lm * _cc2real(z, _m_arr)
    e_total = e_p_lm + e_t_lm  # (lm_max, n_r_max)

    # Total and axisymmetric sums
    ones = torch.ones(lm_max, dtype=torch.float64, device=e_p_lm.device)
    fac = 0.5 * eScale

    e_p = fac * rInt_R(ones @ e_p_lm).item()
    e_t = fac * rInt_R(ones @ e_t_lm).item()
    e_p_as = fac * rInt_R(_m0_f @ e_p_lm).item()
    e_t_as = fac * rInt_R(_m0_f @ e_t_lm).item()
    e_kin = e_p + e_t

    # Per-l energy decomposition via scatter_add
    e_lr = torch.zeros(l_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
    e_lr.scatter_add_(0, _l_expanded, e_total)

    # Convective (m != 0) per-l
    m_ne0 = (st_lm2m != 0).unsqueeze(1)  # (lm_max, 1)
    e_conv = e_total * m_ne0
    e_lr_c = torch.zeros(l_max + 1, n_r_max, dtype=torch.float64, device=e_total.device)
    e_lr_c.scatter_add_(0, _l_expanded, e_conv)

    # Rossby / Reynolds numbers
    vol = float(vol_oc)
    Re = math.sqrt(2.0 * e_kin / vol) if e_kin > 0.0 else 0.0
    e_kin_conv = e_kin - e_p_as - e_t_as
    ReConv = math.sqrt(2.0 * e_kin_conv / vol) if e_kin_conv > 0.0 else 0.0

    # l_non_rot = ek < 0 (step_time.py)
    l_non_rot = ek < 0.0
    if l_non_rot:
        Ro = 0.0
        RoConv = 0.0
    else:
        Ro = Re * ek
        RoConv = ReConv * ek

    # Magnetic Reynolds number (nVarCond=0 path)
    Rm = Re * prmag if prmag != 0 else Re

    # Length scales: dl = pi*E/EL — batched rInt_R over all l
    e_l_all = fac * rInt_R(e_lr[1:, :])  # (l_max,) — radial integral per l
    l_range = torch.arange(1, l_max + 1, dtype=torch.float64, device=e_total.device)
    E = e_l_all.sum().item()
    EL = (l_range * e_l_all).sum().item()

    e_l_c_all = fac * rInt_R(e_lr_c[1:, :])  # (l_max,)
    Ec = e_l_c_all.sum().item()
    ELc = (l_range * e_l_c_all).sum().item()

    dl = math.pi * E / EL if EL != 0.0 else 0.0
    dlc = math.pi * Ec / ELc if ELc != 0.0 else 0.0

    # Local Rossby
    Rol = Ro / dl if dl != 0.0 else Ro
    RolC = RoConv / dlc if dlc != 0.0 else RoConv

    return USquare(e_p, e_t, e_p_as, e_t_as, Ro, Rm, Rol, dl, RolC, dlc)


def write_u_square_line(f, time, usq):
    """Write one line to u_square.TAG matching Fortran '(1P,ES20.12,10ES16.8)'."""
    f.write(f"{time:20.12E}{usq.e_p:16.8E}{usq.e_t:16.8E}"
            f"{usq.e_p_as:16.8E}{usq.e_t_as:16.8E}"
            f"{usq.Ro:16.8E}{usq.Rm:16.8E}{usq.Rol:16.8E}"
            f"{usq.dl:16.8E}{usq.RolC:16.8E}{usq.dlc:16.8E}\n")


# --- Hemisphere diagnostic (outMisc.f90 get_hemi / outHemi) ---

# Hemisphere mask: True = Northern hemisphere (sorted theta index < n_theta_max/2)
from .horizontal_data import n_theta_cal2ord, gauss_grid, O_sin_theta_E2_grid, _grid_idx
# Hemisphere mask: True = Northern hemisphere in interleaved grid-space order.
# _grid_idx[i] gives the sorted (geographic) theta index for interleaved position i.
# Sorted indices 0..n_theta_max/2-1 are Northern hemisphere.
_north_mask = (_grid_idx < n_theta_max // 2)  # (n_theta_max,) — even indices = North

Hemi = namedtuple('Hemi', 'hemi_vr hemi_ekin hemi_br hemi_emag hemi_cmb ekin_total emag_total')


def get_hemi(vr, vt, vp, br, bt, bp):
    """Compute hemisphere diagnostics (outMisc.f90 get_hemi + outHemi).

    Args:
        vr, vt, vp: velocity fields (n_r_max, n_theta_max, n_phi_max) from batched SHT
        br, bt, bp: magnetic fields (n_r_max, n_theta_max, n_phi_max) or None if not l_mag

    Returns:
        Hemi namedtuple with 7 components matching hemi.TAG cols 2-8.
    """
    phiNorm = 2.0 * math.pi / float(n_phi_max)

    # Gauss weights: (n_theta_max,) -> (1, n_theta_max, 1) for broadcasting
    gw = (phiNorm * gauss_grid).unsqueeze(0).unsqueeze(2)  # (1, n_theta_max, 1)
    # O_sin_theta_E2 in grid order: (n_theta_max,) -> (1, n_theta_max, 1)
    oste2 = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)
    # or2: (n_r_max,) -> (n_r_max, 1, 1)
    or2_3d = or2.unsqueeze(1).unsqueeze(2)
    # orho1: (n_r_max,) -> (n_r_max, 1, 1)
    orho1_3d = orho1.unsqueeze(1).unsqueeze(2)

    # Hemisphere mask: (1, n_theta_max, 1)
    north = _north_mask.unsqueeze(0).unsqueeze(2)

    # --- Velocity hemisphere ---
    # en = 0.5 * orho1 * (or2*vr^2 + O_sin_theta_E2*(vt^2 + vp^2))
    en_v = 0.5 * orho1_3d * (or2_3d * vr * vr + oste2 * (vt * vt + vp * vp))
    # vrabs = orho1 * |vr|
    vrabs_v = orho1_3d * vr.abs()

    # Weighted per-radial-level, summed over phi: (n_r_max, n_theta_max, n_phi_max)
    en_v_weighted = gw * en_v    # (n_r_max, n_theta_max, n_phi_max)
    vrabs_v_weighted = gw * vrabs_v

    # Sum over phi, then split hemisphere, then sum over theta -> (n_r_max,)
    # North
    ekin_r_N = en_v_weighted[:, _north_mask, :].sum(dim=(1, 2))
    ekin_r_S = en_v_weighted[:, ~_north_mask, :].sum(dim=(1, 2))
    vrabs_r_N = vrabs_v_weighted[:, _north_mask, :].sum(dim=(1, 2))
    vrabs_r_S = vrabs_v_weighted[:, ~_north_mask, :].sum(dim=(1, 2))

    # Radial integration
    ekinN = eScale * rInt_R(ekin_r_N).item()
    ekinS = eScale * rInt_R(ekin_r_S).item()
    vrabsN = vScale * rInt_R(vrabs_r_N).item()
    vrabsS = vScale * rInt_R(vrabs_r_S).item()

    ekin_total = ekinN + ekinS
    if ekin_total > 0.0:
        hemi_ekin = abs(ekinN - ekinS) / ekin_total
        hemi_vr = abs(vrabsN - vrabsS) / (vrabsN + vrabsS)
    else:
        hemi_ekin = 0.0
        hemi_vr = 0.0

    # --- Magnetic hemisphere ---
    if br is not None:
        en_b = 0.5 * (or2_3d * br * br + oste2 * (bt * bt + bp * bp))
        brabs_b = br.abs()

        en_b_weighted = gw * en_b
        brabs_b_weighted = gw * brabs_b

        emag_r_N = en_b_weighted[:, _north_mask, :].sum(dim=(1, 2))
        emag_r_S = en_b_weighted[:, ~_north_mask, :].sum(dim=(1, 2))
        brabs_r_N = brabs_b_weighted[:, _north_mask, :].sum(dim=(1, 2))
        brabs_r_S = brabs_b_weighted[:, ~_north_mask, :].sum(dim=(1, 2))

        emagN = LFfac * eScale * rInt_R(emag_r_N).item()
        emagS = LFfac * eScale * rInt_R(emag_r_S).item()
        brabsN = rInt_R(brabs_r_N).item()
        brabsS = rInt_R(brabs_r_S).item()

        emag_total = emagN + emagS
        if emag_total > 0.0:
            hemi_emag = abs(emagN - emagS) / emag_total
            hemi_br = abs(brabsN - brabsS) / (brabsN + brabsS)
            # CMB is nR=0 (first radial level)
            brN_cmb = brabs_b_weighted[0, _north_mask, :].sum().item()
            brS_cmb = brabs_b_weighted[0, ~_north_mask, :].sum().item()
            if brN_cmb + brS_cmb > 0.0:
                hemi_cmb = abs(brN_cmb - brS_cmb) / (brN_cmb + brS_cmb)
            else:
                hemi_cmb = 0.0
        else:
            hemi_emag = 0.0
            hemi_br = 0.0
            hemi_cmb = 0.0
    else:
        emag_total = 0.0
        hemi_emag = 0.0
        hemi_br = 0.0
        hemi_cmb = 0.0

    return Hemi(hemi_vr, hemi_ekin, hemi_br, hemi_emag, hemi_cmb,
                ekin_total, emag_total)


def write_hemi_line(f, time, hemi):
    """Write one line to hemi.TAG matching Fortran '(1P,ES20.12,7ES16.8)'."""
    f.write(f"{time:20.12E}{_round_off(hemi.hemi_vr, 1.0):16.8E}"
            f"{_round_off(hemi.hemi_ekin, 1.0):16.8E}"
            f"{_round_off(hemi.hemi_br, 1.0):16.8E}"
            f"{_round_off(hemi.hemi_emag, 1.0):16.8E}"
            f"{_round_off(hemi.hemi_cmb, 1.0):16.8E}"
            f"{hemi.ekin_total:16.8E}{hemi.emag_total:16.8E}\n")


# --- Helicity diagnostics (outMisc.f90 get_helicity / outHelicity) ---

Helicity = namedtuple('Helicity',
                       'HelN HelS HelRMSN HelRMSS HelnaN HelnaS HelnaRMSN HelnaRMSS')


def get_helicity(vr, vt, vp, cvr, dvrdt, dvrdp, dvtdr, dvpdr):
    """Compute helicity diagnostics (outMisc.f90 get_helicity + outHelicity).

    All inputs shape (n_r_max, n_theta_max, n_phi_max) from batched SHTs.
    cvr = curl_r(z), dvrdt/dvrdp from pol_to_grad_spat(w),
    dvtdr/dvpdr from torpol_to_spat(dw, ddw, dz).

    Returns Helicity namedtuple with 8 components matching helicity.TAG cols 2-9.
    """
    # Reshape radial quantities: (n_r_max,) -> (n_r_max, 1, 1)
    or4_3d = or4.unsqueeze(1).unsqueeze(2)
    or2_3d = or2.unsqueeze(1).unsqueeze(2)
    orho2_3d = orho2.unsqueeze(1).unsqueeze(2)
    beta_3d = r_beta.unsqueeze(1).unsqueeze(2)

    # O_sin_theta_E2 in grid order: (1, n_theta_max, 1)
    oste2 = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)

    phiNorm = 1.0 / float(n_phi_max)

    # --- Phi-averaged fields (axisymmetric) ---
    # Mean over phi: (n_r_max, n_theta_max, n_phi_max) -> (n_r_max, n_theta_max, 1)
    vras = vr.mean(dim=2, keepdim=True)
    vtas = vt.mean(dim=2, keepdim=True)
    vpas = vp.mean(dim=2, keepdim=True)
    cvras = cvr.mean(dim=2, keepdim=True)
    dvrdtas = dvrdt.mean(dim=2, keepdim=True)
    dvrdpas = dvrdp.mean(dim=2, keepdim=True)
    dvtdras = dvtdr.mean(dim=2, keepdim=True)
    dvpdras = dvpdr.mean(dim=2, keepdim=True)

    # NOTE: Fortran uses sum/n_phi_max which is the same as mean. But the phiNorm
    # normalization is 1/n_phi_max applied to the sums. Since Fortran does
    # sum_over_phi / n_phi_max, that equals mean. ✓

    # --- Non-axisymmetric fields with beta corrections ---
    vrna = vr - vras
    cvrna = cvr - cvras
    vtna = vt - vtas
    vpna = vp - vpas
    dvrdpna = dvrdp - dvrdpas
    # dvpdrna = dvpdr - beta*vp - (dvpdras - beta*vpas)
    dvpdrna = dvpdr - beta_3d * vp - dvpdras + beta_3d * vpas
    # dvtdrna = dvtdr - beta*vt - (dvtdras - beta*vtas)
    dvtdrna = dvtdr - beta_3d * vt - dvtdras + beta_3d * vtas
    dvrdtna = dvrdt - dvrdtas

    # --- Full helicity at each point ---
    # Hel = or4*orho2*vr*cvr + or2*orho2*O_sin_theta_E2 * (
    #   vt*(or2*dvrdp - dvpdr + beta*vp) + vp*(dvtdr - beta*vt - or2*dvrdt))
    Hel = (or4_3d * orho2_3d * vr * cvr +
           or2_3d * orho2_3d * oste2 * (
               vt * (or2_3d * dvrdp - dvpdr + beta_3d * vp) +
               vp * (dvtdr - beta_3d * vt - or2_3d * dvrdt)))

    # --- Non-axisymmetric helicity ---
    Helna = (or4_3d * orho2_3d * vrna * cvrna +
             or2_3d * orho2_3d * oste2 * (
                 vtna * (or2_3d * dvrdpna - dvpdrna) +
                 vpna * (dvtdrna - or2_3d * dvrdtna)))

    # --- Weighted sums per hemisphere ---
    # Weight: phiNorm * gauss_grid(nTheta)
    gw = (phiNorm * gauss_grid).unsqueeze(0).unsqueeze(2)  # (1, n_theta_max, 1)

    Hel_w = gw * Hel         # (n_r_max, n_theta_max, n_phi_max)
    Hel2_w = gw * Hel * Hel
    Helna_w = gw * Helna
    Helna2_w = gw * Helna * Helna

    # Sum over phi and theta, split by hemisphere -> (n_r_max,)
    HelN_r = Hel_w[:, _north_mask, :].sum(dim=(1, 2))
    HelS_r = Hel_w[:, ~_north_mask, :].sum(dim=(1, 2))
    Hel2N_r = Hel2_w[:, _north_mask, :].sum(dim=(1, 2))
    Hel2S_r = Hel2_w[:, ~_north_mask, :].sum(dim=(1, 2))
    HelnaN_r = Helna_w[:, _north_mask, :].sum(dim=(1, 2))
    HelnaS_r = Helna_w[:, ~_north_mask, :].sum(dim=(1, 2))
    Helna2N_r = Helna2_w[:, _north_mask, :].sum(dim=(1, 2))
    Helna2S_r = Helna2_w[:, ~_north_mask, :].sum(dim=(1, 2))
    # HelEAAS: North gets +Hel, South gets -Hel
    HelEA_r = HelN_r - HelS_r  # (n_r_max,)

    # --- Radial integration (outHelicity) ---
    # Integration weight: r^2, factor 2*pi, volume norm = vol_oc/2 per hemisphere
    r2 = r * r
    half_vol = float(vol_oc) / 2.0
    fac = 2.0 * math.pi / half_vol

    HelN = fac * rInt_R(HelN_r * r2).item()
    HelS = fac * rInt_R(HelS_r * r2).item()
    HelnaN = fac * rInt_R(HelnaN_r * r2).item()
    HelnaS = fac * rInt_R(HelnaS_r * r2).item()
    HelEA = 2.0 * math.pi / float(vol_oc) * rInt_R(HelEA_r * r2).item()
    HelRMSN = math.sqrt(abs(fac * rInt_R(Hel2N_r * r2).item()))
    HelRMSS = math.sqrt(abs(fac * rInt_R(Hel2S_r * r2).item()))
    HelnaRMSN = math.sqrt(abs(fac * rInt_R(Helna2N_r * r2).item()))
    HelnaRMSS = math.sqrt(abs(fac * rInt_R(Helna2S_r * r2).item()))

    # Relative helicity: Hel / HelRMS per hemisphere
    HelRMS = HelRMSN + HelRMSS
    HelnaRMS = HelnaRMSN + HelnaRMSS

    if HelnaRMS != 0.0:
        HelnaN = HelnaN / HelnaRMSN if HelnaRMSN != 0.0 else 0.0
        HelnaS = HelnaS / HelnaRMSS if HelnaRMSS != 0.0 else 0.0
    else:
        HelnaN = 0.0
        HelnaS = 0.0

    if HelRMS != 0.0:
        HelN = HelN / HelRMSN if HelRMSN != 0.0 else 0.0
        HelS = HelS / HelRMSS if HelRMSS != 0.0 else 0.0
    else:
        HelN = 0.0
        HelS = 0.0

    return Helicity(HelN, HelS, HelRMSN, HelRMSS, HelnaN, HelnaS, HelnaRMSN, HelnaRMSS)


def write_helicity_line(f, time, hel):
    """Write one line to helicity.TAG matching Fortran '(1P,ES20.12,8ES16.8)'."""
    f.write(f"{time:20.12E}{hel.HelN:16.8E}{hel.HelS:16.8E}"
            f"{hel.HelRMSN:16.8E}{hel.HelRMSS:16.8E}"
            f"{hel.HelnaN:16.8E}{hel.HelnaS:16.8E}"
            f"{hel.HelnaRMSN:16.8E}{hel.HelnaRMSS:16.8E}\n")


# --- Power diagnostics (power.f90 get_power / get_visc_heat) ---

from .horizontal_data import cosn_theta_E2_grid

Power = namedtuple('Power',
                    'buoy buoy_chem z10ICB_term z10CMB_term viscDiss ohmDiss '
                    'powerMA powerIC powerDiff eDiffInt_norm')


def get_visc_heat(vr, vt, vp, cvr, dvrdr, dvrdt, dvrdp, dvtdr, dvtdp, dvpdr, dvpdp):
    """Compute viscous heating at all radial levels (power.f90 get_visc_heat).

    All inputs shape (n_r_max, n_theta_max, n_phi_max) from batched SHTs.
    Returns viscASr (n_r_max,): viscous heating per radial level (before eScale*rInt_R).
    """
    phiNorm = 2.0 * math.pi / float(n_phi_max)

    # Reshape radial quantities: (n_r_max,) -> (n_r_max, 1, 1)
    or2_3d = or2.unsqueeze(1).unsqueeze(2)
    or1_3d = or1.unsqueeze(1).unsqueeze(2)
    orho1_3d = orho1.unsqueeze(1).unsqueeze(2)
    visc_3d = r_visc.unsqueeze(1).unsqueeze(2)
    beta_3d = r_beta.unsqueeze(1).unsqueeze(2)
    r_3d = r.unsqueeze(1).unsqueeze(2)

    # Theta quantities: (1, n_theta_max, 1)
    csn2 = cosn_theta_E2_grid.unsqueeze(0).unsqueeze(2)
    oste2 = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)

    # Gauss weights: (1, n_theta_max, 1)
    gw = gauss_grid.unsqueeze(0).unsqueeze(2)

    # Strain rate terms (matching Fortran get_visc_heat exactly)
    # Term 1: 2*(dvrdr - (2*or1 + beta)*vr)^2
    t1 = 2.0 * (dvrdr - (2.0 * or1_3d + beta_3d) * vr) ** 2

    # Term 2: 2*(csn2*vt + dvpdp + dvrdr - or1*vr)^2
    t2 = 2.0 * (csn2 * vt + dvpdp + dvrdr - or1_3d * vr) ** 2

    # Term 3: 2*(dvpdp + csn2*vt + or1*vr)^2
    # Wait — re-reading Fortran: this is a separate term. Let me re-check.
    # Fortran line: two*(dvpdp + csn2*vt + or1*vr)**2
    t3 = 2.0 * (dvpdp + csn2 * vt + or1_3d * vr) ** 2

    # Term 6 (numbering from Fortran comments): (2*dvtdp + cvr - 2*csn2*vp)^2
    t6 = (2.0 * dvtdp + cvr - 2.0 * csn2 * vp) ** 2

    # Term 4: O_sin_theta_E2 * (r*dvtdr - (2+beta*r)*vt + or1*dvrdt)^2
    t4 = oste2 * (r_3d * dvtdr - (2.0 + beta_3d * r_3d) * vt + or1_3d * dvrdt) ** 2

    # Term 5: O_sin_theta_E2 * (r*dvpdr - (2+beta*r)*vp + or1*dvrdp)^2
    t5 = oste2 * (r_3d * dvpdr - (2.0 + beta_3d * r_3d) * vp + or1_3d * dvrdp) ** 2

    # Correction: -2/3*(beta*vr)^2
    t_corr = -2.0 / 3.0 * (beta_3d * vr) ** 2

    # Total viscous heating
    vh = or2_3d * orho1_3d * visc_3d * (t1 + t2 + t3 + t6 + t4 + t5 + t_corr)

    # Sum over phi and theta with gauss weights
    viscASr = (phiNorm * gw * vh).sum(dim=(1, 2))  # (n_r_max,)

    return viscASr


def get_power_spectral(w, s, xi, b, ddb, aj, dj):
    """Compute spectral contributions to power budget (buoyancy, ohmic dissipation).

    Args:
        w: poloidal velocity (lm_max, n_r_max)
        s: entropy (lm_max, n_r_max)
        xi: composition (lm_max, n_r_max) or None
        b, ddb, aj, dj: magnetic field (lm_max, n_r_max) or None

    Returns:
        (buoy, buoy_chem, curlB2): scalars (radially integrated)
    """
    dLh_lm = dLh.unsqueeze(1)  # (lm_max, 1)
    or2_r = or2.unsqueeze(0)   # (1, n_r_max)
    rgrav_r = rgrav.unsqueeze(0)  # (1, n_r_max)

    # --- Buoyancy power ---
    buoy = 0.0
    if l_heat:
        # buoy_r(nR) = Σ_lm eScale * dLh * BuoFac * rgrav(nR) * cc22real(w, s, m)
        buoy_lm = eScale * dLh_lm * BuoFac * rgrav_r * _cc22real(w, s, _m_arr)
        ones = torch.ones(lm_max, dtype=torch.float64, device=w.device)
        buoy = rInt_R(ones @ buoy_lm).item()

    # --- Chemical buoyancy ---
    buoy_chem = 0.0
    if l_chemical_conv and xi is not None:
        buoy_chem_lm = eScale * dLh_lm * ChemFac * rgrav_r * _cc22real(w, xi, _m_arr)
        ones = torch.ones(lm_max, dtype=torch.float64, device=w.device)
        buoy_chem = rInt_R(ones @ buoy_chem_lm).item()

    # --- Ohmic dissipation ---
    curlB2 = 0.0
    if l_mag and b is not None:
        lambda_r = r_lambda.unsqueeze(0)  # (1, n_r_max)
        # laplace = dLh*or2*b - ddb
        laplace = dLh_lm * or2_r * b - ddb
        # curlB2_r = Σ_lm LFfac*opm*eScale * dLh * lambda * (dLh*or2*cc2real(aj) + cc2real(dj) + cc2real(laplace))
        curlB2_lm = (LFfac * opm * eScale * dLh_lm * lambda_r *
                     (dLh_lm * or2_r * _cc2real(aj, _m_arr) +
                      _cc2real(dj, _m_arr) +
                      _cc2real(laplace, _m_arr)))
        ones = torch.ones(lm_max, dtype=torch.float64, device=b.device)
        curlB2 = rInt_R(ones @ curlB2_lm).item()

    return buoy, buoy_chem, curlB2


def write_power_line(f, time, pw):
    """Write one line to power.TAG matching Fortran '(1P,ES20.12,10ES16.8)'."""
    f.write(f"{time:20.12E}{pw.buoy:16.8E}{pw.buoy_chem:16.8E}"
            f"{pw.z10ICB_term:16.8E}{pw.z10CMB_term:16.8E}"
            f"{pw.viscDiss:16.8E}{pw.ohmDiss:16.8E}"
            f"{pw.powerMA:16.8E}{pw.powerIC:16.8E}"
            f"{pw.powerDiff:16.8E}{pw.eDiffInt_norm:16.8E}\n")


# --- Output diagnostic sweep: batched SHTs for grid-space diagnostics ---

from .sht import (torpol_to_spat, pol_to_curlr_spat, pol_to_grad_spat,
                  scal_to_spat)
from .horizontal_data import dPhi
from .radial_functions import or3 as _or3
from .precision import DTYPE

# Precomputed dLh for Q input in torpol_to_spat
_dLh_2d = dLh.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
_dPhi_2d = dPhi.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
# Precomputed radial factors for dvrdr (n_r_max,) -> (1, n_r_max)
_or2_2d = or2.unsqueeze(0)
_or3_2d = _or3.unsqueeze(0)


def output_diag_sweep(w, dw, ddw, z, dz,
                      b=None, db=None, aj=None,
                      need_hemi=True, need_helicity=True, need_visc_heat=True):
    """Batched SHTs at all radial levels for output diagnostics.

    Returns dict with computed diagnostics (Hemi, Helicity, viscASr).
    Skips SHTs/computations not needed.
    """
    result = {}

    # --- Velocity SHTs (all radial levels) ---
    # torpol_to_spat(dLh*w, dw, z) -> vr, vt, vp
    Q_vel = _dLh_2d * w
    vr_all, vt_all, vp_all = torpol_to_spat(Q_vel, dw, z)

    # --- Hemi: only needs vr, vt, vp (and optionally br, bt, bp) ---
    if need_hemi:
        br_all = bt_all = bp_all = None
        if l_mag and b is not None:
            Q_mag = _dLh_2d * b
            br_all, bt_all, bp_all = torpol_to_spat(Q_mag, db, aj)
        result['hemi'] = get_hemi(vr_all, vt_all, vp_all,
                                  br_all, bt_all, bp_all)

    # --- Additional SHTs for helicity and visc_heat ---
    if need_helicity or need_visc_heat:
        # cvr from pol_to_curlr_spat(z)
        cvr_all = pol_to_curlr_spat(z)

        # dvrdt, dvrdp from pol_to_grad_spat(w)
        dvrdt_all, dvrdp_all = pol_to_grad_spat(w)

        # dvtdr, dvpdr from torpol_to_spat(dLh*dw, ddw, dz)
        Q_dv = _dLh_2d * dw
        _, dvtdr_all, dvpdr_all = torpol_to_spat(Q_dv, ddw, dz)

    if need_helicity:
        result['helicity'] = get_helicity(
            vr_all, vt_all, vp_all, cvr_all,
            dvrdt_all, dvrdp_all, dvtdr_all, dvpdr_all)

    if need_visc_heat:
        # dvtdp, dvpdp from torpol_to_dphspat(dw, z) — inline via batched torpol_to_spat
        Slm_dph = _dPhi_2d * dw
        Tlm_dph = _dPhi_2d * z
        Qlm_dph = torch.zeros_like(Slm_dph)
        _, dvtdp_raw, dvpdp_raw = torpol_to_spat(Qlm_dph, Slm_dph, Tlm_dph)
        # Multiply by 1/sin²θ
        Ost2_grid = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)
        dvtdp_all = dvtdp_raw * Ost2_grid
        dvpdp_all = dvpdp_raw * Ost2_grid

        # dvrdr = dvrdrc from torpol_to_spat(dw, ddw, dz) = scal_to_spat(dLh*dw)
        # This is the RAW SHT output, NOT the physical dvr/dr.
        # The get_visc_heat formula applies its own or1/beta corrections.
        # (Fortran rIter.f90 line 551-553: torpol_to_spat(dw, ddw, dz, dvrdrc, ...))
        dvrdr_all = scal_to_spat(_dLh_2d * dw)

        result['viscASr'] = get_visc_heat(
            vr_all, vt_all, vp_all, cvr_all,
            dvrdr_all, dvrdt_all, dvrdp_all,
            dvtdr_all, dvtdp_all, dvpdr_all, dvpdp_all)

    return result


def get_elsAnel(b, db, aj):
    """Radially-integrated Elsasser number (unfactored, matching magnetic_energy.f90:342,455).

    els_r = (e_p_r + e_t_r) * orho1 * sigma, then rInt_R.
    """
    e_p_r, _, e_t_r, _, _ = get_e_mag_radial(b, db, aj)
    els_r = (e_p_r + e_t_r) * orho1 * sigma
    return rInt_R(els_r).item()


def get_e_mag_cmb(b, db, aj):
    """Total magnetic energy at CMB (FACTORED: 0.5*LFfac applied).

    e_mag_cmb = fac * (e_p_r[0] + e_t_r[0])
    Fortran: e_cmb = e_p_r_global(n_r_cmb) + e_t_r_global(n_r_cmb), then fac*e_cmb
    """
    e_p_r, _, e_t_r, _, _ = get_e_mag_radial(b, db, aj)
    fac = 0.5 * LFfac  # eScale=1.0
    return (fac * (e_p_r[0] + e_t_r[0])).item()


def get_par_data(ek, em, dip_cols, dlm_v, dlm_b,
                 elsAnel_val, e_mag_cmb_val):
    """Compute 19 par.TAG value columns (cols 2-20).

    Returns: list of 19 floats matching Fortran par.TAG columns 2-20.
    """
    e_kin = ek.e_p + ek.e_t
    e_kin_nas = e_kin - ek.e_p_as - ek.e_t_as
    # Epsilon guard (output.f90:704-705)
    if abs(e_kin_nas) <= 10.0 * _eps * mass or e_kin_nas < 0.0:
        e_kin_nas = 0.0

    vol = float(vol_oc)
    Re = math.sqrt(2.0 * e_kin / vol) / math.sqrt(mass)  # eScale=1
    ReConv = math.sqrt(2.0 * e_kin_nas / vol) / math.sqrt(mass)
    Ro = Re * ekScaled
    RoConv = ReConv * ekScaled

    # Magnetic Reynolds (nVarCond=0)
    Rm = Re * prmag if prmag != 0 else Re

    # Elsasser
    if l_mag:
        El = elsAnel_val / vol
        # ElCmb: Fortran output.f90:741
        # ElCmb = two * e_mag_cmb / surf_cmb / LFfac * sigma(n_r_cmb) * orho1(n_r_cmb) / eScale
        # e_mag_cmb is FACTORED (= fac * (e_p_r+e_t_r)[0], fac=0.5*LFfac)
        # So: ElCmb = 2 * 0.5*LFfac*(e_p_r+e_t_r)[0] / surf_cmb / LFfac * sigma[0] * orho1[0]
        #           = (e_p_r+e_t_r)[0] * sigma[0] * orho1[0] / surf_cmb
        ElCmb = (2.0 * e_mag_cmb_val / float(surf_cmb) / LFfac *
                 float(sigma[0]) * float(orho1[0]))
    else:
        El = 0.0
        ElCmb = 0.0

    # Lengthscales
    dlV = dlm_v[0]      # dl
    dmV = dlm_v[2]      # dm
    dlVc = dlm_v[3]     # dlc
    dlVPolPeak = dlm_v[4]  # dlPolPeak

    if l_mag and dlm_b is not None:
        dlB = dlm_b[0]
        dmB = dlm_b[2]
    else:
        dlB = 0.0
        dmB = 0.0

    # Rossby lengthscale
    Rol = Ro / dlV if dlV != 0.0 else Ro
    RolC = RoConv / dlVc if dlVc != 0.0 else RoConv

    # Dipolarity (reuse from dip_cols)
    if dip_cols is not None:
        Dip = dip_cols[2]     # e_dipole_ax / e_mag_total (col 4 in dipole.TAG = index 2)
        DipCMB = dip_cols[3]  # e_dipole_ax_cmb / e_cmb (col 5 = index 3)
    else:
        Dip = 0.0
        DipCMB = 0.0

    # Not computed (Geos, dpV, dzV, lvDiss, lbDiss, ReEquat)
    Geos = 0.0
    dpV = 0.0
    dzV = 0.0
    lvDiss = 0.0
    lbDiss = 0.0
    ReEquat = 0.0

    # Fortran column order (output.f90:773-788):
    # Rm, El, Rol, Geos, Dip, DipCMB, dlV, dmV, dpV, dzV,
    # lvDiss, lbDiss, dlB, dmB, ElCmb, RolC, dlVc, dlVPolPeak, ReEquat
    return [Rm, El, Rol, Geos, Dip, DipCMB,
            dlV, dmV, dpV, dzV, lvDiss, lbDiss,
            dlB, dmB, ElCmb, RolC, dlVc, dlVPolPeak, ReEquat]


def update_par_means(rm_ms, rol_ms, urol_ms, dlv_ms, dlvc_ms, dlpp_ms,
                     ekinR_factored, dlVR, dlVRc, dlPolPeakR, dt, total_time):
    """Update 6 parR MeanSD accumulators (outPar.f90:254-279).

    ekinR_factored = 0.5 * (e_p_r + e_t_r) -- MUST have the 0.5 factor.
    """
    _or2 = or2
    _mass = mass

    ReR = torch.sqrt(2.0 * ekinR_factored * _or2 / (4.0 * math.pi * _mass))
    RoR = ReR * ekScaled
    RolR = torch.where(dlVR != 0.0, RoR / dlVR, RoR)

    # RmR: outPar.f90 line 264 (l_mag_nl path)
    RmR = ReR * prmag * sigma * r * r

    dlv_ms.compute(dlVR, dt, total_time)       # line 269
    dlvc_ms.compute(dlVRc, dt, total_time)     # line 270
    rol_ms.compute(RolR, dt, total_time)       # line 271
    urol_ms.compute(RolR, dt, total_time)      # line 275 (l_anel=False: same as RolR)

    # Second modification: outPar.f90 line 277
    RmR = RmR * torch.sqrt(_mass * orho1) * _or2
    rm_ms.compute(RmR, dt, total_time)         # line 278
    dlpp_ms.compute(dlPolPeakR, dt, total_time)  # line 279


def get_lorentz_torque_ic(b, db, aj):
    """Compute Lorentz torque on IC from spectral magnetic fields.

    Does a single-level inverse SHT at ICB to get brc, bpc, then
    integrates gauss * brc * bpc. Matches get_lorentz_torque in outRot.f90.
    """
    from .sht import torpol_to_spat
    from .horizontal_data import gauss_grid, dLh as _dLh_hd
    from .params import n_phi_max, n_r_max
    _dLh_cd = _dLh_hd.to(CDTYPE).unsqueeze(1)

    icb = n_r_max - 1
    # Single-level inverse SHT at ICB
    Q = _dLh_cd * b[:, icb:icb + 1]
    S = db[:, icb:icb + 1]
    T = aj[:, icb:icb + 1]
    brc, _, bpc = torpol_to_spat(Q, S, T)
    brc = brc[0]  # (n_theta, n_phi)
    bpc = bpc[0]

    fac = LFfac * 2.0 * math.pi / float(n_phi_max)
    return fac * (gauss_grid.unsqueeze(1) * brc * bpc).sum().item()


def get_viscous_torque(z10, dz10, r_bnd, beta_bnd, visc_bnd):
    """Compute viscous torque at a boundary (outRot.f90 get_viscous_torque).

    Args:
        z10: real value of z(l=1,m=0) at boundary
        dz10: real value of dz/dr(l=1,m=0) at boundary
        r_bnd: radius of boundary
        beta_bnd: d(ln rho0)/dr at boundary (0 for Boussinesq)
        visc_bnd: kinematic viscosity at boundary

    Returns:
        viscous torque (scalar float)
    """
    return -4.0 * math.sqrt(pi / 3.0) * visc_bnd * r_bnd * (
        (2.0 + beta_bnd * r_bnd) * z10 - r_bnd * dz10)


def write_rot_line(f, time, omega_ic, lorentz_torque_ic, viscous_torque_ic,
                   omega_ma, lorentz_torque_ma, viscous_torque_ma,
                   gravi_torque_ic):
    """Write one line to rot.TAG matching Fortran '(1P,2X,ES20.12,7ES16.8)'."""
    f.write(f"  {time:20.12E}{omega_ic:16.8E}{lorentz_torque_ic:16.8E}"
            f"{viscous_torque_ic:16.8E}{omega_ma:16.8E}"
            f"{lorentz_torque_ma:16.8E}{viscous_torque_ma:16.8E}"
            f"{gravi_torque_ic:16.8E}\n")


def write_par_line(f, time, cols):
    """Write one line to par.TAG matching Fortran '(ES20.12,19ES16.8)'."""
    f.write(f"{time:20.12E}")
    for v in cols:
        f.write(f"{v:16.8E}")
    f.write("\n")


def write_parR_file(path, rm_ms, rol_ms, urol_ms, dlv_ms, dlvc_ms, dlpp_ms):
    """Write parR.TAG matching Fortran '(ES20.10,6ES15.7,6ES13.5)'."""
    means = [rm_ms, rol_ms, urol_ms, dlv_ms, dlvc_ms, dlpp_ms]
    mean_refs = [m.mean.max().item() for m in means]
    sd_refs = [m.SD.max().item() for m in means]
    with open(path, 'w') as f:
        for nR in range(n_r_max):
            f.write(f"{r[nR].item():20.10E}")
            for i, m in enumerate(means):
                f.write(f"{_round_off(m.mean[nR].item(), mean_refs[i]):15.7E}")
            for i, m in enumerate(means):
                f.write(f"{_round_off(m.SD[nR].item(), sd_refs[i]):13.5E}")
            f.write("\n")
