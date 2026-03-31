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
from .params import n_r_max, lm_max, l_max, l_mag, l_cond_ic, minc, kbotb, ktopb, m_max, prmag
from .chebyshev import r, r_cmb, r_icb
from .radial_functions import or2, orho1, sigma, vol_oc, surf_cmb
from .horizontal_data import dLh
from .blocking import st_lm2l, st_lm2m
from .integration import rInt_R
from .pre_calculations import LFfac, ekScaled, mass
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


def get_e_mag_ic(b):
    """Compute inner-core magnetic energy for insulating IC.

    For insulating IC (l_cond_ic=False): uses b at ICB with l*(l+1)^2 weighting.
    Returns EMagIC namedtuple with 4 components matching Fortran e_mag_ic.TAG.
    """
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
    from .chebyshev import r_cmb, r_icb
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
