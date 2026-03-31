"""CFL Courant condition check matching courant.f90 (XSH_COURANT=0 path).

Computes Courant time step limits from grid-space velocity and magnetic fields,
and decides whether dt needs adjustment with hysteresis.
"""

import torch

from .precision import DTYPE, DEVICE
from .params import (n_r_max, n_theta_max,
                     l_mag, l_mag_LF, l_mag_kin, l_cour_alf_damp)
from .radial_functions import or4, or2, orho1, orho2, delxr2, delxh2
from .horizontal_data import O_sin_theta_E2_grid
from .pre_calculations import LFfac, opm


# Broadcast helpers (precomputed once at import)
_or4_3 = or4.reshape(n_r_max, 1, 1)
_or2_3 = or2.reshape(n_r_max, 1, 1)
_orho1_3 = orho1.reshape(n_r_max, 1, 1)
_orho2_3 = orho2.reshape(n_r_max, 1, 1)
_Ost2 = O_sin_theta_E2_grid.reshape(1, n_theta_max, 1)


@torch.compile(disable=(DEVICE.type == "mps"))
def _courant_mag_damp(vrc, vtc, vpc, brc, btc, bpc, cf2, af2, valri2, valhi2):
    """CFL with magnetic field + Alfven damping (l_cour_alf_damp=True)."""
    vflr2 = _orho2_3 * vrc * vrc
    vflh2 = (vtc * vtc + vpc * vpc) * _Ost2 * _orho2_3

    valr = brc * brc * LFfac * _orho1_3
    valr2 = valr * valr / (valr + valri2)

    valh2 = (btc * btc + bpc * bpc) * LFfac * _Ost2 * _orho1_3
    valh2m = valh2 * valh2 / (valh2 + valhi2)

    vr2max = (_or4_3 * (cf2 * vflr2 + af2 * valr2)).amax(dim=(1, 2))
    vh2max = (_or2_3 * (cf2 * vflh2 + af2 * valh2m)).amax(dim=(1, 2))

    return vr2max, vh2max


@torch.compile(disable=(DEVICE.type == "mps"))
def _courant_mag_nodamp(vrc, vtc, vpc, brc, btc, bpc, cf2, af2):
    """CFL with magnetic field, no Alfven damping (l_cour_alf_damp=False).

    When valri2=0, valr^2/(valr+0) = valr for valr>0 but NaN for valr=0.
    Fortran's max() ignores NaN but PyTorch's amax() propagates it.
    Use valr directly (equivalent when damping disabled).
    """
    vflr2 = _orho2_3 * vrc * vrc
    vflh2 = (vtc * vtc + vpc * vpc) * _Ost2 * _orho2_3

    valr = brc * brc * LFfac * _orho1_3
    valh2 = (btc * btc + bpc * bpc) * LFfac * _Ost2 * _orho1_3

    vr2max = (_or4_3 * (cf2 * vflr2 + af2 * valr)).amax(dim=(1, 2))
    vh2max = (_or2_3 * (cf2 * vflh2 + af2 * valh2)).amax(dim=(1, 2))

    return vr2max, vh2max


@torch.compile(disable=(DEVICE.type == "mps"))
def _courant_nomag(vrc, vtc, vpc, cf2):
    """CFL without magnetic field (XSH_COURANT=0)."""
    vflr2 = _orho2_3 * vrc * vrc
    vflh2 = (vtc * vtc + vpc * vpc) * _Ost2 * _orho2_3

    vr2max = (cf2 * _or4_3 * vflr2).amax(dim=(1, 2))
    vh2max = (cf2 * _or2_3 * vflh2).amax(dim=(1, 2))

    return vr2max, vh2max


def courant_check(vrc, vtc, vpc, brc=None, btc=None, bpc=None,
                  courfac=2.5, alffac=1.0):
    """Compute per-radial-level Courant time steps dtrkc and dthkc.

    Args:
        vrc, vtc, vpc: velocity grid-space arrays (n_r, n_theta, n_phi)
        brc, btc, bpc: magnetic field arrays, or None if no magnetic field
        courfac, alffac: CFL factors from the time scheme

    Returns:
        (dtrkc, dthkc): per-level Courant time steps, shape (n_r_max,)
    """
    cf2 = courfac * courfac

    use_mag = (l_mag and l_mag_LF and not l_mag_kin
               and brc is not None)

    if use_mag:
        af2 = alffac * alffac

        if l_cour_alf_damp:
            valri2 = ((0.5 * (1.0 + opm)) ** 2 / delxr2).reshape(n_r_max, 1, 1)
            valhi2 = ((0.5 * (1.0 + opm)) ** 2 / delxh2).reshape(n_r_max, 1, 1)
            vr2max, vh2max = _courant_mag_damp(vrc, vtc, vpc, brc, btc, bpc,
                                                cf2, af2, valri2, valhi2)
        else:
            vr2max, vh2max = _courant_mag_nodamp(vrc, vtc, vpc, brc, btc, bpc,
                                                  cf2, af2)
    else:
        vr2max, vh2max = _courant_nomag(vrc, vtc, vpc, cf2)

    # dtrkc/dthkc per level: sqrt(delx2/v2max) where v2max > 0, else large
    large = torch.tensor(1e10, dtype=DTYPE, device=DEVICE)
    dtrkc = torch.where(vr2max > 0, torch.sqrt(delxr2 / vr2max), large)
    dthkc = torch.where(vh2max > 0, torch.sqrt(delxh2 / vh2max), large)

    return dtrkc, dthkc


def dt_courant(dt, dtMax, dtrkc, dthkc):
    """Decide whether to change dt based on Courant limits.

    Matches courant.f90 subroutine dt_courant (lines 279-346).

    Args:
        dt: current time step (float)
        dtMax: maximum allowed time step (float)
        dtrkc: per-level radial Courant step, shape (n_r_max,)
        dthkc: per-level horizontal Courant step, shape (n_r_max,)

    Returns:
        (l_new_dt, dt_new): whether dt changed, and the new dt value
    """
    # Single GPU->CPU sync: min of both arrays
    dtrkc_min = dtrkc.min().item()
    dthkc_min = dthkc.min().item()

    dt_fac = 2.0
    dtMin = min(dtrkc_min, dthkc_min)
    dt_2 = min(0.5 * (1.0 / dt_fac + 1.0) * dtMin, dtMax)

    if dt > dtMax:
        return True, dtMax
    elif dt > dtMin:
        return True, dt_2
    elif dt_fac * dt < dtMin and dt < dtMax:
        return True, dt_2
    else:
        return False, dt
