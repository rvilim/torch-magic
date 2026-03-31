"""Spectral-space time derivative assembly matching get_td.f90.

Assembles explicit (nonlinear + Coriolis) RHS terms for each field equation
from QST-decomposed nonlinear products. All operations vectorized over
(lm, nR) — no Python loops.

The "finish_exp" steps (radial derivatives of dVSrLM, dVxBhLM) are handled
in the update modules (Phase 7).
"""

import torch

from .precision import CDTYPE, DEVICE
from .params import n_r_max, lm_max, l_max
from .chebyshev import r
from .radial_functions import or1, or2, or4
from .blocking import st_lm2l, st_lm2lmS, st_lm2lmA
from .horizontal_data import dLh, dPhi, dTheta2S, dTheta2A, dTheta3S, dTheta3A, dTheta4S, dTheta4A
from .pre_calculations import CorFac

# --- Precompute broadcast arrays ---
# Radial profiles: (n_r_max,) → (1, n_r_max)
_or1 = or1.unsqueeze(0)      # (1, n_r_max)
_or2 = or2.unsqueeze(0)
_or4 = or4.unsqueeze(0)
_r2 = (r * r).unsqueeze(0)   # (1, n_r_max)

# LM-dependent arrays: (lm_max,) → (lm_max, 1)
_dLh = dLh.to(CDTYPE).unsqueeze(1)       # (lm_max, 1)
_dPhi = dPhi.unsqueeze(1)                 # (lm_max, 1) complex
_dTh2S = dTheta2S.to(CDTYPE).unsqueeze(1)
_dTh2A = dTheta2A.to(CDTYPE).unsqueeze(1)
_dTh3S = dTheta3S.to(CDTYPE).unsqueeze(1)
_dTh3A = dTheta3A.to(CDTYPE).unsqueeze(1)
_dTh4S = dTheta4S.to(CDTYPE).unsqueeze(1)
_dTh4A = dTheta4A.to(CDTYPE).unsqueeze(1)

# l+1 coupling indices and mask (for l=l_max, lmA=-1 → invalid)
_lmA = st_lm2lmA.clamp(min=0)  # safe gather index
_mask_A = (st_lm2lmA >= 0).to(CDTYPE).unsqueeze(1)  # (lm_max, 1)

# l-1 coupling indices (always valid — dummy self-index for l=m with zero coefficient)
_lmS = st_lm2lmS


def _gather_A(field):
    """Gather l+1 neighbor values, zeroing invalid (l=l_max) modes."""
    return field[_lmA, :] * _mask_A


def _gather_S(field):
    """Gather l-1 neighbor values."""
    return field[_lmS, :]


def get_dwdt(AdvrLM, dw, z):
    """Explicit terms for poloidal velocity equation.

    dwdt = (1/r²) * AdvrLM + Coriolis_pol

    Coriolis_pol = 2*CorFac/r * (i*m*dw/dr + dTheta2A*z(l+1) - dTheta2S*z(l-1))

    Args:
        AdvrLM: (lm_max, n_r_max) complex — Q-component of (Adv + LF)
        dw: (lm_max, n_r_max) complex — radial derivative of poloidal potential
        z: (lm_max, n_r_max) complex — toroidal potential

    Returns:
        dwdt: (lm_max, n_r_max) complex
    """
    # Nonlinear term
    dwdt = _or2 * AdvrLM

    # Coriolis: 2*CorFac/r * (i*m*dw + dTheta2A*z_A - dTheta2S*z_S)
    z_A = _gather_A(z)
    z_S = _gather_S(z)
    CorPol = 2.0 * CorFac * _or1 * (
        _dPhi * dw + _dTh2A * z_A - _dTh2S * z_S
    )
    dwdt = dwdt + CorPol

    # Zero boundary points (nBc=2 for no-slip)
    dwdt[:, 0] = 0.0
    dwdt[:, -1] = 0.0

    return dwdt


def get_dzdt(AdvpLM, w, dw, z):
    """Explicit terms for toroidal velocity equation.

    dzdt = dLh * AdvpLM + Coriolis_tor

    Coriolis_tor = 2*CorFac/r² * (i*m*z + dTheta3A*dw(l+1) + (1/r)*dTheta4A*w(l+1)
                                       + dTheta3S*dw(l-1) - (1/r)*dTheta4S*w(l-1))

    Args:
        AdvpLM: (lm_max, n_r_max) complex — T-component of (Adv + LF)
        w, dw, z: (lm_max, n_r_max) complex — current fields

    Returns:
        dzdt: (lm_max, n_r_max) complex
    """
    # Nonlinear term
    dzdt = _dLh * AdvpLM

    # Coriolis
    dw_A = _gather_A(dw)
    w_A = _gather_A(w)
    dw_S = _gather_S(dw)
    w_S = _gather_S(w)

    CorTor = 2.0 * CorFac * _or2 * (
        _dPhi * z
        + _dTh3A * dw_A + _or1 * _dTh4A * w_A
        + _dTh3S * dw_S - _or1 * _dTh4S * w_S
    )
    dzdt = dzdt + CorTor

    # Zero boundary points
    dzdt[:, 0] = 0.0
    dzdt[:, -1] = 0.0

    return dzdt


def get_dpdt(AdvtLM, w, dw, z):
    """Explicit terms for pressure equation.

    dpdt = -dLh * AdvtLM + Coriolis_p

    Coriolis_p = 2*CorFac/r² * (-i*m*(dw/dr + dLh/r*w) + dTheta3A*z(l+1) + dTheta3S*z(l-1))

    Args:
        AdvtLM: (lm_max, n_r_max) complex — S-component of (Adv + LF)
        w, dw, z: (lm_max, n_r_max) complex — current fields

    Returns:
        dpdt: (lm_max, n_r_max) complex
    """
    # Nonlinear term
    dpdt = -_dLh * AdvtLM

    # Coriolis
    z_A = _gather_A(z)
    z_S = _gather_S(z)

    CorPol = 2.0 * CorFac * _or2 * (
        -_dPhi * (dw + _or1 * _dLh * w)
        + _dTh3A * z_A
        + _dTh3S * z_S
    )
    dpdt = dpdt + CorPol

    # Zero boundary points
    dpdt[:, 0] = 0.0
    dpdt[:, -1] = 0.0

    return dpdt


def get_dsdt(VStLM, dVSrLM):
    """Partial explicit terms for entropy equation (before finish_exp).

    dsdt_partial = dLh * VStLM   (horizontal divergence of u*s)
    dVSrLM is passed through for finish_exp_entropy.

    Args:
        VStLM: (lm_max, n_r_max) complex — S-component of spat_to_qst(VS)
        dVSrLM: (lm_max, n_r_max) complex — Q-component (ur*s in SH)

    Returns:
        dsdt_partial: (lm_max, n_r_max) complex
        dVSrLM: (lm_max, n_r_max) complex — unchanged, for finish_exp
    """
    dsdt_partial = _dLh * VStLM

    # Zero boundary points
    dsdt_partial[:, 0] = 0.0
    dsdt_partial[:, -1] = 0.0

    # dVSrLM at boundaries should be zero
    dVSrLM = dVSrLM.clone()
    dVSrLM[:, 0] = 0.0
    dVSrLM[:, -1] = 0.0

    return dsdt_partial, dVSrLM


def get_dxidt(VXitLM, dVXirLM):
    """Partial explicit terms for composition equation (before finish_exp).

    Structurally identical to get_dsdt: dxidt_partial = dLh * VXitLM.

    Args:
        VXitLM: (lm_max, n_r_max) complex — S-component of spat_to_qst(VXi)
        dVXirLM: (lm_max, n_r_max) complex — Q-component (ur*xi in SH)

    Returns:
        dxidt_partial: (lm_max, n_r_max) complex
        dVXirLM: (lm_max, n_r_max) complex — for finish_exp_comp
    """
    dxidt_partial = _dLh * VXitLM

    # Zero boundary points
    dxidt_partial[:, 0] = 0.0
    dxidt_partial[:, -1] = 0.0

    # dVXirLM at boundaries should be zero
    dVXirLM = dVXirLM.clone()
    dVXirLM[:, 0] = 0.0
    dVXirLM[:, -1] = 0.0

    return dxidt_partial, dVXirLM


def get_dbdt(VxBrLM, VxBtLM, VxBpLM):
    """Explicit terms for magnetic field equations (djdt partial, before finish_exp).

    dbdt = dLh * VxBpLM                    (poloidal magnetic)
    djdt_partial = dLh/r⁴ * VxBrLM        (toroidal magnetic, partial)
    dVxBhLM = -dLh * VxBtLM * r²          (auxiliary for finish_exp_mag)

    Args:
        VxBrLM, VxBtLM, VxBpLM: (lm_max, n_r_max) complex — QST of (v×B)

    Returns:
        dbdt: (lm_max, n_r_max) complex
        djdt_partial: (lm_max, n_r_max) complex
        dVxBhLM: (lm_max, n_r_max) complex — for finish_exp_mag
    """
    dbdt = _dLh * VxBpLM
    djdt_partial = _dLh * _or4 * VxBrLM
    dVxBhLM = -_dLh * VxBtLM * _r2

    # Zero boundary points for dbdt, djdt
    dbdt[:, 0] = 0.0
    dbdt[:, -1] = 0.0
    djdt_partial[:, 0] = 0.0
    djdt_partial[:, -1] = 0.0

    # dVxBhLM at boundaries: still computed for interior (Fortran does this)
    # but l=0 is zeroed
    dVxBhLM[0, 0] = 0.0
    dVxBhLM[0, -1] = 0.0

    return dbdt, djdt_partial, dVxBhLM
