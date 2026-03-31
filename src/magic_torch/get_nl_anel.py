"""Non-curl advection + viscous heating for anelastic runs (get_nl.f90).

Advection uses u·∇u formulation (l_adv_curl=.false.) instead of (curl u)×u.
Viscous heating: strain-rate-squared dissipation.
No magnetic field (mode=1, l_mag=False) — Lorentz/VxB/Ohmic omitted.

All operations vectorized: inputs shape (n_r, n_theta, n_phi).
"""

import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, n_theta_max, n_phi_max
from .chebyshev import r
from .radial_functions import (
    or1, or2, or4, beta, orho1, otemp1, visc, ViscHeatFac,
)
from .horizontal_data import O_sin_theta_E2_grid, cosn_theta_E2_grid
from .constants import two, third


# Broadcast helpers: (n_r_max,) → (n_r_max, 1, 1)
_or1_3 = or1.reshape(n_r_max, 1, 1)
_or2_3 = or2.reshape(n_r_max, 1, 1)
_or4_3 = or4.reshape(n_r_max, 1, 1)
_r_3 = r.reshape(n_r_max, 1, 1)
_beta_3 = beta.reshape(n_r_max, 1, 1)
_orho1_3 = orho1.reshape(n_r_max, 1, 1)
_otemp1_3 = otemp1.reshape(n_r_max, 1, 1)
_visc_3 = visc.reshape(n_r_max, 1, 1)
_vhf_3 = torch.tensor(ViscHeatFac, dtype=DTYPE, device=DEVICE)

# Theta-dependent (grid layout): (n_theta_max,) → (1, n_theta_max, 1)
_Ost2 = O_sin_theta_E2_grid.reshape(1, n_theta_max, 1)
_cosn = cosn_theta_E2_grid.reshape(1, n_theta_max, 1)


@torch.compile(disable=DEVICE.type == "mps")
def get_nl_anel(
    vrc, vtc, vpc,
    dvrdrc, dvtdrc, dvpdrc,
    dvrdtc, dvrdpc,
    dvtdpc, dvpdpc,
    cvrc,
    sc, xic,
):
    """Compute non-curl advection + viscous heating in grid space.

    All inputs shape (n_r, n_theta, n_phi) where n_r may be < n_r_max (bulk).

    Returns:
        Advr, Advt, Advp: momentum equation RHS (non-curl advection, no Lorentz)
        VSr, VSt, VSp: entropy advection vector v*S
        VXir, VXit, VXip: composition advection vector v*Xi
        heatTerms: viscous heating (grid-space scalar)
    """
    # --- Non-curl advection: u·∇u (get_nl.f90:276-307) ---
    # Advr = -or2*orho1*(vrc*(dvrdrc - (2/r+beta)*vrc) + Ost2*(vtc*(dvrdtc - r*vtc) + vpc*(dvrdpc - r*vpc)))
    Advr = -_or2_3 * _orho1_3 * (
        vrc * (dvrdrc - (two * _or1_3 + _beta_3) * vrc)
        + _Ost2 * (
            vtc * (dvrdtc - _r_3 * vtc)
            + vpc * (dvrdpc - _r_3 * vpc)
        )
    )

    # Advt = or4*orho1*(-vrc*(dvtdrc - beta*vtc) + vtc*(cosn*vtc + dvpdpc + dvrdrc) + vpc*(cosn*vpc - dvtdpc))
    Advt = _or4_3 * _orho1_3 * (
        -vrc * (dvtdrc - _beta_3 * vtc)
        + vtc * (_cosn * vtc + dvpdpc + dvrdrc)
        + vpc * (_cosn * vpc - dvtdpc)
    )

    # Advp = or4*orho1*(-vrc*(dvpdrc - beta*vpc) - vtc*(dvtdpc + cvrc) - vpc*dvpdpc)
    Advp = _or4_3 * _orho1_3 * (
        -vrc * (dvpdrc - _beta_3 * vpc)
        - vtc * (dvtdpc + cvrc)
        - vpc * dvpdpc
    )

    # --- Entropy advection: v*S ---
    VSr = vrc * sc
    VSt = _or2_3 * vtc * sc
    VSp = _or2_3 * vpc * sc

    # --- Composition advection: v*Xi ---
    VXir = vrc * xic
    VXit = _or2_3 * vtc * xic
    VXip = _or2_3 * vpc * xic

    # --- Viscous heating (get_nl.f90:402-425) ---
    # 6 strain rate components + compressibility correction
    prefac = _vhf_3 * _or4_3 * _orho1_3 * _otemp1_3 * _visc_3
    beta_r = _beta_3 * _r_3  # beta(nR) * r(nR)

    e_rr = dvrdrc - (two * _or1_3 + _beta_3) * vrc              # (1)
    e_tt = _cosn * vtc + dvpdpc + dvrdrc - _or1_3 * vrc          # (2)
    e_pp = dvpdpc + _cosn * vtc + _or1_3 * vrc                   # (3)
    e_tp = two * dvtdpc + cvrc - two * _cosn * vpc               # (6)
    e_rt = _r_3 * dvtdrc - (two + beta_r) * vtc + _or1_3 * dvrdtc  # (4)
    e_rp = _r_3 * dvpdrc - (two + beta_r) * vpc + _or1_3 * dvrdpc  # (5)

    heatTerms = prefac * (
        two * e_rr * e_rr
        + two * e_tt * e_tt
        + two * e_pp * e_pp
        + e_tp * e_tp
        + _Ost2 * (e_rt * e_rt + e_rp * e_rp)
        - two * third * (_beta_3 * vrc) ** 2
    )

    return Advr, Advt, Advp, VSr, VSt, VSp, VXir, VXit, VXip, heatTerms
