"""Physical-space nonlinear products matching get_nl.f90.

Computes pointwise products on the (n_r_max, n_theta_max, n_phi_max) grid.
All operations are fully vectorized over radial levels.

Boussinesq benchmark: beta=0, orho1=1, l_adv_curl=.true.
Advection uses curl formulation: Adv = -(curl u) × u
"""

import torch

from .precision import DTYPE, DEVICE
from .params import n_r_max, n_theta_max, n_phi_max
from .radial_scheme import r
from .radial_functions import or1, or2, or4
from .horizontal_data import O_sin_theta_E2_grid, cosn_theta_E2_grid
from .pre_calculations import LFfac


# Broadcast helpers: (n_r_max,) → (n_r_max, 1, 1) for grid-space broadcasting
_or4_3 = or4.reshape(n_r_max, 1, 1)

# Theta-dependent (grid layout): (n_theta_max,) → (1, n_theta_max, 1)
_Ost2 = O_sin_theta_E2_grid.reshape(1, n_theta_max, 1)

# Radial for entropy/induction
_or2_3 = or2.reshape(n_r_max, 1, 1)


@torch.compile
def get_nl(vrc, vtc, vpc, cvrc, cvtc, cvpc,
           sc, brc, btc, bpc, cbrc, cbtc, cbpc, xic):
    """Compute all nonlinear products in grid space.

    Uses l_adv_curl=.true. formulation: advection = -(curl u) × u.

    All inputs shape (n_r_max, n_theta_max, n_phi_max).

    Args:
        vrc, vtc, vpc: velocity components
        cvrc, cvtc, cvpc: vorticity (curl u) components
        sc: entropy
        brc, btc, bpc: magnetic field components
        cbrc, cbtc, cbpc: current density (curl B) components
        xic: composition (zeros when inactive)

    Returns:
        Advr, Advt, Advp: momentum equation RHS (advection + Lorentz)
        VSr, VSt, VSp: entropy advection vector v*S
        VxBr, VxBt, VxBp: induction v×B
        VXir, VXit, VXip: composition advection vector v*Xi
    """
    # --- Lorentz force: (curl B) × B ---
    LFr = LFfac * _Ost2 * (cbtc * bpc - cbpc * btc)
    LFt = LFfac * _or4_3 * (cbpc * brc - cbrc * bpc)
    LFp = LFfac * _or4_3 * (cbrc * btc - cbtc * brc)

    # --- Advection: -(curl u) × u  (l_adv_curl=.true.) ---
    Advr = -_Ost2 * (cvtc * vpc - cvpc * vtc)
    Advt = -_or4_3 * (cvpc * vrc - cvrc * vpc)
    Advp = -_or4_3 * (cvrc * vtc - cvtc * vrc)

    # --- Add Lorentz force to advection ---
    Advr = Advr + LFr
    Advt = Advt + LFt
    Advp = Advp + LFp

    # --- Entropy advection: v*S (divergence taken spectrally) ---
    VSr = vrc * sc
    VSt = _or2_3 * vtc * sc
    VSp = _or2_3 * vpc * sc

    # --- Induction: v × B (Boussinesq: orho1=1) ---
    VxBr = _Ost2 * (vtc * bpc - vpc * btc)
    VxBt = _or4_3 * (vpc * brc - vrc * bpc)
    VxBp = _or4_3 * (vrc * btc - vtc * brc)

    # --- Composition advection: v*Xi (same structure as entropy) ---
    VXir = vrc * xic
    VXit = _or2_3 * vtc * xic
    VXip = _or2_3 * vpc * xic

    return Advr, Advt, Advp, VSr, VSt, VSp, VxBr, VxBt, VxBp, VXir, VXit, VXip
