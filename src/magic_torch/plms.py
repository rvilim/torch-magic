"""Associated Legendre polynomials matching Fortran's plms.f90.

Computes fully normalized (norm=2) Plm(cos θ) and sin(θ)*dPlm/dθ
using the three-term recurrence relation. These are stored in Plm matrices
used by the SHT module.

The normalization is: ∫ |Y_lm|^2 dΩ = 1 (fully normalized).
This uses the 1/sqrt(4*pi) prefactor (osq4pi) for norm=2.
No Condon-Shortley phase.
"""

import math
import torch

from .precision import DTYPE, DEVICE
from .params import l_max, m_max, m_min, minc, lm_max, n_theta_max
from .constants import osq4pi, one, two
from .horizontal_data import theta_ord, gauss


def plm_theta(theta: float, max_degree: int, min_order: int,
              max_order: int, m0: int, norm: int = 2):
    """Compute Plm and sin(θ)*dPlm/dθ at a single colatitude.

    Args:
        theta: colatitude in radians
        max_degree: maximum spherical harmonic degree
        min_order: minimum order
        max_order: maximum order
        m0: basic wavenumber (minc)
        norm: normalization (2 = fully normalized)

    Returns:
        plma: Plm values, shape (lm_max,)
        dtheta_plma: sin(θ)*dPlm/dθ values, shape (lm_max,)
    """
    dnorm = 1.0
    if norm == 2:
        dnorm = osq4pi
    sq2 = math.sqrt(2.0)

    plma = torch.zeros(lm_max, dtype=DTYPE, device="cpu")
    dtheta_plma = torch.zeros(lm_max, dtype=DTYPE, device="cpu")

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    pos = -1
    for m in range(min_order, max_order + 1, m0):
        # Compute P_m^m starting value
        fac = 1.0
        for j in range(3, 2 * m + 2, 2):
            fac = fac * j / (j - 1)
        plm = math.sqrt(fac)
        if sin_theta != 0.0:
            plm = plm * sin_theta ** m
        elif m != 0:
            plm = 0.0

        # Store l=m
        l = m
        if norm == 1:
            dnorm = 1.0 / math.sqrt(2 * l + 1)
            if m != 0:
                dnorm = sq2 * dnorm

        pos += 1
        plma[pos] = dnorm * plm

        plm1 = 0.0

        # l > m using recurrence
        for l in range(m + 1, max_degree + 1):
            plm2 = plm1
            plm1 = plm
            plm = (cos_theta * math.sqrt(float((2 * l - 1) * (2 * l + 1))
                                         / float((l - m) * (l + m))) * plm1
                   - math.sqrt(float((2 * l + 1) * (l + m - 1) * (l - m - 1))
                               / float((2 * l - 3) * (l - m) * (l + m))) * plm2)
            if norm == 1:
                dnorm = 1.0 / math.sqrt(2 * l + 1)
                if m != 0:
                    dnorm = sq2 * dnorm

            pos += 1
            plma[pos] = dnorm * plm

        # Extra P_{l_max+1}^m for theta derivative calculation
        l = max_degree + 1
        plm2 = plm1
        plm1 = plm
        if (l - m) * (l + m) > 0:
            plm = (cos_theta * math.sqrt(float((2 * l - 1) * (2 * l + 1))
                                         / float((l - m) * (l + m))) * plm1
                   - math.sqrt(float((2 * l + 1) * (l + m - 1) * (l - m - 1))
                               / float((2 * l - 3) * (l - m) * (l + m))) * plm2)
        else:
            plm = 0.0
        if norm == 1:
            dnorm = 1.0 / math.sqrt(2 * l + 1)
            if m != 0:
                dnorm = sq2 * dnorm
        dtheta_plma[pos] = dnorm * plm  # temporary storage for P_{l_max+1}

    # Now compute sin(θ)*dPlm/dθ using the recurrence
    pos = -1
    for m in range(min_order, max_order + 1, m0):
        # l=m contribution
        l = m
        pos += 1
        if m < max_degree:
            if norm == 0 or norm == 2:
                dtheta_plma[pos] = l / math.sqrt(2 * l + 3) * plma[pos + 1]
            elif norm == 1:
                dtheta_plma[pos] = l / math.sqrt(2 * l + 1) * plma[pos + 1]
        else:
            if norm == 0 or norm == 2:
                dtheta_plma[pos] = l / math.sqrt(2 * l + 3) * dtheta_plma[pos]
            elif norm == 1:
                dtheta_plma[pos] = l / math.sqrt(2 * l + 1) * dtheta_plma[pos]

        # l = m+1 to max_degree-1
        for l in range(m + 1, max_degree):
            pos += 1
            if norm == 0 or norm == 2:
                dtheta_plma[pos] = (
                    l * math.sqrt(float((l + m + 1) * (l - m + 1))
                                  / float((2 * l + 1) * (2 * l + 3))) * plma[pos + 1]
                    - (l + 1) * math.sqrt(float((l + m) * (l - m))
                                          / float((2 * l - 1) * (2 * l + 1))) * plma[pos - 1]
                )
            elif norm == 1:
                dtheta_plma[pos] = (
                    l * math.sqrt(float((l + m + 1) * (l - m + 1))) * plma[pos + 1]
                    - (l + 1) * math.sqrt(float((l + m) * (l - m))) * plma[pos - 1]
                ) / float(2 * l + 1)

        # l = max_degree (uses stored P_{l_max+1} in dtheta_plma[pos])
        if m < max_degree:
            l = max_degree
            pos += 1
            if norm == 0 or norm == 2:
                dtheta_plma[pos] = (
                    l * math.sqrt(float((l + m + 1) * (l - m + 1))
                                  / float((2 * l + 1) * (2 * l + 3))) * dtheta_plma[pos]
                    - (l + 1) * math.sqrt(float((l + m) * (l - m))
                                          / float((2 * l - 1) * (2 * l + 1))) * plma[pos - 1]
                )
            elif norm == 1:
                dtheta_plma[pos] = (
                    l * math.sqrt(float((l + m + 1) * (l - m + 1))) * dtheta_plma[pos]
                    - (l + 1) * math.sqrt(float((l + m) * (l - m))) * plma[pos - 1]
                ) / float(2 * l + 1)

    return plma, dtheta_plma


def build_plm_matrices():
    """Build Plm and dPlm matrices for all northern hemisphere Gauss points.

    Returns:
        Plm: shape (lm_max, n_theta_max//2)
        dPlm: shape (lm_max, n_theta_max//2)
        wPlm: shape (lm_max, n_theta_max//2) — 2*pi*gauss*Plm
        wdPlm: shape (lm_max, n_theta_max//2) — 2*pi*gauss*dPlm
    """
    n_theta_NHS = n_theta_max // 2
    # Build on CPU (plm_theta uses scalar Python loops)
    Plm = torch.zeros(lm_max, n_theta_NHS, dtype=DTYPE, device="cpu")
    dPlm = torch.zeros(lm_max, n_theta_NHS, dtype=DTYPE, device="cpu")
    wPlm = torch.zeros(lm_max, n_theta_NHS, dtype=DTYPE, device="cpu")
    wdPlm = torch.zeros(lm_max, n_theta_NHS, dtype=DTYPE, device="cpu")

    # theta_ord and gauss may be on DEVICE; extract values on CPU
    _theta = theta_ord.cpu()
    _gauss = gauss.cpu()

    for n_theta in range(n_theta_NHS):
        colat = _theta[n_theta].item()
        plma, dtheta_plma = plm_theta(colat, l_max, m_min, m_max, minc, norm=2)
        Plm[:, n_theta] = plma
        dPlm[:, n_theta] = dtheta_plma
        wPlm[:, n_theta] = 2.0 * math.pi * _gauss[n_theta] * plma
        wdPlm[:, n_theta] = 2.0 * math.pi * _gauss[n_theta] * dtheta_plma

    return Plm.to(DEVICE), dPlm.to(DEVICE), wPlm.to(DEVICE), wdPlm.to(DEVICE)


# Build the matrices at module load time
Plm, dPlm, wPlm, wdPlm = build_plm_matrices()

# Build lStart/lStop index arrays (0-based)
lStart = torch.zeros(n_theta_max, dtype=torch.long, device="cpu")  # oversized but fine
lStop = torch.zeros(n_theta_max, dtype=torch.long, device="cpu")

_lstart_list = []
_lstop_list = []
_pos = 0
for mc in range(n_theta_max):
    m = mc * minc
    if m > m_max:
        break
    _lstart_list.append(_pos)
    _lstop_list.append(_pos + l_max - m)
    _pos += l_max - m + 1

n_m_actual = len(_lstart_list)
lStart_list = _lstart_list
lStop_list = _lstop_list
