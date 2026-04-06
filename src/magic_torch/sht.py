"""Spherical Harmonic Transforms using Plm matrices + FFT.

Matches shtransforms.f90 (native matrix-multiply implementation).
All transforms work on a single radial level or batched over radial levels.

Conventions:
- Spectral fields use standard (m-major) LM ordering: (lm_max,) complex
- Grid fields: (n_theta_max, n_phi_max) real
- NHS = northern hemisphere (n_theta_max/2 points)
- Equatorial symmetry: even lm → ES, odd lm → EA
  f(θ_N) = ES + EA, f(θ_S) = ES - EA
"""

import math
import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import (l_max, m_max, m_min, minc, n_theta_max, n_phi_max,
                     n_m_max, lm_max)
from .constants import ci, half
from .plms import Plm, dPlm, wPlm, wdPlm, lStart_list, lStop_list
from .horizontal_data import O_sin_theta_E2, O_sin_theta_E2_grid, dLh, dPhi

# Precomputed sign vector for ES/EA parity: ES(even idx)=-1, EA(odd idx)=+1
# Used in torpol_to_spat to vectorize the inner l-loop.
_max_n_lm = l_max + 1  # max modes for m=0
_parity_sign = torch.ones(_max_n_lm, 1, dtype=CDTYPE, device=DEVICE)
_parity_sign[0::2] = -1.0

# Inverse dLh for spat_to_sphertor division: 1/[l(l+1)], zero for l=0
_inv_dLh = torch.zeros(lm_max, 1, dtype=CDTYPE, device=DEVICE)
_inv_dLh[1:, 0] = 1.0 / dLh[1:].to(CDTYPE)

# --- Precomputed per-m arrays to avoid repeated work inside m-loops ---
# ES/EA index arrays per m-mode
_es_idx_m = []
_ea_idx_m = []
# Complex-typed Plm slices per m-mode (avoids .to(CDTYPE) every call)
_Plm_c_m = []      # Plm[ls:lmS+1] as CDTYPE
_dPlm_c_m = []     # dPlm[ls:lmS+1] as CDTYPE
_wPlm_c_m = []     # wPlm[ls:lmS+1] as CDTYPE
_wdPlm_c_m = []    # wdPlm[ls:lmS+1] as CDTYPE
# PlmG = dPlm - m*Plm, PlmC = dPlm + m*Plm (complex, for torpol_to_spat)
_PlmG_c_m = []
_PlmC_c_m = []
# Per-m metadata
_m_val = []   # m value
_ls_m = []    # lStart index
_lmS_m = []   # lmS index (adjusted for lcut=l_max)
_n_lm_m = []  # number of l-modes
_dm_m = []    # float(m)

for _mc in range(n_m_max):
    _m = _mc * minc
    _ls = lStart_list[_mc]
    _le = lStop_list[_mc]
    _lmS = _le  # lcut=l_max → lmS = le - l_max + l_max = le
    _nlm = _lmS - _ls + 1

    _m_val.append(_m)
    _ls_m.append(_ls)
    _lmS_m.append(_lmS)
    _n_lm_m.append(_nlm)
    _dm_m.append(float(_m))

    _es_idx_m.append(torch.arange(0, _nlm, 2, device=DEVICE))
    _ea_idx_m.append(torch.arange(1, _nlm, 2, device=DEVICE))

    _plm_slice = Plm[_ls:_lmS + 1, :].to(CDTYPE)
    _dplm_slice = dPlm[_ls:_lmS + 1, :].to(CDTYPE)
    _Plm_c_m.append(_plm_slice)
    _dPlm_c_m.append(_dplm_slice)
    _wPlm_c_m.append(wPlm[_ls:_lmS + 1, :].to(CDTYPE))
    _wdPlm_c_m.append(wdPlm[_ls:_lmS + 1, :].to(CDTYPE))
    _PlmG_c_m.append(_dplm_slice - float(_m) * _plm_slice)
    _PlmC_c_m.append(_dplm_slice + float(_m) * _plm_slice)

# --- Precomputed padded matrices for loop-free SHT ---
# Eliminates the m-loop by padding all per-m Plm matrices to the same size
# and using a single batched matmul across all azimuthal modes.

_NHS = n_theta_max // 2

# Spectral gather index: (n_m_max, max_nlm)
# Maps (mc, k) → flat lm index. Padding positions use index 0 (safe because
# padded matrix rows are zero, so the gathered value doesn't affect the result).
_spec_gather = torch.zeros(n_m_max, _max_n_lm, dtype=torch.long, device=DEVICE)
for _mc in range(n_m_max):
    _nlm = _n_lm_m[_mc]
    _spec_gather[_mc, :_nlm] = torch.arange(_ls_m[_mc], _lmS_m[_mc] + 1, device=DEVICE)

# Result-to-lm index for inverse SHT scatter: maps each lm index to its
# position in the flattened (n_m_max * max_nlm) padded result array.
_result_to_lm = torch.zeros(lm_max, dtype=torch.long, device=DEVICE)
for _mc in range(n_m_max):
    _ls = _ls_m[_mc]
    _nlm = _n_lm_m[_mc]
    _result_to_lm[_ls:_ls + _nlm] = torch.arange(
        _mc * _max_n_lm, _mc * _max_n_lm + _nlm, device=DEVICE)


def _build_padded_T_r(per_m_list):
    """Build (n_m_max, NHS, max_nlm) float64 from list of (nlm, NHS) complex matrices."""
    P = torch.zeros(n_m_max, _NHS, _max_n_lm, dtype=DTYPE, device=DEVICE)
    for mc in range(n_m_max):
        nlm = _n_lm_m[mc]
        P[mc, :, :nlm] = per_m_list[mc].real.T
    return P


def _build_padded_r(per_m_list):
    """Build (n_m_max, max_nlm, NHS) float64 from list of (nlm, NHS) complex matrices."""
    P = torch.zeros(n_m_max, _max_n_lm, _NHS, dtype=DTYPE, device=DEVICE)
    for mc in range(n_m_max):
        nlm = _n_lm_m[mc]
        P[mc, :nlm, :] = per_m_list[mc].real
    return P


# Sign vectors for ES/EA parity (applied to max_nlm dimension)
# ES (even index) = (l-m) even, EA (odd index) = (l-m) odd
_sign_es_neg_r = torch.ones(_max_n_lm, dtype=DTYPE, device=DEVICE)
_sign_es_neg_r[0::2] = -1.0  # -1 for ES, +1 for EA

_sign_ea_neg_r = -_sign_es_neg_r  # +1 for ES, -1 for EA (for S hemisphere)

# Complex sign vector for post-bmm combination in spat_to_sphertor
_sign_ea_neg = _sign_ea_neg_r.to(CDTYPE)

# --- Forward SHT padded matrices: float64 (n_m_max, NHS, max_nlm) ---
# All Plm matrices are real; storing as float64 and using view_as_real bmm
# gives 2-6x speedup over complex128 bmm.
_P_plm_T_r = _build_padded_T_r(_Plm_c_m)
_P_PlmG_T_r = _build_padded_T_r(_PlmG_c_m)
_P_PlmC_T_r = _build_padded_T_r(_PlmC_c_m)

_P_plm_signS_T_r = _P_plm_T_r * _sign_ea_neg_r
_P_PlmC_sign_T_r = _P_PlmC_T_r * _sign_es_neg_r
_P_PlmG_sign_T_r = _P_PlmG_T_r * _sign_es_neg_r

# --- Padded matrices for scal_to_grad_spat (batched gradient SHT) ---
# dPlm: derivative of associated Legendre functions (parity-flipped vs Plm)
_P_dplm_T_r = _build_padded_T_r(_dPlm_c_m)
_P_dplm_signS_T_r = _P_dplm_T_r * _sign_es_neg_r  # ES spectral = EA spatial → negate for S

# m*Plm: for phi gradient. Same parity as Plm.
_mPlm_c_m = [_dm_m[mc] * _Plm_c_m[mc] for mc in range(n_m_max)]
_P_mPlm_T_r = _build_padded_T_r(_mPlm_c_m)
_P_mPlm_signS_T_r = _P_mPlm_T_r * _sign_ea_neg_r  # same parity sign as Plm

# Stacked matrix for torpol_to_spat: single bmm across all 6 component types
_torpol_mats_r = torch.cat([
    _P_plm_T_r, _P_plm_signS_T_r,
    _P_PlmG_T_r, _P_PlmC_T_r,
    _P_PlmC_sign_T_r, _P_PlmG_sign_T_r,
], dim=0)  # (6*n_m_max, NHS, max_nlm) float64

# --- Inverse SHT padded matrices: float64 (n_m_max, max_nlm, NHS) ---
_wPlm_pad_r = _build_padded_r(_wPlm_c_m)
_wdPlm_pad_r = _build_padded_r(_wdPlm_c_m)
_wPlm_sign_pad_r = _wPlm_pad_r * _sign_ea_neg_r.unsqueeze(1)

# For spat_to_sphertor: wPlm_dm = -ci*dm*wPlm is purely imaginary.
# Store the imaginary coefficient: -dm*wPlm (float64).
_wPlm_dm_imag_m = [(-_dm_m[mc]) * _wPlm_c_m[mc].real for mc in range(n_m_max)]
_wPlm_dm_imag_pad_r = torch.zeros(n_m_max, _max_n_lm, _NHS, dtype=DTYPE, device=DEVICE)
for _mc in range(n_m_max):
    _nlm = _n_lm_m[_mc]
    _wPlm_dm_imag_pad_r[_mc, :_nlm, :] = _wPlm_dm_imag_m[_mc]

# spat_to_sphertor: separate real (wdPlm) and imaginary (wPlm_dm) matrix groups
# Real group: 4 copies of wdPlm (for input positions [f2N, f1N, f2S, f1S])
_sphertor_D_r = torch.cat([_wdPlm_pad_r] * 4, dim=0)
# Imag coeff group: 4 copies of -dm*wPlm (for input positions [f1N, f2N, f1S, f2S])
_sphertor_C_r = torch.cat([_wPlm_dm_imag_pad_r] * 4, dim=0)

# Free intermediate matrices only used to build the stacked versions above
del _P_PlmG_T_r, _P_PlmC_T_r, _P_PlmC_sign_T_r, _P_PlmG_sign_T_r
del _wdPlm_pad_r, _wPlm_dm_imag_pad_r
del _Plm_c_m, _dPlm_c_m, _wPlm_c_m, _wdPlm_c_m, _PlmG_c_m, _PlmC_c_m
del _mPlm_c_m, _wPlm_dm_imag_m


def scal_to_spat(Slm: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Scalar spherical harmonic to spatial grid.

    Args:
        Slm: shape (lm_max,) or (lm_max, n_batch), complex spectral coefficients
        lcut: maximum degree to include (default: l_max)

    Returns:
        sc: shape (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real grid values
    """
    if lcut is None:
        lcut = l_max

    batched = Slm.dim() == 2
    if not batched:
        Slm = Slm.unsqueeze(1)  # (lm_max, 1)

    n_batch = Slm.shape[1]

    # Gather spectral coefficients for all m-modes: (n_m_max, max_nlm, n_batch)
    Q_pad = Slm[_spec_gather]

    # Real bmm: Plm matrices are real, use view_as_real for ~3x speedup
    Q_ri = torch.view_as_real(Q_pad).flatten(-2)       # (n_m_max, max_nlm, 2*n_batch) float64
    sN_ri = torch.bmm(_P_plm_T_r, Q_ri)               # (n_m_max, NHS, 2*n_batch) float64
    sS_ri = torch.bmm(_P_plm_signS_T_r, Q_ri)         # (n_m_max, NHS, 2*n_batch) float64
    sN = torch.view_as_complex(sN_ri.unflatten(-1, (-1, 2)))  # (n_m_max, NHS, n_batch) complex
    sS = torch.view_as_complex(sS_ri.unflatten(-1, (-1, 2)))

    # Assemble frequency-domain array: (n_theta_max, n_phi_max//2+1, n_batch)
    tmp = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmp[0::2, :n_m_max, :] = sN.permute(1, 0, 2)
    tmp[1::2, :n_m_max, :] = sS.permute(1, 0, 2)

    sc = torch.fft.irfft(tmp, n=n_phi_max, dim=1, norm="forward")

    if batched:
        sc = sc.permute(2, 0, 1)
    else:
        sc = sc.squeeze(2)

    return sc


def scal_to_SH(sc: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Spatial grid to scalar spherical harmonic coefficients.

    Args:
        sc: shape (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real
        lcut: maximum degree (default: l_max)

    Returns:
        Slm: shape (lm_max,) or (lm_max, n_batch), complex spectral coefficients
    """
    if lcut is None:
        lcut = l_max

    batched = sc.dim() == 3
    if not batched:
        sc = sc.unsqueeze(0)

    n_batch = sc.shape[0]

    # Forward FFT along phi
    f1TM = torch.fft.rfft(sc, dim=2, norm="forward")  # (n_batch, n_theta_max, n_phi_max//2+1)

    # Extract N/S hemispheres for all m-modes: (n_batch, NHS, n_m_max)
    f1_N = f1TM[:, 0::2, :n_m_max]
    f1_S = f1TM[:, 1::2, :n_m_max]

    # Rearrange to (n_m_max, NHS, n_batch) for bmm
    f1N = f1_N.permute(2, 1, 0).contiguous()
    f1S = f1_S.permute(2, 1, 0).contiguous()

    # Real bmm: wPlm matrices are real, use view_as_real for ~3x speedup
    f1N_ri = torch.view_as_real(f1N).flatten(-2)   # (n_m_max, NHS, 2*n_batch) float64
    f1S_ri = torch.view_as_real(f1S).flatten(-2)   # (n_m_max, NHS, 2*n_batch) float64
    RN_ri = torch.bmm(_wPlm_pad_r, f1N_ri)         # (n_m_max, max_nlm, 2*n_batch) float64
    RS_ri = torch.bmm(_wPlm_sign_pad_r, f1S_ri)    # (n_m_max, max_nlm, 2*n_batch) float64

    # Combine and scatter to output
    result = torch.view_as_complex((RN_ri + RS_ri).unflatten(-1, (-1, 2)))
    Slm = result.reshape(-1, n_batch)[_result_to_lm]  # (lm_max, n_batch)

    if not batched:
        Slm = Slm.squeeze(1)

    return Slm


def torpol_to_spat(Qlm: torch.Tensor, Slm: torch.Tensor, Tlm: torch.Tensor,
                   lcut: int = None):
    """Vector SHT: Q,S,T (poloidal/toroidal) to (br, btheta, bphi) on grid.

    This matches native_qst_to_spat in shtransforms.f90.

    Args:
        Qlm, Slm, Tlm: shape (lm_max,) or (lm_max, n_batch), complex spectral coefficients

    Returns:
        (brc, btc, bpc): each (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real
    """
    if lcut is None:
        lcut = l_max

    batched = Qlm.dim() == 2
    if not batched:
        Qlm = Qlm.unsqueeze(1)
        Slm = Slm.unsqueeze(1)
        Tlm = Tlm.unsqueeze(1)

    n_batch = Qlm.shape[1]

    # Build helper vectors: (lm_max, n_batch)
    bhG = Slm - ci * Tlm
    bhC = Slm + ci * Tlm
    bhG[0] = 0.0
    bhC[0] = 0.0

    # Gather spectral coefficients for all m-modes: (n_m_max, max_nlm, n_batch)
    Q_pad = Qlm[_spec_gather]
    G_pad = bhG[_spec_gather]
    C_pad = bhC[_spec_gather]

    # Stack inputs matching _torpol_mats_r order:
    # [plm_N, plm_S, PlmG, PlmC, PlmC_sign, PlmG_sign]
    inputs_c = torch.cat([Q_pad, Q_pad, G_pad, C_pad, G_pad, C_pad], dim=0)

    # Real bmm: Plm matrices are real, use view_as_real for ~6x speedup
    inputs_ri = torch.view_as_real(inputs_c).flatten(-2)
    results_ri = torch.bmm(_torpol_mats_r, inputs_ri)
    results = torch.view_as_complex(results_ri.unflatten(-1, (-1, 2)))

    # Unpack: (6, n_m_max, NHS, n_batch)
    brN, brS, N1, N2, S1, S2 = results.view(6, n_m_max, _NHS, n_batch).unbind(0)

    # Assemble frequency-domain arrays
    tmpr = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmpt = torch.zeros_like(tmpr)
    tmpp = torch.zeros_like(tmpr)

    # Radial
    tmpr[0::2, :n_m_max, :] = brN.permute(1, 0, 2)
    tmpr[1::2, :n_m_max, :] = brS.permute(1, 0, 2)

    # Angular: theta = 0.5*(N1+N2), phi = -ci*0.5*(N1-N2) for each hemisphere
    half_N1pN2 = half * (N1 + N2)
    half_S1pS2 = half * (S1 + S2)
    half_N1mN2 = half * (N1 - N2)
    half_S1mS2 = half * (S1 - S2)

    tmpt[0::2, :n_m_max, :] = half_N1pN2.permute(1, 0, 2)
    tmpt[1::2, :n_m_max, :] = half_S1pS2.permute(1, 0, 2)
    tmpp[0::2, :n_m_max, :] = (-ci * half_N1mN2).permute(1, 0, 2)
    tmpp[1::2, :n_m_max, :] = (-ci * half_S1mS2).permute(1, 0, 2)

    brc = torch.fft.irfft(tmpr, n=n_phi_max, dim=1, norm="forward")
    btc = torch.fft.irfft(tmpt, n=n_phi_max, dim=1, norm="forward")
    bpc = torch.fft.irfft(tmpp, n=n_phi_max, dim=1, norm="forward")

    if batched:
        brc = brc.permute(2, 0, 1)
        btc = btc.permute(2, 0, 1)
        bpc = bpc.permute(2, 0, 1)
    else:
        brc = brc.squeeze(2)
        btc = btc.squeeze(2)
        bpc = bpc.squeeze(2)

    return brc, btc, bpc


def scal_to_grad_spat(Slm: torch.Tensor, lcut: int = None):
    """Scalar to gradient on spatial grid: ds/dθ and (1/sinθ)*ds/dφ.

    Matches native_sph_to_grad_spat. Supports batched input.

    Args:
        Slm: shape (lm_max,) or (lm_max, n_batch), complex spectral coefficients
        lcut: maximum degree to include (default: l_max)

    Returns:
        (gradtc, gradpc): each (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max) real
    """
    if lcut is None:
        lcut = l_max

    batched = Slm.dim() == 2
    if not batched:
        Slm = Slm.unsqueeze(1)

    n_batch = Slm.shape[1]

    # Gather spectral coefficients: (n_m_max, max_nlm, n_batch)
    Q_pad = Slm[_spec_gather]

    # view_as_real bmm: float64 dgemm for ~3x speedup
    Q_ri = torch.view_as_real(Q_pad).flatten(-2)  # (n_m_max, max_nlm, 2*n_batch)

    # Theta gradient: dPlm.T @ Q (parity-flipped: S hemisphere negates ES spectral)
    tN_ri = torch.bmm(_P_dplm_T_r, Q_ri)          # (n_m_max, NHS, 2*n_batch)
    tS_ri = torch.bmm(_P_dplm_signS_T_r, Q_ri)    # (n_m_max, NHS, 2*n_batch)
    tN = torch.view_as_complex(tN_ri.unflatten(-1, (-1, 2)))  # (n_m_max, NHS, n_batch)
    tS = torch.view_as_complex(tS_ri.unflatten(-1, (-1, 2)))

    # Phi gradient: ci * m * Plm.T @ Q (same parity as scal_to_spat)
    pN_ri = torch.bmm(_P_mPlm_T_r, Q_ri)          # (n_m_max, NHS, 2*n_batch)
    pS_ri = torch.bmm(_P_mPlm_signS_T_r, Q_ri)    # (n_m_max, NHS, 2*n_batch)
    pN = torch.view_as_complex(pN_ri.unflatten(-1, (-1, 2)))
    pS = torch.view_as_complex(pS_ri.unflatten(-1, (-1, 2)))

    # Assemble interleaved theta grid: N at even, S at odd
    tmpt = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmpt[0::2, :n_m_max, :] = tN.permute(1, 0, 2)
    tmpt[1::2, :n_m_max, :] = tS.permute(1, 0, 2)

    tmpp = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmpp[0::2, :n_m_max, :] = (ci * pN).permute(1, 0, 2)
    tmpp[1::2, :n_m_max, :] = (ci * pS).permute(1, 0, 2)

    gradtc = torch.fft.irfft(tmpt, n=n_phi_max, dim=1, norm="forward")
    gradpc = torch.fft.irfft(tmpp, n=n_phi_max, dim=1, norm="forward")

    if batched:
        gradtc = gradtc.permute(2, 0, 1)
        gradpc = gradpc.permute(2, 0, 1)
    else:
        gradtc = gradtc.squeeze(2)
        gradpc = gradpc.squeeze(2)

    return gradtc, gradpc


def spat_to_SH(sc: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Alias for scal_to_SH (spatial → spectral for scalars)."""
    return scal_to_SH(sc, lcut)


def spat_to_sphertor(vt: torch.Tensor, vp: torch.Tensor, lcut: int = None):
    """Vector SHT: (vtheta, vphi) on grid → spheroidal (S) and toroidal (T) coefficients.

    Matches native_spat_to_sph_tor in shtransforms.f90.

    Args:
        vt, vp: shape (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real

    Returns:
        (Slm, Tlm): each (lm_max,) or (lm_max, n_batch), complex
    """
    if lcut is None:
        lcut = l_max

    batched = vt.dim() == 3
    if not batched:
        vt = vt.unsqueeze(0)
        vp = vp.unsqueeze(0)

    n_batch = vt.shape[0]

    # FFT along phi: (n_batch, n_theta, n_phi) → (n_batch, n_theta, n_phi//2+1)
    f2TM = torch.fft.rfft(vt, dim=2, norm="forward")
    f1TM = torch.fft.rfft(vp, dim=2, norm="forward")

    # Multiply by 1/sin²(θ) — grid-layout (interleaved N/S)
    ost2 = O_sin_theta_E2_grid.unsqueeze(0).unsqueeze(2)  # (1, n_theta, 1)
    f1TM = f1TM * ost2
    f2TM = f2TM * ost2

    # Extract N/S hemispheres for all m-modes: (n_batch, NHS, n_m_max)
    f1_N = f1TM[:, 0::2, :n_m_max]
    f1_S = f1TM[:, 1::2, :n_m_max]
    f2_N = f2TM[:, 0::2, :n_m_max]
    f2_S = f2TM[:, 1::2, :n_m_max]

    # Rearrange to (n_m_max, NHS, n_batch) for bmm
    f1N = f1_N.permute(2, 1, 0).contiguous()
    f1S = f1_S.permute(2, 1, 0).contiguous()
    f2N = f2_N.permute(2, 1, 0).contiguous()
    f2S = f2_S.permute(2, 1, 0).contiguous()

    # Real bmm: separate real (wdPlm=D) and imaginary (wPlm_dm=j*C) matrix groups.
    # D (real): positions [1,2,5,6], inputs [f2N, f1N, f2S, f1S]
    # C (imag coeff -dm*wPlm): positions [0,3,4,7], inputs [f1N, f2N, f1S, f2S]
    D_inputs = torch.cat([f2N, f1N, f2S, f1S], dim=0)  # (4*nm, NHS, nb) complex
    D_ri = torch.view_as_real(D_inputs).flatten(-2)      # (4*nm, NHS, 2*nb) float64
    D_out = torch.view_as_complex(
        torch.bmm(_sphertor_D_r, D_ri).unflatten(-1, (-1, 2)))
    D_parts = D_out.view(4, n_m_max, _max_n_lm, n_batch)  # [Bf2N, Bf1N, Bf2S, Bf1S]

    # For j*C @ x: result = complex(-C@x_imag, C@x_real)
    # Achieved by feeding [-x_imag, x_real] interleaved as the "real" input
    C_inputs = torch.cat([f1N, f2N, f1S, f2S], dim=0)
    C_swapped = torch.stack([-C_inputs.imag, C_inputs.real], dim=-1).flatten(-2)
    C_out = torch.view_as_complex(
        torch.bmm(_sphertor_C_r, C_swapped).unflatten(-1, (-1, 2)))
    C_parts = C_out.view(4, n_m_max, _max_n_lm, n_batch)  # [Af1N, Af2N, Af1S, Af2S]

    # Combine using ES/EA parity:
    # Slm = (Af1N + Bf2N) + sign * (Af1S - Bf2S)
    # Tlm = (-Bf1N + Af2N) + sign * (Bf1S + Af2S)
    sign = _sign_ea_neg.unsqueeze(1)  # (max_nlm, 1) complex

    Slm_pad = (C_parts[0] + D_parts[0]) + sign * (C_parts[2] - D_parts[2])
    Tlm_pad = (-D_parts[1] + C_parts[1]) + sign * (D_parts[3] + C_parts[3])

    # Scatter to output using precomputed index
    Slm = Slm_pad.reshape(-1, n_batch)[_result_to_lm]
    Tlm = Tlm_pad.reshape(-1, n_batch)[_result_to_lm]

    # Division by l(l+1)
    Slm *= _inv_dLh
    Tlm *= _inv_dLh

    if not batched:
        Slm = Slm.squeeze(1)
        Tlm = Tlm.squeeze(1)

    return Slm, Tlm


def sphtor_to_spat(Slm: torch.Tensor, Tlm: torch.Tensor,
                   lcut: int = None):
    """Spheroidal/toroidal (S, T) to (vtheta, vphi) on spatial grid.

    Matches SHsphtor_to_spat_l — angular-only synthesis (no radial component).

    Args:
        Slm, Tlm: shape (lm_max,) complex spectral coefficients

    Returns:
        (vtc, vpc): each shape (n_theta_max, n_phi_max) real grid values
    """
    Qlm = torch.zeros_like(Slm)
    _, vtc, vpc = torpol_to_spat(Qlm, Slm, Tlm, lcut)
    return vtc, vpc


def pol_to_curlr_spat(Qlm: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Radial component of curl of poloidal field: l(l+1) * Qlm → scalar synthesis.

    Matches pol_to_curlr_spat in shtns.f90: dQlm = dLh * Qlm, then SH_to_spat.

    Args:
        Qlm: shape (lm_max,) or (lm_max, n_batch), complex

    Returns:
        cvrc: shape (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real
    """
    dLh_c = dLh.to(CDTYPE)
    if Qlm.dim() == 2:
        dQlm = dLh_c.unsqueeze(1) * Qlm
    else:
        dQlm = dLh_c * Qlm
    return scal_to_spat(dQlm, lcut)


def pol_to_grad_spat(Slm: torch.Tensor, lcut: int = None):
    """Gradient of poloidal scalar: Qlm = dLh * Slm → gradient synthesis.

    Matches pol_to_grad_spat in shtns.f90: premultiply by l(l+1), then SHsph_to_spat.

    Args:
        Slm: shape (lm_max,) or (lm_max, n_batch), complex

    Returns:
        (gradtc, gradpc): each (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max) real
    """
    dLh_c = dLh.to(CDTYPE)
    if Slm.dim() == 2:
        Qlm = dLh_c.unsqueeze(1) * Slm
    else:
        Qlm = dLh_c * Slm
    return scal_to_grad_spat(Qlm, lcut)


def torpol_to_dphspat(dWlm: torch.Tensor, Zlm: torch.Tensor,
                      lcut: int = None):
    """Phi derivative of toroidal/poloidal field.

    Matches torpol_to_dphspat in shtns.f90:
    Slm = ci*m*dWlm, Tlm = ci*m*Zlm → sphtor_to_spat → multiply by O_sin_theta_E2.

    Args:
        dWlm, Zlm: shape (lm_max,) complex

    Returns:
        (dvtdp, dvpdp): each (n_theta_max, n_phi_max) real
    """
    Slm = dPhi * dWlm  # dPhi = ci * m
    Tlm = dPhi * Zlm
    dvtdp, dvpdp = sphtor_to_spat(Slm, Tlm, lcut)
    dvtdp = dvtdp * O_sin_theta_E2_grid.unsqueeze(1)
    dvpdp = dvpdp * O_sin_theta_E2_grid.unsqueeze(1)
    return dvtdp, dvpdp


def torpol_to_curl_spat(or2, Blm: torch.Tensor, ddBlm: torch.Tensor,
                        Jlm: torch.Tensor, dJlm: torch.Tensor,
                        lcut: int = None):
    """Curl of toroidal/poloidal magnetic field.

    Matches torpol_to_curl_spat in shtns.f90:
    Qlm = dLh*Jlm, Tlm = or2*dLh*Blm - ddBlm → QST synthesis with Slm=dJlm.

    Args:
        or2: scalar or tensor (n_batch,) — 1/r² at radial level(s)
        Blm, ddBlm, Jlm, dJlm: shape (lm_max,) or (lm_max, n_batch), complex

    Returns:
        (cvrc, cvtc, cvpc): each (n_theta_max, n_phi_max) or (n_batch, n_theta_max, n_phi_max), real
    """
    dLh_c = dLh.to(CDTYPE)
    if Blm.dim() == 2:
        dLh_2d = dLh_c.unsqueeze(1)  # (lm_max, 1)
        or2_t = torch.as_tensor(or2, dtype=CDTYPE, device=DEVICE).unsqueeze(0)  # (1, n_batch)
        Qlm = dLh_2d * Jlm
        Tlm = or2_t * dLh_2d * Blm - ddBlm
    else:
        Qlm = dLh_c * Jlm
        Tlm = float(or2) * dLh_c * Blm - ddBlm
    return torpol_to_spat(Qlm, dJlm.clone(), Tlm, lcut)


def spat_to_qst(f: torch.Tensor, g: torch.Tensor, h: torch.Tensor,
                lcut: int = None):
    """Spatial to QST (poloidal/toroidal) decomposition.

    Matches spat_to_qst in shtns.f90:
    qLM = scal_to_SH(f), (sLM, tLM) = spat_to_sphertor(g, h).

    Args:
        f: shape (n_theta_max, n_phi_max) real — radial component
        g, h: shape (n_theta_max, n_phi_max) real — theta, phi components

    Returns:
        (qLM, sLM, tLM): each (lm_max,) complex
    """
    qLM = scal_to_SH(f, lcut)
    sLM, tLM = spat_to_sphertor(g, h, lcut)
    return qLM, sLM, tLM
