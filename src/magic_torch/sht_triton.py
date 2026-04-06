"""On-the-fly Triton SHT kernel using Legendre recurrence.

Prototype: scalar forward SHT (scal_to_spat) only.
Computes Plm values via 3-term recurrence in registers instead of
loading precomputed Plm matrices from HBM.

Usage:
    from magic_torch.sht_triton import scal_to_spat_triton
    grid_values = scal_to_spat_triton(Slm)  # same API as sht.scal_to_spat
"""

import math
import torch
import triton
import triton.language as tl

from .precision import DTYPE, CDTYPE, DEVICE
from .params import l_max, lm_max, n_m_max, n_theta_max, n_phi_max, minc
from .horizontal_data import cosTheta, sinTheta

# Northern hemisphere theta values (sorted, for Plm evaluation)
_NHS = n_theta_max // 2
_cos_theta_N = cosTheta[:_NHS].to(DEVICE)

# --- Precompute recurrence coefficients and seed table ---

# Recurrence: P(l+1,m) = a(l+1,m) * cos(θ) * P(l,m) - b(l+1,m) * P(l-1,m)
# Indexed by flat lm index. a[lm] and b[lm] for the step producing P(l,m)
# from P(l-1,m) and P(l-2,m). At lm=lStart (l=m), these are unused (seed).
_a_coeff = torch.zeros(lm_max, dtype=DTYPE, device="cpu")
_b_coeff = torch.zeros(lm_max, dtype=DTYPE, device="cpu")

# Seed table: un-normalized P_m^m(θ) = sqrt(fac_m) * sin(θ)^m
# Precomputed for each (mc, theta_idx) to avoid tl.math.pow in kernel.
_seed_table = torch.zeros(n_m_max, _NHS, dtype=DTYPE, device="cpu")

# Per-m metadata
_lm_start = torch.zeros(n_m_max, dtype=torch.int32, device="cpu")
_n_lm = torch.zeros(n_m_max, dtype=torch.int32, device="cpu")

# Fill recurrence coefficients and seed table
_pos = 0
_sin_N = sinTheta[:_NHS].numpy()
for _mc in range(n_m_max):
    _m = _mc * minc
    _nlm = l_max - _m + 1
    _lm_start[_mc] = _pos
    _n_lm[_mc] = _nlm

    # Seed: sqrt(fac) * sin^m
    _fac = 1.0
    for _j in range(3, 2 * _m + 2, 2):
        _fac = _fac * _j / (_j - 1)
    _seed_val = math.sqrt(_fac)
    for _t in range(_NHS):
        if _sin_N[_t] != 0.0 or _m == 0:
            _seed_table[_mc, _t] = _seed_val * (_sin_N[_t] ** _m if _m > 0 else 1.0)
        else:
            _seed_table[_mc, _t] = 0.0

    # Recurrence coefficients for l = m+1 .. l_max
    for _l in range(_m + 1, l_max + 1):
        _k = _l - _m
        _lm_idx = _pos + _k
        _lm_prod = (_l - _m) * (_l + _m)
        _a_coeff[_lm_idx] = math.sqrt((2 * _l - 1) * (2 * _l + 1) / _lm_prod)
        if _l > _m + 1:
            _b_coeff[_lm_idx] = math.sqrt(
                (2 * _l + 1) * (_l + _m - 1) * (_l - _m - 1)
                / ((2 * _l - 3) * _lm_prod))

    _pos += _nlm

# Move to GPU
_a_coeff = _a_coeff.to(DEVICE)
_b_coeff = _b_coeff.to(DEVICE)
_seed_table = _seed_table.to(DEVICE)
_cos_theta_N = _cos_theta_N.contiguous()
_lm_start = _lm_start.to(DEVICE)
_n_lm = _n_lm.to(DEVICE)

# Normalization constant (applied once to final accumulator)
_osq4pi = 1.0 / math.sqrt(4.0 * math.pi)

# Compile-time max loop bound (must be >= l_max+1)
_MAX_NLM = l_max + 1


@triton.jit
def _scal_to_spat_kernel(
    slm_re_ptr, slm_im_ptr,
    out_N_re_ptr, out_N_im_ptr,
    out_S_re_ptr, out_S_im_ptr,
    a_ptr, b_ptr,
    seed_ptr,
    cos_ptr,
    lm_start_ptr,
    n_lm_ptr,
    n_batch: tl.constexpr,
    NHS: tl.constexpr,
    MAX_NLM: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    mc = tl.program_id(0)
    th = tl.program_id(1)
    bb = tl.program_id(2)

    batch_offs = bb * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    mask = batch_offs < n_batch

    cos_th = tl.load(cos_ptr + th)
    lm0 = tl.load(lm_start_ptr + mc).to(tl.int64)
    nlm = tl.load(n_lm_ptr + mc).to(tl.int64)

    # Seed: un-normalized P_m^m(θ)
    P_curr = tl.load(seed_ptr + mc * NHS + th)
    P_prev = 0.0

    # Load spectral coefficient for l=m
    base = lm0 * n_batch + batch_offs
    f_re = tl.load(slm_re_ptr + base, mask=mask, other=0.0)
    f_im = tl.load(slm_im_ptr + base, mask=mask, other=0.0)

    # Accumulate un-normalized Plm * f_lm
    acc_N_re = P_curr * f_re
    acc_N_im = P_curr * f_im
    sign = 1.0  # (-1)^(l-m) = +1 at l=m (k=0)
    acc_S_re = P_curr * f_re
    acc_S_im = P_curr * f_im

    # Recurrence loop: k = l - m, from 1 to nlm-1
    for k in range(1, MAX_NLM):
        if k < nlm:
            lm_k = lm0 + k
            a = tl.load(a_ptr + lm_k)
            b = tl.load(b_ptr + lm_k)
            P_next = a * cos_th * P_curr - b * P_prev
            P_prev = P_curr
            P_curr = P_next

            off_k = lm_k * n_batch + batch_offs
            f_re = tl.load(slm_re_ptr + off_k, mask=mask, other=0.0)
            f_im = tl.load(slm_im_ptr + off_k, mask=mask, other=0.0)

            acc_N_re += P_curr * f_re
            acc_N_im += P_curr * f_im
            sign = -sign
            acc_S_re += sign * P_curr * f_re
            acc_S_im += sign * P_curr * f_im

    # Apply normalization: osq4pi = 1/sqrt(4*pi)
    osq4pi: tl.constexpr = 0.28209479177387814
    acc_N_re *= osq4pi
    acc_N_im *= osq4pi
    acc_S_re *= osq4pi
    acc_S_im *= osq4pi

    # Write output: (mc, th, batch)
    out_off = (mc * NHS + th) * n_batch + batch_offs
    tl.store(out_N_re_ptr + out_off, acc_N_re, mask=mask)
    tl.store(out_N_im_ptr + out_off, acc_N_im, mask=mask)
    tl.store(out_S_re_ptr + out_off, acc_S_re, mask=mask)
    tl.store(out_S_im_ptr + out_off, acc_S_im, mask=mask)


def scal_to_spat_triton(Slm: torch.Tensor) -> torch.Tensor:
    """Forward scalar SHT via Triton on-the-fly Plm recurrence.

    Drop-in replacement for sht.scal_to_spat. Same input/output API.
    """
    batched = Slm.dim() == 2
    if not batched:
        Slm = Slm.unsqueeze(1)
    n_batch = Slm.shape[1]

    # Split complex → contiguous real/imag
    slm_re = Slm.real.contiguous()
    slm_im = Slm.imag.contiguous()

    # Allocate output as separate real/imag (no complex in Triton)
    out_N_re = torch.empty(n_m_max, _NHS, n_batch, dtype=DTYPE, device=DEVICE)
    out_N_im = torch.empty_like(out_N_re)
    out_S_re = torch.empty_like(out_N_re)
    out_S_im = torch.empty_like(out_N_re)

    BATCH_BLOCK = min(32, n_batch)
    grid = (n_m_max, _NHS, triton.cdiv(n_batch, BATCH_BLOCK))
    _scal_to_spat_kernel[grid](
        slm_re, slm_im,
        out_N_re, out_N_im, out_S_re, out_S_im,
        _a_coeff, _b_coeff, _seed_table, _cos_theta_N,
        _lm_start, _n_lm,
        n_batch, _NHS, _MAX_NLM, BATCH_BLOCK=BATCH_BLOCK,
    )

    # Assemble frequency-domain array + irfft (same as sht.py)
    sN = torch.complex(out_N_re, out_N_im)
    sS = torch.complex(out_S_re, out_S_im)
    tmp = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmp[0::2, :n_m_max, :] = sN.permute(1, 0, 2)
    tmp[1::2, :n_m_max, :] = sS.permute(1, 0, 2)
    sc = torch.fft.irfft(tmp, n=n_phi_max, dim=1, norm="forward")

    if batched:
        return sc.permute(2, 0, 1)
    return sc.squeeze(2)
