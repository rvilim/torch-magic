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
_cos_theta_N = cosTheta[:_NHS].to(DEVICE).contiguous()

# --- Precompute recurrence coefficients and seed table ---

_a_coeff = torch.zeros(lm_max, dtype=DTYPE, device="cpu")
_b_coeff = torch.zeros(lm_max, dtype=DTYPE, device="cpu")

# Seed table: un-normalized P_m^m(θ) = sqrt(fac_m) * sin(θ)^m
_seed_table = torch.zeros(n_m_max, _NHS, dtype=DTYPE, device="cpu")

# Per-m metadata
_lm_start = torch.zeros(n_m_max, dtype=torch.int32, device="cpu")
_n_lm = torch.zeros(n_m_max, dtype=torch.int32, device="cpu")

# Fill recurrence coefficients and seed table
_sin_N = sinTheta[:_NHS].cpu().numpy()
_pos = 0
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
_lm_start = _lm_start.to(DEVICE)
_n_lm = _n_lm.to(DEVICE)

_osq4pi = 1.0 / math.sqrt(4.0 * math.pi)
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
    n_batch,
    NHS: tl.constexpr,
    MAX_NLM: tl.constexpr,
    THETA_BLOCK: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    """Each program handles one m-mode and a tile of (THETA_BLOCK × BATCH_BLOCK).

    The Plm recurrence is vectorized across THETA_BLOCK theta points — each
    point has its own P_prev/P_curr state but shares the same a,b coefficients
    and spectral data. This gives THETA_BLOCK × BATCH_BLOCK independent FMAs
    per recurrence step, providing enough ILP to hide the serial l-dependency.
    """
    mc = tl.program_id(0)
    th_block = tl.program_id(1)
    bb = tl.program_id(2)

    th_offs = th_block * THETA_BLOCK + tl.arange(0, THETA_BLOCK)  # (THETA_BLOCK,)
    th_mask = th_offs < NHS
    batch_offs = bb * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)  # (BATCH_BLOCK,)
    batch_mask = batch_offs < n_batch

    # Load cos(theta) for this theta tile: (THETA_BLOCK,)
    cos_th = tl.load(cos_ptr + th_offs, mask=th_mask, other=0.0).to(tl.float64)

    lm0 = tl.load(lm_start_ptr + mc).to(tl.int64)
    nlm = tl.load(n_lm_ptr + mc).to(tl.int64)

    # Seed: un-normalized P_m^m(θ) for each theta in tile: (THETA_BLOCK,)
    P_curr = tl.load(seed_ptr + mc * NHS + th_offs, mask=th_mask, other=0.0).to(tl.float64)
    P_prev = tl.zeros([THETA_BLOCK], dtype=tl.float64)

    # Load spectral coefficient for l=m: (BATCH_BLOCK,)
    base = lm0 * n_batch + batch_offs
    f_re = tl.load(slm_re_ptr + base, mask=batch_mask, other=0.0).to(tl.float64)
    f_im = tl.load(slm_im_ptr + base, mask=batch_mask, other=0.0).to(tl.float64)

    # Accumulate: (THETA_BLOCK, BATCH_BLOCK) via outer product
    # P_curr[:, None] * f_re[None, :] gives the 2D tile
    acc_N_re = P_curr[:, None] * f_re[None, :]  # (THETA_BLOCK, BATCH_BLOCK)
    acc_N_im = P_curr[:, None] * f_im[None, :]
    sign = tl.full([1], 1.0, dtype=tl.float64)
    acc_S_re = P_curr[:, None] * f_re[None, :]
    acc_S_im = P_curr[:, None] * f_im[None, :]

    for k in range(1, MAX_NLM):
        if k < nlm:
            lm_k = lm0 + k
            a = tl.load(a_ptr + lm_k).to(tl.float64)
            b = tl.load(b_ptr + lm_k).to(tl.float64)

            # Recurrence vectorized across theta: (THETA_BLOCK,)
            P_next = a * cos_th * P_curr - b * P_prev
            P_prev = P_curr
            P_curr = P_next

            # Load spectral data: (BATCH_BLOCK,)
            off_k = lm_k * n_batch + batch_offs
            f_re = tl.load(slm_re_ptr + off_k, mask=batch_mask, other=0.0).to(tl.float64)
            f_im = tl.load(slm_im_ptr + off_k, mask=batch_mask, other=0.0).to(tl.float64)

            # Outer product accumulate: (THETA_BLOCK, BATCH_BLOCK)
            P_2d = P_curr[:, None]
            acc_N_re += P_2d * f_re[None, :]
            acc_N_im += P_2d * f_im[None, :]
            sign = sign * tl.full([1], -1.0, dtype=tl.float64)
            sP_2d = sign * P_2d
            acc_S_re += sP_2d * f_re[None, :]
            acc_S_im += sP_2d * f_im[None, :]

    # Apply normalization
    osq4pi: tl.constexpr = 0.28209479177387814
    acc_N_re *= osq4pi
    acc_N_im *= osq4pi
    acc_S_re *= osq4pi
    acc_S_im *= osq4pi

    # Write output: (mc, th, batch) layout
    # Output is (n_m_max, NHS, n_batch). Use 2D offset grid for the tile.
    # out_offs[i, j] = (mc * NHS + (th_block*THETA_BLOCK + i)) * n_batch + (bb*BATCH_BLOCK + j)
    th_out = th_block * THETA_BLOCK + tl.arange(0, THETA_BLOCK)  # (THETA_BLOCK,)
    out_offs = (mc * NHS + th_out[:, None]) * n_batch + batch_offs[None, :]  # (THETA_BLOCK, BATCH_BLOCK)
    out_mask = th_mask[:, None] & batch_mask[None, :]  # (THETA_BLOCK, BATCH_BLOCK)
    tl.store(out_N_re_ptr + out_offs, acc_N_re, mask=out_mask)
    tl.store(out_N_im_ptr + out_offs, acc_N_im, mask=out_mask)
    tl.store(out_S_re_ptr + out_offs, acc_S_re, mask=out_mask)
    tl.store(out_S_im_ptr + out_offs, acc_S_im, mask=out_mask)


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

    # Allocate output as separate real/imag
    out_N_re = torch.empty(n_m_max, _NHS, n_batch, dtype=DTYPE, device=DEVICE)
    out_N_im = torch.empty_like(out_N_re)
    out_S_re = torch.empty_like(out_N_re)
    out_S_im = torch.empty_like(out_N_re)

    # Tile sizes: each program handles THETA_BLOCK theta points × BATCH_BLOCK batch elements
    THETA_BLOCK = 16
    BATCH_BLOCK = min(64, n_batch)
    grid = (n_m_max,
            triton.cdiv(_NHS, THETA_BLOCK),
            triton.cdiv(n_batch, BATCH_BLOCK))
    _scal_to_spat_kernel[grid](
        slm_re, slm_im,
        out_N_re, out_N_im, out_S_re, out_S_im,
        _a_coeff, _b_coeff, _seed_table, _cos_theta_N,
        _lm_start, _n_lm,
        n_batch, _NHS, _MAX_NLM,
        THETA_BLOCK=THETA_BLOCK, BATCH_BLOCK=BATCH_BLOCK,
    )

    # Assemble frequency-domain array + irfft
    sN = torch.complex(out_N_re, out_N_im)
    sS = torch.complex(out_S_re, out_S_im)
    tmp = torch.zeros(n_theta_max, n_phi_max // 2 + 1, n_batch, dtype=CDTYPE, device=DEVICE)
    tmp[0::2, :n_m_max, :] = sN.permute(1, 0, 2)
    tmp[1::2, :n_m_max, :] = sS.permute(1, 0, 2)
    sc = torch.fft.irfft(tmp, n=n_phi_max, dim=1, norm="forward")

    if batched:
        return sc.permute(2, 0, 1)
    return sc.squeeze(2)
