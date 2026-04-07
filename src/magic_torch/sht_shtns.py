"""SHTns GPU backend for spherical harmonic transforms. [v4 no-robert-no-sin]

Drop-in replacements for the 4 hot-path SHT functions in sht.py.
Uses SHTns's JIT-compiled CUDA kernels with on-the-fly Legendre recurrence.

Data layout conversion:
- Spectral: our (lm_max, n_batch) → SHTns (n_batch, lm_max) via .T.contiguous()
- Spatial: SHTns (n_batch, n_phi, n_theta_sorted) → our (n_batch, n_theta_interleaved, n_phi)
  via transpose + _grid_idx permutation

SHTns handles sin(theta) weighting internally for vector transforms.
SHTns does NOT divide by l(l+1) — we apply _inv_dLh after spat_to_sphertor.
Robert form is NOT enabled — our code expects raw physical velocity/magnetic fields.
"""

import ctypes
import torch
import shtns

from .precision import DTYPE, CDTYPE, DEVICE
from .params import l_max, lm_max, n_theta_max, n_phi_max, n_m_max, minc, radial_chunk_size
from .horizontal_data import _grid_idx, n_theta_cal2ord, dLh

# Inverse dLh for spat_to_sphertor post-processing: 1/[l(l+1)], zero for l=0
_inv_dLh = torch.zeros(lm_max, 1, dtype=CDTYPE, device=DEVICE)
_inv_dLh[1:, 0] = 1.0 / dLh[1:].to(CDTYPE)

# --- SHTns initialization ---

# Load the CUDA SHTns library for ctypes access to shtns_set_many
try:
    import _shtns_cuda as _shtns_mod
    _lib = ctypes.CDLL(_shtns_mod.__file__)
    _lib.shtns_set_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    _lib.shtns_set_many.restype = ctypes.c_int
    _has_cuda_shtns = True
except (ImportError, OSError):
    _has_cuda_shtns = False

if not _has_cuda_shtns:
    raise ImportError("SHTns CUDA module not available")


def _make_shtns_config(n_batch):
    """Create an SHTns config for a given batch size."""
    sh = shtns.sht(l_max, l_max, mres=minc,
                   norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)
    # Do NOT enable Robert form — SHTns outputs raw physical fields without it,
    # which matches our bmm convention. Verified by unit test diagnostic.
    # Set batching via ctypes
    cfg_ptr = ctypes.c_void_p(int(sh.this))
    ret = _lib.shtns_set_many(cfg_ptr, n_batch, sh.nlm)
    if ret <= 0:
        raise RuntimeError(f"shtns_set_many({n_batch}, {sh.nlm}) failed: ret={ret}")
    # Set grid with GPU — use our grid sizes
    assert n_theta_max % 4 == 0, f"n_theta_max={n_theta_max} must be divisible by 4 for SHTns GPU"
    sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)
    return sh


# Create configs for common batch sizes
_chunk_size = radial_chunk_size if radial_chunk_size > 0 else 32
_configs = {}  # n_batch -> shtns config


def _get_config(n_batch):
    """Get or create an SHTns config for the given batch size."""
    if n_batch not in _configs:
        _configs[n_batch] = _make_shtns_config(n_batch)
    return _configs[n_batch]


# --- Layout conversion helpers ---

def _spec_to_shtns(slm):
    """Convert spectral (lm_max, n_batch) complex128 → (n_batch, lm_max) contiguous."""
    return slm.T.contiguous()


def _spec_from_shtns(slm_t, n_batch):
    """Convert spectral (n_batch, lm_max) → (lm_max, n_batch)."""
    return slm_t.T.contiguous()


def _spat_from_shtns(out_shtns):
    """Convert spatial (n_batch, n_phi, n_theta_sorted) → (n_batch, n_theta_interleaved, n_phi).

    SHTns outputs theta in sorted geographic order (N pole → S pole).
    Our code uses interleaved N/S: [θ_N[0], θ_S[0], θ_N[1], θ_S[1], ...].
    """
    # Transpose: (n_batch, n_phi, n_theta) → (n_batch, n_theta, n_phi)
    out = out_shtns.transpose(-1, -2)
    # Permute: sorted → interleaved via _grid_idx
    return out[:, _grid_idx, :]


def _spat_to_shtns(grid_data):
    """Convert spatial (n_batch, n_theta_interleaved, n_phi) → (n_batch, n_phi, n_theta_sorted).

    Inverse of _spat_from_shtns.
    """
    # Permute: interleaved → sorted via n_theta_cal2ord
    out = grid_data[:, n_theta_cal2ord, :]
    # Transpose: (n_batch, n_theta, n_phi) → (n_batch, n_phi, n_theta)
    return out.transpose(-1, -2).contiguous()


# --- SHT functions ---

def scal_to_spat(Slm: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Scalar SHT: spectral → spatial grid via SHTns GPU."""
    batched = Slm.dim() == 2
    if not batched:
        Slm = Slm.unsqueeze(1)
    n_batch = Slm.shape[1]

    sh = _get_config(n_batch)
    slm_gpu = _spec_to_shtns(Slm)
    out_gpu = torch.empty(n_batch, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE)

    torch.cuda.synchronize()
    sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
    torch.cuda.synchronize()

    result = _spat_from_shtns(out_gpu)
    if not batched:
        result = result.squeeze(0)
    return result


def scal_to_SH(sc: torch.Tensor, lcut: int = None) -> torch.Tensor:
    """Scalar SHT: spatial grid → spectral via SHTns GPU."""
    batched = sc.dim() == 3
    if not batched:
        sc = sc.unsqueeze(0)
    n_batch = sc.shape[0]

    sh = _get_config(n_batch)
    sc_shtns = _spat_to_shtns(sc)
    slm_gpu = torch.empty(n_batch, lm_max, dtype=CDTYPE, device=DEVICE)

    torch.cuda.synchronize()
    sh.cu_spat_to_SH(sc_shtns.data_ptr(), slm_gpu.data_ptr())
    torch.cuda.synchronize()

    result = _spec_from_shtns(slm_gpu, n_batch)
    if not batched:
        result = result.squeeze(1)
    return result


def torpol_to_spat(Qlm: torch.Tensor, Slm: torch.Tensor, Tlm: torch.Tensor,
                   lcut: int = None):
    """Vector SHT: Q,S,T → (br, btheta, bphi) via SHTns GPU.

    SHTns handles the bhG/bhC hemisphere decomposition internally.
    Callers pass the same Q, S, T as the bmm version (dLh pre-multiply
    done in step_time.py, not here).
    """
    batched = Qlm.dim() == 2
    if not batched:
        Qlm = Qlm.unsqueeze(1)
        Slm = Slm.unsqueeze(1)
        Tlm = Tlm.unsqueeze(1)
    n_batch = Qlm.shape[1]

    sh = _get_config(n_batch)
    q_gpu = _spec_to_shtns(Qlm)
    s_gpu = _spec_to_shtns(Slm)
    t_gpu = _spec_to_shtns(Tlm)
    vr = torch.empty(n_batch, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE)
    vt = torch.empty_like(vr)
    vp = torch.empty_like(vr)

    torch.cuda.synchronize()
    sh.cu_SHqst_to_spat(q_gpu.data_ptr(), s_gpu.data_ptr(), t_gpu.data_ptr(),
                         vr.data_ptr(), vt.data_ptr(), vp.data_ptr())
    torch.cuda.synchronize()

    brc = _spat_from_shtns(vr)
    btc = _spat_from_shtns(vt)
    bpc = _spat_from_shtns(vp)

    if not batched:
        brc = brc.squeeze(0)
        btc = btc.squeeze(0)
        bpc = bpc.squeeze(0)
    return brc, btc, bpc


def spat_to_sphertor(vt: torch.Tensor, vp: torch.Tensor, lcut: int = None):
    """Vector SHT: (vtheta, vphi) → spheroidal (S) and toroidal (T) via SHTns GPU.

    SHTns handles 1/sin(theta) weighting internally — do NOT pre-multiply by O_sin_theta_E2.
    SHTns does NOT divide by l(l+1) — we apply _inv_dLh after.
    """
    batched = vt.dim() == 3
    if not batched:
        vt = vt.unsqueeze(0)
        vp = vp.unsqueeze(0)
    n_batch = vt.shape[0]

    sh = _get_config(n_batch)
    vt_shtns = _spat_to_shtns(vt)
    vp_shtns = _spat_to_shtns(vp)
    slm_gpu = torch.empty(n_batch, lm_max, dtype=CDTYPE, device=DEVICE)
    tlm_gpu = torch.empty_like(slm_gpu)

    torch.cuda.synchronize()
    sh.cu_spat_to_SHsphtor(vt_shtns.data_ptr(), vp_shtns.data_ptr(),
                            slm_gpu.data_ptr(), tlm_gpu.data_ptr())
    torch.cuda.synchronize()

    Slm = _spec_from_shtns(slm_gpu, n_batch)
    Tlm = _spec_from_shtns(tlm_gpu, n_batch)

    # SHTns does NOT divide by l(l+1) — apply manually
    Slm = Slm * _inv_dLh
    Tlm = Tlm * _inv_dLh

    if not batched:
        Slm = Slm.squeeze(1)
        Tlm = Tlm.squeeze(1)
    return Slm, Tlm
