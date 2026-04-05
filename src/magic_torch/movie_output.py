"""Movie file output matching Fortran out_movie_file.f90.

Writes Fortran sequential unformatted binary files (big-endian, float32)
readable by MagIC's python/magic/movie.py Movie class.

Supported movie types:
- "Br CMB"  — radial B-field at CMB (n_surface=1, field_type=1)
- "Vr EQ"   — radial velocity at equator (n_surface=2, field_type=4)
- "T EQ"    — temperature at equator (n_surface=2, field_type=7)
- "Br EQ"   — radial B-field at equator (n_surface=2, field_type=1)
"""

import math
import struct

import numpy as np
import torch

from .params import (n_r_max, n_r_ic_max, n_theta_max, n_phi_max,
                     lm_max, l_max, minc, ra, ek, pr, prmag, radratio,
                     raxi, sc)
from .radial_scheme import r
from .radial_functions import or1, or2
from .horizontal_data import theta_ord, n_theta_cal2ord, _grid_idx
from .pre_calculations import tScale


# ============================================================
# Binary I/O helpers (Fortran sequential unformatted, big-endian)
# ============================================================

def _fort_write(f, data_bytes):
    """Write a Fortran sequential unformatted record (big-endian markers)."""
    n = len(data_bytes)
    f.write(struct.pack('>i', n))
    f.write(data_bytes)
    f.write(struct.pack('>i', n))


def _be_i32(*vals):
    """Pack int32 values as big-endian bytes."""
    return struct.pack('>' + 'i' * len(vals), *vals)


def _be_f64(val):
    """Pack float64 value as big-endian bytes."""
    return struct.pack('>d', float(val))


def _be_f32(*vals):
    """Pack float32 values as big-endian bytes."""
    return struct.pack('>' + 'f' * len(vals), *[float(v) for v in vals])


def _arr_be_f32(tensor):
    """Convert 1-D tensor to big-endian float32 bytes."""
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    return arr.astype('>f4').tobytes()


# ============================================================
# Movie string parsing
# ============================================================

# Field name → (field_type, file_prefix)
_FIELD_MAP = {
    'BR': (1, 'Br'),
    'BT': (2, 'Bt'),  # Btheta
    'BP': (3, 'Bp'),  # Bphi
    'VR': (4, 'Vr'),
    'VT': (5, 'Vt'),
    'VP': (6, 'Vp'),
    'T':  (7, 'T'),
    'TEMPERATURE': (7, 'T'),
    'S':  (7, 'T'),   # entropy = temperature in Boussinesq
}

# Surface name → (n_surface, const_func, file_suffix)
_SURFACE_MAP = {
    'CMB': (1, lambda: 1.0, 'CMB'),
    'EQ':  (2, lambda: (180.0 / math.pi) * float(theta_ord[n_theta_max // 2 - 1].cpu()), 'EQU'),
    'EQUAT': (2, lambda: (180.0 / math.pi) * float(theta_ord[n_theta_max // 2 - 1].cpu()), 'EQU'),
}


def parse_movie_string(s):
    """Parse a movie string like 'Br CMB' into (field_type, n_surface, const, filename).

    Returns:
        field_type: int (1=Br, 4=Vr, 7=T, etc.)
        n_surface: int (1=r-const, 2=theta-const)
        const: float (surface constant value)
        filename: str (e.g., 'Br_CMB_mov')
    """
    parts = s.strip().upper().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid movie string: '{s}' (need FIELD SURFACE)")

    field_str = parts[0]
    surface_str = parts[1]

    if field_str not in _FIELD_MAP:
        raise ValueError(f"Unknown movie field: '{field_str}'. Supported: {list(_FIELD_MAP.keys())}")
    field_type, field_prefix = _FIELD_MAP[field_str]

    if surface_str not in _SURFACE_MAP:
        raise ValueError(f"Unknown movie surface: '{surface_str}'. Supported: {list(_SURFACE_MAP.keys())}")
    n_surface, const_func, surf_suffix = _SURFACE_MAP[surface_str]

    const = const_func()
    filename = f"{field_prefix}_{surf_suffix}_mov"

    return field_type, n_surface, const, filename


# ============================================================
# File header
# ============================================================

def write_movie_header(f, n_surface, n_fields, const, field_types, runid="MagIC-PyTorch"):
    """Write the movie file header (10 records).

    Args:
        f: file object (binary write mode)
        n_surface: int (1=r-const, 2=theta-const)
        n_fields: int (number of fields per frame)
        const: float (surface constant — 1.0 for CMB, ~90 for equator)
        field_types: list of int (field type codes)
        runid: str (run identifier, padded to 64 bytes)
    """
    r_cmb = float(r[0].cpu())
    n_r_mov_tot = n_r_max + n_r_ic_max  # OC + IC radial points
    n_s_max = 0  # cylindrical grid size (0 for non-geos movies)

    # Record 1: version
    _fort_write(f, _be_i32(3))

    # Record 2: n_surface, n_fields
    _fort_write(f, _be_i32(n_surface, n_fields))

    # Record 3: const (float64!)
    _fort_write(f, _be_f64(const))

    # Record 4: field type codes
    _fort_write(f, _be_i32(*field_types))

    # Record 5: runid (64 bytes, padded)
    runid_bytes = runid.encode('ascii')[:64].ljust(64, b' ')
    _fort_write(f, runid_bytes)

    # Record 6: grid dimensions
    _fort_write(f, _be_i32(n_r_mov_tot, n_r_max, n_r_ic_max,
                            n_theta_max, n_phi_max, n_s_max, minc))

    # Record 7: physics params (float32)
    _fort_write(f, _be_f32(ra, ek, pr, raxi, sc, prmag, radratio, tScale))

    # Record 8: radial grid (OC + IC, normalized by r_cmb)
    r_oc = r.cpu().to(torch.float32) / r_cmb
    # IC radii: simple linear grid for insulating IC (placeholder)
    r_ic = torch.linspace(float(r[-1].cpu()) / r_cmb, 0.0, n_r_ic_max,
                          dtype=torch.float32)
    r_all = torch.cat([r_oc, r_ic])
    _fort_write(f, _arr_be_f32(r_all))

    # Record 9: theta_ord (sorted colatitudes)
    _fort_write(f, _arr_be_f32(theta_ord.cpu()))

    # Record 10: phi
    phi = torch.linspace(0, 2 * math.pi * (1 - 1.0 / n_phi_max),
                         n_phi_max, dtype=torch.float32)
    _fort_write(f, _arr_be_f32(phi))


# ============================================================
# Frame writing
# ============================================================

def write_movie_frame(f, time, field_data_list):
    """Write one movie frame (time + field arrays).

    Args:
        f: file object
        time: float (simulation time)
        field_data_list: list of numpy arrays (one per field), each float32
    """
    # Time record
    _fort_write(f, _be_f32(time))

    # One record per field
    for data in field_data_list:
        _fort_write(f, data.astype('>f4').tobytes())


# ============================================================
# Field extraction from grid-space data
# ============================================================

# Equatorial theta index in interleaved (calculation) order
_theta_eq_idx = int(_grid_idx[n_theta_max // 2 - 1].cpu().item())

# Theta reordering for surface movies: interleaved → sorted
_cal2ord = n_theta_cal2ord.cpu().numpy()


def extract_surface_field(field_type, nR, vr, vt, vp, br, bt, bp, sr):
    """Extract a 2-D field at radius index nR for a surface movie.

    Args:
        field_type: int (1=Br, 4=Vr, 7=T, etc.)
        nR: int (radial index)
        vr, vt, vp: (n_r, n_theta, n_phi) or None
        br, bt, bp: (n_r, n_theta, n_phi) or None
        sr: (n_r, n_theta, n_phi) or None

    Returns:
        (n_theta, n_phi) numpy float32 array in SORTED theta order
    """
    _or2 = float(or2[nR].cpu())
    _or1_val = float(or1[nR].cpu())

    if field_type == 1:    # Br
        data = _or2 * br[nR]
    elif field_type == 2:  # Btheta
        data = _or1_val * bt[nR]
    elif field_type == 3:  # Bphi
        data = _or1_val * bp[nR]
    elif field_type == 4:  # Vr
        data = _or2 * vr[nR]
    elif field_type == 5:  # Vtheta
        data = _or1_val * vt[nR]
    elif field_type == 6:  # Vphi
        data = _or1_val * vp[nR]
    elif field_type == 7:  # Temperature
        data = sr[nR]
    else:
        raise ValueError(f"Unsupported field_type {field_type} for surface movie")

    # Reorder from interleaved to sorted theta
    arr = data.detach().cpu().to(torch.float32).numpy()
    return arr[_cal2ord]


def extract_equat_field(field_type, vr, vt, vp, br, bt, bp, sr):
    """Extract a 2-D field at the equatorial theta for all radial levels.

    Args:
        field_type: int (1=Br, 4=Vr, 7=T, etc.)
        vr, vt, vp: (n_r, n_theta, n_phi) or None
        br, bt, bp: (n_r, n_theta, n_phi) or None
        sr: (n_r, n_theta, n_phi) or None

    Returns:
        (n_r, n_phi) numpy float32 array
    """
    _or2_all = or2.cpu().to(torch.float32).numpy()
    _or1_all = or1.cpu().to(torch.float32).numpy()
    eq = _theta_eq_idx

    if field_type == 1:    # Br
        data = br[:, eq, :]
        scale = _or2_all
    elif field_type == 2:  # Btheta
        data = bt[:, eq, :]
        scale = _or1_all
    elif field_type == 3:  # Bphi
        data = bp[:, eq, :]
        scale = _or1_all
    elif field_type == 4:  # Vr
        data = vr[:, eq, :]
        scale = _or2_all
    elif field_type == 5:  # Vtheta
        data = vt[:, eq, :]
        scale = _or1_all
    elif field_type == 6:  # Vphi
        data = vp[:, eq, :]
        scale = _or1_all
    elif field_type == 7:  # Temperature
        data = sr[:, eq, :]
        scale = None
    else:
        raise ValueError(f"Unsupported field_type {field_type} for equatorial movie")

    arr = data.detach().cpu().to(torch.float32).numpy()
    if scale is not None:
        arr = arr * scale[:, np.newaxis]
    return arr
