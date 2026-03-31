"""Read/write Fortran MagIC checkpoint files (version 2–5, stream access).

Supports both MULTISTEP (CNAB2) and DIRK (BPR353) checkpoints.
Fields are returned in standard (st_map) LM ordering — no reordering needed.
Endianness is auto-detected from the version field.
"""

import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CheckpointData:
    """All data from a MagIC checkpoint file."""

    # Header
    version: int = 0
    time: float = 0.0
    family: str = ""
    nexp: int = 0
    nimp: int = 0
    nold: int = 0
    dt: np.ndarray = field(default_factory=lambda: np.array([]))
    n_time_step: int = 0

    # Physical params
    ra: float = 0.0
    pr: float = 0.0
    raxi: float = 0.0
    sc: float = 0.0
    prmag: float = 0.0
    ek: float = 0.0
    stef: float = 0.0
    radratio: float = 0.0
    sigma_ratio: float = 0.0

    # Grid params
    n_r_max: int = 0
    n_theta_max: int = 0
    n_phi_tot: int = 0
    minc: int = 0
    nalias: int = 0
    n_r_ic_max: int = 0
    l_max: int = 0
    m_min: int = 0
    m_max: int = 0
    n_cheb_max: int = 0  # from rscheme n_max

    # Radial grid
    r: np.ndarray = field(default_factory=lambda: np.array([]))

    # Rotation params
    omega_ic1: float = 0.0
    omega_ma1: float = 0.0

    # Logicals
    l_heat: bool = False
    l_chemical_conv: bool = False
    l_phase_field: bool = False
    l_mag: bool = False
    l_press_store: bool = False
    l_cond_ic: bool = False

    # OC fields — shape (lm_max, n_r_max), complex128
    w: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    s: Optional[np.ndarray] = None
    xi: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    aj: Optional[np.ndarray] = None

    # IC fields — shape (lm_max, n_r_ic_max), complex128
    b_ic: Optional[np.ndarray] = None
    aj_ic: Optional[np.ndarray] = None

    # Detected endianness
    endian: str = "<"


def _compute_lm_max(l_max: int, m_max: int, minc: int) -> int:
    """Compute lm_max for given truncation parameters."""
    count = 0
    for m in range(0, m_max + 1, minc):
        count += l_max - m + 1
    return count


def read_checkpoint(path: str) -> CheckpointData:
    """Read a Fortran MagIC checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    CheckpointData
        All fields and metadata from the checkpoint.
    """
    ck = CheckpointData()

    with open(path, "rb") as f:
        # --- Detect endianness from version field ---
        raw = f.read(4)
        v_le = struct.unpack("<i", raw)[0]
        v_be = struct.unpack(">i", raw)[0]
        if 2 <= v_le <= 5:
            endian = "<"
            ck.version = v_le
        elif 2 <= v_be <= 5:
            endian = ">"
            ck.version = v_be
        else:
            raise ValueError(
                f"Cannot detect endianness: LE={v_le}, BE={v_be} "
                f"(expected version 2–5)")
        ck.endian = endian

        # --- Time ---
        ck.time = struct.unpack(f"{endian}d", f.read(8))[0]

        # --- Time scheme info ---
        ck.family = f.read(10).decode("ascii").strip()
        ck.nexp, ck.nimp, ck.nold = struct.unpack(f"{endian}iii", f.read(12))

        # dt array: DIRK stores 1 value, MULTISTEP stores nexp values
        dt_size = 1 if ck.family == "DIRK" else ck.nexp
        ck.dt = np.frombuffer(f.read(dt_size * 8), dtype=f"{endian}f8").copy()

        # --- n_time_step ---
        ck.n_time_step = struct.unpack(f"{endian}i", f.read(4))[0]

        # --- Physical params (9 doubles for v3+; 8 for v2 without stef) ---
        if ck.version >= 3:
            vals = struct.unpack(f"{endian}9d", f.read(72))
            (ck.ra, ck.pr, ck.raxi, ck.sc, ck.prmag,
             ck.ek, ck.stef, ck.radratio, ck.sigma_ratio) = vals
        else:
            vals = struct.unpack(f"{endian}8d", f.read(64))
            (ck.ra, ck.pr, ck.raxi, ck.sc, ck.prmag,
             ck.ek, ck.radratio, ck.sigma_ratio) = vals
            ck.stef = 0.0

        # --- Grid params ---
        (ck.n_r_max, ck.n_theta_max, ck.n_phi_tot,
         ck.minc, ck.nalias, ck.n_r_ic_max) = struct.unpack(
            f"{endian}6i", f.read(24))

        # v4+: l_max, m_min, m_max
        if ck.version >= 4:
            ck.l_max, ck.m_min, ck.m_max = struct.unpack(
                f"{endian}3i", f.read(12))
        else:
            # Derive from grid params (pre-v4 didn't store these)
            ck.l_max = ck.nalias * ck.n_phi_tot // 60
            ck.m_min = 0
            ck.m_max = ck.l_max

        # --- Radial scheme ---
        rscheme_version = f.read(72).decode("ascii").strip()
        n_max, _order_boundary = struct.unpack(f"{endian}2i", f.read(8))
        _alph1, _alph2 = struct.unpack(f"{endian}2d", f.read(16))
        ck.n_cheb_max = n_max  # rscheme_oc%n_max for cheb

        # --- Radial grid (v2+) ---
        ck.r = np.frombuffer(
            f.read(ck.n_r_max * 8), dtype=f"{endian}f8").copy()

        # --- Scalar time derivatives (MULTISTEP only) ---
        if ck.family == "MULTISTEP":
            # domega_ic_dt: expl(2..nexp), impl(2..nimp), old(2..nold)
            _skip_scalars = (max(0, ck.nexp - 1) + max(0, ck.nimp - 1)
                             + max(0, ck.nold - 1))
            # domega_ma_dt: same
            _skip_scalars *= 2

            if ck.version < 5:
                # lorentz_torque_ic_dt + lorentz_torque_ma_dt
                _skip_scalars += 2 * (max(0, ck.nexp - 1) + max(0, ck.nimp - 1)
                                      + max(0, ck.nold - 1))

            f.read(_skip_scalars * 8)
        # DIRK: no scalar time derivatives written

        # --- Omega parameters (always, 12 doubles) ---
        omegas = struct.unpack(f"{endian}12d", f.read(96))
        ck.omega_ic1 = omegas[0]
        ck.omega_ma1 = omegas[6]

        # --- Logical flags (4 bytes each for gfortran default logical) ---
        if ck.version <= 1:
            raw_l = struct.unpack(f"{endian}4i", f.read(16))
            ck.l_heat = bool(raw_l[0])
            ck.l_chemical_conv = bool(raw_l[1])
            ck.l_mag = bool(raw_l[2])
            ck.l_cond_ic = bool(raw_l[3])
            ck.l_press_store = True
            ck.l_phase_field = False
        elif ck.version == 2:
            raw_l = struct.unpack(f"{endian}5i", f.read(20))
            ck.l_heat = bool(raw_l[0])
            ck.l_chemical_conv = bool(raw_l[1])
            ck.l_mag = bool(raw_l[2])
            ck.l_press_store = bool(raw_l[3])
            ck.l_cond_ic = bool(raw_l[4])
            ck.l_phase_field = False
        else:  # v3+
            raw_l = struct.unpack(f"{endian}6i", f.read(24))
            ck.l_heat = bool(raw_l[0])
            ck.l_chemical_conv = bool(raw_l[1])
            ck.l_phase_field = bool(raw_l[2])
            ck.l_mag = bool(raw_l[3])
            ck.l_press_store = bool(raw_l[4])
            ck.l_cond_ic = bool(raw_l[5])

        # --- Compute lm_max ---
        lm_max = _compute_lm_max(ck.l_max, ck.m_max, ck.minc)

        # --- Read OC fields ---
        def _read_field(n_r: int) -> np.ndarray:
            """Read one complex(cp) field array, shape (lm_max, n_r)."""
            nbytes = lm_max * n_r * 16  # complex128
            arr = np.frombuffer(f.read(nbytes), dtype=f"{endian}c16").copy()
            # Fortran column-major: lm varies fastest → reshape as (lm, nr)
            return arr.reshape((lm_max, n_r), order="F")

        def _skip_multistep_arrays(n_r: int):
            """Skip expl/impl/old arrays for MULTISTEP fields."""
            if ck.family == "MULTISTEP":
                n_skip = (max(0, ck.nexp - 1) + max(0, ck.nimp - 1)
                          + max(0, ck.nold - 1))
                f.read(n_skip * lm_max * n_r * 16)

        def _read_oc_field() -> np.ndarray:
            arr = _read_field(ck.n_r_max)
            _skip_multistep_arrays(ck.n_r_max)
            return arr

        # w and z always present
        ck.w = _read_oc_field()
        ck.z = _read_oc_field()

        if ck.l_press_store:
            ck.p = _read_oc_field()

        if ck.l_heat:
            ck.s = _read_oc_field()

        if ck.l_chemical_conv:
            ck.xi = _read_oc_field()

        if ck.l_phase_field:
            ck.phi = _read_oc_field()

        if ck.l_mag:
            ck.b = _read_oc_field()
            ck.aj = _read_oc_field()

        # --- Read IC fields ---
        if ck.l_mag and ck.l_cond_ic:
            def _read_ic_field() -> np.ndarray:
                arr = _read_field(ck.n_r_ic_max)
                _skip_multistep_arrays(ck.n_r_ic_max)
                return arr

            ck.b_ic = _read_ic_field()
            ck.aj_ic = _read_ic_field()

        # Verify we consumed exactly the right amount
        remaining = f.read(1)
        if remaining:
            pos = f.tell() - 1
            raise ValueError(
                f"Checkpoint has {pos} bytes consumed but file continues "
                f"(expected EOF at {pos})")

    return ck


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_field(f, tensor, endian: str):
    """Write a complex128 tensor as Fortran column-major binary."""
    import torch
    arr = tensor.detach().cpu().to(torch.complex128).numpy()
    arr = arr.astype(f"{endian}c16")
    f.write(arr.flatten(order="F").tobytes())


def _write_one_field(f, tensor, dfdt, family: str, nexp: int, nimp: int,
                     nold: int, endian: str):
    """Write one field + its multistep derivative arrays.

    For MULTISTEP: field, then expl[:,:,i] for i in 1..nexp-1,
    impl[:,:,i] for i in 1..nimp-1, old[:,:,i] for i in 1..nold-1.
    For DIRK: field only.
    """
    _write_field(f, tensor, endian)
    if family == "MULTISTEP":
        for i in range(1, nexp):
            _write_field(f, dfdt.expl[:, :, i], endian)
        for i in range(1, nimp):
            _write_field(f, dfdt.impl[:, :, i], endian)
        for i in range(1, nold):
            _write_field(f, dfdt.old[:, :, i], endian)


def write_checkpoint_fortran(path: str, sim_time: float, n_time_step: int,
                             endian: str = ">") -> None:
    """Write a Fortran-compatible MagIC v5 checkpoint file.

    Reads all state from module globals (fields.*, dt_fields.*, params.*,
    time_scheme.tscheme, chebyshev.r, pre_calculations.tScale).

    Parameters
    ----------
    path : str
        Output file path.
    sim_time : float
        Current simulation time (dimensionless).
    n_time_step : int
        Current time step number (Fortran convention: n_steps + 1).
    endian : str
        Byte order: '>' for big-endian, '<' for little-endian.
    """
    from . import fields, dt_fields, params
    from .time_scheme import tscheme
    from .chebyshev import r as cheb_r
    from .pre_calculations import tScale

    with open(path, "wb") as f:
        # --- Version ---
        f.write(struct.pack(f"{endian}i", 5))

        # --- Time (scaled) ---
        f.write(struct.pack(f"{endian}d", sim_time * tScale))

        # --- Time scheme info ---
        family_bytes = tscheme.family.ljust(10).encode("ascii")
        f.write(family_bytes)
        f.write(struct.pack(f"{endian}iii", tscheme.nexp, tscheme.nimp,
                            tscheme.nold))

        # --- dt (scaled) ---
        if tscheme.family == "DIRK":
            f.write(struct.pack(f"{endian}d",
                                tscheme.dt[0].item() * tScale))
        else:
            for i in range(tscheme.nexp):
                f.write(struct.pack(f"{endian}d",
                                    tscheme.dt[i].item() * tScale))

        # --- n_time_step ---
        f.write(struct.pack(f"{endian}i", n_time_step))

        # --- Physical params (9 doubles) ---
        f.write(struct.pack(f"{endian}9d",
                            params.ra, params.pr, params.raxi, params.sc,
                            params.prmag, params.ek, params.stef,
                            params.radratio, params.sigma_ratio))

        # --- Grid params (6 ints) ---
        f.write(struct.pack(f"{endian}6i",
                            params.n_r_max, params.n_theta_max,
                            params.n_phi_tot, params.minc, params.nalias,
                            params.n_r_ic_max))

        # --- l_max, m_min, m_max ---
        f.write(struct.pack(f"{endian}3i",
                            params.l_max, params.m_min, params.m_max))

        # --- Radial scheme ---
        rscheme_str = "cheb".ljust(72).encode("ascii")
        f.write(rscheme_str)
        f.write(struct.pack(f"{endian}2i", params.n_cheb_max, 0))
        f.write(struct.pack(f"{endian}2d", params.alph1, params.alph2))

        # --- Radial grid ---
        import torch
        r_np = cheb_r.detach().cpu().to(torch.float64).numpy()
        r_np = r_np.astype(f"{endian}f8")
        f.write(r_np.tobytes())

        # --- Scalar time derivatives (MULTISTEP only) ---
        if tscheme.family == "MULTISTEP":
            # domega_ic_dt: expl[1], (no impl[1] since nimp=1), (no old[1])
            for i in range(1, tscheme.nexp):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ic_dt.expl[i].item()))
            for i in range(1, tscheme.nimp):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ic_dt.impl[i].item()))
            for i in range(1, tscheme.nold):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ic_dt.old[i].item()))
            # domega_ma_dt
            for i in range(1, tscheme.nexp):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ma_dt.expl[i].item()))
            for i in range(1, tscheme.nimp):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ma_dt.impl[i].item()))
            for i in range(1, tscheme.nold):
                f.write(struct.pack(
                    f"{endian}d",
                    dt_fields.domega_ma_dt.old[i].item()))

        # --- 12 omega doubles ---
        f.write(struct.pack(f"{endian}12d",
                            params.omega_ic1, params.omegaOsz_ic1,
                            params.tOmega_ic1,
                            params.omega_ic2, params.omegaOsz_ic2,
                            params.tOmega_ic2,
                            params.omega_ma1, params.omegaOsz_ma1,
                            params.tOmega_ma1,
                            params.omega_ma2, params.omegaOsz_ma2,
                            params.tOmega_ma2))

        # --- 6 logicals (4-byte ints, gfortran convention) ---
        l_press_store = True  # always True (not l_double_curl)
        for flag in [params.l_heat, params.l_chemical_conv, False,
                     params.l_mag, l_press_store, params.l_cond_ic]:
            f.write(struct.pack(f"{endian}i", 1 if flag else 0))

        # --- Fields ---
        family = tscheme.family
        ne, ni, no_ = tscheme.nexp, tscheme.nimp, tscheme.nold

        # w + dwdt
        _write_one_field(f, fields.w_LMloc, dt_fields.dwdt,
                         family, ne, ni, no_, endian)
        # z + dzdt
        _write_one_field(f, fields.z_LMloc, dt_fields.dzdt,
                         family, ne, ni, no_, endian)
        # p + dpdt (l_press_store=True always)
        _write_one_field(f, fields.p_LMloc, dt_fields.dpdt,
                         family, ne, ni, no_, endian)
        # s + dsdt (if l_heat)
        if params.l_heat:
            _write_one_field(f, fields.s_LMloc, dt_fields.dsdt,
                             family, ne, ni, no_, endian)
        # xi + dxidt (if l_chemical_conv)
        if params.l_chemical_conv:
            _write_one_field(f, fields.xi_LMloc, dt_fields.dxidt,
                             family, ne, ni, no_, endian)
        # b + dbdt (if l_mag)
        if params.l_mag:
            _write_one_field(f, fields.b_LMloc, dt_fields.dbdt,
                             family, ne, ni, no_, endian)
            _write_one_field(f, fields.aj_LMloc, dt_fields.djdt,
                             family, ne, ni, no_, endian)
        # IC fields (if l_mag and l_cond_ic)
        if params.l_mag and params.l_cond_ic:
            _write_one_field(f, fields.b_ic, dt_fields.dbdt_ic,
                             family, ne, ni, no_, endian)
            _write_one_field(f, fields.aj_ic, dt_fields.djdt_ic,
                             family, ne, ni, no_, endian)
