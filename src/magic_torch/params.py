"""Parameters for MagIC benchmarks, matching Fortran input.nml + truncation.

Configuration is read from the config dict (set via ``config.configure()``)
with fallback to environment variables for backward compatibility.
"""

import os

from .config import _config


def _prime_decomposition(n):
    """Find smallest m >= n with only factors 2, 3, 5 (matches truncation.f90)."""
    dist_min = 1e9
    best = n
    for i in range(13):
        for j in range(7):
            for k in range(7):
                val = 2**i * 3**j * 5**k
                dist = val - n
                if 0 <= dist < dist_min:
                    dist_min = dist
                    best = val
    return best


def _cfg(key, env_key, default):
    """Read config value: config dict first, then env var fallback."""
    if key in _config:
        return _config[key]
    val = os.environ.get(env_key)
    if val is not None:
        return val
    return default


def _cfg_int(key, env_key, default):
    return int(_cfg(key, env_key, default))


def _cfg_float(key, env_key, default):
    return float(_cfg(key, env_key, default))


def _cfg_bool(key, env_key, default):
    val = _cfg(key, env_key, default)
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def _cfg_str(key, env_key, default):
    return str(_cfg(key, env_key, default))


# --- Grid parameters ---
l_max = _cfg_int("l_max", "MAGIC_LMAX", 16)
m_max = l_max
m_min = 0
minc = _cfg_int("minc", "MAGIC_MINC", 1)
nalias = 20

n_r_max = _cfg_int("n_r_max", "MAGIC_NR", 2 * l_max + 1)
n_cheb_max = _cfg_int("n_cheb_max", "MAGIC_NCHEBMAX", n_r_max)
n_r_ic_max = (n_r_max + 1) // 2
n_cheb_ic_max = _cfg_int("n_cheb_ic_max", "MAGIC_NCHEBICMAX", n_r_ic_max)

# --- Derived grid parameters (from truncation.f90 initialize_truncation) ---
n_phi_tot = _prime_decomposition(2 * ((30 * l_max) // nalias))
n_phi_max = n_phi_tot // minc
n_theta_max = n_phi_tot // 2
n_m_max = m_max // minc + 1

# lm_max: general formula for arbitrary minc
# For minc=1: reduces to (l_max+1)*(l_max+2)//2
lm_max = sum(l_max - m + 1 for m in range(0, m_max + 1, minc))

# --- Physical parameters ---
ra = _cfg_float("ra", "MAGIC_RA", 1.0e5)
ek = _cfg_float("ek", "MAGIC_EK", 1.0e-3)
pr = _cfg_float("pr", "MAGIC_PR", 1.0)
prmag = _cfg_float("prmag", "MAGIC_PRMAG", 5.0)
radratio = _cfg_float("radratio", "MAGIC_RADRATIO", 0.35)
raxi = _cfg_float("raxi", "MAGIC_RAXI", 0.0)
sc = _cfg_float("sc", "MAGIC_SC", 10.0)

# Mode: 0=full MHD, 1=convection only, 7=Couette flow (no heat, no mag)
mode = _cfg_int("mode", "MAGIC_MODE", 0)

# Anelastic parameters
strat = _cfg_float("strat", "MAGIC_STRAT", 0.0)
polind = _cfg_float("polind", "MAGIC_POLIND", 2.0)
g0 = _cfg_float("g0", "MAGIC_G0", 0.0)
g1 = _cfg_float("g1", "MAGIC_G1", 0.0)
g2 = _cfg_float("g2", "MAGIC_G2", 0.0)

# Boundary conditions
ktops = _cfg_int("ktops", "MAGIC_KTOPS", 1)    # 1=fixed entropy, 2=fixed flux
kbots = _cfg_int("kbots", "MAGIC_KBOTS", 1)    # 1=fixed entropy, 2=fixed flux
ktopxi = _cfg_int("ktopxi", "MAGIC_KTOPXI", 1) # 1=fixed composition, 2=fixed flux
kbotxi = _cfg_int("kbotxi", "MAGIC_KBOTXI", 1) # 1=fixed composition, 2=fixed flux
ktopv = _cfg_int("ktopv", "MAGIC_KTOPV", 2)  # 1=stress-free, 2=no-slip
kbotv = _cfg_int("kbotv", "MAGIC_KBOTV", 2)  # 1=stress-free, 2=no-slip
kbotb = _cfg_int("kbotb", "MAGIC_KBOTB", 1)  # 1=insulating, 3=conducting IC

# --- Control parameters ---
n_time_steps = 1000
dtmax = _cfg_float("dtmax", "MAGIC_DTMAX", 1.0e-4)
alpha = _cfg_float("alpha", "MAGIC_ALPHA", 0.6)
courfac = _cfg_float("courfac", "MAGIC_COURFAC", 1e3)
alffac = _cfg_float("alffac", "MAGIC_ALFFAC", 1e3)
intfac = _cfg_float("intfac", "MAGIC_INTFAC", 1e3)
radial_chunk_size = _cfg_int("radial_chunk_size", "MAGIC_RADIAL_CHUNK", 0)
l_profile = _cfg_bool("profile", "MAGIC_PROFILE", False)

# --- Time scheme ---
time_scheme = _cfg_str("time_scheme", "MAGIC_TIME_SCHEME", "CNAB2")

# --- Start field ---
init_b1 = 3
amp_b1 = 5.0
init_v1 = _cfg_int("init_v1", "MAGIC_INIT_V1", 0)
init_s1 = _cfg_int("init_s1", "MAGIC_INIT_S1", 404)
amp_s1 = _cfg_float("amp_s1", "MAGIC_AMP_S1", 0.1)
_start_file_val = _cfg_str("start_file", "MAGIC_START_FILE", "")
l_start_file = _start_file_val != ""
start_file = _start_file_val

# --- Logic flags for this benchmark ---
l_mag = (mode not in (1, 7))
l_heat = (mode not in (7,))
l_conv = True
radial_scheme = _cfg_str("radial_scheme", "MAGIC_RADIAL_SCHEME", "chebyshev").upper()
l_finite_diff = (radial_scheme == "FD")
fd_order = _cfg_int("fd_order", "MAGIC_FD_ORDER", 2)
fd_order_bound = _cfg_int("fd_order_bound", "MAGIC_FD_ORDER_BOUND", 2)
fd_stretch = _cfg_float("fd_stretch", "MAGIC_FD_STRETCH", 0.3)
fd_ratio = _cfg_float("fd_ratio", "MAGIC_FD_RATIO", 0.1)
l_axi = False
l_chemical_conv = (raxi != 0.0)
l_double_curl = l_finite_diff  # FD forces double-curl (Fortran Namelists.f90:300)
l_anel = strat > 0.0
l_adv_curl = not l_anel  # anelastic uses non-curl advection
l_correct_AMz = _cfg_bool("l_correct_AMz", "MAGIC_L_CORRECT_AMZ", False)
l_correct_AMe = _cfg_bool("l_correct_AMe", "MAGIC_L_CORRECT_AME", False)
l_isothermal = False
l_single_matrix = False
l_RMS = False
l_mag_LF = l_mag       # Lorentz force active when magnetic field is active
l_mag_kin = False       # kinematic dynamo: False for all benchmarks
l_cour_alf_damp = _cfg_bool("l_cour_alf_damp", "MAGIC_L_COUR_ALF_DAMP", True)

# --- Magnetic field memory ---
lMagMem = 1 if l_mag else 0
n_r_maxMag = n_r_max if l_mag else 0
lm_maxMag = lm_max if l_mag else 0
l_maxMag = l_max if l_mag else 0

# --- Inner core ---
sigma_ratio = _cfg_float("sigma_ratio", "MAGIC_SIGMA_RATIO", 0.0)
nRotIC = _cfg_int("nRotIC", "MAGIC_NROTIC", 0)
nRotMa = 0

# --- Derived IC flags ---
l_cond_ic = sigma_ratio > 0.0
l_rot_ic = nRotIC != 0
l_rot_ma = nRotMa != 0
l_SRIC = nRotIC == -1

# --- Total radial points (OC + IC when conducting) ---
n_r_tot = n_r_max + n_r_ic_max if l_cond_ic else n_r_max

# --- Output ---
n_log_step = _cfg_int("n_log_step", "MAGIC_N_LOG_STEP", 1)
l_power = _cfg_bool("l_power", "MAGIC_L_POWER", False)
l_hel = _cfg_bool("l_hel", "MAGIC_L_HEL", False)
l_hemi = _cfg_bool("l_hemi", "MAGIC_L_HEMI", False)
ktopb = 1  # insulating magnetic top BC (vacuum at CMB)

# --- Chebyshev mapping parameters (num_param.f90) ---
alph1 = _cfg_float("alph1", "MAGIC_ALPH1", 0.8)
alph2 = _cfg_float("alph2", "MAGIC_ALPH2", 0.0)

# --- Stefan number (unused in benchmark) ---
stef = _cfg_float("stef", "MAGIC_STEF", 0.0)

# --- Omega parameters ---
omega_ic1 = _cfg_float("omega_ic1", "MAGIC_OMEGA_IC1", 0.0)
omegaOsz_ic1 = 0.0
tOmega_ic1 = 0.0
omega_ic2 = 0.0
omegaOsz_ic2 = 0.0
tOmega_ic2 = 0.0
omega_ma1 = 0.0
omegaOsz_ma1 = 0.0
tOmega_ma1 = 0.0
omega_ma2 = 0.0
omegaOsz_ma2 = 0.0
tOmega_ma2 = 0.0
