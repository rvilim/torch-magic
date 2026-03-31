"""Parameters for MagIC benchmarks, matching Fortran input.nml + truncation.

All physics/grid parameters are configurable via environment variables,
set by run.py from YAML config before this module is imported.
"""

import os


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


def _env_int(key, default):
    return int(os.environ.get(key, str(default)))


def _env_float(key, default):
    return float(os.environ.get(key, str(default)))


def _env_bool(key, default):
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_str(key, default):
    return os.environ.get(key, default)


# --- Grid parameters (from input.nml &grid, overridable via env vars) ---
l_max = _env_int("MAGIC_LMAX", 16)
m_max = l_max
m_min = 0
minc = _env_int("MAGIC_MINC", 1)
nalias = 20

n_r_max = _env_int("MAGIC_NR", 2 * l_max + 1)
n_cheb_max = _env_int("MAGIC_NCHEBMAX", n_r_max)
n_r_ic_max = (n_r_max + 1) // 2
n_cheb_ic_max = _env_int("MAGIC_NCHEBICMAX", n_r_ic_max)

# --- Derived grid parameters (from truncation.f90 initialize_truncation) ---
n_phi_tot = _prime_decomposition(2 * ((30 * l_max) // nalias))
n_phi_max = n_phi_tot // minc
n_theta_max = n_phi_tot // 2
n_m_max = m_max // minc + 1

# lm_max: general formula for arbitrary minc
# For minc=1: reduces to (l_max+1)*(l_max+2)//2
lm_max = sum(l_max - m + 1 for m in range(0, m_max + 1, minc))

# --- Physical parameters (from input.nml &phys_param) ---
ra = _env_float("MAGIC_RA", 1.0e5)
ek = _env_float("MAGIC_EK", 1.0e-3)
pr = _env_float("MAGIC_PR", 1.0)
prmag = _env_float("MAGIC_PRMAG", 5.0)
radratio = _env_float("MAGIC_RADRATIO", 0.35)
raxi = _env_float("MAGIC_RAXI", 0.0)
sc = _env_float("MAGIC_SC", 10.0)  # Fortran default: Namelists.f90:1376

# Mode: 0=full MHD, 1=convection only (no magnetic field)
mode = _env_int("MAGIC_MODE", 0)

# Anelastic parameters
strat = _env_float("MAGIC_STRAT", 0.0)
polind = _env_float("MAGIC_POLIND", 2.0)
g0 = _env_float("MAGIC_G0", 0.0)
g1 = _env_float("MAGIC_G1", 0.0)
g2 = _env_float("MAGIC_G2", 0.0)

# Boundary conditions
ktops = 1  # fixed entropy at top
kbots = 1  # fixed entropy at bottom
ktopxi = 1  # fixed composition at top
kbotxi = 1  # fixed composition at bottom
ktopv = _env_int("MAGIC_KTOPV", 2)  # 1=stress-free, 2=no-slip
kbotv = _env_int("MAGIC_KBOTV", 2)  # 1=stress-free, 2=no-slip
kbotb = _env_int("MAGIC_KBOTB", 1)  # 1=insulating, 3=conducting IC

# --- Control parameters (from input.nml &control) ---
n_time_steps = 1000
dtmax = _env_float("MAGIC_DTMAX", 1.0e-4)
alpha = _env_float("MAGIC_ALPHA", 0.6)
courfac = _env_float("MAGIC_COURFAC", 1e3)   # Fortran default 1e3; scheme overrides when >= 1e3
alffac = _env_float("MAGIC_ALFFAC", 1e3)    # Fortran default 1e3; scheme overrides when >= 1e3
intfac = _env_float("MAGIC_INTFAC", 1e3)    # Fortran default 1e3; scheme overrides when >= 1e3

# --- Time scheme ---
time_scheme = _env_str("MAGIC_TIME_SCHEME", "CNAB2")

# --- Start field (from input.nml &start_field) ---
init_b1 = 3
amp_b1 = 5.0
init_s1 = _env_int("MAGIC_INIT_S1", 404)
amp_s1 = _env_float("MAGIC_AMP_S1", 0.1)
l_start_file = _env_str("MAGIC_START_FILE", "") != ""
start_file = _env_str("MAGIC_START_FILE", "")

# --- Logic flags for this benchmark ---
l_mag = (mode != 1)
l_heat = True
l_conv = True
l_finite_diff = False
l_axi = False
l_chemical_conv = (raxi != 0.0)
l_double_curl = False
l_anel = strat > 0.0
l_adv_curl = not l_anel  # anelastic uses non-curl advection
l_correct_AMz = _env_str("MAGIC_L_CORRECT_AMZ", "false").lower() in ("true", "1", "yes")
l_correct_AMe = _env_str("MAGIC_L_CORRECT_AME", "false").lower() in ("true", "1", "yes")
l_isothermal = False
l_single_matrix = False
l_RMS = False
l_mag_LF = l_mag       # Lorentz force active when magnetic field is active
l_mag_kin = False       # kinematic dynamo: False for all benchmarks
l_cour_alf_damp = _env_bool("MAGIC_L_COUR_ALF_DAMP", True)  # Fortran default: .true.

# --- Magnetic field memory ---
lMagMem = 1 if l_mag else 0
n_r_maxMag = n_r_max if l_mag else 0
lm_maxMag = lm_max if l_mag else 0
l_maxMag = l_max if l_mag else 0

# --- Inner core ---
sigma_ratio = _env_float("MAGIC_SIGMA_RATIO", 0.0)
nRotIC = _env_int("MAGIC_NROTIC", 0)
nRotMa = 0

# --- Derived IC flags ---
l_cond_ic = sigma_ratio > 0.0
l_rot_ic = nRotIC > 0

# --- Total radial points (OC + IC when conducting) ---
n_r_tot = n_r_max + n_r_ic_max if l_cond_ic else n_r_max

# --- Output ---
n_log_step = 1
ktopb = 1  # insulating magnetic top BC (vacuum at CMB)

# --- Chebyshev mapping parameters (num_param.f90) ---
alph1 = _env_float("MAGIC_ALPH1", 0.8)
alph2 = _env_float("MAGIC_ALPH2", 0.0)

# --- Stefan number (unused in benchmark) ---
stef = _env_float("MAGIC_STEF", 0.0)

# --- Omega parameters (init_fields.f90 namelist, 12 doubles) ---
omega_ic1 = _env_float("MAGIC_OMEGA_IC1", 0.0)
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
