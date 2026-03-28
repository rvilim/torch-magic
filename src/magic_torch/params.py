"""Parameters for the dynamo_benchmark case, matching Fortran input.nml + truncation."""

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


# --- Grid parameters (from input.nml &grid, overridable via env vars) ---
l_max = int(os.environ.get("MAGIC_LMAX", "16"))
m_max = l_max
m_min = 0
minc = 1
nalias = 20

n_r_max = int(os.environ.get("MAGIC_NR", str(2 * l_max + 1)))
n_cheb_max = n_r_max
n_r_ic_max = (n_r_max + 1) // 2
n_cheb_ic_max = n_r_ic_max

# --- Derived grid parameters (from truncation.f90 initialize_truncation) ---
n_phi_tot = _prime_decomposition(2 * ((30 * l_max) // nalias))
n_phi_max = n_phi_tot // minc
n_theta_max = n_phi_tot // 2
n_m_max = m_max // minc + 1

# lm_max = (l_max+1)*(l_max+2)/2
lm_max = (l_max + 1) * (l_max + 2) // 2

n_r_tot = n_r_max  # no conducting inner core in this benchmark path

# --- Physical parameters (from input.nml &phys_param) ---
ra = 1.0e5
ek = 1.0e-3
pr = 1.0
prmag = 5.0
radratio = 0.35

# Boundary conditions
ktops = 1  # fixed entropy at top
kbots = 1  # fixed entropy at bottom
ktopv = 2  # no-slip top
kbotv = 2  # no-slip bottom

# --- Control parameters (from input.nml &control) ---
n_time_steps = 1000
dtmax = 1.0e-4
alpha = 0.6
courfac = 2.5
alffac = 1.0

# --- Start field (from input.nml &start_field) ---
init_b1 = 3
amp_b1 = 5.0
init_s1 = 404  # 0404 octal -> but Fortran reads as integer 404
amp_s1 = 0.1

# --- Logic flags for this benchmark ---
l_mag = True
l_cond_ic = False
l_heat = True
l_conv = True
l_finite_diff = False
l_axi = False
l_chemical_conv = False
l_double_curl = False
l_anel = False  # Boussinesq
l_isothermal = False
l_single_matrix = False
l_RMS = False

# --- Magnetic field memory ---
lMagMem = 1  # l_mag is True
n_r_maxMag = n_r_max
lm_maxMag = lm_max
l_maxMag = l_max

# --- Inner core ---
sigma_ratio = 0.0
nRotIC = 0
nRotMa = 0

# --- Output ---
n_log_step = 1
