"""Parameters for the dynamo_benchmark case, matching Fortran input.nml + truncation."""

# --- Grid parameters (from input.nml &grid) ---
n_r_max = 33
n_cheb_max = 33
l_max = 16
m_max = 16
m_min = 0
minc = 1
nalias = 20
n_r_ic_max = 17
n_cheb_ic_max = 17

# --- Derived grid parameters (from truncation.f90 initialize_truncation) ---
# l_max given (!=0), so:
#   n_theta_max = (30*l_max)/nalias = (30*16)/20 = 24
#   n_phi_tot = 2*n_theta_max = 48  (already good prime decomposition: 2^4 * 3)
#   n_phi_max = n_phi_tot/minc = 48
n_theta_max = 24
n_phi_tot = 48
n_phi_max = 48
n_m_max = m_max // minc + 1  # 17

# lm_max = sum_{m=0}^{16} (16-m+1) = 17*18/2 = 153
lm_max = 153

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
