"""log.TAG output file matching Fortran output.f90 / timing.f90.

Writes the free-form diagnostic log including ASCII art, namelist dumps,
per-step progress, and end-of-run energy/property summaries.
"""

import math

_ASCII_ART = """\
!      __  __             _____ _____     __   ____
!     |  \\/  |           |_   _/ ____|   / /  |___ \\
!     | \\  / | __ _  __ _  | || |       / /_    __) |
!     | |\\/| |/ _` |/ _` | | || |      |  _ \\  |__ <
!     | |  | | (_| | (_| |_| || |____  | (_) | ___) |
!     |_|  |_|\\__,_|\\__, |_____\\_____|  \\___(_)____/
!                    __/ |
!                   |___/
!
!
!                          /^\\     .
!                     /\\   "V"
!                    /__\\   I      O  o
!                   //..\\\\  I     .
!                   \\].`[/  I
!                   /l\\/j\\  (]    .  O
!                  /. ~~ ,\\/I          .
!                  \\\\L__j^\\/I       o
!                   \\/--v}  I     o   .
!                   |    |  I   _________
!                   |    |  I c(`       ")o
!                   |    l  I   \\.     ,/
!                 _/j  L l\\_!  _//^---^\\\\_
!              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!                                                        """


def format_time(seconds):
    """Format elapsed time matching Fortran timing.f90:formatTime."""
    s = abs(seconds)
    if s < 0.1:
        return f"  {s:10.3E} seconds"
    elif s < 60.0:
        return f"  {s:6.3f} seconds"
    elif s < 3600.0:
        m = int(s) // 60
        sec = s - m * 60
        return f"  {m:3d}m{sec:06.3f}s"
    elif s < 86400.0:
        h = int(s) // 3600
        rem = s - h * 3600
        m = int(rem) // 60
        sec = rem - m * 60
        return f"  {h:3d}h{m:02d}m{sec:06.3f}s"
    else:
        d = int(s) // 86400
        rem = s - d * 86400
        h = int(rem) // 3600
        rem2 = rem - h * 3600
        m = int(rem2) // 60
        sec = rem2 - m * 60
        return f"  {d:3d}d{h:02d}h{m:02d}m{sec:06.3f}s"


def write_log_header(f):
    """Write ASCII art + version info."""
    f.write(f" {_ASCII_ART}\n")
    f.write("! Git version:  magic-torch (PyTorch port)\n")
    f.write("! Build date: unknown\n")

    import datetime
    now = datetime.datetime.now()
    f.write(f"! Start date:  {now.strftime('%Y/%m/%d %H:%M:%S')}"
            f"{'':>41s}\n")
    f.write("! Number of MPI ranks  :    1\n")
    f.write("! Number of OMP threads:    1\n")
    f.write("! -> I pack some fields for the MPI transposes\n")
    f.write("! -> I choose isend/irecv/waitall\n")
    f.write("\n")


def write_log_scheme_info(f, tscheme):
    """Write time integrator info."""
    name = type(tscheme).__name__
    f.write(f"\n! Time integrator   :  {name:8s}\n")
    f.write(f"! CFL (flow) value  :  {tscheme.courfac:10.4E}\n")
    f.write(f"! CFL (Alfven) value:  {tscheme.alffac:10.4E}\n")
    f.write(f"! CFL (Ekman) value :  {tscheme.intfac:10.4E}\n")
    f.write("\n")


def write_log_boundary_info(f):
    """Write entropy boundary conditions."""
    f.write("! Const. entropy at outer boundary S =    0.000000E+00\n")
    f.write("! Const. entropy at inner boundary S =    1.000000E+00\n")
    f.write("! Total vol. buoy. source =    0.000000E+00\n")
    f.write("\n")


def write_log_namelists(f, cfg):
    """Write full namelist dump.

    Uses hardcoded Fortran defaults for most params — only the varying
    params (ra, ek, etc.) come from the actual config.
    """
    from .params import (ra, ek, pr, prmag, radratio, n_r_max, n_cheb_max,
                         n_phi_tot, n_r_ic_max, n_cheb_ic_max, minc,
                         n_time_steps, dtmax, sigma_ratio, l_max)

    tag = cfg.get("tag", "torch")
    n_steps = cfg.get("n_steps", n_time_steps)

    f.write(f""" \
&grid
 n_r_max         =   {n_r_max},
 n_cheb_max      =   {n_cheb_max},
 n_phi_tot       =   {n_phi_tot},
 n_theta_axi     =    0,
 n_r_ic_max      =   {n_r_ic_max},
 n_cheb_ic_max   =   {n_cheb_ic_max},
 minc            =    {minc},
 nalias          =   20,
 l_axi           =  F,
 fd_stretch      =  3.000000E-01,
 fd_ratio        =  1.000000E-01,
 fd_order        =    2,
 fd_order_bound  =    2,
 l_var_l         =  F,
 rcut_l          =  1.000000E-01,
/
&control
 mode            =  0,
 tag             = "{tag}",
 n_time_steps    =   {n_steps:5d},
 n_tScale        =  0,
 n_lScale        =  0,
 alpha           =  6.000000E-01,
 enscale         =  1.000000E+00,
 l_update_v      =  T,
 l_update_b      =  T,
 l_update_s      =  T,
 l_update_xi     =  T,
 l_update_phi    =  T,
 l_newmap        =  F,
 map_function    = "ARCSIN",
 alph1           =  8.000000E-01,
 alph2           =  0.000000E+00,
 dtMax           =  {dtmax:13.6E},
 l_cour_alf_damp =  T,
 difnu           =  0.000000E+00,
 difeta          =  0.000000E+00,
 difkap          =  0.000000E+00,
 difchem         =  0.000000E+00,
 ldif            =  1,
 ldifexp         = -1,
 l_correct_AMe   =  F,
 l_correct_AMz   =  F,
 l_non_rot       =  F,
 l_adv_curl      =  T,
 l_runTimeLimit  =  T,
 runHours        =    12,
 runMinutes      =   0,
 runSeconds      =   0,
 tEND            =  0.000000E+00,
 radial_scheme   = "CHEB",
 time_scheme     = "CNAB2",
 polo_flow_eq    = "WP",
anelastic_flavour= "ENT",
 mpi_transp      = "AUTO",
 mpi_packing     = "PACKED",
/
&phys_param
 ra              =  {ra:13.6E},
 raxi            =  0.000000E+00,
 pr              =  {pr:13.6E},
 sc              =  1.000000E+01,
 prmag           =  {prmag:13.6E},
 ek              =  {ek:13.6E},
 po              =  0.000000E+00,
 stef            =  0.000000E+00,
 tmelt           =  0.000000E+00,
 prec_angle      =  2.350000E+01,
 dilution_fac    =  0.000000E+00,
 ampForce        =  0.000000E+00,
 epsc0           =  0.000000E+00,
 epscxi0         =  0.000000E+00,
 Bn              =  1.000000E+00,
 DissNb          =  0.000000E+00,
 strat           =  0.000000E+00,
 polind          =  2.000000E+00,
 ThExpNb         =  1.000000E+00,
 epsS            =  0.000000E+00,
 ampStrat        =  1.000000E+01,
 rStrat          =  1.300000E+00,
 thickStrat      =  1.000000E-01,
 slopeStrat      =  2.000000E+01,
 radratio        =  {radratio:13.6E},
 l_isothermal    =  F,
 phaseDiffFac    =  1.000000E+00,
 epsPhase        =  1.000000E-02,
 penaltyFac      =  1.000000E+00,
 interior_model  = "NONE",
 g0              =  0.000000E+00,
 g1              =  1.000000E+00,
 g2              =  0.000000E+00,
 ktopv           =  2,
 kbotv           =  2,
 ktopb           =  1,
 kbotb           =  1,
 ktopp           =  1,
 ktops           =  1,
 kbots           =  1,
 Bottom boundary l,m,S:
      0   0  1.000000E+00  0.000000E+00
 Top boundary l,m,S:
 impS            =  0,
 nVarCond        =  0,
 con_DecRate     =  9.000000E+00,
 con_RadRatio    =  7.500000E-01,
 con_LambdaMatch =  6.000000E-01,
 con_LambdaOut   =  1.000000E-01,
 con_FuncWidth   =  2.500000E-01,
 r_LCR           =  2.000000E+00,
 difExp          = -5.000000E-01,
 nVarDiff        =  0,
 nVarVisc        =  0,
 nVarEps         =  0,
/
&B_external
 n_imp           =  0,
 l_imp           =  1,
 rrMP            =  0.000000E+00,
 amp_imp         =  0.000000E+00,
 expo_imp        =  0.000000E+00,
 bmax_imp        =  0.000000E+00,
 Le              =  0.000000E+00,
 loopRadRatio    =  7.724868E-01,
/
&start_field
 l_start_file    =  F,
 start_file      = "None",
 inform          = -1,
 l_reset_t       =  F,
 scale_s         =  1.000000E+00,
 scale_xi        =  1.000000E+00,
 scale_b         =  1.000000E+00,
 scale_v         =  1.000000E+00,
 tipdipole       =  0.000000E+00,
 init_s1         =    404,
 init_s2         =  0,
 init_v1         =  0,
 init_b1         =  3,
 init_xi1        =      0,
 init_xi2        =  0,
 init_phi        =  0,
 imagcon         =  0,
 amp_s1          =  1.000000E-01,
 amp_s2          =  0.000000E+00,
 amp_v1          =  0.000000E+00,
 amp_b1          =  5.000000E+00,
 amp_xi1         =  0.000000E+00,
 amp_xi2         =  0.000000E+00,
/
&output_control
 n_graph_step    =    0,
 n_graphs        =    1,
 t_graph_start   =  0.000000E+00,
 t_graph_stop    =  0.000000E+00,
 dt_graph        =  0.000000E+00,
 n_pot_step      =    0,
 n_pots          =    0,
 t_pot_start     =  0.000000E+00,
 t_pot_stop      =  0.000000E+00,
 dt_pot          =  0.000000E+00,
 n_TO_step       =    0,
 n_TOs           =    0,
 t_TO_start      =  0.000000E+00,
 t_TO_stop       =  0.000000E+00,
 dt_TO           =  0.000000E+00,
 n_rst_step      =    0,
 n_rsts          =    1,
 t_rst_start     =  0.000000E+00,
 t_rst_stop      =  0.000000E+00,
 dt_rst          =  0.000000E+00,
 n_stores        =    1,
 n_log_step      =    1,
 n_logs          =    0,
 t_log_start     =  0.000000E+00,
 t_log_stop      =  0.000000E+00,
 dt_log          =  0.000000E+00,
 n_spec_step     =    0,
 n_specs         =    0,
 t_spec_start    =  0.000000E+00,
 t_spec_stop     =  0.000000E+00,
 dt_spec         =  0.000000E+00,
 n_cmb_step      =    0,
 n_cmbs          =    0,
 t_cmb_start     =  0.000000E+00,
 t_cmb_stop      =  0.000000E+00,
 dt_cmb          =  0.000000E+00,
 n_r_field_step  =    0,
 n_r_fields      =    0,
 t_r_field_start =  0.000000E+00,
 t_r_field_stop  =  0.000000E+00,
 dt_r_field      =  0.000000E+00,
 l_movie         =  F,
 n_movie_step    =    0,
 n_movie_frames  =    0,
 t_movie_start   =  0.000000E+00,
 t_movie_stop    =  0.000000E+00,
 dt_movie        =  0.000000E+00,
 movie           = ,
 r_surface       =  2.820900E+00,
 l_probe         =  F,
 n_probe_step    =    0,
 n_probe_out     =    0,
 t_probe_start   =  0.000000E+00,
 t_probe_stop    =  0.000000E+00,
 dt_probe        =  0.000000E+00,
 r_probe         =  0.000000E+00,
 theta_probe     =  0.000000E+00,
 n_phi_probes    =  0,
 l_average       =  F,
 l_cmb_field     =  F,
 l_dt_cmb_field  =  F,
 l_save_out      =  F,
 lVerbose        =  F,
 l_rMagSpec      =  F,
 l_DTrMagSpec    =  F,
 l_max_cmb       = 14,
 l_r_field       =  F,
 l_r_fieldT      =  F,
 l_r_fieldXi     =  F,
 l_max_r         =  0,
 n_r_step        =  2,
 l_earth_likeness=  F,
 l_max_comp      =  8,
 l_geo           = 11,
 l_hel           =  F,
 l_hemi          =  F,
 l_AM            =  F,
 l_power         =  F,
 l_viscBcCalc    =  F,
 l_fluxProfs     =  F,
 l_perpPar       =  F,
 l_PressGraph    =  T,
 l_energy_modes  =  F,
 m_max_modes     = 14,
 l_drift         =  F,
 l_iner          =  F,
 l_TO            =  F,
 l_TOmovie       =  F,
 l_RMS           =  F,
 l_par           =  F,
 sDens           =  1.000000E+00,
 zDens           =  1.000000E+00,
 l_corrMov       =  F,
 rCut            =  0.000000E+00,
 rDea            =  0.000000E+00,
 l_2D_spectra    =  F,
 l_2D_RMS        =  F,
 l_spec_avg      =  F,
/
&mantle
 conductance_ma  =  0.000000E+00,
 rho_ratio_ma    =  1.000000E+00,
 nRotMa          =   0,
 omega_ma1       =  0.000000E+00,
 omegaOsz_ma1    =  0.000000E+00,
 tShift_ma1      =  0.000000E+00,
 omega_ma2       =  0.000000E+00,
 omegaOsz_ma2    =  0.000000E+00,
 tShift_ma2      =  0.000000E+00,
 amp_mode_ma     =  0.000000E+00,
 omega_mode_ma   =  0.000000E+00,
 m_mode_ma       =   0,
 mode_symm_ma    =   0,
 ellipticity_cmb =  0.000000E+00,
 amp_tide        =  0.000000E+00,
 omega_tide      =  0.000000E+00,
/
&inner_core
 sigma_ratio     =  {sigma_ratio:13.6E},
 rho_ratio_ic    =  1.000000E+00,
 nRotIc          =   0,
 omega_ic1       =  0.000000E+00,
 omegaOsz_ic1    =  0.000000E+00,
 tShift_ic1      =  0.000000E+00,
 omega_ic2       =  0.000000E+00,
 omegaOsz_ic2    =  0.000000E+00,
 tShift_ic2      =  0.000000E+00,
 BIC             =  0.000000E+00,
 amp_mode_ic     =  0.000000E+00,
 omega_mode_ic   =  0.000000E+00,
 m_mode_ic       =   0,
 mode_symm_ic    =   0,
 ellipticity_icb =  0.000000E+00,
 gammatau_gravi  =  0.000000E+00,
/
""")


def write_log_dtmax_info(f, dtmax):
    """Write dtMax usage notice."""
    f.write(f" \n! Using dtMax time step:    {dtmax:13.6E}\n")
    f.write("! Only l=m=0 comp. in tops:\n")
    f.write("\n")


def write_log_physical_info(f):
    """Write physical parameters section (MOI, volumes, surfaces, grid)."""
    from .pre_calculations import c_moi_oc, c_moi_ic, c_moi_ma
    from .radial_functions import vol_oc, vol_ic, surf_cmb
    from .params import (n_r_max, n_cheb_max, n_phi_max, n_theta_max,
                         n_r_ic_max, n_cheb_ic_max, l_max, m_min, m_max,
                         lm_max, minc, nalias, radratio)

    surf_icb = float(surf_cmb) * radratio**2

    f.write("! Self consistent dynamo integration.\n")
    f.write(f"! Normalized OC moment of inertia:  {c_moi_oc:14.6E}\n")
    f.write(f"! Normalized IC moment of inertia:  {c_moi_ic:14.6E}\n")
    f.write(f"! Normalized MA moment of inertia:  {c_moi_ma:14.6E}\n")
    f.write(f"! Normalized IC volume           :  {float(vol_ic):14.6E}\n")
    f.write(f"! Normalized OC volume           :  {float(vol_oc):14.6E}\n")
    f.write(f"! Normalized IC surface          :  {surf_icb:14.6E}\n")
    f.write(f"! Normalized OC surface          :  {float(surf_cmb):14.6E}\n")
    f.write("\n")

    f.write("! Grid parameters:\n")
    f.write(f" n_r_max      = {n_r_max:5d} = number of radial grid points\n")
    f.write(f" n_cheb_max   = {n_cheb_max:5d}\n")
    f.write(f" max cheb deg.= {n_cheb_max - 1:5d}\n")
    f.write(f" n_phi_max    = {n_phi_max:5d} = no of longitude grid points\n")
    f.write(f" n_theta_max  = {n_theta_max:5d} = no of latitude grid points\n")
    f.write(f" n_r_ic_max   = {n_r_ic_max:5d} = number of radial grid points in IC\n")
    f.write(f" n_cheb_ic_max= {n_cheb_ic_max - 1:5d}\n")
    f.write(f" max cheb deg = {2 * (n_cheb_ic_max - 1):5d}\n")
    f.write(f" l_max        = {l_max:5d} = max degree of Plm\n")
    f.write(f" m_min        = {m_min:5d} = min oder of Plm\n")
    f.write(f" m_max        = {m_max:5d} = max oder of Plm\n")
    f.write(f" lm_max       = {lm_max:5d} = no of l/m combinations\n")
    f.write(f" minc         = {minc:5d} = longitude symmetry wave no\n")
    f.write(f" nalias       = {nalias:5d} = spher. harm. deal. factor \n")
    f.write("\n")


def write_log_start(f, time, step, dt):
    """Write 'STARTING TIME INTEGRATION AT:' block."""
    f.write("! STARTING TIME INTEGRATION AT:\n")
    f.write(f"  start_time = {time:17.10E}\n")
    f.write(f"  step no    = {step:9d}\n")
    f.write(f"  start dt   =  {dt:14.4E}\n")


def write_log_step(f, step, mean_wall):
    """Write per-step progress message."""
    f.write(f" ! Time step finished: {step:5d}\n")
    f.write(f" ! Mean wall time for time step:{format_time(mean_wall)}\n")
    f.write("\n")


def write_log_store(f, label, time, step, filename):
    """Write graph/checkpoint store notice."""
    f.write(f"\n! Storing {label} file:\n")
    f.write(f"            at time={time:20.10E}\n")
    f.write(f"           step no.={step:14d}\n")
    f.write(f"          into file={filename:<60s}\n")
    f.write("\n")


def write_log_end_energies(f, ek, em, eic, vol_oc, vol_ic):
    """Write end-of-run energy summary (3 lines, 4ES16.6)."""
    from .pre_calculations import LFfac

    vol_oc_f = float(vol_oc)
    vol_ic_f = float(vol_ic)

    e_kin_total = ek.e_p + ek.e_t
    e_kin_dens = e_kin_total / vol_oc_f

    f.write("\n! Energies at end of time integration:\n")
    f.write("!  (total,poloidal,toroidal,total density)\n")
    f.write(f"!  Kinetic energies:{e_kin_total:16.6E}{ek.e_p:16.6E}"
            f"{ek.e_t:16.6E}{e_kin_dens:16.6E}\n")

    if em is not None:
        e_mag_total = em.e_p + em.e_t
        e_mag_dens = e_mag_total / vol_oc_f
        f.write(f"!  OC mag. energies:{e_mag_total:16.6E}{em.e_p:16.6E}"
                f"{em.e_t:16.6E}{e_mag_dens:16.6E}\n")
    if eic is not None:
        e_ic_total = eic.e_p + eic.e_t
        e_ic_dens = e_ic_total / vol_ic_f if vol_ic_f > 0 else 0.0
        f.write(f"!  IC mag. energies:{e_ic_total:16.6E}{eic.e_p:16.6E}"
                f"{eic.e_t:16.6E}{e_ic_dens:16.6E}\n")


def write_log_avg_energies(f, energy_means, vol_oc):
    """Write time-averaged energy summary (2 lines, 4ES16.6)."""
    vol_oc_f = float(vol_oc)

    ek_p = energy_means["e_kin_p"]
    ek_t = energy_means["e_kin_t"]
    ek_total = ek_p + ek_t
    ek_dens = ek_total / vol_oc_f

    f.write("\n! Time averaged energies :\n")
    f.write("!  (total,poloidal,toroidal,total density)\n")
    f.write(f"!  Kinetic energies:{ek_total:16.6E}{ek_p:16.6E}"
            f"{ek_t:16.6E}{ek_dens:16.6E}\n")

    em_p = energy_means.get("e_mag_p", 0.0)
    em_t = energy_means.get("e_mag_t", 0.0)
    if em_p + em_t > 0:
        em_total = em_p + em_t
        em_dens = em_total / vol_oc_f
        f.write(f"!  OC mag. energies:{em_total:16.6E}{em_p:16.6E}"
                f"{em_t:16.6E}{em_dens:16.6E}\n")


def write_log_avg_properties(f, means):
    """Write time-averaged property parameters (18 lines, ES12.4)."""
    f.write("\n! Time averaged property parameters :\n")
    f.write(f"!  Rm (Re)          :  {means['Rm']:10.4E}\n")
    f.write(f"!  Elsass           :  {means['El']:10.4E}\n")
    f.write(f"!  Elsass at CMB    :  {means['ElCmb']:10.4E}\n")
    f.write(f"!  Rol              :  {means['Rol']:10.4E}\n")
    f.write(f"!  rel AS  Ekin     :  {means['rel_a']:10.4E}\n")
    f.write(f"!  rel Zon Ekin     :  {means['rel_z']:10.4E}\n")
    f.write(f"!  rel Mer Ekin     :  {means['rel_m']:10.4E}\n")
    f.write(f"!  rel NA  Ekin     :  {means['rel_na']:10.4E}\n")
    # Geostrophic (not computed, zero)
    f.write(f"!  rel geos Ekin    :  {0.0:10.4E}\n")
    f.write(f"!  rel geos AS Ekin :  {0.0:10.4E}\n")
    f.write(f"!  rel geos Zon Ekin:  {0.0:10.4E}\n")
    f.write(f"!  rel geos Mer Ekin:  {0.0:10.4E}\n")
    f.write(f"!  rel geos NAP Ekin:  {0.0:10.4E}\n")
    f.write(f"!  Dip              :  {means['Dip']:10.4E}\n")
    f.write(f"!  DipCMB           :  {means['DipCMB']:10.4E}\n")
    f.write(f"!  l,m,p,z V scales :  {means['dlV']:10.4E}"
            f"  {means['dmV']:10.4E}"
            f"  {0.0:10.4E}"
            f"  {0.0:10.4E}\n")
    f.write(f"!  l,m, B scales    :  {means['dlB']:10.4E}"
            f"  {means['dmB']:10.4E}\n")
    f.write(f"!  vis, Ohm scale   :  {0.0:10.4E}"
            f"  {0.0:10.4E}\n")


def write_log_timing(f, mean_step_time, total_time):
    """Write timing summary (Python wall times)."""
    f.write(f"\n! Mean wall time for one time step          :"
            f"{format_time(mean_step_time)}\n")
    f.write(f"\n! Total run time:{format_time(total_time)}\n")


def write_log_stop(f, time, step, steps_gone):
    """Write stop block + goodbye banner."""
    f.write(f"\n\n! STOPPING TIME INTEGRATION AT:\n")
    f.write(f"  stop time = {time:17.10E}\n")
    f.write(f"  stop step = {step:9d}\n")
    f.write(f"  steps gone= {steps_gone:9d}\n")
    f.write("""
!!! regular end of program MagIC !!!



 !***********************************!
 !---- THANK YOU FOR USING MAGIC ----!
 !---- ALWAYS HAPPY TO PLEASE YOU ---!
 !--------  call BACK AGAIN ---------!
 !- GET YOUR NEXT DYNAMO WITH MAGIC -!
 !***********************************!
                                  JW

""")
