"""Anelastic matrix roundtrip runner: set env vars, build matrices, verify A @ inv ≈ I.

Runs in a subprocess to isolate anelastic env vars from other tests.
Reconstructs the original matrix A for each l (before LU/preconditioning),
then checks that A @ inv_by_l[l] ≈ I for representative l values.

Outputs JSON with roundtrip errors to stdout.
"""
import os
os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_STRAT"] = "5.0"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "0.0"
os.environ["MAGIC_G2"] = "1.0"
os.environ["MAGIC_RA"] = "1.48638035e5"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_KTOPV"] = "1"
os.environ["MAGIC_KBOTV"] = "1"
os.environ["MAGIC_ALPHA"] = "0.6"

import json
import torch

from magic_torch.precision import DTYPE
from magic_torch.params import n_r_max, l_max, n_cheb_max
from magic_torch.constants import two, third, three, four
from magic_torch.chebyshev import rMat, drMat, d2rMat, d3rMat, rnorm, boundary_fac
from magic_torch.radial_functions import (
    or1, or2, or3, beta, dbeta, dLtemp0, dLkappa, kappa, visc, dLvisc,
)
from magic_torch.horizontal_data import dLh, hdif_S, hdif_V
from magic_torch.pre_calculations import opr
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, build_all_matrices
from magic_torch.time_scheme import tscheme

# Initialize everything
initialize_fields()
setup_initial_state()
dt = 1e-4
initialize_dt(dt)
tscheme.set_weights()
build_all_matrices()

N = n_r_max
wimp_lin0 = tscheme.wimp_lin[0].item()

# Common CPU tensors
cpu = torch.device("cpu")
_rMat = rMat.to(cpu)
_drMat = drMat.to(cpu)
_d2rMat = d2rMat.to(cpu)
_d3rMat = d3rMat.to(cpu)
_or1 = or1.to(cpu)
_or2 = or2.to(cpu)
_or3 = or3.to(cpu)
_beta = beta.to(cpu)
_dbeta = dbeta.to(cpu)
_visc = visc.to(cpu)
_dLvisc = dLvisc.to(cpu)
_kappa = kappa.to(cpu)
_dLtemp0 = dLtemp0.to(cpu)
_dLkappa = dLkappa.to(cpu)
_rnorm = rnorm.to(cpu) if isinstance(rnorm, torch.Tensor) else rnorm
_bfac = boundary_fac.to(cpu) if isinstance(boundary_fac, torch.Tensor) else boundary_fac
or1_col = _or1.unsqueeze(1)
or2_col = _or2.unsqueeze(1)
or3_col = _or3.unsqueeze(1)
beta_col = _beta.unsqueeze(1)
dbeta_col = _dbeta.unsqueeze(1)
visc_col = _visc.unsqueeze(1)
dLvisc_col = _dLvisc.unsqueeze(1)
d1_coeff_s = (_beta + _dLtemp0 + two * _or1 + _dLkappa).unsqueeze(1)
kappa_col = _kappa.unsqueeze(1)

eye = torch.eye(N, dtype=DTYPE, device=cpu)
eye2N = torch.eye(2 * N, dtype=DTYPE, device=cpu)

test_ls = [1, 5, 10, l_max]

results = {}


def _reconstruct_sMat(l):
    """Reconstruct the original sMat for degree l (before LU/preconditioning)."""
    dL = float(l * (l + 1))
    hdif_l = hdif_S[l].item()
    dat = _rnorm * (_rMat - wimp_lin0 * opr * hdif_l * kappa_col * (
        _d2rMat + d1_coeff_s * _drMat - dL * or2_col * _rMat
    ))
    dat[0, :] = _rnorm * _rMat[0, :]
    dat[N - 1, :] = _rnorm * _rMat[N - 1, :]
    if n_cheb_max < N:
        dat[0, n_cheb_max:N] = 0.0
        dat[N - 1, n_cheb_max:N] = 0.0
    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N - 1] = dat[:, N - 1] * _bfac
    return dat


def _reconstruct_zMat(l):
    """Reconstruct the original zMat for degree l (before LU/preconditioning)."""
    dL = float(l * (l + 1))
    hdif_l = hdif_V[l].item()
    from magic_torch.params import ktopv, kbotv

    dat = _rnorm * dL * or2_col * (
        _rMat - wimp_lin0 * hdif_l * visc_col * (
            _d2rMat
            + (dLvisc_col - beta_col) * _drMat
            - (dLvisc_col * beta_col + two * dLvisc_col * or1_col
               + dL * or2_col + dbeta_col + two * beta_col * or1_col) * _rMat
        )
    )
    if ktopv == 1:  # stress-free
        dat[0, :] = _rnorm * (_drMat[0, :] - (two * _or1[0] + _beta[0]) * _rMat[0, :])
    else:
        dat[0, :] = _rnorm * _rMat[0, :]
    if kbotv == 1:  # stress-free
        dat[N - 1, :] = _rnorm * (_drMat[N - 1, :] - (two * _or1[N - 1] + _beta[N - 1]) * _rMat[N - 1, :])
    else:
        dat[N - 1, :] = _rnorm * _rMat[N - 1, :]
    if n_cheb_max < N:
        dat[0, n_cheb_max:N] = 0.0
        dat[N - 1, n_cheb_max:N] = 0.0
    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N - 1] = dat[:, N - 1] * _bfac
    return dat


def _reconstruct_wpMat(l):
    """Reconstruct the original wpMat for degree l (before LU/preconditioning)."""
    from magic_torch.params import ktopv, kbotv
    dL = float(l * (l + 1))
    hdif_l = hdif_V[l].item()

    dat = torch.zeros(2 * N, 2 * N, dtype=DTYPE, device=cpu)

    # w-row, w-col
    dat[1:N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
        _rMat[1:N-1] - wimp_lin0 * hdif_l * visc_col[1:N-1] * (
            _d2rMat[1:N-1]
            + (two * dLvisc_col[1:N-1] - third * beta_col[1:N-1]) * _drMat[1:N-1]
            - (dL * or2_col[1:N-1]
               + four * third * (dLvisc_col[1:N-1] * beta_col[1:N-1]
                                 + (three * dLvisc_col[1:N-1] + beta_col[1:N-1])
                                 * or1_col[1:N-1]
                                 + dbeta_col[1:N-1]))
            * _rMat[1:N-1]
        )
    )

    # w-row, p-col
    dat[1:N-1, N:] = _rnorm * wimp_lin0 * (
        _drMat[1:N-1] - beta_col[1:N-1] * _rMat[1:N-1]
    )

    # p-row, w-col
    dat[N+1:2*N-1, :N] = _rnorm * dL * or2_col[1:N-1] * (
        -_drMat[1:N-1] - wimp_lin0 * hdif_l * visc_col[1:N-1] * (
            -_d3rMat[1:N-1]
            + (beta_col[1:N-1] - dLvisc_col[1:N-1]) * _d2rMat[1:N-1]
            + (dL * or2_col[1:N-1] + dbeta_col[1:N-1]
               + dLvisc_col[1:N-1] * beta_col[1:N-1]
               + two * (dLvisc_col[1:N-1] + beta_col[1:N-1]) * or1_col[1:N-1])
            * _drMat[1:N-1]
            - dL * or2_col[1:N-1]
            * (two * or1_col[1:N-1] + dLvisc_col[1:N-1]
               + two * third * beta_col[1:N-1])
            * _rMat[1:N-1]
        )
    )

    # p-row, p-col
    dat[N+1:2*N-1, N:] = -_rnorm * wimp_lin0 * dL * or2_col[1:N-1] * _rMat[1:N-1]

    # BCs
    dat[0, :N] = _rnorm * _rMat[0, :]
    dat[N-1, :N] = _rnorm * _rMat[N-1, :]

    if ktopv == 1:  # stress-free
        dat[N, :N] = _rnorm * (
            _d2rMat[0, :] - (two * _or1[0] + _beta[0]) * _drMat[0, :]
        )
    else:
        dat[N, :N] = _rnorm * _drMat[0, :]

    if kbotv == 1:  # stress-free
        dat[2*N-1, :N] = _rnorm * (
            _d2rMat[N-1, :] - (two * _or1[N-1] + _beta[N-1]) * _drMat[N-1, :]
        )
    else:
        dat[2*N-1, :N] = _rnorm * _drMat[N-1, :]

    if n_cheb_max < N:
        for row in [0, N-1, N, 2*N-1]:
            dat[row, n_cheb_max:N] = 0.0
            dat[row, N+n_cheb_max:2*N] = 0.0

    for blk_start in [0, N]:
        dat[:, blk_start] = dat[:, blk_start] * _bfac
        dat[:, blk_start + N - 1] = dat[:, blk_start + N - 1] * _bfac

    return dat


def _reconstruct_p0Mat():
    """Reconstruct the original p0Mat (before LU)."""
    from magic_torch.radial_functions import ViscHeatFac, ThExpNb
    _l_p0_integ_bc = (ViscHeatFac * ThExpNb != 0.0)

    dat = torch.zeros(N, N, dtype=DTYPE, device=cpu)
    dat[1:, :] = _rnorm * (_drMat[1:, :] - _beta[1:].unsqueeze(1) * _rMat[1:, :])

    if _l_p0_integ_bc:
        from magic_torch.init_fields import _cheb_integ_kernel_f64
        from magic_torch.radial_functions import ogrun, alpha0
        from magic_torch.chebyshev import r
        from magic_torch.cosine_transform import costf
        work_phys = ThExpNb * ViscHeatFac * ogrun * alpha0 * r * r
        work = costf(work_phys.to(cpu))
        work = work * _rnorm
        work[0] = _bfac * work[0]
        work[N - 1] = _bfac * work[N - 1]
        K = _cheb_integ_kernel_f64(N)
        dat[0, :] = (K @ work).to(DTYPE)
    else:
        dat[0, :] = _rnorm * _rMat[0, :]

    if n_cheb_max < N:
        dat[0, n_cheb_max:] = 0.0

    dat[:, 0] = dat[:, 0] * _bfac
    dat[:, N - 1] = dat[:, N - 1] * _bfac
    return dat


# === Test sMat roundtrip ===
from magic_torch.update_s import _s_inv_by_l
s_errors = {}
for l in test_ls:
    A = _reconstruct_sMat(l)
    inv_l = _s_inv_by_l[l].to(cpu)
    prod = A @ inv_l
    err = (prod - eye).abs().max().item()
    s_errors[str(l)] = err
results["sMat"] = s_errors

# === Test zMat roundtrip ===
from magic_torch.update_z import _z_inv_by_l
z_errors = {}
for l in test_ls:
    A = _reconstruct_zMat(l)
    inv_l = _z_inv_by_l[l].to(cpu)
    prod = A @ inv_l
    err = (prod - eye).abs().max().item()
    z_errors[str(l)] = err
results["zMat"] = z_errors

# === Test wpMat roundtrip ===
from magic_torch.update_wp import _wp_inv_by_l
wp_errors = {}
for l in test_ls:
    A = _reconstruct_wpMat(l)
    inv_l = _wp_inv_by_l[l].to(cpu)
    prod = A @ inv_l
    err = (prod - eye2N).abs().max().item()
    wp_errors[str(l)] = err
results["wpMat"] = wp_errors

# === Test p0Mat roundtrip ===
from magic_torch.update_wp import _p0Mat_inv
A_p0 = _reconstruct_p0Mat()
inv_p0 = _p0Mat_inv.to(cpu)
prod_p0 = A_p0 @ inv_p0
p0_err = (prod_p0 - eye).abs().max().item()
results["p0Mat"] = {"0": p0_err}

# Print JSON to stdout
print(json.dumps(results))
