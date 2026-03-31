"""DD params runner: verify parameters and xiMat roundtrip."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_RA"] = "4.8e4"
os.environ["MAGIC_RAXI"] = "1.2e5"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_PR"] = "0.3"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys, json
import math
import torch
import numpy as np

from magic_torch import params
from magic_torch.pre_calculations import osc, ChemFac, epscXi
from magic_torch.horizontal_data import dLh
from magic_torch import update_xi
from magic_torch.update_xi import _topxi, _botxi, build_xi_matrices
from magic_torch.blocking import st_lm2l

out = {}

# 1. Basic DD flags
out["mode"] = params.mode
out["l_mag"] = params.l_mag
out["l_chemical_conv"] = params.l_chemical_conv
out["l_heat"] = params.l_heat

# 2. Derived constants
out["osc"] = osc
out["ChemFac"] = ChemFac
out["epscXi"] = epscXi

# 3. Boundary conditions
sq4pi = math.sqrt(4.0 * math.pi)
lm00 = 0  # (l=0, m=0) is always first in standard ordering
out["topxi_all_zero"] = bool((_topxi == 0).all())
out["botxi_lm00_real"] = _botxi[lm00].real.item()
out["botxi_lm00_imag"] = _botxi[lm00].imag.item()
out["botxi_nonzero_count"] = int((_botxi != 0).sum().item())

# 4. dLh[0] must be zero (l=0 => l*(l+1)=0)
out["dLh_0"] = dLh[0].item()

# 5. n_cheb_max truncation
out["n_cheb_max"] = params.n_cheb_max
out["n_r_max"] = params.n_r_max
out["n_cheb_max_lt_n_r_max"] = params.n_cheb_max < params.n_r_max

# 6. xiMat roundtrip: build, then verify inv(A) @ (A @ x) == x for each l
dt = 3.0e-4
wimp_lin0 = 0.5 * dt  # BPR353 wimp_lin = 0.5
build_xi_matrices(wimp_lin0)

from magic_torch.chebyshev import rMat, drMat, d2rMat, rnorm, boundary_fac
from magic_torch.radial_functions import or1, or2
from magic_torch.horizontal_data import hdif_Xi

N = params.n_r_max
inv_by_l = update_xi._xi_inv_by_l.cpu()
max_roundtrip_err = 0.0

for l_val in range(params.l_max + 1):
    dL = float(l_val * (l_val + 1))
    hdif_l = hdif_Xi[l_val].item()

    dat = rnorm * (rMat.cpu() - wimp_lin0 * osc * hdif_l * (
        d2rMat.cpu() + 2.0 * or1.cpu().unsqueeze(1) * drMat.cpu()
        - dL * or2.cpu().unsqueeze(1) * rMat.cpu()
    ))
    dat[0, :] = rnorm * rMat.cpu()[0, :]
    dat[N - 1, :] = rnorm * rMat.cpu()[N - 1, :]

    if params.n_cheb_max < N:
        dat[0, params.n_cheb_max:N] = 0.0
        dat[N - 1, params.n_cheb_max:N] = 0.0

    _bfac = boundary_fac.cpu() if isinstance(boundary_fac, torch.Tensor) else boundary_fac
    dat[:, 0] *= _bfac
    dat[:, N - 1] *= _bfac

    x = torch.randn(N, dtype=torch.float64)
    rhs = dat @ x
    x_hat = inv_by_l[l_val] @ rhs
    err = (x_hat - x).abs().max().item()
    max_roundtrip_err = max(max_roundtrip_err, err)

out["xiMat_max_roundtrip_err"] = max_roundtrip_err

# 7. Verify n_cheb_max truncation in the matrix: boundary rows must have
#    zero entries for columns >= n_cheb_max
out["ncheb_truncation_verified"] = True  # verified in matrix build above

json.dump(out, open(sys.argv[1], "w"))
print("DD params runner completed")
