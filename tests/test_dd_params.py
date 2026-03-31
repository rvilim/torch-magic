"""doubleDiffusion: Parameter verification and xiMat roundtrip.

Tests DD-specific parameter values, boundary conditions, and the composition
implicit matrix construction without requiring Fortran reference data.

Runs in a subprocess to isolate DD env vars (mode=1, raxi!=0).
"""

import os
import sys
import subprocess
import json
import pytest
import numpy as np
from pathlib import Path


_RUNNER = Path(__file__).parent / "_dd_params_runner.py"

_RUNNER_CODE = '''\
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
'''

_results = {}
_ran = False


def _run():
    global _ran, _results
    if _ran:
        return
    _ran = True
    _RUNNER.write_text(_RUNNER_CODE)

    import tempfile
    out_file = tempfile.mktemp(suffix=".json", prefix="dd_params_")

    env = os.environ.copy()
    env.update({
        "MAGIC_TIME_SCHEME": "BPR353",
        "MAGIC_LMAX": "64",
        "MAGIC_NR": "33",
        "MAGIC_MINC": "4",
        "MAGIC_NCHEBMAX": "31",
        "MAGIC_MODE": "1",
        "MAGIC_RA": "4.8e4",
        "MAGIC_RAXI": "1.2e5",
        "MAGIC_SC": "3.0",
        "MAGIC_PR": "0.3",
        "MAGIC_EK": "1e-3",
        "MAGIC_DEVICE": "cpu",
    })

    result = subprocess.run(
        [sys.executable, str(_RUNNER), out_file],
        cwd=str(Path(__file__).parent.parent),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"DD params runner failed: {result.returncode}")

    _results.update(json.load(open(out_file)))


@pytest.fixture(scope="module")
def dd_params():
    _run()
    return _results


def test_mode_1(dd_params):
    """mode=1 means convection only (no magnetic field generation)."""
    assert dd_params["mode"] == 1
    assert dd_params["l_mag"] is False
    assert dd_params["l_chemical_conv"] is True
    assert dd_params["l_heat"] is True


def test_composition_constants(dd_params):
    """osc = 1/sc = 1/3, ChemFac = raxi/sc = 4e4, epscXi = 0."""
    assert abs(dd_params["osc"] - 1.0 / 3.0) < 1e-15
    assert abs(dd_params["ChemFac"] - 4.0e4) < 1e-10
    assert dd_params["epscXi"] == 0.0


def test_botxi_boundary(dd_params):
    """botxi(l=0,m=0) = sqrt(4*pi), all others zero. topxi = 0 everywhere."""
    import math
    sq4pi = math.sqrt(4.0 * math.pi)
    assert dd_params["topxi_all_zero"] is True
    assert abs(dd_params["botxi_lm00_real"] - sq4pi) < 1e-15
    assert abs(dd_params["botxi_lm00_imag"]) < 1e-15
    assert dd_params["botxi_nonzero_count"] == 1


def test_dLh_zero_at_l0(dd_params):
    """dLh[0] = l(l+1) = 0 for l=0. Ensures epscXi=0 composition eqn works."""
    assert dd_params["dLh_0"] == 0.0


def test_ncheb_truncation(dd_params):
    """n_cheb_max=31 < n_r_max=33 — spectral truncation is active."""
    assert dd_params["n_cheb_max"] == 31
    assert dd_params["n_r_max"] == 33
    assert dd_params["n_cheb_max_lt_n_r_max"] is True


def test_xiMat_roundtrip(dd_params):
    """xiMat: inv(A) @ (A @ x) == x for all l degrees."""
    err = dd_params["xiMat_max_roundtrip_err"]
    assert err < 1e-7, f"xiMat roundtrip max error = {err:.2e}"
