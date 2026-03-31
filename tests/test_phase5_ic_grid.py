"""Phase 5: IC radial grid — compare against Fortran reference.

Tests the inner core Chebyshev grid arrays:
- r_ic, O_r_ic (radial grid and inverse)
- cheb_ic, dcheb_ic, d2cheb_ic (even Chebyshev polynomials and derivatives)
- dr_top_ic (real-space derivative weights at ICB)
- cheb_norm_ic, dr_fac_ic (normalization constants)

Runs in a subprocess with MAGIC_SIGMA_RATIO=1.0 to activate l_cond_ic.
Compares against Fortran reference in samples/dynamo_benchmark_condIC/fortran_ref/.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

CONDIC_REF = Path(__file__).parent.parent.parent / "samples" / "dynamo_benchmark_condIC" / "fortran_ref"

_RUNNER = Path(__file__).parent / "_ic_grid_runner.py"


def _ensure_runner():
    if _RUNNER.exists():
        return
    _RUNNER.write_text('''\
"""IC grid runner: set env vars, import, dump IC grid arrays."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"

import sys
import numpy as np

from magic_torch.radial_functions import (
    r_ic, O_r_ic, cheb_ic, dcheb_ic, d2cheb_ic,
    dr_top_ic, cheb_norm_ic, dr_fac_ic,
)

out_dir = sys.argv[1]
np.save(os.path.join(out_dir, "r_ic.npy"), r_ic.cpu().numpy())
np.save(os.path.join(out_dir, "O_r_ic.npy"), O_r_ic.cpu().numpy())
np.save(os.path.join(out_dir, "cheb_ic.npy"), cheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dcheb_ic.npy"), dcheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "d2cheb_ic.npy"), d2cheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dr_top_ic.npy"), dr_top_ic.cpu().numpy())
np.save(os.path.join(out_dir, "cheb_norm_ic.npy"), np.array(cheb_norm_ic))
np.save(os.path.join(out_dir, "dr_fac_ic.npy"), np.array(dr_fac_ic))

print("IC grid runner completed")
''')


_results = {}
_ran = False


def _run_ic_grid():
    global _ran
    if _ran:
        return
    _ran = True
    _ensure_runner()

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="ic_grid_")

    env = os.environ.copy()
    env["MAGIC_SIGMA_RATIO"] = "1.0"
    env["MAGIC_KBOTB"] = "3"
    env["MAGIC_DEVICE"] = "cpu"

    result = subprocess.run(
        [sys.executable, str(_RUNNER), out_dir],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"IC grid runner failed: {result.returncode}")

    for name in ["r_ic", "O_r_ic", "cheb_ic", "dcheb_ic", "d2cheb_ic",
                  "dr_top_ic", "cheb_norm_ic", "dr_fac_ic"]:
        _results[name] = np.load(os.path.join(out_dir, f"{name}.npy"))


# Tolerances per array. The even Chebyshev recursion (depth 17) accumulates
# FP differences between gfortran -O3 FMA and Python separate multiply-add:
# - dcheb_ic: mismatches at values near zero (~1e-14); max abs diff ~3e-13
# - d2cheb_ic: 2nd derivative amplifies; max rel diff ~2.4e-14, max abs ~2.4e-11
# All other arrays match to 1e-14 or better.
_TOLERANCES = {
    "r_ic":         (1e-14, 1e-14),
    "O_r_ic":       (1e-14, 1e-14),
    "cheb_ic":      (1e-14, 1e-14),
    "dcheb_ic":     (5e-13, 1e-12),   # values near zero, FMA accumulation
    "d2cheb_ic":    (5e-11, 5e-14),   # 2nd deriv, depth-17 recursion FMA diff
    "dr_top_ic":    (1e-14, 1e-14),
    "cheb_norm_ic": (1e-14, 1e-14),
    "dr_fac_ic":    (1e-14, 1e-14),
}


@pytest.mark.parametrize("name", [
    "r_ic", "O_r_ic", "cheb_ic", "dcheb_ic", "d2cheb_ic",
    "dr_top_ic", "cheb_norm_ic", "dr_fac_ic",
])
def test_ic_grid(name):
    """Compare IC grid array against Fortran reference."""
    _run_ic_grid()
    ref = np.load(CONDIC_REF / f"{name}.npy")
    actual = _results[name]
    atol, rtol = _TOLERANCES[name]
    np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol,
                               err_msg=f"IC grid {name} mismatch")
