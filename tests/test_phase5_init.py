"""Phase 5: Verify initial fields against Fortran reference.

Calls initialize_fields() + setup_initial_state() then compares field arrays
against Fortran dumps.

Note: Fortran field arrays use snake LM ordering; conftest.py reorders them to
standard ordering for comparison against PyTorch.

Tolerances: ps_cond solves a 66x66 linear system via LU factorization. Different
FP operation ordering between Fortran (gfortran -O3) and PyTorch accumulates
errors at machine epsilon (~2.2e-16). For p with values ~6e4, absolute errors
reach ~3e-10 (relative ~5e-15). We use rtol=1e-13 to accommodate this.
"""

import torch
from conftest import load_ref
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state
from magic_torch import fields


_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        initialize_fields()
        setup_initial_state()
        _initialized = True


def test_s_init():
    _ensure_init()
    ref = load_ref("s_init")
    torch.testing.assert_close(fields.s_LMloc, ref, atol=1e-14, rtol=1e-14)


def test_b_init():
    _ensure_init()
    ref = load_ref("b_init")
    torch.testing.assert_close(fields.b_LMloc, ref, atol=1e-14, rtol=1e-14)


def test_db_init():
    """db is computed from b via get_ddr in setup_initial_state."""
    _ensure_init()
    ref = load_ref("db_init")
    # db involves Chebyshev differentiation of b; FP accumulation gives ~4e-12
    torch.testing.assert_close(fields.db_LMloc, ref, atol=1e-11, rtol=1e-11)


def test_dw_init():
    """dw is computed from w via get_dddr in setup_initial_state."""
    _ensure_init()
    ref = load_ref("dw_init")
    torch.testing.assert_close(fields.dw_LMloc, ref, atol=1e-14, rtol=1e-14)


def test_aj_init():
    _ensure_init()
    ref = load_ref("aj_init")
    torch.testing.assert_close(fields.aj_LMloc, ref, atol=1e-14, rtol=1e-14)


def test_w_init():
    _ensure_init()
    ref = load_ref("w_init")
    torch.testing.assert_close(fields.w_LMloc, ref, atol=0, rtol=1e-14)


def test_z_init():
    _ensure_init()
    ref = load_ref("z_init")
    torch.testing.assert_close(fields.z_LMloc, ref, atol=0, rtol=1e-14)


def test_p_init():
    _ensure_init()
    ref = load_ref("p_init")
    torch.testing.assert_close(fields.p_LMloc, ref, atol=1e-9, rtol=1e-13)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Phase 5: init field tests done")
