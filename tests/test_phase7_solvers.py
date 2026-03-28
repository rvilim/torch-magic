"""Phase 7: Verify implicit solver inputs against Fortran reference.

Runs the full pipeline through radial_loop (init → setup → explicit assembly),
then verifies the dt_field components (old, impl, expl) and IMEX RHS assembly
for z, w, p equations against Fortran dumps.

The reference data for dt_field components was dumped from the Fortran at the
point after set_imex_rhs but before the implicit solve, during step 1 (initial
state + radial_loop). This is separate from the nR17 grid-space dumps which
are from step 2.

Tolerances for explicit terms (dwdt_expl, dzdt_expl) reflect accumulated FP
differences through the SHT → nonlinear → spectral → get_td chain. The final
step output (phase 8) matches to machine precision, confirming correctness.
"""

import torch
from conftest import load_ref
from magic_torch.params import dtmax


_initialized = False


def _ensure_init():
    global _initialized
    if _initialized:
        return

    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import (
        setup_initial_state, initialize_dt, build_all_matrices, radial_loop,
    )
    from magic_torch.time_scheme import tscheme

    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)
    tscheme.dt[0] = dtmax
    tscheme.dt[1] = dtmax
    tscheme.set_weights()
    build_all_matrices()
    radial_loop()
    _initialized = True


# === dt_field component tests ===

def test_dzdt_old():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dzdt_old")
    torch.testing.assert_close(dt_fields.dzdt.old[:, :, 0], ref, atol=1e-13, rtol=1e-12)


def test_dzdt_impl():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dzdt_impl")
    torch.testing.assert_close(dt_fields.dzdt.impl[:, :, 0], ref, atol=1e-13, rtol=1e-12)


def test_dzdt_expl():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dzdt_expl")
    # FP accumulation through SHT → get_nl → spat_to_qst → get_dzdt chain
    torch.testing.assert_close(dt_fields.dzdt.expl[:, :, 0], ref, atol=1e-9, rtol=1e-9)


def test_dwdt_old():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dwdt_old")
    torch.testing.assert_close(dt_fields.dwdt.old[:, :, 0], ref, atol=1e-13, rtol=1e-12)


def test_dwdt_impl():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dwdt_impl")
    # FP accumulation in implicit diffusion terms (ddw involves 2nd derivative)
    torch.testing.assert_close(dt_fields.dwdt.impl[:, :, 0], ref, atol=1e-7, rtol=1e-7)


def test_dwdt_expl():
    _ensure_init()
    from magic_torch import dt_fields
    ref = load_ref("dwdt_expl")
    # FP accumulation through full nonlinear + Coriolis + buoyancy chain
    torch.testing.assert_close(dt_fields.dwdt.expl[:, :, 0], ref, atol=1e-5, rtol=1e-5)


# === IMEX RHS tests ===

def test_z_imex_rhs():
    _ensure_init()
    from magic_torch import dt_fields
    from magic_torch.time_scheme import tscheme
    ref = load_ref("z_imex_rhs")
    rhs = tscheme.set_imex_rhs(dt_fields.dzdt)
    torch.testing.assert_close(rhs, ref, atol=1e-13, rtol=1e-12)


def test_w_imex_rhs():
    _ensure_init()
    from magic_torch import dt_fields
    from magic_torch.time_scheme import tscheme
    ref = load_ref("w_imex_rhs")
    rhs = tscheme.set_imex_rhs(dt_fields.dwdt)
    # IMEX RHS: wimp*old + wimp_lin*impl + wexp[0]*expl[0] + wexp[1]*expl[1]
    # expl error (~1e-6) multiplied by wexp[0]=1.5*dt(~1.5e-5) gives ~1.5e-11
    torch.testing.assert_close(rhs, ref, atol=1e-9, rtol=1e-9)


def test_p_imex_rhs():
    _ensure_init()
    from magic_torch import dt_fields
    from magic_torch.time_scheme import tscheme
    ref = load_ref("p_imex_rhs")
    rhs = tscheme.set_imex_rhs(dt_fields.dpdt)
    torch.testing.assert_close(rhs, ref, atol=1e-7, rtol=1e-12)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Phase 7: solver tests done")
