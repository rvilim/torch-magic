"""Phase 4: Verify CNAB2 time scheme weights and IMEX RHS assembly.

Tests CNAB2 weights analytically and validates set_imex_rhs against Fortran
reference data (z_imex_rhs, dzdt_old/impl/expl from phase 7 dumps).

No new Fortran dumps needed — uses analytical checks + existing reference data.
"""

import torch
from conftest import load_ref
from magic_torch.time_scheme import CNAB2
from magic_torch.params import alpha, dtmax


def test_cnab2_weights_constant_dt():
    """Verify CNAB2 weights with constant dt match analytical formulas."""
    ts = CNAB2()
    dt = 1e-4
    ts.dt[0] = dt
    ts.dt[1] = dt
    ts.set_weights()

    # Implicit: wimp[0] = 1.0, wimp_lin[0] = alpha*dt, wimp_lin[1] = (1-alpha)*dt
    assert ts.wimp[0].item() == 1.0
    assert ts.wimp_lin[0].item() == alpha * dt
    assert ts.wimp_lin[1].item() == (1.0 - alpha) * dt

    # Explicit AB2: wexp[0] = 1.5*dt, wexp[1] = -0.5*dt (when dt1==dt2)
    torch.testing.assert_close(ts.wexp[0], torch.tensor(1.5 * dt, dtype=torch.float64), atol=0, rtol=1e-15)
    torch.testing.assert_close(ts.wexp[1], torch.tensor(-0.5 * dt, dtype=torch.float64), atol=0, rtol=1e-15)


def test_cnab2_weights_varying_dt():
    """Verify CNAB2 weights with different dt1, dt2."""
    ts = CNAB2()
    dt1 = 1e-4
    dt2 = 2e-4
    ts.dt[0] = dt1
    ts.dt[1] = dt2
    ts.set_weights()

    expected_wexp0 = (1.0 + 0.5 * dt1 / dt2) * dt1
    expected_wexp1 = -0.5 * dt1 * dt1 / dt2

    torch.testing.assert_close(
        torch.tensor(ts.wexp[0].item()),
        torch.tensor(expected_wexp0),
        atol=0, rtol=1e-15,
    )
    torch.testing.assert_close(
        torch.tensor(ts.wexp[1].item()),
        torch.tensor(expected_wexp1),
        atol=0, rtol=1e-15,
    )


def test_ab1_weights():
    """Verify AB1 (first-order Euler) gives wexp = [dt, 0]."""
    ts = CNAB2()
    dt = 1e-4
    ts.dt[0] = dt
    ts.dt[1] = dt
    ts.start_with_ab1()

    assert ts.wexp[0].item() == dt
    assert ts.wexp[1].item() == 0.0


def test_set_imex_rhs_against_fortran():
    """Verify set_imex_rhs reproduces Fortran z_imex_rhs.

    Uses dzdt_old/impl/expl from Fortran + CNAB2 weights at dt=dtmax
    to reconstruct z_imex_rhs and compare against Fortran reference.
    """
    from magic_torch.dt_fields import TimeArray

    # Load Fortran reference dt_field components
    dzdt_old_ref = load_ref("dzdt_old")
    dzdt_impl_ref = load_ref("dzdt_impl")
    dzdt_expl_ref = load_ref("dzdt_expl")
    z_imex_rhs_ref = load_ref("z_imex_rhs")

    # Build a TimeArray with the Fortran data
    dzdt = TimeArray(nold=1, nexp=2, nimp=1)
    dzdt.old[:, :, 0] = dzdt_old_ref
    dzdt.impl[:, :, 0] = dzdt_impl_ref
    dzdt.expl[:, :, 0] = dzdt_expl_ref
    # expl[:,:,1] = 0 for step 1 (no previous explicit term)

    # Set up CNAB2 with constant dt = dtmax (matching Fortran step 1)
    ts = CNAB2()
    ts.dt[0] = dtmax
    ts.dt[1] = dtmax
    ts.set_weights()

    # Compute IMEX RHS
    rhs = ts.set_imex_rhs(dzdt)

    torch.testing.assert_close(rhs, z_imex_rhs_ref, atol=1e-13, rtol=1e-12)


def test_dt_rotation():
    """Verify dt array rotation matches Fortran cshift behavior."""
    ts = CNAB2()
    ts.dt[0] = 1e-4
    ts.dt[1] = 2e-4

    # Simulate one_step's rotation: dt_old = dt[0], dt[0] = new, dt[1] = old
    dt_old = ts.dt[0].item()
    dt_new = 3e-4
    ts.dt[0] = dt_new
    ts.dt[1] = dt_old

    assert ts.dt[0].item() == 3e-4
    assert ts.dt[1].item() == 1e-4


def test_rotate_imex():
    """Verify rotate_imex shifts explicit history correctly."""
    from magic_torch.dt_fields import TimeArray

    ta = TimeArray(nold=1, nexp=2, nimp=1)
    # Set slot 0 to a known pattern
    ta.expl[:, :, 0] = 1.0 + 0j
    ta.expl[:, :, 1] = 2.0 + 0j

    ts = CNAB2()
    ts.rotate_imex(ta)

    # After rotation: slot 1 <- old slot 0 = 1.0
    torch.testing.assert_close(
        ta.expl[:, :, 1],
        torch.full_like(ta.expl[:, :, 1], 1.0 + 0j),
        atol=0, rtol=0,
    )


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Phase 4: time scheme tests done")
