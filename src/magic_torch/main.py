"""Main driver for the MagIC dynamo benchmark in PyTorch.

Runs the Christensen et al. 2001 benchmark case 1:
Ra=1e5, Ek=1e-3, Pr=1, Pm=5, l_max=16, n_r_max=33.
"""

import time as timeit

from .init_fields import initialize_fields
from .step_time import setup_initial_state, one_step, initialize_dt
from .output import get_e_kin, get_e_mag
from . import fields
from .params import n_time_steps, dtmax


def run(n_steps=None, dt=None, print_every=100):
    """Run the dynamo benchmark.

    Args:
        n_steps: number of time steps (default: n_time_steps from params)
        dt: time step size
        print_every: print diagnostics every N steps
    """
    if n_steps is None:
        n_steps = n_time_steps
    if dt is None:
        dt = dtmax

    # Initialize fields
    initialize_fields()

    # Compute initial derivatives and time arrays
    setup_initial_state()

    # Initialize dt array (Fortran sets all slots to dtMax from scratch)
    initialize_dt(dt)

    # Print initial diagnostics
    e_kin_pol, e_kin_tor = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    e_mag_pol, e_mag_tor = get_e_mag(
        fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
        fields.aj_LMloc, fields.dj_LMloc)
    print(f"Step {0:5d}: e_kin = {e_kin_pol + e_kin_tor:.6e} "
          f"(pol={e_kin_pol:.6e}, tor={e_kin_tor:.6e}), "
          f"e_mag = {e_mag_pol + e_mag_tor:.6e} "
          f"(pol={e_mag_pol:.6e}, tor={e_mag_tor:.6e})")

    # Time integration
    t_start = timeit.time()
    sim_time = 0.0

    for n in range(1, n_steps + 1):
        one_step(n, dt)
        sim_time += dt

        if n % print_every == 0 or n == 1 or n == n_steps:
            e_kin_pol, e_kin_tor = get_e_kin(
                fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
                fields.z_LMloc, fields.dz_LMloc)
            e_mag_pol, e_mag_tor = get_e_mag(
                fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
                fields.aj_LMloc, fields.dj_LMloc)
            elapsed = timeit.time() - t_start
            print(f"Step {n:5d}: e_kin = {e_kin_pol + e_kin_tor:.6e} "
                  f"(pol={e_kin_pol:.6e}, tor={e_kin_tor:.6e}), "
                  f"e_mag = {e_mag_pol + e_mag_tor:.6e} "
                  f"(pol={e_mag_pol:.6e}, tor={e_mag_tor:.6e}), "
                  f"t={sim_time:.6e}, wall={elapsed:.1f}s")

    elapsed = timeit.time() - t_start
    print(f"\nDone: {n_steps} steps in {elapsed:.1f}s "
          f"({elapsed/n_steps:.3f}s/step)")

    return e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor


if __name__ == "__main__":
    run(n_steps=1, print_every=1)
