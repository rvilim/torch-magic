"""Main driver for the MagIC dynamo benchmark in PyTorch.

Supports YAML-driven config, CSV energy logging, and checkpointing.
"""

import csv
import os
import time as timeit

import torch

from .init_fields import initialize_fields
from .step_time import setup_initial_state, one_step, initialize_dt, build_all_matrices
from .output import get_e_kin, get_e_mag
from . import fields
from . import dt_fields
from .params import n_time_steps, dtmax
from .time_scheme import tscheme


def _get_energies():
    """Compute all four energy components."""
    e_kin_pol, e_kin_tor = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    e_mag_pol, e_mag_tor = get_e_mag(
        fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
        fields.aj_LMloc, fields.dj_LMloc)
    return e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor


_FIELD_NAMES = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "p_LMloc", "dp_LMloc",
    "s_LMloc", "ds_LMloc",
    "b_LMloc", "db_LMloc", "ddb_LMloc",
    "aj_LMloc", "dj_LMloc", "ddj_LMloc",
]

_DT_NAMES = ["dsdt", "dwdt", "dzdt", "dpdt", "dbdt", "djdt"]


def save_checkpoint(path, step, sim_time, cfg):
    """Save simulation state to a .pt file."""
    state = {
        "step": step,
        "sim_time": sim_time,
        "cfg": cfg,
        "tscheme_dt": tscheme.dt.clone(),
    }
    # Field tensors
    for name in _FIELD_NAMES:
        state[f"fields.{name}"] = getattr(fields, name).clone()
    state["fields.omega_ic"] = fields.omega_ic
    state["fields.omega_ma"] = fields.omega_ma

    # TimeArray objects
    for name in _DT_NAMES:
        ta = getattr(dt_fields, name)
        state[f"dt.{name}.impl"] = ta.impl.clone()
        state[f"dt.{name}.expl"] = ta.expl.clone()
        state[f"dt.{name}.old"] = ta.old.clone()

    torch.save(state, path)


def load_checkpoint(path):
    """Load simulation state from a .pt file.

    Returns (step, sim_time).
    """
    state = torch.load(path, weights_only=False)

    # Restore fields
    for name in _FIELD_NAMES:
        getattr(fields, name).copy_(state[f"fields.{name}"])
    fields.omega_ic = state["fields.omega_ic"]
    fields.omega_ma = state["fields.omega_ma"]

    # Restore dt arrays
    for name in _DT_NAMES:
        ta = getattr(dt_fields, name)
        ta.impl.copy_(state[f"dt.{name}.impl"])
        ta.expl.copy_(state[f"dt.{name}.expl"])
        ta.old.copy_(state[f"dt.{name}.old"])

    # Restore time scheme dt
    tscheme.dt.copy_(state["tscheme_dt"])
    tscheme.set_weights()

    # Rebuild implicit matrices for the restored dt
    build_all_matrices()

    return state["step"], state["sim_time"]


def run(cfg=None):
    """Run the dynamo benchmark.

    Args:
        cfg: dict from YAML config, or None for legacy defaults.
    """
    if cfg is None:
        cfg = {}

    n_steps = cfg.get("n_steps", n_time_steps)
    dt = cfg.get("dt", dtmax)
    log_every = cfg.get("log_every", 100)
    checkpoint_every = cfg.get("checkpoint_every", 0)
    output_dir = cfg.get("output_dir", "./output")
    restart = cfg.get("restart", None)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize or restore
    if restart:
        start_step, sim_time = load_checkpoint(restart)
        print(f"Restored from checkpoint: step {start_step}, sim_time={sim_time:.6e}")
    else:
        start_step = 0
        sim_time = 0.0
        initialize_fields()
        setup_initial_state()
        initialize_dt(dt)

    # Open CSV log
    log_path = os.path.join(output_dir, "log.csv")
    csv_columns = [
        "step", "sim_time", "e_kin_pol", "e_kin_tor",
        "e_mag_pol", "e_mag_tor", "wall_elapsed", "ms_per_step",
    ]
    appending = restart and os.path.exists(log_path)
    log_file = open(log_path, "a" if appending else "w", newline="")
    writer = csv.writer(log_file)
    if not appending:
        writer.writerow(csv_columns)

    def write_row(step, sim_time, wall_elapsed, steps_done):
        e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor = _get_energies()
        ms = (wall_elapsed / steps_done * 1000) if steps_done > 0 else 0.0
        writer.writerow([
            step, f"{sim_time:.10e}",
            f"{e_kin_pol:.15e}", f"{e_kin_tor:.15e}",
            f"{e_mag_pol:.15e}", f"{e_mag_tor:.15e}",
            f"{wall_elapsed:.3f}", f"{ms:.3f}",
        ])
        log_file.flush()
        print(f"Step {step:6d}: e_kin={e_kin_pol + e_kin_tor:.6e} "
              f"(pol={e_kin_pol:.6e} tor={e_kin_tor:.6e}) "
              f"e_mag={e_mag_pol + e_mag_tor:.6e} "
              f"(pol={e_mag_pol:.6e} tor={e_mag_tor:.6e}) "
              f"t={sim_time:.6e} wall={wall_elapsed:.1f}s")

    # Step 0 row (only if fresh start)
    if not restart:
        write_row(0, 0.0, 0.0, 0)

    # Time integration
    t_start = timeit.time()

    for n in range(start_step + 1, n_steps + 1):
        one_step(n, dt)
        sim_time += dt
        steps_done = n - start_step
        elapsed = timeit.time() - t_start

        should_log = (n % log_every == 0) or n == start_step + 1 or n == n_steps
        if should_log:
            write_row(n, sim_time, elapsed, steps_done)

        if checkpoint_every and (n % checkpoint_every == 0 or n == n_steps):
            cp_path = os.path.join(output_dir, f"checkpoint_{n:06d}.pt")
            save_checkpoint(cp_path, n, sim_time, cfg)
            print(f"  Saved checkpoint: {cp_path}")

    log_file.close()
    elapsed = timeit.time() - t_start
    steps_done = n_steps - start_step
    if steps_done > 0:
        print(f"\nDone: {steps_done} steps in {elapsed:.1f}s "
              f"({elapsed / steps_done * 1000:.1f}ms/step)")

    return _get_energies()


if __name__ == "__main__":
    run()
