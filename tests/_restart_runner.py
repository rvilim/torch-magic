"""Restart test runner: run continuous or from checkpoint.

Usage:
    python _restart_runner.py continuous <n_steps> <outdir>
    python _restart_runner.py restart <checkpoint_path> <n_steps> <outdir>

Dumps all field snapshots at every step to <outdir>/<field>_step<N>.npy.
"""
import os
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import (
    setup_initial_state, one_step, initialize_dt, build_all_matrices,
)
from magic_torch.main import save_checkpoint, load_checkpoint
from magic_torch import fields, dt_fields
from magic_torch.params import dtmax, l_mag, l_cond_ic, l_chemical_conv
from magic_torch.time_scheme import tscheme

mode = sys.argv[1]
FIELD_NAMES = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "p_LMloc", "dp_LMloc",
    "s_LMloc", "ds_LMloc",
]
if l_mag:
    FIELD_NAMES += [
        "b_LMloc", "db_LMloc", "ddb_LMloc",
        "aj_LMloc", "dj_LMloc", "ddj_LMloc",
    ]
if l_chemical_conv:
    FIELD_NAMES += ["xi_LMloc", "dxi_LMloc"]

IC_FIELD_NAMES = []
if l_cond_ic:
    IC_FIELD_NAMES = ["b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"]


def dump_fields(out_dir, step):
    """Save all field arrays and derivative state for one step."""
    for name in FIELD_NAMES:
        arr = getattr(fields, name).cpu().to(torch.complex128).numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    for name in IC_FIELD_NAMES:
        arr = getattr(fields, name).cpu().to(torch.complex128).numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    # Also save omega_ic
    np.save(os.path.join(out_dir, f"omega_ic_step{step}.npy"),
            np.array(fields.omega_ic))
    # Save derivative arrays (to verify they're restored correctly)
    dt_names = ["dwdt", "dzdt", "dpdt", "dsdt", "dbdt", "djdt",
                "dbdt_ic", "djdt_ic", "dxidt"]
    for dtname in dt_names:
        ta = getattr(dt_fields, dtname, None)
        if ta is None:
            continue
        for slot in ("expl", "impl", "old"):
            arr = getattr(ta, slot).cpu().to(torch.complex128).numpy()
            np.save(os.path.join(out_dir, f"{dtname}_{slot}_step{step}.npy"), arr)
    # Save scalar derivatives (domega_ic_dt, domega_ma_dt)
    for sname in ("domega_ic_dt", "domega_ma_dt"):
        ts = getattr(dt_fields, sname)
        for slot in ("expl", "impl", "old"):
            arr = getattr(ts, slot).cpu().numpy()
            np.save(os.path.join(out_dir, f"{sname}_{slot}_step{step}.npy"), arr)


if mode == "continuous":
    n_steps = int(sys.argv[2])
    out_dir = sys.argv[3]
    checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else None
    checkpoint_at = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    os.makedirs(out_dir, exist_ok=True)

    initialize_fields()
    setup_initial_state()
    dt = dtmax
    initialize_dt(dt)

    for step in range(1, n_steps + 1):
        one_step(step, dt)
        dump_fields(out_dir, step)

        if checkpoint_dir and step == checkpoint_at:
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_path = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.pt")
            save_checkpoint(cp_path, step, step * dt, {})
            print(f"Saved checkpoint at step {step}: {cp_path}")

    print(f"Continuous run: {n_steps} steps done")

elif mode == "restart":
    checkpoint_path = sys.argv[2]
    n_steps = int(sys.argv[3])
    out_dir = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    # Must initialize modules before loading checkpoint
    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)

    # Now load checkpoint (restores fields, dt arrays, tscheme.dt, rebuilds matrices)
    start_step, sim_time = load_checkpoint(checkpoint_path)
    dt = tscheme.dt[0].item()
    print(f"Loaded checkpoint: step={start_step}, sim_time={sim_time:.6e}, dt={dt:.6e}")

    for step in range(start_step + 1, start_step + n_steps + 1):
        one_step(step, dt)
        dump_fields(out_dir, step)

    print(f"Restart run: {n_steps} steps from step {start_step}")

else:
    raise ValueError(f"Unknown mode: {mode}")
