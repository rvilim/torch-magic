#!/usr/bin/env python3
"""Launch MagIC simulation on a Modal cloud GPU.

Usage:
    cd magic-torch
    MAGIC_MODAL_GPU=H100 uv run modal run run_modal.py --config configs/dynamo_benchmark_modal.yaml
"""

import os
from datetime import datetime

import modal
import yaml

# GPU type is set at module level so the decorator can read it.
# Modal passes CLI args via local_entrypoint, but the decorator is evaluated
# at import time — so we read from an env var that the user can also set.
GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")

app = modal.App("magic-torch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pyyaml", "tensorboard")
    .add_local_dir("src", remote_path="/root/src")
)

vol = modal.Volume.from_name("magic-output", create_if_missing=True)
input_vol = modal.Volume.from_name("magic-input", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,
    volumes={"/output": vol, "/input": input_vol},
)
def run_remote(cfg: dict, run_name: str):
    """Run the simulation inside a Modal container."""
    import sys

    # Set env vars BEFORE importing magic_torch
    _CFG_TO_ENV = {
        "l_max": "MAGIC_LMAX",
        "n_r_max": "MAGIC_NR",
        "n_cheb_max": "MAGIC_NCHEBMAX",
        "minc": "MAGIC_MINC",
        "n_cheb_ic_max": "MAGIC_NCHEBICMAX",
        "sigma_ratio": "MAGIC_SIGMA_RATIO",
        "kbotb": "MAGIC_KBOTB",
        "nRotIC": "MAGIC_NROTIC",
        "ra": "MAGIC_RA",
        "time_scheme": "MAGIC_TIME_SCHEME",
        "dt": "MAGIC_DTMAX",
        "mode": "MAGIC_MODE",
        "raxi": "MAGIC_RAXI",
        "sc": "MAGIC_SC",
        "strat": "MAGIC_STRAT",
        "polind": "MAGIC_POLIND",
        "g0": "MAGIC_G0",
        "g1": "MAGIC_G1",
        "g2": "MAGIC_G2",
        "ktopv": "MAGIC_KTOPV",
        "kbotv": "MAGIC_KBOTV",
        "alpha": "MAGIC_ALPHA",
        "ek": "MAGIC_EK",
        "pr": "MAGIC_PR",
        "l_correct_AMz": "MAGIC_L_CORRECT_AMZ",
        "l_correct_AMe": "MAGIC_L_CORRECT_AME",
        "init_s1": "MAGIC_INIT_S1",
        "amp_s1": "MAGIC_AMP_S1",
        "prmag": "MAGIC_PRMAG",
        "radial_scheme": "MAGIC_RADIAL_SCHEME",
        "fd_order": "MAGIC_FD_ORDER",
        "fd_order_bound": "MAGIC_FD_ORDER_BOUND",
        "fd_stretch": "MAGIC_FD_STRETCH",
        "fd_ratio": "MAGIC_FD_RATIO",
        "radratio": "MAGIC_RADRATIO",
        "ktops": "MAGIC_KTOPS",
        "kbots": "MAGIC_KBOTS",
        "ktopxi": "MAGIC_KTOPXI",
        "kbotxi": "MAGIC_KBOTXI",
        "intfac": "MAGIC_INTFAC",
    }
    for key, env_var in _CFG_TO_ENV.items():
        if key in cfg:
            os.environ[env_var] = str(cfg[key])
    os.environ["MAGIC_DEVICE"] = "cuda"

    sys.path.insert(0, "/root/src")

    # Override output dir to the persistent volume
    output_dir = f"/output/{run_name}"
    cfg["output_dir"] = output_dir
    cfg["device"] = "cuda"

    from magic_torch.main import run

    result = run(cfg)
    vol.commit()
    return result


@app.local_entrypoint()
def main(config: str):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Upload Fortran checkpoint to input volume if needed
    fortran_restart = cfg.get("fortran_restart")
    if fortran_restart and not fortran_restart.startswith("/input/"):
        local_path = fortran_restart
        remote_name = os.path.basename(local_path)
        print(f"Uploading checkpoint {local_path} to Modal input volume...")
        in_vol = modal.Volume.from_name("magic-input", create_if_missing=True)
        with in_vol.batch_upload(force=True) as batch:
            batch.put_file(local_path, f"/{remote_name}")
        cfg["fortran_restart"] = f"/input/{remote_name}"
        print(f"  Uploaded as /input/{remote_name}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Launching Modal run: {run_name} (GPU: {GPU_TYPE})")

    result = run_remote.remote(cfg, run_name)

    # Download output from volume
    local_dir = f"./output/modal_{run_name}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nSimulation complete. Downloading output to {local_dir}/")
    _download_from_volume(run_name, local_dir)

    if result:
        e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor = result
        print(f"\nFinal energies:")
        print(f"  e_kin = {e_kin_pol + e_kin_tor:.6e} (pol={e_kin_pol:.6e}, tor={e_kin_tor:.6e})")
        print(f"  e_mag = {e_mag_pol + e_mag_tor:.6e} (pol={e_mag_pol:.6e}, tor={e_mag_tor:.6e})")


def _download_from_volume(run_name: str, local_dir: str):
    """Download run output from Modal volume to local directory."""
    try:
        vol_ref = modal.Volume.from_name("magic-output")
        for entry in vol_ref.listdir(f"/{run_name}"):
            remote_path = f"/{run_name}/{entry.path}"
            local_path = os.path.join(local_dir, entry.path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in vol_ref.read_file(remote_path):
                    f.write(chunk)
            print(f"  Downloaded: {entry.path}")
    except Exception as e:
        print(f"  Warning: Could not download output: {e}")
        print(f"  Files are still available in Modal volume 'magic-output' at /{run_name}/")
