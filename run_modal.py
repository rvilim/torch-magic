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
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir("src", remote_path="/root/src")
)

vol = modal.Volume.from_name("magic-output", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,
    volumes={"/output": vol},
)
def run_remote(cfg: dict, run_name: str):
    """Run the simulation inside a Modal container."""
    import sys

    # Set env vars BEFORE importing magic_torch
    if "l_max" in cfg:
        os.environ["MAGIC_LMAX"] = str(cfg["l_max"])
    if "n_r_max" in cfg:
        os.environ["MAGIC_NR"] = str(cfg["n_r_max"])
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
