#!/usr/bin/env python3
"""CLI launcher for the MagIC PyTorch dynamo simulation.

Reads a YAML config, sets env vars BEFORE importing magic_torch
(which initializes params/modules at import time), then runs.

Usage: uv run python run.py config.yaml
"""

import argparse
import os
import sys

import yaml

# Mapping from YAML config keys to env var names.
# run.py sets these before any magic_torch import so that params.py picks them up.
_CONFIG_ENV_MAP = {
    "l_max": "MAGIC_LMAX",
    "n_r_max": "MAGIC_NR",
    "n_cheb_max": "MAGIC_NCHEBMAX",
    "minc": "MAGIC_MINC",
    "device": "MAGIC_DEVICE",
    "ra": "MAGIC_RA",
    "ek": "MAGIC_EK",
    "pr": "MAGIC_PR",
    "prmag": "MAGIC_PRMAG",
    "radratio": "MAGIC_RADRATIO",
    "dtmax": "MAGIC_DTMAX",
    "dt": "MAGIC_DTMAX",  # alias: dt in config -> dtmax param
    "alpha": "MAGIC_ALPHA",
    "time_scheme": "MAGIC_TIME_SCHEME",
    "sigma_ratio": "MAGIC_SIGMA_RATIO",
    "nRotIC": "MAGIC_NROTIC",
    "kbotb": "MAGIC_KBOTB",
    "n_cheb_ic_max": "MAGIC_NCHEBICMAX",
    "start_file": "MAGIC_START_FILE",
    "mode": "MAGIC_MODE",
    "raxi": "MAGIC_RAXI",
    "sc": "MAGIC_SC",
}


def main():
    parser = argparse.ArgumentParser(description="MagIC PyTorch dynamo runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Set env vars before any magic_torch import
    for key, env_var in _CONFIG_ENV_MAP.items():
        if key in cfg and cfg[key] is not None and cfg[key] != "":
            os.environ[env_var] = str(cfg[key])

    # Now safe to import magic_torch
    from magic_torch.main import run

    run(cfg)


if __name__ == "__main__":
    main()
