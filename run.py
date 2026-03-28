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


def main():
    parser = argparse.ArgumentParser(description="MagIC PyTorch dynamo runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Set env vars before any magic_torch import
    if "l_max" in cfg:
        os.environ["MAGIC_LMAX"] = str(cfg["l_max"])
    if "n_r_max" in cfg:
        os.environ["MAGIC_NR"] = str(cfg["n_r_max"])
    if "device" in cfg and cfg["device"]:
        os.environ["MAGIC_DEVICE"] = str(cfg["device"])

    # Now safe to import magic_torch
    from magic_torch.main import run

    run(cfg)


if __name__ == "__main__":
    main()
