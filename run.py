#!/usr/bin/env python3
"""CLI launcher for the MagIC PyTorch dynamo simulation.

Reads a YAML config, calls configure() BEFORE importing magic_torch
(which initializes params/modules at import time), then runs.

Usage: uv run python run.py config.yaml
"""

import argparse
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="MagIC PyTorch dynamo runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Handle dt -> dtmax alias
    if "dt" in cfg and "dtmax" not in cfg:
        cfg["dtmax"] = cfg.pop("dt")

    # Configure before any magic_torch import
    from magic_torch.config import configure
    configure(cfg)

    # Now safe to import magic_torch
    from magic_torch.main import run

    run(cfg)


if __name__ == "__main__":
    main()
