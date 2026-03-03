#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from simulations.perimeter_start.sim_views import VIEW_REGISTRY


def parse_args() -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--view", choices=tuple(VIEW_REGISTRY.keys()), default="dcpa-sampled-episode")
    selected, _ = probe.parse_known_args()

    p = argparse.ArgumentParser(description="Simulation sandbox for manual behavior checks")
    p.add_argument("--view", choices=tuple(VIEW_REGISTRY.keys()), default="dcpa-sampled-episode")
    VIEW_REGISTRY[selected.view].register_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    VIEW_REGISTRY[args.view].run(args)


if __name__ == "__main__":
    main()
