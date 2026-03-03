#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np

from simulations.scenarios import SIMULATION_REGISTRY


def parse_args() -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--simulation", choices=tuple(SIMULATION_REGISTRY.keys()), default="dcpa-sampled-episode")
    selected, _ = probe.parse_known_args()

    p = argparse.ArgumentParser(description="Simulation sandbox for trying scenario variants without RL training")
    p.add_argument("--simulation", choices=tuple(SIMULATION_REGISTRY.keys()), default="dcpa-sampled-episode")
    SIMULATION_REGISTRY[selected.simulation].register_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    SIMULATION_REGISTRY[args.simulation].run(args)


if __name__ == "__main__":
    main()
