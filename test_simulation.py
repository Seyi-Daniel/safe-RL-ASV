#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np

from sim_views import VIEW_REGISTRY


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
