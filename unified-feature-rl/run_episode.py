#!/usr/bin/env python3
"""Compatibility wrapper: use root-level run_episode.py."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "run_episode.py"), run_name="__main__")
