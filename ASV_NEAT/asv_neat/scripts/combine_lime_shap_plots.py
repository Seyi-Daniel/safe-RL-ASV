#!/usr/bin/env python3
"""Combine LIME and SHAP plot images into 2x2 grids across scenarios."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

from PIL import Image

RUDDER_PLOT_PATTERN = re.compile(r"explanation_rudder_(\d+)\.png")
THROTTLE_PLOT_PATTERN = re.compile(r"explanation_throttle_(\d+)\.png")


def _find_matching_dirs(base: Path, patterns: Sequence[re.Pattern[str]]) -> dict[str, Path]:
    """Return mapping of relative scenario names to directories containing all patterns."""

    scenario_dirs: dict[str, Path] = {}
    search_dirs: List[Path] = [base]
    search_dirs.extend(child for child in base.rglob("*") if child.is_dir())

    for candidate in search_dirs:
        if all(
            any(pattern.fullmatch(item.name) for item in candidate.iterdir() if item.is_file())
            for pattern in patterns
        ):
            scenario_key = candidate.relative_to(base).as_posix() or "root"
            scenario_dirs[scenario_key] = candidate

    return scenario_dirs


def _discover_plot_dirs(base: Path) -> dict[str, Path]:
    return _find_matching_dirs(base, [RUDDER_PLOT_PATTERN, THROTTLE_PLOT_PATTERN])


def _extract_plot_steps(plot_dir: Path) -> List[int]:
    steps = []
    for plot_path in plot_dir.iterdir():
        if not plot_path.is_file():
            continue
        match = RUDDER_PLOT_PATTERN.fullmatch(plot_path.name) or THROTTLE_PLOT_PATTERN.fullmatch(plot_path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _resize_plot_to_height(path: Path, target_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_height = max(1, target_height)
    target_width = max(1, int(image.width * target_height / max(image.height, 1)))
    return image.resize((target_width, target_height), Image.LANCZOS)


def _pad_plot_width(image: Image.Image, target_width: int) -> Image.Image:
    if image.width >= target_width:
        return image
    canvas = Image.new("RGB", (target_width, image.height), color=(255, 255, 255))
    offset_x = (target_width - image.width) // 2
    canvas.paste(image, (offset_x, 0))
    return canvas


def _compose_plot_grid(
    lime_rudder_path: Path,
    lime_throttle_path: Path,
    shap_rudder_path: Path,
    shap_throttle_path: Path,
) -> Image.Image:
    lime_rudder = Image.open(lime_rudder_path).convert("RGB")
    lime_throttle = Image.open(lime_throttle_path).convert("RGB")
    shap_rudder = Image.open(shap_rudder_path).convert("RGB")
    shap_throttle = Image.open(shap_throttle_path).convert("RGB")

    top_height = max(lime_rudder.height, shap_rudder.height)
    bottom_height = max(lime_throttle.height, shap_throttle.height)

    lime_rudder = _resize_plot_to_height(lime_rudder_path, top_height)
    shap_rudder = _resize_plot_to_height(shap_rudder_path, top_height)
    lime_throttle = _resize_plot_to_height(lime_throttle_path, bottom_height)
    shap_throttle = _resize_plot_to_height(shap_throttle_path, bottom_height)

    left_width = max(lime_rudder.width, lime_throttle.width)
    right_width = max(shap_rudder.width, shap_throttle.width)

    lime_rudder = _pad_plot_width(lime_rudder, left_width)
    lime_throttle = _pad_plot_width(lime_throttle, left_width)
    shap_rudder = _pad_plot_width(shap_rudder, right_width)
    shap_throttle = _pad_plot_width(shap_throttle, right_width)

    grid_width = left_width + right_width
    grid_height = top_height + bottom_height
    canvas = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    canvas.paste(lime_rudder, (0, 0))
    canvas.paste(shap_rudder, (left_width, 0))
    canvas.paste(lime_throttle, (0, top_height))
    canvas.paste(shap_throttle, (left_width, top_height))

    return canvas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lime-dir", type=Path, required=True, help="Path to root LIME reports directory")
    parser.add_argument("--shap-dir", type=Path, required=True, help="Path to root SHAP reports directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined plot outputs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    lime_dir: Path = args.lime_dir
    shap_dir: Path = args.shap_dir
    output_dir: Path = args.output_dir

    lime_plot_dirs = _discover_plot_dirs(lime_dir)
    shap_plot_dirs = _discover_plot_dirs(shap_dir)

    shared_keys = sorted(set(lime_plot_dirs) & set(shap_plot_dirs))

    if not shared_keys:
        raise RuntimeError("No matching scenarios with LIME and SHAP plot images were found.")

    total_generated = 0
    for scenario_key in shared_keys:
        lime_plot_dir = lime_plot_dirs[scenario_key]
        shap_plot_dir = shap_plot_dirs[scenario_key]

        lime_steps = _extract_plot_steps(lime_plot_dir)
        shap_steps = _extract_plot_steps(shap_plot_dir)
        candidate_steps = sorted(set(lime_steps) | set(shap_steps))

        if not candidate_steps:
            continue

        scenario_output_dir = output_dir / scenario_key
        combined_plot_dir = scenario_output_dir / "combined_plots"
        combined_plot_dir.mkdir(parents=True, exist_ok=True)
        scenario_output_dir.mkdir(parents=True, exist_ok=True)

        generated_for_scenario = 0
        for step in candidate_steps:
            rudder_name = f"explanation_rudder_{step:03d}.png"
            throttle_name = f"explanation_throttle_{step:03d}.png"

            lime_rudder_path = lime_plot_dir / rudder_name
            lime_throttle_path = lime_plot_dir / throttle_name
            shap_rudder_path = shap_plot_dir / rudder_name
            shap_throttle_path = shap_plot_dir / throttle_name

            plot_required_paths = [lime_rudder_path, lime_throttle_path, shap_rudder_path, shap_throttle_path]
            if not all(path.exists() for path in plot_required_paths):
                continue

            plot_grid = _compose_plot_grid(
                lime_rudder_path,
                lime_throttle_path,
                shap_rudder_path,
                shap_throttle_path,
            )
            plot_output_path = combined_plot_dir / f"plots_{step:03d}.png"
            plot_grid.save(plot_output_path)
            generated_for_scenario += 1

        if generated_for_scenario:
            total_generated += generated_for_scenario
            print(
                f"Saved {generated_for_scenario} plot grids to {combined_plot_dir} "
                f"for scenario '{scenario_key}'"
            )

    if not total_generated:
        raise RuntimeError("No plot grids were generated; ensure required plots exist for matching steps.")


if __name__ == "__main__":
    main()
