#!/usr/bin/env python3
"""Combine LIME and SHAP explanation outputs into a single animation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

import imageio.v2 as imageio
from PIL import Image

FRAME_FILENAME_PATTERN = re.compile(r"frame_(\d+)\.png")
RUDDER_PLOT_PATTERN = re.compile(r"explanation_rudder_(\d+)\.png")
THROTTLE_PLOT_PATTERN = re.compile(r"explanation_throttle_(\d+)\.png")


def _find_matching_dir(base: Path, patterns: Sequence[re.Pattern[str]]) -> Path:
    """Return the first directory under ``base`` containing files for all patterns."""

    search_dirs: List[Path] = [base]
    search_dirs.extend([child for child in base.iterdir() if child.is_dir()])

    for candidate in search_dirs:
        if all(any(pattern.fullmatch(item.name) for item in candidate.iterdir() if item.is_file()) for pattern in patterns):
            return candidate
    pattern_descriptions = ", ".join(p.pattern for p in patterns)
    raise FileNotFoundError(f"Could not find directory under {base} with files matching: {pattern_descriptions}")


def _discover_frame_dir(base: Path) -> Path | None:
    try:
        return _find_matching_dir(base, [FRAME_FILENAME_PATTERN])
    except FileNotFoundError:
        return None


def _discover_plot_dir(base: Path) -> Path:
    return _find_matching_dir(base, [RUDDER_PLOT_PATTERN, THROTTLE_PLOT_PATTERN])


def _extract_steps(frame_dir: Path | None) -> List[int]:
    if frame_dir is None:
        return []
    steps = []
    for frame_path in frame_dir.iterdir():
        if not frame_path.is_file():
            continue
        match = FRAME_FILENAME_PATTERN.fullmatch(frame_path.name)
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


def _compose_frame(
    scene_path: Path,
    lime_rudder_path: Path,
    lime_throttle_path: Path,
    shap_rudder_path: Path,
    shap_throttle_path: Path,
) -> Image.Image:
    scene = Image.open(scene_path).convert("RGB")
    top_height = scene.height // 2
    bottom_height = scene.height - top_height

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

    margin = 10
    canvas_width = scene.width + margin + grid_width
    canvas_height = max(scene.height, grid_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    scene_y = (canvas_height - scene.height) // 2 if canvas_height > scene.height else 0
    canvas.paste(scene, (0, scene_y))

    grid_x = scene.width + margin
    grid_y = (canvas_height - grid_height) // 2 if canvas_height > grid_height else 0

    canvas.paste(lime_rudder, (grid_x, grid_y))
    canvas.paste(shap_rudder, (grid_x + left_width, grid_y))
    canvas.paste(lime_throttle, (grid_x, grid_y + top_height))
    canvas.paste(shap_throttle, (grid_x + left_width, grid_y + top_height))

    return canvas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lime-dir", type=Path, required=True, help="Path to LIME scenario output directory")
    parser.add_argument("--shap-dir", type=Path, required=True, help="Path to SHAP scenario output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined outputs")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the final GIF")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    lime_dir: Path = args.lime_dir
    shap_dir: Path = args.shap_dir
    output_dir: Path = args.output_dir
    fps: int = args.fps

    lime_frame_dir = _discover_frame_dir(lime_dir)
    lime_plot_dir = _discover_plot_dir(lime_dir)
    shap_frame_dir = _discover_frame_dir(shap_dir)
    shap_plot_dir = _discover_plot_dir(shap_dir)

    steps = _extract_steps(lime_frame_dir) or _extract_steps(shap_frame_dir)
    if not steps:
        raise RuntimeError("No frame images found in the provided directories.")

    combined_dir = output_dir / "combined_frames"
    combined_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: List[Path] = []

    for step in steps:
        frame_name = f"frame_{step:03d}.png"
        rudder_name = f"explanation_rudder_{step:03d}.png"
        throttle_name = f"explanation_throttle_{step:03d}.png"

        scene_path = lime_frame_dir / frame_name if lime_frame_dir else None
        if scene_path is None or not scene_path.exists():
            scene_path = shap_frame_dir / frame_name if shap_frame_dir else None
        lime_rudder_path = lime_plot_dir / rudder_name
        lime_throttle_path = lime_plot_dir / throttle_name
        shap_rudder_path = shap_plot_dir / rudder_name
        shap_throttle_path = shap_plot_dir / throttle_name

        plot_required_paths = [
            lime_rudder_path,
            lime_throttle_path,
            shap_rudder_path,
            shap_throttle_path,
        ]

        if scene_path is None or not scene_path.exists():
            continue
        if not all(path.exists() for path in plot_required_paths):
            continue

        composite = _compose_frame(
            scene_path,
            lime_rudder_path,
            lime_throttle_path,
            shap_rudder_path,
            shap_throttle_path,
        )
        output_path = combined_dir / f"combined_{step:03d}.png"
        composite.save(output_path)
        combined_frames.append(output_path)

    if not combined_frames:
        raise RuntimeError("No combined frames were generated; ensure required frames and plots exist.")

    images = [imageio.imread(frame_path) for frame_path in combined_frames]
    gif_path = output_dir / "lime_shap_explanation_animation.gif"
    imageio.mimsave(gif_path, images, fps=fps)

    print(f"Saved {len(combined_frames)} frames to {combined_dir}")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
