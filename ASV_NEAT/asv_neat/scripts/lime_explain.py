#!/usr/bin/env python3
"""Generate LIME explanations and visualisations for a saved NEAT controller."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'imageio' package is required to generate videos.") from exc

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'matplotlib' package is required to plot explanations.") from exc

try:  # pragma: no cover - optional dependency
    from lime import lime_tabular
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'lime' package is required to run the explanation script.") from exc

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'Pillow' package is required to compose visualisations.") from exc

try:  # pragma: no cover - optional dependency
    import pygame
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'pygame' package is required to capture render frames.") from exc

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the explanation script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat.cli_helpers import SCENARIO_KIND_CHOICES  # noqa: E402
from asv_neat.paths import default_winner_path  # noqa: E402
from asv_neat.utils import helm_label_from_rudder_cmd  # noqa: E402

FEATURE_NAMES: List[str] = [
    "x_goal_TV",
    "y_goal_TV",
    "speed_TV",
    "heading_TV",
    "x_TV",
    "y_TV",
    "x_goal_ASV",
    "y_goal_ASV",
    "speed_ASV",
    "heading_ASV",
    "x_ASV",
    "y_ASV",
]

FRAME_DIRNAME = "frames"
PLOT_DIRNAME = "plots"
COMBINED_DIRNAME = "combined_frames"
RUDDER_ANIMATION_FILENAME = "explanation_rudder_animation.gif"
THROTTLE_ANIMATION_FILENAME = "explanation_throttle_animation.gif"


THROTTLE_LABELS: List[str] = ["hold speed", "accelerate", "decelerate"]


def _format_rudder_angle(angle_rad: float) -> str:
    angle_deg = math.degrees(angle_rad)
    if abs(angle_deg) < 1e-3:
        return "0.0"
    return f"{angle_deg:+.1f}"


def _throttle_distribution(throttle_val: float) -> np.ndarray:
    scaled = max(0.0, min(1.0, throttle_val)) * 2.0
    logits = -np.square(scaled - np.arange(3, dtype=float))
    logits = logits - logits.max()
    exp = np.exp(logits)
    denom = exp.sum()
    if denom <= 0.0:
        return np.asarray([1 / 3] * 3, dtype=float)
    return exp / denom


class LimeRudderWrapper:
    """Regression-style adapter exposing the rudder output."""

    def __init__(self, network: neat.nn.FeedForwardNetwork) -> None:
        self._network = network

    def predict(self, data: np.ndarray) -> np.ndarray:
        preds = []
        for row in data:
            outputs = np.asarray(self._network.activate(row.tolist()), dtype=float)
            rudder = float(outputs[0]) if outputs.ndim > 0 and outputs.size > 0 else 0.0
            preds.append(rudder)
        return np.asarray(preds, dtype=float)


class LimeThrottleWrapper:
    """Classification adapter for the discrete throttle choice."""

    def __init__(self, network: neat.nn.FeedForwardNetwork) -> None:
        self._network = network

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        predictions = []
        for row in data:
            outputs = np.asarray(self._network.activate(row.tolist()), dtype=float)
            throttle_val = float(outputs[1]) if outputs.ndim > 0 and outputs.size > 1 else 0.0
            predictions.append(_throttle_distribution(throttle_val))
        return np.asarray(predictions, dtype=float)
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python configuration file that matches the saved genome.",
    )
    parser.add_argument(
        "--winner",
        type=Path,
        default=None,
        help="Path to the pickled winning genome. Defaults to winners/<scenario>_winner.pkl.",
    )
    parser.add_argument(
        "--scenario-kind",
        choices=SCENARIO_KIND_CHOICES,
        default="all",
        help="Select which encounter family should be explained.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "captured_episodes",
        help="Directory containing pre-captured traces and frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "lime_reports",
        help="Directory where explanation artefacts will be written.",
    )
    return parser


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_frame(path: Path, surface) -> None:
    if surface is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def _plot_explanation(step_data: dict, output_path: Path, *, title: str) -> None:
    weights = {item["feature"]: item["weight"] for item in step_data["feature_attributions"]}
    values = [weights.get(name, 0.0) for name in FEATURE_NAMES]
    colors = ["#3CB371" if weight >= 0 else "#D95F02" for weight in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(FEATURE_NAMES, values, color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("LIME weight", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_explanations(explanations: List[dict], plot_dir: Path) -> List[tuple[int, Path, Path]]:
    plot_paths: List[tuple[int, Path, Path]] = []
    for item in explanations:
        step = item["step"]
        rudder_path = plot_dir / f"explanation_rudder_{step:03d}.png"
        throttle_path = plot_dir / f"explanation_throttle_{step:03d}.png"
        rudder_info = item["rudder"]
        actual_rudder = float(rudder_info.get("actual_rudder", 0.0))
        helm_label = rudder_info.get("helm_label") or "keep_straight"
        _plot_explanation(
            rudder_info,
            rudder_path,
            title=(
                f"Step: {step:03d}  Rudder angle: {_format_rudder_angle(actual_rudder)}° "
                f"Turn action: {helm_label.replace('_', ' ')}"
            ),
        )
        _plot_explanation(
            item["throttle"],
            throttle_path,
            title=(
                f"Step: {step:03d}  Throttle action: "
                f"{THROTTLE_LABELS[item['throttle']['prediction']]}"
            ),
        )
        plot_paths.append((step, rudder_path, throttle_path))
    return plot_paths


def _combine_images(scene_path: Path, plot_path: Path, output_path: Path) -> None:
    scene = Image.open(scene_path).convert("RGB")
    plot = Image.open(plot_path).convert("RGB")
    height = max(scene.height, plot.height)
    canvas = Image.new("RGB", (scene.width + plot.width, height), color=(0, 0, 0))
    canvas.paste(scene, (0, 0))
    canvas.paste(plot, (scene.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _combine_frames(
    frame_dir: Path, plot_dir: Path, combined_dir: Path, *, kind: str
) -> List[Path]:
    combined: List[Path] = []
    for frame_path in sorted(frame_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[-1])):
        step = frame_path.stem.split("_")[-1]
        plot_path = plot_dir / f"explanation_{kind}_{step}.png"
        if not plot_path.exists():
            continue
        output_path = combined_dir / f"combined_{kind}_{step}.png"
        _combine_images(frame_path, plot_path, output_path)
        combined.append(output_path)
    return combined


def _write_animation(frame_paths: List[Path], output_path: Path, fps: int = 8) -> None:
    if not frame_paths:
        return
    images = [imageio.imread(path) for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps)


def _generate_lime(
    scenario_dir: Path,
    trace: List[dict],
    network: neat.nn.FeedForwardNetwork,
) -> List[dict]:
    if not trace:
        return []

    features = np.asarray(
        [item.get("features") or item.get("obs") or [] for item in trace], dtype=float
    )
    rudder_wrapper = LimeRudderWrapper(network)
    throttle_wrapper = LimeThrottleWrapper(network)
    rudder_explainer = lime_tabular.LimeTabularExplainer(
        training_data=features,
        feature_names=FEATURE_NAMES,
        mode="regression",
        discretize_continuous=False,
    )
    throttle_explainer = lime_tabular.LimeTabularExplainer(
        training_data=features,
        feature_names=FEATURE_NAMES,
        class_names=THROTTLE_LABELS,
        discretize_continuous=False,
    )

    explanations: List[dict] = []
    for item in trace:
        features = item.get("features") or item.get("obs") or []
        feature_vec = np.asarray(features, dtype=float)
        rudder_pred = float(rudder_wrapper.predict(np.asarray([feature_vec]))[0])
        rudder_exp = rudder_explainer.explain_instance(
            feature_vec,
            rudder_wrapper.predict,
            num_features=len(FEATURE_NAMES),
        )
        rudder_map = next(iter(rudder_exp.as_map().values()), [])
        rudder_attr = [
            {
                "feature": FEATURE_NAMES[idx],
                "weight": float(weight),
                "value": float(feature_vec[idx]),
            }
            for idx, weight in rudder_map
        ]

        throttle_probs = throttle_wrapper.predict_proba(np.asarray([feature_vec]))[0]
        throttle_label = int(np.argmax(throttle_probs))
        throttle_exp = throttle_explainer.explain_instance(
            feature_vec,
            throttle_wrapper.predict_proba,
            num_features=len(FEATURE_NAMES),
            top_labels=1,
        )
        throttle_map = throttle_exp.as_map().get(throttle_label, [])
        throttle_attr = [
            {
                "feature": FEATURE_NAMES[idx],
                "weight": float(weight),
                "value": float(feature_vec[idx]),
            }
            for idx, weight in throttle_map
        ]

        rudder_cmd = float(item.get("rudder_cmd", rudder_pred))
        helm_label = item.get("helm_label") or helm_label_from_rudder_cmd(rudder_cmd)
        actual_rudder = float(item.get("agent_state", {}).get("rudder", 0.0))

        explanation = {
            "step": item["step"],
            "rudder": {
                "prediction": rudder_pred,
                "rudder_cmd": rudder_cmd,
                "helm_label": helm_label,
                "actual_rudder": actual_rudder,
                "feature_attributions": rudder_attr,
            },
            "throttle": {
                "prediction": throttle_label,
                "probabilities": throttle_probs.tolist(),
                "feature_attributions": throttle_attr,
            },
        }
        explanations.append(explanation)
        _write_json(scenario_dir / f"lime_step_{item['step']:03d}.json", explanation)

    _write_json(scenario_dir / "lime_summary.json", explanations)
    return explanations


def _load_winner(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _iter_scenario_dirs(data_dir: Path, scenario_kind: str) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    scenario_dirs: List[Path] = []
    for path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        metadata_path = path / "metadata.json"
        trace_path = path / "trace.json"
        if not metadata_path.exists() or not trace_path.exists():
            continue
        metadata = _load_json(metadata_path)
        if scenario_kind != "all" and metadata.get("scenario_kind") != scenario_kind:
            continue
        scenario_dirs.append(path)
    if not scenario_dirs:
        raise RuntimeError(
            f"No scenarios found in {data_dir} for kind '{scenario_kind}'."
        )
    return scenario_dirs


def explain_scenarios(
    *,
    winner_path: Path,
    config_path: Path,
    data_dir: Path,
    scenario_kind: str,
    output_dir: Path,
) -> None:
    winner = _load_winner(winner_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    network = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    for idx, scenario_path in enumerate(
        _iter_scenario_dirs(data_dir, scenario_kind), start=1
    ):
        metadata = _load_json(scenario_path / "metadata.json")
        trace = _load_json(scenario_path / "trace.json")

        scenario_dir = output_dir / scenario_path.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = scenario_path / FRAME_DIRNAME
        plot_dir = scenario_dir / PLOT_DIRNAME
        combined_dir = scenario_dir / COMBINED_DIRNAME

        _write_json(scenario_dir / "metadata.json", metadata)
        _write_json(scenario_dir / "trace.json", trace)

        explanations = _generate_lime(scenario_dir, trace, network)
        _plot_explanations(explanations, plot_dir)
        combined_rudder_frames = _combine_frames(
            frame_dir, plot_dir, combined_dir / "rudder", kind="rudder"
        )
        combined_throttle_frames = _combine_frames(
            frame_dir, plot_dir, combined_dir / "throttle", kind="throttle"
        )
        _write_animation(combined_rudder_frames, scenario_dir / RUDDER_ANIMATION_FILENAME)
        _write_animation(
            combined_throttle_frames, scenario_dir / THROTTLE_ANIMATION_FILENAME
        )
        steps = metadata.get("metrics", {}).get("steps", 0)
        cost = metadata.get("episode_cost", 0.0)
        kind = metadata.get("scenario_kind", "unknown")
        print(
            f"Scenario {idx:02d} [{kind}] steps={steps:4d} "
            f"cost={cost:7.2f} outputs saved to {scenario_dir}"
        )


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_path = args.winner
    if winner_path is None:
        winner_path = default_winner_path(args.scenario_kind)
    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    explain_scenarios(
        winner_path=winner_path,
        config_path=args.config,
        data_dir=args.data_dir,
        scenario_kind=args.scenario_kind,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
