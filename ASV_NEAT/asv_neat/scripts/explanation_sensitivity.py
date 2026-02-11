#!/usr/bin/env python3
"""Run feature-perturbation sensitivity tests using LIME and SHAP summaries."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the sensitivity script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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

THROTTLE_LABELS: List[str] = ["hold_speed", "accelerate", "decelerate"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python config file that matches the saved genome.",
    )
    parser.add_argument(
        "--winner",
        type=Path,
        default=None,
        help="Path to the pickled winning genome. Defaults to winners/crossing_winner.pkl.",
    )
    parser.add_argument(
        "--lime-dir",
        type=Path,
        required=True,
        help="Root folder containing LIME scenario report subdirectories.",
    )
    parser.add_argument(
        "--shap-dir",
        type=Path,
        required=True,
        help="Root folder containing SHAP scenario report subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "sensitivity_reports",
        help="Directory where sensitivity report JSON files will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of most influential features to test per output (default: 3).",
    )
    parser.add_argument(
        "--perturbation-pcts",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Percent perturbations applied in both directions (default: 1 2 3).",
    )
    return parser


def _load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_winner(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _iter_scenarios(lime_dir: Path, shap_dir: Path) -> List[tuple[Path, Path]]:
    if not lime_dir.exists():
        raise FileNotFoundError(f"LIME directory '{lime_dir}' does not exist.")
    if not shap_dir.exists():
        raise FileNotFoundError(f"SHAP directory '{shap_dir}' does not exist.")

    lime_scenarios = {
        p.name: p
        for p in sorted(p for p in lime_dir.iterdir() if p.is_dir())
        if (p / "lime_summary.json").exists()
    }
    shap_scenarios = {
        p.name: p
        for p in sorted(p for p in shap_dir.iterdir() if p.is_dir())
        if (p / "shap_summary.json").exists()
    }
    scenario_names = sorted(set(lime_scenarios).intersection(shap_scenarios))
    if not scenario_names:
        raise RuntimeError(
            "No common scenario folders found with both lime_summary.json and shap_summary.json."
        )
    return [(lime_scenarios[name], shap_scenarios[name]) for name in scenario_names]


def _network_outputs(network: neat.nn.FeedForwardNetwork, features: Sequence[float]) -> dict:
    outputs = np.asarray(network.activate(list(features)), dtype=float)
    rudder = float(outputs[0]) if outputs.ndim > 0 and outputs.size > 0 else 0.0
    throttle_raw = float(outputs[1]) if outputs.ndim > 0 and outputs.size > 1 else 0.0
    throttle_cmd = min(2, max(0, int(round(max(0.0, min(1.0, throttle_raw)) * 2.0))))
    return {
        "rudder": rudder,
        "rudder_abs": abs(rudder),
        "helm_label": helm_label_from_rudder_cmd(rudder),
        "throttle_raw": throttle_raw,
        "throttle_command": throttle_cmd,
        "throttle_label": THROTTLE_LABELS[throttle_cmd],
    }


def _attrib_value(item: dict) -> float:
    if "weight" in item:
        return float(item["weight"])
    if "shap_value" in item:
        return float(item["shap_value"])
    return 0.0


def _top_features(
    step_output: dict,
    top_k: int,
    feature_values: Dict[str, float],
) -> List[dict]:
    attributions = step_output.get("feature_attributions", [])
    ordered = sorted(attributions, key=lambda x: abs(_attrib_value(x)), reverse=True)
    top_features = []
    for rank, item in enumerate(ordered[:top_k], start=1):
        feature_name = item["feature"]
        top_features.append(
            {
                "rank": rank,
                "feature": feature_name,
                "attribution": _attrib_value(item),
                "attribution_abs": abs(_attrib_value(item)),
                "baseline_feature_value": float(feature_values[feature_name]),
            }
        )
    return top_features


def _extract_feature_values(*step_payloads: dict) -> Dict[str, float]:
    feature_values: Dict[str, float] = {}
    for payload in step_payloads:
        for output_name in ("rudder", "throttle"):
            attrs = payload.get(output_name, {}).get("feature_attributions", [])
            for item in attrs:
                feature = item.get("feature")
                if feature is None:
                    continue
                if feature not in feature_values:
                    feature_values[feature] = float(item.get("value", 0.0))
    missing = [name for name in FEATURE_NAMES if name not in feature_values]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Unable to recover a full feature vector from summary JSON. Missing: "
            f"{missing_str}"
        )
    return feature_values


def _run_perturbations(
    network: neat.nn.FeedForwardNetwork,
    baseline_features: np.ndarray,
    baseline_outputs: dict,
    feature_name: str,
    percentages: Iterable[float],
) -> List[dict]:
    idx = FEATURE_NAMES.index(feature_name)
    baseline_value = float(baseline_features[idx])

    perturbations: List[dict] = []
    for pct in percentages:
        frac = float(pct) / 100.0
        for direction, sign in (("decrease", -1.0), ("increase", 1.0)):
            factor = 1.0 + sign * frac
            perturbed_features = baseline_features.copy()
            perturbed_features[idx] = baseline_value * factor

            perturbed_outputs = _network_outputs(network, perturbed_features)
            rudder_delta = perturbed_outputs["rudder"] - baseline_outputs["rudder"]
            throttle_raw_delta = (
                perturbed_outputs["throttle_raw"] - baseline_outputs["throttle_raw"]
            )

            perturbations.append(
                {
                    "direction": direction,
                    "percent": float(pct),
                    "factor": factor,
                    "baseline_feature_value": baseline_value,
                    "perturbed_feature_value": float(perturbed_features[idx]),
                    "outputs": perturbed_outputs,
                    "deltas": {
                        "rudder_delta": rudder_delta,
                        "rudder_abs_delta": abs(perturbed_outputs["rudder_abs"] - baseline_outputs["rudder_abs"]),
                        "rudder_magnitude_delta": abs(rudder_delta),
                        "helm_label_changed": perturbed_outputs["helm_label"]
                        != baseline_outputs["helm_label"],
                        "throttle_raw_delta": throttle_raw_delta,
                        "throttle_command_delta": int(perturbed_outputs["throttle_command"])
                        - int(baseline_outputs["throttle_command"]),
                        "throttle_label_changed": perturbed_outputs["throttle_label"]
                        != baseline_outputs["throttle_label"],
                    },
                }
            )
    return perturbations


def _step_index(items: list[dict]) -> Dict[int, dict]:
    return {int(item["step"]): item for item in items}


def _build_step_report(
    network: neat.nn.FeedForwardNetwork,
    step: int,
    lime_step: dict,
    shap_step: dict,
    top_k: int,
    percentages: Iterable[float],
) -> dict:
    feature_values = _extract_feature_values(lime_step, shap_step)
    baseline_vec = np.asarray([feature_values[name] for name in FEATURE_NAMES], dtype=float)
    baseline_outputs = _network_outputs(network, baseline_vec)

    explainers = {}
    for explainer_name, step_payload in (("lime", lime_step), ("shap", shap_step)):
        outputs = {}
        for output_name in ("rudder", "throttle"):
            top_features = _top_features(
                step_payload[output_name],
                top_k,
                feature_values,
            )
            sensitivity_rows = []
            for feature_item in top_features:
                sensitivity_rows.append(
                    {
                        **feature_item,
                        "perturbations": _run_perturbations(
                            network,
                            baseline_vec,
                            baseline_outputs,
                            feature_item["feature"],
                            percentages,
                        ),
                    }
                )
            outputs[output_name] = {
                "top_features": top_features,
                "sensitivity": sensitivity_rows,
            }
        explainers[explainer_name] = outputs

    return {
        "step": step,
        "baseline": {
            "features": {
                FEATURE_NAMES[idx]: float(baseline_vec[idx]) for idx in range(len(FEATURE_NAMES))
            },
            "outputs": baseline_outputs,
        },
        "explainers": explainers,
    }


def run_sensitivity(
    *,
    winner_path: Path,
    config_path: Path,
    lime_dir: Path,
    shap_dir: Path,
    output_dir: Path,
    top_k: int,
    perturbation_pcts: Sequence[float],
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

    overall_report = []
    for idx, (lime_scenario_dir, shap_scenario_dir) in enumerate(
        _iter_scenarios(lime_dir, shap_dir), start=1
    ):
        lime_summary = _load_json(lime_scenario_dir / "lime_summary.json")
        shap_summary = _load_json(shap_scenario_dir / "shap_summary.json")
        metadata_path = lime_scenario_dir / "metadata.json"
        metadata = _load_json(metadata_path) if metadata_path.exists() else {}

        lime_by_step = _step_index(lime_summary)
        shap_by_step = _step_index(shap_summary)
        common_steps = sorted(set(lime_by_step).intersection(shap_by_step))
        if not common_steps:
            raise RuntimeError(
                f"Scenario '{lime_scenario_dir.name}' has no overlapping LIME/SHAP steps."
            )

        step_reports = [
            _build_step_report(
                network,
                step,
                lime_by_step[step],
                shap_by_step[step],
                top_k,
                perturbation_pcts,
            )
            for step in common_steps
        ]

        scenario_report = {
            "scenario": lime_scenario_dir.name,
            "metadata": metadata,
            "settings": {
                "top_k": top_k,
                "perturbation_percents": [float(x) for x in perturbation_pcts],
            },
            "steps": step_reports,
        }
        scenario_out = output_dir / lime_scenario_dir.name / "sensitivity_summary.json"
        _write_json(scenario_out, scenario_report)
        overall_report.append(
            {
                "scenario": lime_scenario_dir.name,
                "steps": len(step_reports),
                "output": str(scenario_out),
            }
        )
        print(
            f"Scenario {idx:02d} [{lime_scenario_dir.name}] processed {len(step_reports)} steps -> {scenario_out}"
        )

    _write_json(output_dir / "sensitivity_index.json", overall_report)


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_path = args.winner if args.winner is not None else default_winner_path("crossing")
    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    run_sensitivity(
        winner_path=winner_path,
        config_path=args.config,
        lime_dir=args.lime_dir,
        shap_dir=args.shap_dir,
        output_dir=output_dir,
        top_k=max(1, args.top_k),
        perturbation_pcts=args.perturbation_pcts,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
