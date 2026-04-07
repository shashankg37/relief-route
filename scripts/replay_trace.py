"""Replay a saved ReliefRoute episode trace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relief_route_env.models import EpisodeTrace, SupplyType


def _print_observation_summary(step_label: str, observation) -> None:
    print(f"\n{step_label}")
    print(
        f"Task={observation.task_id} Theme={observation.scenario_theme} "
        f"Step={observation.current_step}/{observation.max_steps} Score={observation.info.final_score:.3f}"
    )
    print(
        f"Inventory: water={observation.depot_inventory.water} "
        f"food={observation.depot_inventory.food} medicine={observation.depot_inventory.medicine}"
    )
    for zone in observation.zones:
        print(
            f"  {zone.display_name}: remaining "
            f"W={zone.remaining(SupplyType.WATER)} "
            f"F={zone.remaining(SupplyType.FOOD)} "
            f"M={zone.remaining(SupplyType.MEDICINE)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved ReliefRoute trace.")
    parser.add_argument("trace_file", type=Path)
    parser.add_argument("--delay-ms", type=int, default=350)
    args = parser.parse_args()

    trace = EpisodeTrace.model_validate(json.loads(args.trace_file.read_text(encoding="utf-8")))
    print(f"Replay policy: {trace.policy} | task: {trace.task_id} | final_score: {trace.final_score:.4f}")
    _print_observation_summary("Initial observation", trace.initial_observation)

    for step in trace.steps:
        print(f"\nStep {step.step_index}")
        print(step.action.model_dump_json(indent=2))
        print(f"Step reward: {step.total_reward:.3f}")
        if step.observation.info.delivered_this_step:
            for item in step.observation.info.delivered_this_step:
                print(f"  - {item}")
        _print_observation_summary("Observation", step.observation)
        time.sleep(max(0, args.delay_ms) / 1000.0)


if __name__ == "__main__":
    main()
