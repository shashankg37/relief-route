"""Run ReliefRoute baseline policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relief_route_env.baseline import (
    greedy_priority_dispatch_action,
    heuristic_dispatch_action,
    random_dispatch_action,
)
from relief_route_env.grader import grade_observation
from server.relief_route_environment import ReliefRouteEnvironment


def _policy_action(policy: str, observation, rng: Random):
    if policy == "heuristic":
        return heuristic_dispatch_action(observation)
    if policy == "greedy":
        return greedy_priority_dispatch_action(observation)
    if policy == "random":
        return random_dispatch_action(observation, rng)
    raise ValueError(f"Unknown policy: {policy}")


def run_task(task_id: str, episodes: int, policy: str, trace_dir: Path | None = None) -> dict:
    scores: list[float] = []
    completion_reasons: list[str] = []
    rng = Random(7)

    for _ in range(episodes):
        env = ReliefRouteEnvironment(task_id=task_id)
        observation = env.reset()
        while not observation.done:
            action = _policy_action(policy, observation, rng)
            observation = env.step(action)
        scores.append(grade_observation(observation.info))
        completion_reasons.append(str(observation.metadata.get("completion_reason", "unknown")))
        if trace_dir is not None:
            env.save_trace(trace_dir / f"{policy}_{task_id}_episode_{len(scores)}.json", policy=policy)

    return {
        "policy": policy,
        "task_id": task_id,
        "episodes": episodes,
        "average_score": round(sum(scores) / max(len(scores), 1), 4),
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "scores": scores,
        "completion_reasons": completion_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReliefRoute baseline policies.")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--policy", choices=["heuristic", "greedy", "random"], default="heuristic")
    parser.add_argument("--trace-dir", type=Path, default=None)
    args = parser.parse_args()

    results = [run_task(task_id, args.episodes, args.policy, args.trace_dir) for task_id in args.tasks]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
