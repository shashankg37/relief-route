from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any

from openai import OpenAI
from openenv.core import GenericEnvClient

from relief_route_env.baseline import heuristic_dispatch_action
from relief_route_env.models import ReliefRouteAction, ReliefRouteObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "relief-route-openenv")
TASK_NAME = os.getenv("RELIEF_ROUTE_TASK", "easy")
BENCHMARK = os.getenv("RELIEF_ROUTE_BENCHMARK", "relief-route")
SCENARIO_SEED = os.getenv("RELIEF_ROUTE_SEED")
SPACE_ID = os.getenv("RELIEF_ROUTE_SPACE_ID", "shash37/relief-route")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MAX_STEPS = _env_int("RELIEF_ROUTE_MAX_STEPS", 12)
TEMPERATURE = _env_float("RELIEF_ROUTE_TEMPERATURE", 0.2)
MAX_TOKENS = _env_int("RELIEF_ROUTE_MAX_TOKENS", 500)
SUCCESS_SCORE_THRESHOLD = _env_float("RELIEF_ROUTE_SUCCESS_THRESHOLD", 0.6)


async def create_env_client() -> GenericEnvClient:
    env_vars = {"RELIEF_ROUTE_TASK": TASK_NAME}
    try:
        return await GenericEnvClient.from_docker_image(
            LOCAL_IMAGE_NAME,
            env_vars=env_vars,
        )
    except Exception as docker_error:
        docker_message = str(docker_error).lower()
        if "docker" not in docker_message:
            raise
    return await GenericEnvClient.from_env(
        SPACE_ID,
        use_docker=False,
        env_vars=env_vars,
    )

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are coordinating humanitarian logistics across disaster-hit and conflict-affected civilian zones.
    Return exactly one JSON object with this schema:
    {
      "commands": [
        {
          "vehicle_id": "string",
          "command_type": "dispatch" | "wait",
          "destination_zone_id": "string or null",
          "supply_type": "water" | "food" | "medicine" | null,
          "quantity": integer
        }
      ]
    }

    Rules:
    - One command per available vehicle.
    - Use only zone ids and vehicle ids present in the observation.
    - If waiting, set only vehicle_id and command_type=wait.
    - Do not include markdown, prose, code fences, or explanations.
    - Focus on urgent medicine first, then food/water fulfillment, then safety and deadlines.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={reward_text}", flush=True)


def compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            text = matches[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    return json.loads(text[start : end + 1])


def fallback_action(observation_dict: dict[str, Any]) -> ReliefRouteAction:
    observation = ReliefRouteObservation.model_validate(observation_dict)
    return heuristic_dispatch_action(observation)


def build_user_prompt(step: int, observation_dict: dict[str, Any], history: list[str]) -> str:
    info = observation_dict.get("info", {})
    summary = {
        "task_id": observation_dict.get("task_id"),
        "scenario_theme": observation_dict.get("scenario_theme"),
        "current_step": observation_dict.get("current_step"),
        "max_steps": observation_dict.get("max_steps"),
        "depot_inventory": observation_dict.get("depot_inventory"),
        "vehicles": observation_dict.get("vehicles", []),
        "zones": observation_dict.get("zones", []),
        "weighted_fulfillment": info.get("weighted_fulfillment"),
        "on_time_coverage": info.get("on_time_coverage"),
        "safety_score": info.get("safety_score"),
        "invalid_action_reasons": info.get("invalid_action_reasons", []),
        "delivered_this_step": info.get("delivered_this_step", []),
    }
    recent_history = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Recent history:
        {recent_history}

        Current observation summary:
        {json.dumps(summary, indent=2)}

        Return the next action as raw JSON only.
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    observation_dict: dict[str, Any],
    history: list[str],
) -> ReliefRouteAction:
    user_prompt = build_user_prompt(step, observation_dict, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = extract_json(content)
        return ReliefRouteAction.model_validate(parsed)
    except Exception:
        return fallback_action(observation_dict)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env: GenericEnvClient | None = None
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    startup_error: str | None = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await create_env_client()
        reset_kwargs: dict[str, Any] = {}
        if SCENARIO_SEED is not None:
            try:
                reset_kwargs["seed"] = int(SCENARIO_SEED)
            except ValueError:
                pass
        result = await env.reset(**reset_kwargs)

        for step in range(1, MAX_STEPS + 1):
            observation_dict = result.observation or {}
            info = observation_dict.get("info", {})
            final_score = float(info.get("final_score", final_score) or 0.0)
            if result.done:
                break

            action = get_model_action(client, step, observation_dict, history)
            action_payload = action.model_dump(exclude_none=True)
            result = await env.step(action_payload)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            observation_dict = result.observation or {}
            info = observation_dict.get("info", {})
            final_score = float(info.get("final_score", final_score) or 0.0)
            invalid_reasons = info.get("invalid_action_reasons") or []
            error = invalid_reasons[0] if invalid_reasons else None
            action_str = compact_json(action_payload)

            log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)
            history.append(f"step {step}: action={action_str} reward={reward:.2f} done={str(result.done).lower()}")

            if result.done:
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        startup_error = str(exc)
        success = False
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass
        if startup_error:
            log_step(
                step=max(steps_taken, 1),
                action="startup",
                reward=0.0,
                done=True,
                error=startup_error,
            )
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as exc:
        try:
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="startup", reward=0.0, done=True, error=str(exc))
            log_end(success=False, steps=0, rewards=[])
        finally:
            raise SystemExit(0)
