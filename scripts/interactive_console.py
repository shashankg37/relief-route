"""Interactive terminal runner for ReliefRoute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relief_route_env.baseline import heuristic_dispatch_action
from relief_route_env.models import (
    CommandType,
    DispatchCommand,
    ReliefRouteAction,
    ReliefRouteObservation,
    SupplyType,
)
from server.relief_route_environment import ReliefRouteEnvironment


def _print_observation(observation: ReliefRouteObservation) -> None:
    print()
    print(
        f"Task: {observation.task_id} | Theme: {observation.scenario_theme} | "
        f"Step: {observation.current_step}/{observation.max_steps}"
    )
    print(f"Prompt: {observation.prompt}")
    print("Depot inventory:")
    print(
        f"  water={observation.depot_inventory.water} "
        f"food={observation.depot_inventory.food} "
        f"medicine={observation.depot_inventory.medicine}"
    )
    print("Vehicles:")
    for vehicle in observation.vehicles:
        status = "available" if vehicle.available else f"en route ({vehicle.eta_remaining} turns left)"
        cargo = ""
        if vehicle.cargo_supply_type is not None and vehicle.cargo_quantity > 0:
            cargo = f", cargo={vehicle.cargo_quantity} {vehicle.cargo_supply_type.value}"
        destination = f", dest={vehicle.destination_zone_id}" if vehicle.destination_zone_id else ""
        print(f"  {vehicle.vehicle_id}: cap={vehicle.capacity}, {status}{destination}{cargo}")
    print("Zones:")
    for zone in observation.zones:
        route_status = "open" if zone.route_is_open(observation.current_step) else f"closed until step {zone.route_reopens_at_step}"
        access_status = (
            "open" if zone.access_is_open(observation.current_step)
            else f"closed outside window {zone.access_window_start}-{zone.access_window_end}"
        )
        print(
            f"  {zone.zone_id} ({zone.display_name}) priority={zone.priority} "
            f"deadline={zone.deadline_step} travel={zone.travel_time} checkpoint={zone.checkpoint_delay} "
            f"risk={zone.route_risk_score:.2f} route={route_status} access={access_status}"
        )
        print(
            f"    remaining: water={zone.remaining(SupplyType.WATER)} "
            f"food={zone.remaining(SupplyType.FOOD)} "
            f"medicine={zone.remaining(SupplyType.MEDICINE)}"
        )
    breakdown = observation.reward_breakdown
    print(
        "Last reward breakdown: "
        f"fulfillment={breakdown.fulfillment_reward:.3f}, "
        f"urgency={breakdown.urgency_reward:.3f}, "
        f"timeliness={breakdown.timeliness_reward:.3f}, "
        f"efficiency={breakdown.efficiency_reward:.3f}, "
        f"safety={breakdown.safety_reward:.3f}, "
        f"progress={breakdown.progress_reward:.3f}, "
        f"terminal={breakdown.terminal_reward:.3f}, "
        f"penalty={breakdown.penalty:.3f}"
    )
    print(
        "Score info: "
        f"weighted_fulfillment={observation.info.weighted_fulfillment:.3f}, "
        f"on_time_coverage={observation.info.on_time_coverage:.3f}, "
        f"efficiency_score={observation.info.efficiency_score:.3f}, "
        f"safety_score={observation.info.safety_score:.3f}, "
        f"final_score={observation.info.final_score:.3f}"
    )
    if observation.info.delivered_this_step:
        print("Deliveries this step:")
        for item in observation.info.delivered_this_step:
            print(f"  - {item}")
    if observation.info.invalid_action_reasons:
        print("Invalid action reasons:")
        for item in observation.info.invalid_action_reasons:
            print(f"  - {item}")
    print()


def _print_help() -> None:
    print("Commands:")
    print("  baseline")
    print("  wait <vehicle_id>")
    print("  dispatch <vehicle_id> <zone_id> <water|food|medicine> <quantity>")
    print("  done")
    print("  state")
    print("  help")
    print("  quit")


def _parse_turn_input(observation: ReliefRouteObservation) -> ReliefRouteAction | None:
    print("Enter commands for this turn. Type 'help' for options.")
    commands: list[DispatchCommand] = []

    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered == "help":
            _print_help()
            continue
        if lowered == "state":
            _print_observation(observation)
            continue
        if lowered == "quit":
            return None
        if lowered == "baseline":
            return heuristic_dispatch_action(observation)
        if lowered == "done":
            return ReliefRouteAction(commands=commands)

        parts = raw.split()
        if parts[0].lower() == "wait" and len(parts) == 2:
            commands.append(DispatchCommand(vehicle_id=parts[1], command_type=CommandType.WAIT))
            print(f"Queued wait for {parts[1]}.")
            continue
        if parts[0].lower() == "dispatch" and len(parts) == 5:
            vehicle_id, zone_id, supply_name, quantity_text = parts[1:]
            try:
                quantity = int(quantity_text)
                command = DispatchCommand(
                    vehicle_id=vehicle_id,
                    command_type=CommandType.DISPATCH,
                    destination_zone_id=zone_id,
                    supply_type=supply_name,
                    quantity=quantity,
                )
            except Exception as exc:
                print(f"Invalid command: {exc}")
                continue
            commands.append(command)
            print(
                f"Queued dispatch: vehicle={vehicle_id}, zone={zone_id}, supply={supply_name}, quantity={quantity}"
            )
            continue

        print("Could not parse that command. Type 'help' for the expected format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play ReliefRoute interactively in the terminal.")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "expert"], default="easy")
    args = parser.parse_args()

    env = ReliefRouteEnvironment(task_id=args.task)
    observation = env.reset()

    print("ReliefRoute interactive demo")
    print("Goal: route humanitarian aid across disaster and conflict-affected regions.")
    _print_help()

    while True:
        _print_observation(observation)
        if observation.done:
            print("Episode complete.")
            break

        action = _parse_turn_input(observation)
        if action is None:
            print("Exiting demo.")
            return

        observation = env.step(action)
        total_reward = (
            observation.reward_breakdown.fulfillment_reward
            + observation.reward_breakdown.urgency_reward
            + observation.reward_breakdown.timeliness_reward
            + observation.reward_breakdown.efficiency_reward
            + observation.reward_breakdown.safety_reward
            + observation.reward_breakdown.progress_reward
            + observation.reward_breakdown.terminal_reward
            + observation.reward_breakdown.penalty
        )
        print(f"Step reward: {total_reward:.3f}")

    print(
        "Final metrics: "
        f"weighted_fulfillment={observation.info.weighted_fulfillment:.3f}, "
        f"on_time_coverage={observation.info.on_time_coverage:.3f}, "
        f"efficiency_score={observation.info.efficiency_score:.3f}, "
        f"safety_score={observation.info.safety_score:.3f}, "
        f"final_score={observation.info.final_score:.3f}"
    )


if __name__ == "__main__":
    main()
