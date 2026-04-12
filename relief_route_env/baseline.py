"""Baseline policies for ReliefRoute."""

from __future__ import annotations

from random import Random

from .models import (
    CommandType,
    DispatchCommand,
    ReliefRouteAction,
    ReliefRouteObservation,
    SupplyType,
    VehicleState,
    ZoneState,
)


SUPPLY_WEIGHTS = {
    SupplyType.WATER: 1.0,
    SupplyType.FOOD: 1.1,
    SupplyType.MEDICINE: 1.5,
}


def _zone_score(zone: ZoneState, supply_type: SupplyType, current_step: int) -> float:
    remaining = zone.remaining(supply_type)
    if remaining <= 0:
        return -1.0
    if not zone.route_is_open(current_step):
        return -1.0
    if not zone.access_is_open(current_step):
        return -1.0

    deadline_pressure = max(1.0, float(zone.deadline_step - current_step))
    score = (remaining * SUPPLY_WEIGHTS[supply_type] * float(zone.priority)) / deadline_pressure
    score = score / float(zone.travel_time + zone.checkpoint_delay)
    score -= zone.route_risk * 0.8
    if zone.conflict_affected and supply_type == SupplyType.MEDICINE:
        score += 0.9
    if zone.priority == 3:
        score += 0.4
    return score


def _choose_zone(
    vehicle: VehicleState,
    observation: ReliefRouteObservation,
    used_zones: set[str],
) -> tuple[ZoneState | None, SupplyType | None, int]:
    best: tuple[float, ZoneState | None, SupplyType | None, int] = (-1.0, None, None, 0)

    for zone in observation.zones:
        if zone.zone_id in used_zones:
            continue
        for supply_type in SupplyType:
            quantity = min(
                vehicle.capacity,
                zone.remaining(supply_type),
                observation.depot_inventory.get(supply_type),
            )
            if quantity <= 0:
                continue
            score = _zone_score(zone, supply_type, observation.current_step)
            if quantity == vehicle.capacity:
                score += 0.1
            if score > best[0]:
                best = (score, zone, supply_type, quantity)
    return best[1], best[2], best[3]


def heuristic_dispatch_action(observation: ReliefRouteObservation) -> ReliefRouteAction:
    commands: list[DispatchCommand] = []
    used_zones: set[str] = set()
    available_vehicles = [vehicle for vehicle in observation.vehicles if vehicle.available]
    available_vehicles.sort(key=lambda item: (-item.capacity, item.vehicle_id))

    for vehicle in available_vehicles:
        zone, supply_type, quantity = _choose_zone(vehicle, observation, used_zones)
        if zone is None or supply_type is None or quantity <= 0:
            commands.append(
                DispatchCommand(vehicle_id=vehicle.vehicle_id, command_type=CommandType.WAIT)
            )
            continue

        commands.append(
            DispatchCommand(
                vehicle_id=vehicle.vehicle_id,
                command_type=CommandType.DISPATCH,
                destination_zone_id=zone.zone_id,
                supply_type=supply_type,
                quantity=quantity,
            )
        )
        used_zones.add(zone.zone_id)

    return ReliefRouteAction(commands=commands)


def greedy_priority_dispatch_action(observation: ReliefRouteObservation) -> ReliefRouteAction:
    commands: list[DispatchCommand] = []
    used_zones: set[str] = set()
    available_vehicles = [vehicle for vehicle in observation.vehicles if vehicle.available]
    available_vehicles.sort(key=lambda item: (-item.capacity, item.vehicle_id))

    zones = sorted(
        observation.zones,
        key=lambda zone: (-zone.priority, zone.deadline_step, zone.travel_time, zone.zone_id),
    )

    for vehicle in available_vehicles:
        chosen_zone: ZoneState | None = None
        chosen_supply: SupplyType | None = None
        chosen_qty = 0

        for zone in zones:
            if zone.zone_id in used_zones:
                continue
            if not zone.route_is_open(observation.current_step):
                continue
            if not zone.access_is_open(observation.current_step):
                continue
            for supply_type in (SupplyType.MEDICINE, SupplyType.WATER, SupplyType.FOOD):
                quantity = min(
                    vehicle.capacity,
                    zone.remaining(supply_type),
                    observation.depot_inventory.get(supply_type),
                )
                if quantity > 0:
                    chosen_zone = zone
                    chosen_supply = supply_type
                    chosen_qty = quantity
                    break
            if chosen_zone is not None:
                break

        if chosen_zone is None or chosen_supply is None:
            commands.append(DispatchCommand(vehicle_id=vehicle.vehicle_id, command_type=CommandType.WAIT))
            continue

        commands.append(
            DispatchCommand(
                vehicle_id=vehicle.vehicle_id,
                command_type=CommandType.DISPATCH,
                destination_zone_id=chosen_zone.zone_id,
                supply_type=chosen_supply,
                quantity=chosen_qty,
            )
        )
        used_zones.add(chosen_zone.zone_id)

    return ReliefRouteAction(commands=commands)


def random_dispatch_action(observation: ReliefRouteObservation, rng: Random) -> ReliefRouteAction:
    commands: list[DispatchCommand] = []
    available_vehicles = [vehicle for vehicle in observation.vehicles if vehicle.available]
    available_vehicles.sort(key=lambda item: item.vehicle_id)

    for vehicle in available_vehicles:
        candidate_commands: list[DispatchCommand] = [
            DispatchCommand(vehicle_id=vehicle.vehicle_id, command_type=CommandType.WAIT)
        ]
        for zone in observation.zones:
            if not zone.route_is_open(observation.current_step):
                continue
            if not zone.access_is_open(observation.current_step):
                continue
            for supply_type in SupplyType:
                quantity = min(
                    vehicle.capacity,
                    zone.remaining(supply_type),
                    observation.depot_inventory.get(supply_type),
                )
                if quantity > 0:
                    candidate_commands.append(
                        DispatchCommand(
                            vehicle_id=vehicle.vehicle_id,
                            command_type=CommandType.DISPATCH,
                            destination_zone_id=zone.zone_id,
                            supply_type=supply_type,
                            quantity=quantity,
                        )
                    )
        commands.append(rng.choice(candidate_commands))

    return ReliefRouteAction(commands=commands)
