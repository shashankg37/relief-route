"""Humanitarian logistics environment for OpenEnv."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from relief_route_env.grader import score_episode
from relief_route_env.models import (
    CommandType,
    DispatchCommand,
    EpisodeTrace,
    ReliefRouteAction,
    ReliefRouteInfo,
    ReliefRouteObservation,
    ReliefRouteState,
    RewardBreakdown,
    SupplyLedger,
    SupplyType,
    TraceStep,
    VehicleState,
    ZoneState,
)
from relief_route_env.tasks import TaskConfig, VehicleTemplate, ZoneTemplate, get_task


SUPPLY_WEIGHTS = {
    SupplyType.WATER: 1.0,
    SupplyType.FOOD: 1.1,
    SupplyType.MEDICINE: 1.5,
}


class ReliefRouteEnvironment(Environment[ReliefRouteAction, ReliefRouteObservation, ReliefRouteState]):
    """Deterministic humanitarian logistics benchmark with dense reward shaping."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self.task_config = get_task(task_id)
        self._initial_observation: ReliefRouteObservation | None = None
        self._trace_steps: list[TraceStep] = []
        self._state = ReliefRouteState(
            episode_id=str(uuid4()),
            task_id=task_id,
            scenario_theme=self.task_config.scenario_theme,  # type: ignore[arg-type]
            max_steps=self.task_config.max_steps,
        )
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> ReliefRouteObservation:
        task_id = kwargs.get("task_id")
        if isinstance(task_id, str) and task_id != self.task_config.task_id:
            self.task_config = get_task(task_id)
        self._state = ReliefRouteState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self.task_config.task_id,
            scenario_theme=self.task_config.scenario_theme,  # type: ignore[arg-type]
            current_step=0,
            max_steps=self.task_config.max_steps,
            depot_inventory=self.task_config.depot_inventory.model_copy(deep=True),
            vehicles=[self._vehicle_from_template(vehicle) for vehicle in self.task_config.vehicles],
            zones=[self._zone_from_template(zone) for zone in self.task_config.zones],
            weighted_total_demand=self._compute_weighted_total_demand(),
            done=False,
            completion_reason="running",
        )
        weighted_fulfillment, on_time_coverage, efficiency_score, safety_score = self._current_metrics()
        self._state.last_score = score_episode(
            weighted_fulfillment,
            on_time_coverage,
            efficiency_score,
            safety_score,
        )
        observation = self._build_observation(RewardBreakdown(), [], 0, 0, [])
        self._initial_observation = observation.model_copy(deep=True)
        self._trace_steps = []
        return observation

    def step(
        self,
        action: ReliefRouteAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> ReliefRouteObservation:
        if self._state.done:
            return self._build_observation(RewardBreakdown(), [], 0, 0, [])

        commands, invalid_count, invalid_reasons = self._normalize_commands(action)
        reward_breakdown = RewardBreakdown()
        idle_vehicle_count = 0

        available_vehicle_ids = {
            vehicle.vehicle_id for vehicle in self._state.vehicles if vehicle.available
        }

        if self._has_remaining_demand():
            for vehicle_id in available_vehicle_ids:
                command = commands[vehicle_id]
                if command.command_type == CommandType.WAIT:
                    idle_vehicle_count += 1
                    continue
                dispatch_valid, invalid_reason = self._apply_dispatch(command)
                if not dispatch_valid:
                    invalid_count += 1
                    if invalid_reason is not None:
                        invalid_reasons.append(invalid_reason)
                    idle_vehicle_count += 1

        completed_reward, delivered_this_step = self._advance_vehicles()
        reward_breakdown.fulfillment_reward += completed_reward["fulfillment"]
        reward_breakdown.urgency_reward += completed_reward["urgency"]
        reward_breakdown.timeliness_reward += completed_reward["timeliness"]
        reward_breakdown.efficiency_reward += completed_reward["efficiency"]
        reward_breakdown.safety_reward += completed_reward["safety"]

        self._state.current_step += 1
        self._state.step_count += 1
        self._state.invalid_action_count += invalid_count
        self._state.idle_vehicle_count += idle_vehicle_count

        reward_breakdown.penalty -= self._backlog_penalty()
        reward_breakdown.penalty -= invalid_count * 0.35
        reward_breakdown.penalty -= idle_vehicle_count * 0.08

        previous_score = self._state.last_score
        self._update_completion_status()
        weighted_fulfillment, on_time_coverage, efficiency_score, safety_score = self._current_metrics()
        final_score = score_episode(weighted_fulfillment, on_time_coverage, efficiency_score, safety_score)
        self._state.last_score = final_score
        progress_delta = final_score - previous_score
        reward_breakdown.progress_reward += max(-0.1, min(0.24, progress_delta * 1.6))

        if self._state.done:
            reward_breakdown.terminal_reward = final_score * 2.0

        normalized_reward = self._normalized_reward(self._total_reward(reward_breakdown))
        observation = self._build_observation(
            reward_breakdown,
            delivered_this_step,
            invalid_count,
            idle_vehicle_count,
            invalid_reasons,
            normalized_reward,
        )
        self._trace_steps.append(
            TraceStep(
                step_index=len(self._trace_steps) + 1,
                action=action.model_copy(deep=True),
                observation=observation.model_copy(deep=True),
                total_reward=normalized_reward,
            )
        )
        return observation

    @property
    def state(self) -> ReliefRouteState:
        return deepcopy(self._state)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ReliefRoute",
            description="Humanitarian aid dispatch benchmark for disaster and conflict-affected regions.",
            version="0.2.0",
            author="shash",
        )

    def get_trace(self, policy: str = "manual") -> EpisodeTrace:
        if self._initial_observation is None:
            self.reset()
        assert self._initial_observation is not None
        return EpisodeTrace(
            task_id=self._state.task_id,
            policy=policy,
            initial_observation=self._initial_observation.model_copy(deep=True),
            steps=[step.model_copy(deep=True) for step in self._trace_steps],
            final_score=round(self._state.last_score, 4),
        )

    def save_trace(self, output_path: str | Path, policy: str = "manual") -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.get_trace(policy=policy).model_dump_json(indent=2), encoding="utf-8")
        return path

    def _vehicle_from_template(self, vehicle: VehicleTemplate) -> VehicleState:
        return VehicleState(
            vehicle_id=vehicle.vehicle_id,
            capacity=vehicle.capacity,
            available=True,
            eta_remaining=0,
        )

    def _zone_from_template(self, zone: ZoneTemplate) -> ZoneState:
        return ZoneState(
            zone_id=zone.zone_id,
            display_name=zone.display_name,
            priority=zone.priority,
            deadline_step=zone.deadline_step,
            travel_time=zone.travel_time,
            route_risk_score=zone.route_risk_score,
            conflict_affected=zone.conflict_affected,
            access_window_start=zone.access_window_start,
            access_window_end=zone.access_window_end,
            checkpoint_delay=zone.checkpoint_delay,
            route_reopens_at_step=zone.route_reopens_at_step,
            displaced_population=zone.displaced_population,
            demand=zone.demand.model_copy(deep=True),
            delivered=SupplyLedger(),
        )

    def _compute_weighted_total_demand(self) -> float:
        total = 0.0
        for zone in self.task_config.zones:
            for supply_type in SupplyType:
                total += (
                    zone.demand.get(supply_type)
                    * SUPPLY_WEIGHTS[supply_type]
                    * float(zone.priority)
                )
        return total

    def _normalize_commands(
        self,
        action: ReliefRouteAction,
    ) -> tuple[dict[str, DispatchCommand], int, list[str]]:
        available_vehicles = {
            vehicle.vehicle_id: vehicle
            for vehicle in self._state.vehicles
            if vehicle.available
        }
        commands_by_vehicle: dict[str, DispatchCommand] = {}
        invalid_count = 0
        invalid_reasons: list[str] = []

        for command in action.commands:
            if command.vehicle_id not in available_vehicles:
                invalid_count += 1
                invalid_reasons.append(f"{command.vehicle_id}: vehicle is not available this turn")
                continue
            if command.vehicle_id in commands_by_vehicle:
                invalid_count += 1
                invalid_reasons.append(f"{command.vehicle_id}: duplicate command for the same vehicle")
                continue
            commands_by_vehicle[command.vehicle_id] = command

        for vehicle_id in available_vehicles:
            if vehicle_id not in commands_by_vehicle:
                commands_by_vehicle[vehicle_id] = DispatchCommand(
                    vehicle_id=vehicle_id,
                    command_type=CommandType.WAIT,
                )
        return commands_by_vehicle, invalid_count, invalid_reasons

    def _apply_dispatch(self, command: DispatchCommand) -> tuple[bool, str | None]:
        vehicle = self._get_vehicle(command.vehicle_id)
        zone = self._get_zone(command.destination_zone_id or "")
        if vehicle is None or zone is None or command.supply_type is None:
            return False, f"{command.vehicle_id}: target zone or supply selection is invalid"
        if not vehicle.available:
            return False, f"{command.vehicle_id}: vehicle is already in transit"
        if not zone.route_is_open(self._state.current_step):
            return False, f"{command.vehicle_id}: route to {zone.zone_id} is closed right now"
        if not zone.access_is_open(self._state.current_step):
            return False, f"{command.vehicle_id}: access window to {zone.zone_id} is closed"
        if command.quantity > vehicle.capacity:
            return False, f"{command.vehicle_id}: quantity exceeds vehicle capacity"
        if self._state.depot_inventory.get(command.supply_type) < command.quantity:
            return False, f"{command.vehicle_id}: insufficient {command.supply_type.value} inventory at depot"
        if zone.remaining(command.supply_type) <= 0:
            return False, f"{command.vehicle_id}: {zone.display_name} does not need more {command.supply_type.value}"

        self._state.depot_inventory.remove(command.supply_type, command.quantity)
        vehicle.available = False
        vehicle.eta_remaining = zone.travel_time + zone.checkpoint_delay
        vehicle.destination_zone_id = zone.zone_id
        vehicle.cargo_supply_type = command.supply_type
        vehicle.cargo_quantity = command.quantity
        self._state.total_dispatched_units += command.quantity
        self._state.cumulative_risk_exposure += zone.route_risk_score * command.quantity
        if zone.route_risk_score >= 0.5:
            self._state.unsafe_dispatches += 1
        return True, None

    def _advance_vehicles(self) -> tuple[dict[str, float], list[str]]:
        reward = {
            "fulfillment": 0.0,
            "urgency": 0.0,
            "timeliness": 0.0,
            "efficiency": 0.0,
            "safety": 0.0,
        }
        delivered_messages: list[str] = []
        delivery_step = self._state.current_step + 1

        for vehicle in self._state.vehicles:
            if vehicle.available:
                continue
            vehicle.eta_remaining = max(0, vehicle.eta_remaining - 1)
            if vehicle.eta_remaining > 0:
                continue

            zone = self._get_zone(vehicle.destination_zone_id or "")
            supply_type = vehicle.cargo_supply_type
            quantity = vehicle.cargo_quantity
            if zone is not None and supply_type is not None and quantity > 0:
                deliverable = min(quantity, zone.remaining(supply_type))
                wasted = quantity - deliverable
                if deliverable > 0:
                    zone.delivered.add(supply_type, deliverable)
                    weighted_units = deliverable * SUPPLY_WEIGHTS[supply_type] * float(zone.priority)
                    self._state.useful_delivered_units += deliverable
                    self._state.weighted_delivered += weighted_units
                    reward["fulfillment"] += deliverable * 0.22
                    reward["urgency"] += weighted_units * 0.12
                    if delivery_step <= zone.deadline_step:
                        self._state.weighted_on_time_delivered += weighted_units
                        reward["timeliness"] += weighted_units * 0.08
                    reward["efficiency"] += min(1.0, deliverable / max(vehicle.capacity, 1)) * 0.12
                    reward["safety"] += max(0.0, 0.16 - (zone.route_risk_score * 0.12))
                    delivered_messages.append(
                        f"{vehicle.vehicle_id} delivered {deliverable} {supply_type.value} to {zone.display_name}"
                    )
                if wasted > 0:
                    self._state.wasted_units += wasted

            vehicle.available = True
            vehicle.destination_zone_id = None
            vehicle.cargo_supply_type = None
            vehicle.cargo_quantity = 0

        return reward, delivered_messages

    def _backlog_penalty(self) -> float:
        penalty = 0.0
        current_time = self._state.current_step + 1
        for zone in self._state.zones:
            for supply_type in SupplyType:
                remaining = zone.remaining(supply_type)
                if remaining <= 0:
                    continue
                weight = SUPPLY_WEIGHTS[supply_type] * float(zone.priority)
                penalty += remaining * weight * 0.015
                if current_time > zone.deadline_step and zone.priority >= 2:
                    penalty += remaining * weight * 0.02
                if zone.conflict_affected and not zone.access_is_open(current_time):
                    penalty += remaining * weight * 0.01
        return penalty

    def _current_metrics(self) -> tuple[float, float, float, float]:
        if self._state.weighted_total_demand <= 0:
            return 1.0, 1.0, 1.0, 1.0

        weighted_fulfillment = min(1.0, self._state.weighted_delivered / self._state.weighted_total_demand)
        on_time_coverage = min(1.0, self._state.weighted_on_time_delivered / self._state.weighted_total_demand)

        dispatch_efficiency = 1.0
        if self._state.total_dispatched_units > 0:
            dispatch_efficiency = self._state.useful_delivered_units / float(self._state.total_dispatched_units)
        efficiency_score = max(
            0.0,
            min(
                1.0,
                dispatch_efficiency
                - (self._state.invalid_action_count * 0.03)
                - (self._state.idle_vehicle_count * 0.015),
            ),
        )

        total_dispatches = max(1, self._state.total_dispatched_units)
        average_risk = self._state.cumulative_risk_exposure / float(total_dispatches)
        unsafe_penalty = (self._state.unsafe_dispatches * 0.25) / total_dispatches
        safety_score = max(0.0, min(1.0, 1.0 - (average_risk * 0.8) - unsafe_penalty))
        return weighted_fulfillment, on_time_coverage, efficiency_score, safety_score

    def _update_completion_status(self) -> None:
        if not self._has_remaining_demand():
            self._state.done = True
            self._state.completion_reason = "demand_met"
            return
        if self._state.current_step >= self._state.max_steps:
            self._state.done = True
            self._state.completion_reason = "time_limit"
            return
        if self._state.depot_inventory.water == 0 and self._state.depot_inventory.food == 0 and self._state.depot_inventory.medicine == 0:
            if all(vehicle.available for vehicle in self._state.vehicles):
                self._state.done = True
                self._state.completion_reason = "inventory_exhausted"

    def _build_observation(
        self,
        reward_breakdown: RewardBreakdown,
        delivered_this_step: list[str],
        invalid_action_count: int,
        idle_vehicle_count: int,
        invalid_action_reasons: list[str],
        normalized_reward: float | None = None,
    ) -> ReliefRouteObservation:
        weighted_fulfillment, on_time_coverage, efficiency_score, safety_score = self._current_metrics()
        current_score = score_episode(
            weighted_fulfillment,
            on_time_coverage,
            efficiency_score,
            safety_score,
        )
        self._state.last_score = current_score
        unmet_critical_zones = [
            zone.display_name
            for zone in self._state.zones
            if zone.priority >= 3 and any(zone.remaining(supply_type) > 0 for supply_type in SupplyType)
        ]
        info = ReliefRouteInfo(
            task_id=self._state.task_id,
            scenario_theme=self._state.scenario_theme,
            completed_deliveries=len(delivered_this_step),
            invalid_action_count=invalid_action_count,
            idle_vehicle_count=idle_vehicle_count,
            weighted_fulfillment=round(weighted_fulfillment, 4),
            on_time_coverage=round(on_time_coverage, 4),
            efficiency_score=round(efficiency_score, 4),
            safety_score=round(safety_score, 4),
            final_score=round(current_score, 4),
            unmet_critical_zones=unmet_critical_zones,
            delivered_this_step=delivered_this_step,
            invalid_action_reasons=invalid_action_reasons,
        )
        if normalized_reward is None:
            normalized_reward = self._normalized_reward(self._total_reward(reward_breakdown))

        return ReliefRouteObservation(
            done=self._state.done,
            reward=round(normalized_reward, 4),
            task_id=self._state.task_id,
            scenario_theme=self._state.scenario_theme,
            current_step=self._state.current_step,
            max_steps=self._state.max_steps,
            depot_inventory=self._state.depot_inventory.model_copy(deep=True),
            vehicles=[vehicle.model_copy(deep=True) for vehicle in self._state.vehicles],
            zones=[zone.model_copy(deep=True) for zone in self._state.zones],
            prompt=self.task_config.prompt,
            reward_breakdown=reward_breakdown,
            info=info,
            metadata={"completion_reason": self._state.completion_reason},
        )

    def _total_reward(self, reward_breakdown: RewardBreakdown) -> float:
        return (
            reward_breakdown.fulfillment_reward
            + reward_breakdown.urgency_reward
            + reward_breakdown.timeliness_reward
            + reward_breakdown.efficiency_reward
            + reward_breakdown.safety_reward
            + reward_breakdown.progress_reward
            + reward_breakdown.terminal_reward
            + reward_breakdown.penalty
        )

    def _normalized_reward(self, raw_reward: float) -> float:
        # Keep per-step reward within the hackathon's requested 0.0-1.0 range
        # while preserving ordering between worse and better actions.
        return max(0.0, min(1.0, raw_reward / 4.0))

    def _has_remaining_demand(self) -> bool:
        return any(
            zone.remaining(supply_type) > 0
            for zone in self._state.zones
            for supply_type in SupplyType
        )

    def _get_vehicle(self, vehicle_id: str) -> VehicleState | None:
        for vehicle in self._state.vehicles:
            if vehicle.vehicle_id == vehicle_id:
                return vehicle
        return None

    def _get_zone(self, zone_id: str) -> ZoneState | None:
        for zone in self._state.zones:
            if zone.zone_id == zone_id:
                return zone
        return None
