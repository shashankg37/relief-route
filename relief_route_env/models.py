"""Typed models for the ReliefRoute environment."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class SupplyType(str, Enum):
    WATER = "water"
    FOOD = "food"
    MEDICINE = "medicine"


class CommandType(str, Enum):
    DISPATCH = "dispatch"
    WAIT = "wait"


class SupplyLedger(BaseModel):
    water: int = Field(default=0, ge=0)
    food: int = Field(default=0, ge=0)
    medicine: int = Field(default=0, ge=0)

    def get(self, supply_type: SupplyType) -> int:
        return getattr(self, supply_type.value)

    def set(self, supply_type: SupplyType, value: int) -> None:
        setattr(self, supply_type.value, max(0, value))

    def add(self, supply_type: SupplyType, quantity: int) -> None:
        self.set(supply_type, self.get(supply_type) + quantity)

    def remove(self, supply_type: SupplyType, quantity: int) -> None:
        self.set(supply_type, self.get(supply_type) - quantity)


class ZoneState(BaseModel):
    zone_id: str
    display_name: str
    priority: int = Field(ge=1, le=3)
    deadline_step: int = Field(ge=1)
    travel_time: int = Field(ge=1)
    route_risk_score: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    conflict_affected: bool = False
    access_window_start: int = Field(default=0, ge=0)
    access_window_end: int | None = Field(default=None, ge=0)
    checkpoint_delay: int = Field(default=0, ge=0)
    route_reopens_at_step: int | None = Field(default=None, ge=0)
    displaced_population: int = Field(default=0, ge=0)
    demand: SupplyLedger = Field(default_factory=SupplyLedger)
    delivered: SupplyLedger = Field(default_factory=SupplyLedger)

    def route_is_open(self, current_step: int) -> bool:
        return self.route_reopens_at_step is None or current_step >= self.route_reopens_at_step

    def access_is_open(self, current_step: int) -> bool:
        if current_step < self.access_window_start:
            return False
        if self.access_window_end is not None and current_step > self.access_window_end:
            return False
        return True

    def remaining(self, supply_type: SupplyType) -> int:
        return max(0, self.demand.get(supply_type) - self.delivered.get(supply_type))


class VehicleState(BaseModel):
    vehicle_id: str
    capacity: int = Field(ge=1)
    available: bool = True
    eta_remaining: int = Field(default=0, ge=0)
    destination_zone_id: str | None = None
    cargo_supply_type: SupplyType | None = None
    cargo_quantity: int = Field(default=0, ge=0)


class DispatchCommand(BaseModel):
    vehicle_id: str
    command_type: CommandType = CommandType.WAIT
    destination_zone_id: str | None = None
    supply_type: SupplyType | None = None
    quantity: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_command(self) -> "DispatchCommand":
        if self.command_type == CommandType.DISPATCH:
            if self.destination_zone_id is None:
                raise ValueError("dispatch commands require a destination_zone_id")
            if self.supply_type is None:
                raise ValueError("dispatch commands require a supply_type")
            if self.quantity <= 0:
                raise ValueError("dispatch commands require quantity > 0")
        else:
            self.destination_zone_id = None
            self.supply_type = None
            self.quantity = 0
        return self


class ReliefRouteAction(Action):
    commands: list[DispatchCommand] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    fulfillment_reward: float = 0.0
    urgency_reward: float = 0.0
    timeliness_reward: float = 0.0
    efficiency_reward: float = 0.0
    safety_reward: float = 0.0
    progress_reward: float = 0.0
    terminal_reward: float = 0.0
    penalty: float = 0.0


class ReliefRouteInfo(BaseModel):
    task_id: str
    scenario_theme: Literal["disaster", "disaster_and_conflict"]
    completed_deliveries: int = 0
    invalid_action_count: int = 0
    idle_vehicle_count: int = 0
    weighted_fulfillment: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    on_time_coverage: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    efficiency_score: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    safety_score: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    final_score: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    unmet_critical_zones: list[str] = Field(default_factory=list)
    delivered_this_step: list[str] = Field(default_factory=list)
    invalid_action_reasons: list[str] = Field(default_factory=list)


class ReliefRouteObservation(Observation):
    task_id: str
    scenario_theme: Literal["disaster", "disaster_and_conflict"]
    current_step: int
    max_steps: int
    depot_inventory: SupplyLedger = Field(default_factory=SupplyLedger)
    vehicles: list[VehicleState] = Field(default_factory=list)
    zones: list[ZoneState] = Field(default_factory=list)
    prompt: str = ""
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    info: ReliefRouteInfo


class TraceStep(BaseModel):
    step_index: int = Field(ge=1)
    action: ReliefRouteAction
    observation: ReliefRouteObservation
    total_reward: float = 0.0


class EpisodeTrace(BaseModel):
    task_id: str
    policy: str = "manual"
    initial_observation: ReliefRouteObservation
    steps: list[TraceStep] = Field(default_factory=list)
    final_score: float = 0.0001


class ReliefRouteState(State):
    task_id: str
    scenario_theme: Literal["disaster", "disaster_and_conflict"] = "disaster"
    current_step: int = 0
    max_steps: int = 0
    depot_inventory: SupplyLedger = Field(default_factory=SupplyLedger)
    vehicles: list[VehicleState] = Field(default_factory=list)
    zones: list[ZoneState] = Field(default_factory=list)
    total_dispatched_units: int = 0
    useful_delivered_units: int = 0
    wasted_units: int = 0
    invalid_action_count: int = 0
    idle_vehicle_count: int = 0
    unsafe_dispatches: int = 0
    cumulative_risk_exposure: float = 0.0
    weighted_total_demand: float = 0.0
    weighted_delivered: float = 0.0
    weighted_on_time_delivered: float = 0.0
    last_score: float = 0.0001
    done: bool = False
    completion_reason: Literal["running", "demand_met", "time_limit", "inventory_exhausted"] = "running"
