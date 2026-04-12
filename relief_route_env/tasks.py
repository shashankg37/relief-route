"""Task definitions for ReliefRoute."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import SupplyLedger


class VehicleTemplate(BaseModel):
    vehicle_id: str
    capacity: int = Field(ge=1)


class ZoneTemplate(BaseModel):
    zone_id: str
    display_name: str
    priority: int = Field(ge=1, le=3)
    deadline_step: int = Field(ge=1)
    travel_time: int = Field(ge=1)
    route_risk: float = Field(default=0.0001, ge=0.0001, le=0.9999)
    conflict_affected: bool = False
    access_window_start: int = Field(default=0, ge=0)
    access_window_end: int | None = Field(default=None, ge=0)
    checkpoint_delay: int = Field(default=0, ge=0)
    route_reopens_at_step: int | None = Field(default=None, ge=0)
    displaced_population: int = Field(default=0, ge=0)
    demand: SupplyLedger = Field(default_factory=SupplyLedger)


class TaskConfig(BaseModel):
    task_id: str
    scenario_theme: str
    max_steps: int
    depot_inventory: SupplyLedger
    vehicles: list[VehicleTemplate]
    zones: list[ZoneTemplate]
    prompt: str


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        scenario_theme="disaster",
        max_steps=6,
        depot_inventory=SupplyLedger(water=6, food=6, medicine=4),
        vehicles=[VehicleTemplate(vehicle_id="van_1", capacity=3)],
        zones=[
            ZoneTemplate(
                zone_id="clinic",
                display_name="Field Clinic",
                priority=3,
                deadline_step=2,
                travel_time=1,
                displaced_population=60,
                demand=SupplyLedger(medicine=3, water=1),
            ),
            ZoneTemplate(
                zone_id="shelter_a",
                display_name="Shelter A",
                priority=2,
                deadline_step=4,
                travel_time=2,
                displaced_population=120,
                demand=SupplyLedger(food=2, water=2),
            ),
            ZoneTemplate(
                zone_id="village_east",
                display_name="Village East",
                priority=1,
                deadline_step=5,
                travel_time=2,
                displaced_population=80,
                demand=SupplyLedger(food=1, water=2),
            ),
        ],
        prompt="Coordinate humanitarian deliveries after a flood. Prioritize medicine and urgent civilian needs before deadlines.",
    ),
    "medium": TaskConfig(
        task_id="medium",
        scenario_theme="disaster",
        max_steps=8,
        depot_inventory=SupplyLedger(water=11, food=10, medicine=6),
        vehicles=[
            VehicleTemplate(vehicle_id="van_1", capacity=3),
            VehicleTemplate(vehicle_id="truck_1", capacity=4),
        ],
        zones=[
            ZoneTemplate(
                zone_id="clinic",
                display_name="Regional Clinic",
                priority=3,
                deadline_step=3,
                travel_time=1,
                displaced_population=90,
                demand=SupplyLedger(medicine=4, water=2),
            ),
            ZoneTemplate(
                zone_id="school_shelter",
                display_name="School Shelter",
                priority=2,
                deadline_step=4,
                travel_time=2,
                displaced_population=180,
                demand=SupplyLedger(food=3, water=3),
            ),
            ZoneTemplate(
                zone_id="bridge_camp",
                display_name="Bridge Camp",
                priority=2,
                deadline_step=5,
                travel_time=3,
                route_reopens_at_step=2,
                displaced_population=140,
                demand=SupplyLedger(food=2, water=3, medicine=1),
            ),
            ZoneTemplate(
                zone_id="hill_village",
                display_name="Hill Village",
                priority=1,
                deadline_step=6,
                travel_time=2,
                displaced_population=110,
                demand=SupplyLedger(food=2, water=2),
            ),
        ],
        prompt="Manage humanitarian logistics after an earthquake with damaged roads and competing urgent needs.",
    ),
    "hard": TaskConfig(
        task_id="hard",
        scenario_theme="disaster_and_conflict",
        max_steps=10,
        depot_inventory=SupplyLedger(water=14, food=12, medicine=8),
        vehicles=[
            VehicleTemplate(vehicle_id="van_1", capacity=3),
            VehicleTemplate(vehicle_id="truck_1", capacity=4),
            VehicleTemplate(vehicle_id="truck_2", capacity=4),
        ],
        zones=[
            ZoneTemplate(
                zone_id="trauma_center",
                display_name="Trauma Center",
                priority=3,
                deadline_step=2,
                travel_time=1,
                    route_risk=0.2,
                conflict_affected=True,
                access_window_start=0,
                access_window_end=4,
                checkpoint_delay=1,
                displaced_population=75,
                demand=SupplyLedger(medicine=5, water=2),
            ),
            ZoneTemplate(
                zone_id="border_shelter",
                display_name="Border Shelter",
                priority=3,
                deadline_step=4,
                travel_time=2,
                    route_risk=0.35,
                conflict_affected=True,
                access_window_start=1,
                access_window_end=7,
                checkpoint_delay=1,
                displaced_population=220,
                demand=SupplyLedger(food=3, water=4, medicine=1),
            ),
            ZoneTemplate(
                zone_id="old_town",
                display_name="Old Town",
                priority=2,
                deadline_step=5,
                travel_time=2,
                    route_risk=0.6,
                conflict_affected=True,
                access_window_start=2,
                access_window_end=6,
                checkpoint_delay=2,
                route_reopens_at_step=3,
                displaced_population=150,
                demand=SupplyLedger(food=3, water=2, medicine=2),
            ),
            ZoneTemplate(
                zone_id="north_camp",
                display_name="North Camp",
                priority=2,
                deadline_step=6,
                travel_time=3,
                    route_risk=0.15,
                displaced_population=170,
                demand=SupplyLedger(food=2, water=3),
            ),
            ZoneTemplate(
                zone_id="river_crossing",
                display_name="River Crossing",
                priority=1,
                deadline_step=8,
                travel_time=3,
                    route_risk=0.4,
                conflict_affected=True,
                access_window_start=4,
                access_window_end=9,
                checkpoint_delay=1,
                displaced_population=95,
                demand=SupplyLedger(food=2, water=2, medicine=1),
            ),
        ],
        prompt="Coordinate humanitarian aid across disaster and conflict-affected zones with access windows, checkpoint delays, and risky corridors.",
    ),
}


def get_task(task_id: str) -> TaskConfig:
    try:
        return TASKS[task_id].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc
