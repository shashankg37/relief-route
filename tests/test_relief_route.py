from relief_route_env.baseline import heuristic_dispatch_action
from relief_route_env.models import CommandType, DispatchCommand, ReliefRouteAction, SupplyType
from relief_route_env.tasks import TASKS
from server.relief_route_environment import ReliefRouteEnvironment


def test_reset_returns_structured_observation() -> None:
    env = ReliefRouteEnvironment(task_id="easy")
    observation = env.reset()
    assert observation.task_id == "easy"
    assert observation.current_step == 0
    assert len(observation.zones) == 3


def test_task_catalog_has_three_levels() -> None:
    assert set(TASKS.keys()) == {"easy", "medium", "hard", "expert"}
    assert TASKS["hard"].scenario_theme == "disaster_and_conflict"
    assert TASKS["expert"].scenario_theme == "disaster_and_conflict"


def test_baseline_is_deterministic_on_medium() -> None:
    env_a = ReliefRouteEnvironment(task_id="medium")
    env_b = ReliefRouteEnvironment(task_id="medium")

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    while not obs_a.done and not obs_b.done:
        action_a = heuristic_dispatch_action(obs_a)
        action_b = heuristic_dispatch_action(obs_b)
        assert action_a.model_dump() == action_b.model_dump()
        obs_a = env_a.step(action_a)
        obs_b = env_b.step(action_b)

    assert obs_a.info.final_score == obs_b.info.final_score


def test_invalid_dispatch_is_penalized() -> None:
    env = ReliefRouteEnvironment(task_id="easy")
    env.reset()
    invalid_action = ReliefRouteAction(
        commands=[
            DispatchCommand(
                vehicle_id="van_1",
                command_type=CommandType.DISPATCH,
                destination_zone_id="clinic",
                supply_type="medicine",
                quantity=99,
            )
        ]
    )
    observation = env.step(invalid_action)
    assert observation.info.invalid_action_count >= 1
    assert observation.reward_breakdown.penalty < 0
    assert any("exceeds vehicle capacity" in reason for reason in observation.info.invalid_action_reasons)


def test_final_score_is_normalized() -> None:
    env = ReliefRouteEnvironment(task_id="expert")
    observation = env.reset()
    while not observation.done:
        observation = env.step(heuristic_dispatch_action(observation))
    assert 0.0 <= observation.info.final_score <= 1.0


def test_trace_capture_contains_steps() -> None:
    env = ReliefRouteEnvironment(task_id="easy")
    observation = env.reset()
    while not observation.done:
        observation = env.step(heuristic_dispatch_action(observation))
    trace = env.get_trace(policy="heuristic")
    assert trace.task_id == "easy"
    assert trace.policy == "heuristic"
    assert len(trace.steps) > 0
    assert trace.steps[-1].observation.done is True


def test_closed_route_is_rejected_before_reopen() -> None:
    env = ReliefRouteEnvironment(task_id="medium")
    env.reset()
    action = ReliefRouteAction(
        commands=[
            DispatchCommand(
                vehicle_id="truck_1",
                command_type=CommandType.DISPATCH,
                destination_zone_id="bridge_camp",
                supply_type="water",
                quantity=3,
            )
        ]
    )
    observation = env.step(action)
    assert observation.info.invalid_action_count >= 1
    assert any("route to bridge_camp is closed" in reason for reason in observation.info.invalid_action_reasons)


def test_access_window_failure_returns_reason() -> None:
    env = ReliefRouteEnvironment(task_id="hard")
    env.reset()
    action = ReliefRouteAction(
        commands=[
            DispatchCommand(
                vehicle_id="van_1",
                command_type=CommandType.DISPATCH,
                destination_zone_id="border_shelter",
                supply_type="water",
                quantity=2,
            )
        ]
    )
    observation = env.step(action)
    assert observation.info.invalid_action_count >= 1
    assert any("access window to border_shelter is closed" in reason for reason in observation.info.invalid_action_reasons)


def test_inventory_exhausted_completion_path() -> None:
    env = ReliefRouteEnvironment(task_id="easy")
    env.reset()
    env._state.depot_inventory.water = 0
    env._state.depot_inventory.food = 0
    env._state.depot_inventory.medicine = 0
    observation = env.step(ReliefRouteAction(commands=[]))
    assert observation.done is True
    assert observation.metadata["completion_reason"] == "inventory_exhausted"


def test_demand_met_completion_path() -> None:
    env = ReliefRouteEnvironment(task_id="easy")
    env.reset()
    for zone in env._state.zones:
        for supply_type in SupplyType:
            zone.delivered.set(supply_type, zone.demand.get(supply_type))
    observation = env.step(ReliefRouteAction(commands=[]))
    assert observation.done is True
    assert observation.metadata["completion_reason"] == "demand_met"


def test_expert_task_has_more_operational_pressure() -> None:
    expert = TASKS["expert"]
    assert expert.max_steps > TASKS["hard"].max_steps
    assert len(expert.vehicles) > len(TASKS["hard"].vehicles)
    assert any(zone.route_risk_score >= 0.5 for zone in expert.zones)
