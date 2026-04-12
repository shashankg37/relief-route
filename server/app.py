"""FastAPI entrypoint for ReliefRoute."""

from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
from threading import Lock
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import Body
from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field

from relief_route_env.baseline import greedy_priority_dispatch_action, heuristic_dispatch_action, random_dispatch_action
from relief_route_env.models import EpisodeTrace, ReliefRouteAction, ReliefRouteObservation
from server.relief_route_environment import ReliefRouteEnvironment

LEADERBOARD_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "leaderboard.json"


def _factory() -> ReliefRouteEnvironment:
    return ReliefRouteEnvironment(task_id=os.getenv("RELIEF_ROUTE_TASK", "easy"))


def _policy_action(policy: str, observation: ReliefRouteObservation, rng: Random) -> ReliefRouteAction:
    if policy == "heuristic":
        return heuristic_dispatch_action(observation)
    if policy == "greedy":
        return greedy_priority_dispatch_action(observation)
    if policy == "random":
        return random_dispatch_action(observation, rng)
    raise ValueError(f"Unknown policy: {policy}")


def _run_policy_episode(task_id: str, policy: str) -> EpisodeTrace:
    env = ReliefRouteEnvironment(task_id=task_id)
    observation = env.reset()
    rng = Random(7)
    while not observation.done:
        observation = env.step(_policy_action(policy, observation, rng))
    return env.get_trace(policy=policy)


def _load_leaderboard() -> dict[str, Any]:
    if not LEADERBOARD_PATH.exists():
        return {"entries": []}
    try:
        return json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"entries": []}


def _save_leaderboard(payload: dict[str, Any]) -> None:
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_leaderboard(task_id: str, results: list[dict[str, Any]]) -> None:
    payload = _load_leaderboard()
    entries = payload.get("entries", [])
    for result in results:
        entries.append(
            {
                "task_id": task_id,
                "policy": result["policy"],
                "final_score": result["final_score"],
                "weighted_fulfillment": result["weighted_fulfillment"],
                "on_time_coverage": result["on_time_coverage"],
                "safety_score": result["safety_score"],
                "completion_reason": result["completion_reason"],
            }
        )
    entries.sort(key=lambda item: (item["final_score"], item["weighted_fulfillment"]), reverse=True)
    payload["entries"] = entries[:18]
    _save_leaderboard(payload)


class SuggestTurnRequest(BaseModel):
    observation: ReliefRouteObservation
    policy: str = Field(default="heuristic")


class DashboardResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class DashboardSessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, ReliefRouteEnvironment] = {}

    def create(self, task_id: str, seed: int | None = None) -> tuple[str, ReliefRouteObservation]:
        env = ReliefRouteEnvironment(task_id=task_id)
        observation = env.reset(task_id=task_id, seed=seed)
        session_id = str(uuid4())
        with self._lock:
            self._sessions[session_id] = env
        return session_id, observation

    def get(self, session_id: str) -> ReliefRouteEnvironment:
        with self._lock:
            env = self._sessions.get(session_id)
        if env is None:
            raise KeyError(session_id)
        return env

    def destroy(self, session_id: str) -> None:
        with self._lock:
            env = self._sessions.pop(session_id, None)
        if env is not None:
            env.close()


DASHBOARD_SESSIONS = DashboardSessionStore()


app = create_app(
    _factory,
    ReliefRouteAction,
    ReliefRouteObservation,
    env_name="relief-route",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReliefRoute Dashboard</title>
  <style>
    :root { --bg:#0b1020; --panel:#121a31; --panel2:#17223f; --text:#eef2ff; --muted:#9aa8cf; --line:#24345f; --strong:#36579f; --danger:#ff6b6b; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Segoe UI",sans-serif; background:radial-gradient(circle at top,#12214f,var(--bg) 40%); color:var(--text); }
    .wrap { max-width:1320px; margin:0 auto; padding:20px; }
    h1,h2,h3,p { margin:0; }
    .hero { display:flex; justify-content:space-between; align-items:end; gap:24px; margin-bottom:16px; }
    .hero p { color:var(--muted); margin-top:6px; max-width:520px; line-height:1.45; }
    .panel,.controls,.metric,.zone,.vehicle,.card { background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02)); border:1px solid var(--line); border-radius:18px; box-shadow:0 16px 32px rgba(0,0,0,.18); }
    .controls { padding:14px; display:grid; grid-template-columns:1fr auto auto auto auto; gap:12px; align-items:end; margin:0 0 16px; }
    .controls label { display:flex; flex-direction:column; gap:6px; color:var(--muted); font-size:14px; }
    button,input,select,textarea { background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:12px; padding:10px 12px; font:inherit; }
    button { cursor:pointer; }
    button.primary { background:linear-gradient(135deg,#2a9d8f,#55c5a7); color:#061317; border:none; font-weight:700; }
    button.secondary { background:#1d2a4d; }
    .grid { display:grid; grid-template-columns:1.18fr 1.12fr; gap:18px; align-items:start; }
    .panel { padding:16px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
    .metric { padding:14px; }
    .label { color:var(--muted); font-size:13px; margin-bottom:6px; }
    .value { font-size:28px; font-weight:700; }
    .inventory { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
    .pill { background:rgba(255,255,255,.05); border:1px solid var(--line); border-radius:999px; padding:8px 12px; color:var(--muted); }
    .zones,.vehicles,.actions { display:grid; gap:12px; }
    .zone,.vehicle,.card,.action-card { padding:14px; }
    .zone-header,.vehicle-header { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:10px; }
    .badge { padding:4px 9px; border-radius:999px; font-size:12px; font-weight:700; }
    .priority-3 { background:rgba(255,107,107,.16); color:#ff9c9c; }
    .priority-2 { background:rgba(255,209,102,.16); color:#ffe2a1; }
    .priority-1 { background:rgba(115,224,169,.16); color:#a3f0c7; }
    .muted { color:var(--muted); }
    .danger { color:var(--danger); }
    .action-card { border-radius:14px; background:rgba(255,255,255,.03); border:1px solid var(--line); }
    .action-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:10px; }
    .helper-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:14px; }
    .helper-actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
    .trace { min-height:220px; background:#091126; border-radius:14px; padding:12px; border:1px solid var(--line); overflow:auto; }
    .trace-entry { padding:10px 0; border-bottom:1px solid rgba(255,255,255,.08); }
    .trace-entry:last-child { border-bottom:none; }
    .map-wrap { margin-bottom:16px; border:1px solid var(--strong); border-radius:18px; overflow:hidden; background:radial-gradient(circle at center,rgba(116,178,255,.26),rgba(16,28,58,.5) 38%,rgba(5,10,22,.98) 78%),linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.015)); }
    .map-legend { display:flex; flex-wrap:wrap; gap:10px; padding:10px 14px; border-bottom:1px solid #264072; color:#d7e4ff; font-size:13px; background:rgba(6,12,28,.88); }
    .legend-dot { display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; vertical-align:middle; }
    .map-stage { width:100%; height:360px; display:block; }
    .map-node-label { font-size:13px; fill:#f8fbff; font-weight:700; paint-order:stroke; stroke:rgba(5,10,22,.9); stroke-width:3px; stroke-linejoin:round; }
    .map-sub-label { font-size:11px; fill:#d5e2ff; paint-order:stroke; stroke:rgba(5,10,22,.9); stroke-width:2px; stroke-linejoin:round; }
    .map-route { stroke-width:4; fill:none; opacity:1; filter:drop-shadow(0 0 10px rgba(255,255,255,.12)); }
    .map-route.closed { stroke-dasharray:8 6; opacity:.9; }
    .map-depot { fill:#5ef0ad; stroke:#041a11; stroke-width:4; }
    .map-zone { stroke:rgba(255,255,255,.88); stroke-width:3; }
    .map-vehicle { fill:#fff5bf; stroke:#473200; stroke-width:3; }
    textarea {
      width:100%;
      min-height:360px;
      height:360px;
      resize:vertical;
      line-height:1.55;
      font-family:"Cascadia Code",Consolas,monospace;
      font-size:13px;
      white-space:pre;
      overflow:auto;
    }
    .modal { position:fixed; inset:0; display:none; align-items:center; justify-content:center; padding:24px; background:rgba(3,8,18,.72); backdrop-filter:blur(8px); z-index:1000; }
    .modal.open { display:flex; }
    .modal-card { width:min(960px,100%); max-height:88vh; overflow:auto; background:linear-gradient(180deg,rgba(20,31,60,.98),rgba(8,13,28,.98)); border:1px solid var(--strong); border-radius:22px; box-shadow:0 28px 60px rgba(0,0,0,.42); padding:20px; }
    .modal-head { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:10px; }
    .compare-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:16px; }
    .score { font-size:28px; font-weight:700; margin:8px 0; }
    .stack { display:grid; gap:14px; }
    .code-box { background:#091126; border:1px solid var(--line); border-radius:14px; padding:12px; overflow:auto; white-space:pre-wrap; font-family:"Cascadia Code",Consolas,monospace; font-size:13px; }
    .top-actions { display:flex; gap:10px; justify-content:flex-end; flex-wrap:wrap; }
    .section-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 10px; }
    .compact-note { color:var(--muted); font-size:13px; }
    @media (max-width:1024px) { .hero,.grid,.controls{grid-template-columns:1fr; display:grid;} .metrics{grid-template-columns:repeat(2,minmax(0,1fr));} .action-row,.helper-grid,.compare-grid{grid-template-columns:1fr;} .top-actions{justify-content:flex-start;} }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>ReliefRoute Mission Control</h1>
        <p>Humanitarian dispatch across disaster and conflict-affected zones.</p>
      </div>
      <div class="top-actions">
        <button class="secondary" id="compareBtn">Compare Baselines</button>
        <button class="secondary" id="replayBtn">Replay Baseline</button>
      </div>
    </div>
    <div class="controls">
      <div>
        <div class="compact-note">Task is set by <code>RELIEF_ROUTE_TASK</code> at server startup.</div>
        <div class="muted" id="statusLine" style="margin-top:8px;">Reset the scenario to begin.</div>
      </div>
      <label>Task<select id="taskSelect"><option value="easy">easy</option><option value="medium">medium</option><option value="hard">hard</option></select></label>
      <label>Replay policy<select id="replayPolicy"><option value="heuristic">heuristic</option><option value="greedy">greedy</option><option value="random">random</option></select></label>
      <label>Assist policy<select id="assistPolicy"><option value="heuristic">heuristic</option><option value="greedy">greedy</option><option value="random">random</option></select></label>
      <button class="secondary" id="suggestBtn">Suggest Turn</button>
      <button class="primary" id="resetBtn">Reset Scenario</button>
    </div>
    <div class="grid">
      <div class="panel">
        <div class="metrics">
          <div class="metric"><div class="label">Step</div><div class="value" id="stepValue">-</div></div>
          <div class="metric"><div class="label">Fulfillment</div><div class="value" id="fulfillmentValue">-</div></div>
          <div class="metric"><div class="label">Safety</div><div class="value" id="safetyValue">-</div></div>
          <div class="metric"><div class="label">Final Score</div><div class="value" id="scoreValue">-</div></div>
        </div>
        <div class="map-wrap">
          <div class="map-legend">
            <span><span class="legend-dot" style="background:#73e0a9;"></span>depot</span>
            <span><span class="legend-dot" style="background:#ff9c9c;"></span>priority 3 zone</span>
            <span><span class="legend-dot" style="background:#ffe2a1;"></span>priority 2 zone</span>
            <span><span class="legend-dot" style="background:#a3f0c7;"></span>priority 1 zone</span>
            <span><span class="legend-dot" style="background:#fff3bf;"></span>vehicle</span>
          </div>
          <svg id="missionMap" class="map-stage" viewBox="0 0 720 360"></svg>
        </div>
        <div class="inventory" id="inventoryPills"></div>
        <div class="section-head">
          <h2>Zone Status</h2>
          <span class="compact-note" id="manualHint">Reset a scenario to load guidance.</span>
        </div>
        <div class="zones" id="zones"></div>
        <div class="section-head" style="margin-top:18px;">
          <h2>Vehicles</h2>
        </div>
        <div class="vehicles" id="vehicles"></div>
      </div>
      <div class="panel">
        <h2>Dispatch Builder</h2>
        <p class="muted" style="margin:8px 0 14px;">Build one command per available vehicle and send the turn.</p>
        <div class="helper-grid">
          <div class="card">
            <strong>Quick Actions</strong>
            <div class="helper-actions">
              <button class="secondary" id="useSuggestedBtn">Use Suggestion</button>
              <button class="secondary" id="clearDraftBtn">Clear Draft</button>
            </div>
          </div>
          <div class="card">
            <strong>Current Objective</strong>
            <div class="muted" style="margin-top:8px;" id="objectiveHint">Reset a scenario to load task-specific guidance.</div>
          </div>
        </div>
        <div class="actions" id="actions"></div>
        <button class="primary" style="margin-top:14px;" id="submitBtn">Send Step</button>
        <div class="section-head" style="margin:18px 0 8px;">
          <h3>Raw Action JSON</h3>
          <span class="compact-note">Edit the full action payload here if the builder feels too cramped.</span>
        </div>
        <textarea id="rawAction" style="min-height:360px;height:360px;"></textarea>
        <h3 style="margin:18px 0 8px;">Mission Log</h3>
        <div class="trace" id="trace"></div>
      </div>
    </div>
  </div>
  <div class="modal" id="compareModal"><div class="modal-card"><div class="modal-head"><div><h2>Baseline Comparison</h2><p class="muted" id="compareSubtitle">Policy results for the current task.</p></div><button class="secondary" data-close="compareModal">Close</button></div><div class="compare-grid" id="compareGrid"><div class="card"><div class="muted">Run comparison to see heuristic, greedy, and random side by side.</div></div></div></div></div>
  <div class="modal" id="replayModal"><div class="modal-card"><div class="modal-head"><div><h2>Replay Timeline</h2><p class="muted" id="replaySubtitle">Baseline turn-by-turn playback.</p></div><button class="secondary" data-close="replayModal">Close</button></div><div class="stack"><div><div class="muted" id="replayStepLabel">No replay loaded yet.</div><input type="range" id="replaySlider" min="0" max="0" value="0" /></div><div class="card"><strong>Chosen Action</strong><div class="code-box" id="replayActionBox">Run a replay to inspect actions.</div></div><div class="card"><strong>Outcome</strong><div class="code-box" id="replayOutcomeBox">Replay output will appear here.</div></div></div></div></div>
  <script>
    let currentObservation = null;
    let currentSessionId = null;
    let lastReplay = null;
    let lastAction = { commands: [] };
    function appendLog(message) { const trace = document.getElementById("trace"); const item = document.createElement("div"); item.className = "trace-entry"; item.textContent = message; trace.prepend(item); }
    function clearLog() { document.getElementById("trace").innerHTML = ""; }
    function setStatus(message, isError = false) { const node = document.getElementById("statusLine"); node.textContent = message; node.className = isError ? "danger" : "muted"; }
    function priorityClass(priority) { return `priority-${priority}`; }
    function zoneRemaining(zone) { return `water ${Math.max(0, zone.demand.water - zone.delivered.water)} | food ${Math.max(0, zone.demand.food - zone.delivered.food)} | medicine ${Math.max(0, zone.demand.medicine - zone.delivered.medicine)}`; }
    function zoneFill(priority) { if (priority === 3) return "#ff7d7d"; if (priority === 2) return "#ffd970"; return "#7df0b6"; }
function routeStroke(zone, currentStep) { if (zone.route_reopens_at_step !== null && currentStep < zone.route_reopens_at_step) return "#8ea2c8"; if (zone.route_risk >= 0.5) return "#ff6262"; if (zone.route_risk >= 0.25) return "#ffd24f"; return "#63f0aa"; }
    function taskHint(observation) { if (observation.task_id === "easy") return "Field Clinic first for medicine, then shift to water and food at Shelter A."; if (observation.task_id === "medium") return "Regional Clinic needs medicine fast; Bridge Camp only opens later, so cover School Shelter meanwhile."; return "Serve Trauma Center and Border Shelter early, but watch access windows and the Old Town route reopening."; }

    function renderMap(observation) {
      const svg = document.getElementById("missionMap");
      const width = 720, height = 360, centerX = 360, centerY = 180, radius = Math.min(width, height) * 0.32;
      const zones = observation.zones || [];
      const positions = zones.map((zone, index) => {
        const angle = ((Math.PI * 2) / Math.max(zones.length, 1)) * index - (Math.PI / 2);
        return { zone, x: centerX + (Math.cos(angle) * radius), y: centerY + (Math.sin(angle) * radius) };
      });
      const routes = positions.map(({ zone, x, y }) => `<line class="map-route ${zone.route_reopens_at_step !== null && observation.current_step < zone.route_reopens_at_step ? "closed" : ""}" x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" stroke="${routeStroke(zone, observation.current_step)}"></line>`).join("");
      const recentDeliveries = new Map();
      (observation.info.delivered_this_step || []).forEach(message => {
        const match = message.match(/^([^ ]+) delivered .* to (.+)$/);
        if (!match) return;
        const zoneMatch = positions.find(item => item.zone.display_name === match[2]);
        if (zoneMatch) recentDeliveries.set(match[1], zoneMatch);
      });
      const lastActionByVehicle = new Map((lastAction?.commands || []).map(command => [command.vehicle_id, command]));
      const zoneNodes = positions.map(({ zone, x, y }) => {
        const remaining = (zone.demand.water - zone.delivered.water) + (zone.demand.food - zone.delivered.food) + (zone.demand.medicine - zone.delivered.medicine);
        return `<circle class="map-zone" cx="${x}" cy="${y}" r="28" fill="${zoneFill(zone.priority)}"></circle><text x="${x}" y="${y - 40}" text-anchor="middle" class="map-node-label">${zone.display_name}</text><text x="${x}" y="${y + 5}" text-anchor="middle" class="map-node-label">P${zone.priority}</text><text x="${x}" y="${y + 20}" text-anchor="middle" class="map-sub-label">rem ${remaining}</text>`;
      }).join("");
      const vehicleNodes = observation.vehicles.map((vehicle, index) => {
        let x = centerX + 42 + (index * 18), y = centerY + 42, stateLabel = "ready";
        if (recentDeliveries.has(vehicle.vehicle_id)) { const match = recentDeliveries.get(vehicle.vehicle_id); x = match.x; y = match.y + 42; stateLabel = "arrived"; }
        else if (vehicle.destination_zone_id) { const match = positions.find(item => item.zone.zone_id === vehicle.destination_zone_id); if (match) { const progress = vehicle.available ? 0.9 : 0.58; x = centerX + ((match.x - centerX) * progress); y = centerY + ((match.y - centerY) * progress); stateLabel = vehicle.available ? "returning" : "en route"; } }
        else if (lastActionByVehicle.has(vehicle.vehicle_id)) { const action = lastActionByVehicle.get(vehicle.vehicle_id); if (action.command_type === "dispatch" && action.destination_zone_id) { const match = positions.find(item => item.zone.zone_id === action.destination_zone_id); if (match) { x = centerX + ((match.x - centerX) * 0.78); y = centerY + ((match.y - centerY) * 0.78); stateLabel = "moving"; } } }
        return `<g><rect class="map-vehicle" x="${x - 12}" y="${y - 8}" width="24" height="16" rx="5"></rect><text x="${x}" y="${y + 28}" text-anchor="middle" class="map-sub-label">${vehicle.vehicle_id}</text><text x="${x}" y="${y + 42}" text-anchor="middle" class="map-sub-label">${stateLabel}</text></g>`;
      }).join("");
      svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect><circle cx="${centerX}" cy="${centerY}" r="${radius + 44}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 8"></circle>${routes}<circle class="map-depot" cx="${centerX}" cy="${centerY}" r="24"></circle><text x="${centerX}" y="${centerY - 34}" text-anchor="middle" class="map-node-label">Depot</text>${zoneNodes}${vehicleNodes}`;
    }

    function buildActionFromCards() {
      const cards = [...document.querySelectorAll(".action-card")];
      return { commands: cards.map(card => { const vehicleId = card.dataset.vehicle; const commandType = card.querySelector(".commandType").value; if (commandType === "wait") return { vehicle_id: vehicleId, command_type: "wait" }; return { vehicle_id: vehicleId, command_type: "dispatch", destination_zone_id: card.querySelector(".zoneId").value, supply_type: card.querySelector(".supplyType").value, quantity: Number(card.querySelector(".quantity").value) }; }) };
    }
    function buildAction() {
      const raw = document.getElementById("rawAction").value.trim();
      if (raw) { try { const parsed = JSON.parse(raw); if (parsed && Array.isArray(parsed.commands)) return parsed; } catch (error) {} }
      return buildActionFromCards();
    }
    function openModal(id) { const node = document.getElementById(id); node.classList.add("open"); }
    function closeModal(id) { const node = document.getElementById(id); node.classList.remove("open"); }
    function unwrapObservation(payload) { return payload && payload.observation ? payload.observation : payload; }
    async function postJson(path, body) { const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); if (!response.ok) throw new Error(await response.text()); return response.json(); }
    async function resetSession() { const task_id = document.getElementById("taskSelect").value; const payload = await postJson("/dashboard/session/reset", { task_id }); if (payload.session_id) currentSessionId = payload.session_id; return unwrapObservation(payload); }
    async function stepSession(action) { if (!currentSessionId) throw new Error("No active dashboard session. Reset the scenario first."); const payload = await fetch(`/dashboard/session/step?session_id=${encodeURIComponent(currentSessionId)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }); if (!payload.ok) throw new Error(await payload.text()); const data = await payload.json(); if (data.error === "unknown_session") throw new Error("Dashboard session expired. Reset the scenario again."); return unwrapObservation(data); }

    function renderObservation(observation) {
      currentObservation = observation;
      if (observation.current_step === 0) lastAction = { commands: [] };
      document.getElementById("taskSelect").value = observation.task_id;
      setStatus(`Loaded ${observation.task_id} at step ${observation.current_step}.`);
      document.getElementById("stepValue").textContent = `${observation.current_step}/${observation.max_steps}`;
      document.getElementById("fulfillmentValue").textContent = `${(observation.info.weighted_fulfillment * 100).toFixed(0)}%`;
      const safetyPct = Math.max(0, Math.min(100, observation.info.safety_score * 100));
      document.getElementById("safetyValue").textContent = `${safetyPct.toFixed(0)}%`;
      document.getElementById("scoreValue").textContent = Number(observation.info.final_score || 0).toFixed(2);
      document.getElementById("manualHint").textContent = taskHint(observation);
      document.getElementById("objectiveHint").textContent = taskHint(observation);
      document.getElementById("inventoryPills").innerHTML = `<div class="pill">Theme: ${observation.scenario_theme}</div><div class="pill">Task: ${observation.task_id}</div><div class="pill">Water: ${observation.depot_inventory.water}</div><div class="pill">Food: ${observation.depot_inventory.food}</div><div class="pill">Medicine: ${observation.depot_inventory.medicine}</div>`;
      renderMap(observation);
    document.getElementById("zones").innerHTML = observation.zones.map(zone => `<div class="zone"><div class="zone-header"><div><strong>${zone.display_name}</strong><div class="muted">${zone.zone_id}</div></div><span class="badge ${priorityClass(zone.priority)}">P${zone.priority}</span></div><div class="${zone.route_risk >= 0.5 ? "danger" : "muted"}">risk ${zone.route_risk.toFixed(2)} | deadline ${zone.deadline_step} | travel ${zone.travel_time} | checkpoint ${zone.checkpoint_delay}</div><div class="muted" style="margin-top:6px;">remaining: ${zoneRemaining(zone)}</div><div class="muted" style="margin-top:6px;">route ${zone.route_reopens_at_step === null || observation.current_step >= zone.route_reopens_at_step ? "open" : `closed until ${zone.route_reopens_at_step}`} | access ${observation.current_step >= zone.access_window_start && (zone.access_window_end === null || observation.current_step <= zone.access_window_end) ? "open" : `window ${zone.access_window_start}-${zone.access_window_end ?? "end"}`} | ${zone.conflict_affected ? "conflict-affected" : "disaster response"}</div></div>`).join("");
      document.getElementById("vehicles").innerHTML = observation.vehicles.map(vehicle => `<div class="vehicle"><div class="vehicle-header"><strong>${vehicle.vehicle_id}</strong><span class="badge ${vehicle.available ? "priority-1" : "priority-2"}">${vehicle.available ? "available" : `ETA ${vehicle.eta_remaining}`}</span></div><div class="muted">capacity ${vehicle.capacity}</div><div class="muted">${vehicle.destination_zone_id ? `destination ${vehicle.destination_zone_id}` : "awaiting dispatch"}</div></div>`).join("");
      const availableVehicles = observation.vehicles.filter(vehicle => vehicle.available);
      document.getElementById("actions").innerHTML = availableVehicles.length ? availableVehicles.map(vehicle => `<div class="action-card" data-vehicle="${vehicle.vehicle_id}"><strong>${vehicle.vehicle_id}</strong><div class="action-row"><select class="commandType"><option value="wait">wait</option><option value="dispatch">dispatch</option></select><select class="zoneId">${observation.zones.map(zone => `<option value="${zone.zone_id}">${zone.display_name}</option>`).join("")}</select><select class="supplyType"><option value="water">water</option><option value="food">food</option><option value="medicine">medicine</option></select><input class="quantity" type="number" min="0" value="${Math.min(vehicle.capacity, 1)}" /></div></div>`).join("") : `<div class="muted">No vehicles available this turn.</div>`;
      document.getElementById("rawAction").value = JSON.stringify(buildActionFromCards(), null, 2);
      if (observation.info.delivered_this_step?.length) observation.info.delivered_this_step.forEach(item => appendLog(item));
      if (observation.info.invalid_action_reasons?.length) observation.info.invalid_action_reasons.forEach(item => appendLog(`Invalid: ${item}`));
      appendLog(`Loaded ${observation.task_id} step ${observation.current_step}.`);
    }

    function renderReplay(trace) {
      lastReplay = trace;
      if (trace.steps.length) { lastAction = trace.steps[trace.steps.length - 1].action; renderObservation(trace.steps[trace.steps.length - 1].observation); }
      else { renderObservation(trace.initial_observation); }
      document.getElementById("replaySubtitle").textContent = `${trace.policy} on ${trace.task_id}, final score ${trace.final_score.toFixed(2)}`;
      const slider = document.getElementById("replaySlider");
      slider.max = String(Math.max(trace.steps.length, 1) - 1);
      slider.value = "0";
      renderReplayStep(0);
      openModal("replayModal");
    }

    function renderReplayStep(index) {
      if (!lastReplay) return;
      if (!lastReplay.steps.length) { document.getElementById("replayStepLabel").textContent = "No replay steps recorded."; document.getElementById("replayActionBox").textContent = JSON.stringify({ commands: [] }, null, 2); document.getElementById("replayOutcomeBox").textContent = JSON.stringify(lastReplay.initial_observation, null, 2); return; }
      const step = lastReplay.steps[Math.max(0, Math.min(index, lastReplay.steps.length - 1))];
      document.getElementById("replayStepLabel").textContent = `Step ${step.step_index} of ${lastReplay.steps.length}`;
      document.getElementById("replayActionBox").textContent = JSON.stringify(step.action, null, 2);
      document.getElementById("replayOutcomeBox").textContent = JSON.stringify({ reward: step.total_reward, delivered: step.observation.info.delivered_this_step, invalid_reasons: step.observation.info.invalid_action_reasons, final_score: step.observation.info.final_score, completion_reason: step.observation.metadata.completion_reason }, null, 2);
    }

    function renderComparison(compareResults) {
      document.getElementById("compareGrid").innerHTML = compareResults.map(result => `<div class="card"><div class="muted">${result.policy}</div><div class="score">${result.final_score.toFixed(2)}</div><div class="muted">fulfillment ${(result.weighted_fulfillment * 100).toFixed(0)}%</div><div class="muted">on-time ${(result.on_time_coverage * 100).toFixed(0)}%</div><div class="muted">safety ${(result.safety_score * 100).toFixed(0)}%</div><div class="muted">completion ${result.completion_reason}</div></div>`).join("");
      openModal("compareModal");
    }

    document.getElementById("resetBtn").addEventListener("click", async () => {
      try { lastAction = { commands: [] }; lastReplay = null; clearLog(); renderObservation(await resetSession()); }
      catch (error) { setStatus(`Reset failed: ${error.message}`, true); appendLog(`Reset failed: ${error.message}`); }
    });
    document.getElementById("submitBtn").addEventListener("click", async () => {
      try { const action = buildAction(); lastAction = action; document.getElementById("rawAction").value = JSON.stringify(action, null, 2); renderObservation(await stepSession(action)); }
      catch (error) { setStatus(`Step failed: ${error.message}`, true); appendLog(`Step failed: ${error.message}`); }
    });
    document.getElementById("suggestBtn").addEventListener("click", async () => {
      if (!currentObservation) { setStatus("Reset a scenario before requesting a suggestion.", true); return; }
      try { const policy = document.getElementById("assistPolicy").value; const suggestion = await postJson("/dashboard/suggest", { observation: currentObservation, policy }); lastAction = suggestion; document.getElementById("rawAction").value = JSON.stringify(suggestion, null, 2); setStatus(`Loaded ${policy} suggestion for the current turn.`); appendLog(`Loaded ${policy} suggestion.`); }
      catch (error) { setStatus(`Suggestion failed: ${error.message}`, true); appendLog(`Suggestion failed: ${error.message}`); }
    });
    document.getElementById("useSuggestedBtn").addEventListener("click", async () => {
      if (!currentObservation) { setStatus("Reset a scenario before requesting a suggestion.", true); return; }
      try { const policy = document.getElementById("assistPolicy").value; const suggestion = await postJson("/dashboard/suggest", { observation: currentObservation, policy }); lastAction = suggestion; document.getElementById("rawAction").value = JSON.stringify(suggestion, null, 2); setStatus(`Loaded ${policy} suggestion for the current turn.`); appendLog(`Loaded ${policy} suggestion.`); }
      catch (error) { setStatus(`Suggestion failed: ${error.message}`, true); appendLog(`Suggestion failed: ${error.message}`); }
    });
    document.getElementById("clearDraftBtn").addEventListener("click", () => { document.getElementById("rawAction").value = JSON.stringify(buildActionFromCards(), null, 2); setStatus("Draft action reset to the current builder values."); });
    document.getElementById("replayBtn").addEventListener("click", async () => {
      const policy = document.getElementById("replayPolicy").value;
      const taskId = currentObservation?.task_id ?? "easy";
      const response = await fetch(`/dashboard/replay?task_id=${encodeURIComponent(taskId)}&policy=${encodeURIComponent(policy)}`);
      if (!response.ok) { setStatus("Replay failed.", true); appendLog("Replay failed."); return; }
      renderReplay(await response.json());
    });
    document.getElementById("compareBtn").addEventListener("click", async () => {
      const taskId = currentObservation?.task_id ?? "easy";
      const response = await fetch(`/dashboard/compare?task_id=${encodeURIComponent(taskId)}`);
      if (!response.ok) { setStatus("Comparison failed.", true); appendLog("Comparison failed."); return; }
      document.getElementById("compareSubtitle").textContent = `Policy results for task ${taskId}.`;
      renderComparison(await response.json());
      setStatus(`Loaded baseline comparison for ${taskId}.`);
    });
    document.getElementById("replaySlider").addEventListener("input", event => renderReplayStep(Number(event.target.value)));
    document.querySelectorAll("[data-close]").forEach(node => node.addEventListener("click", () => closeModal(node.dataset.close)));
    document.querySelectorAll(".modal").forEach(node => node.addEventListener("click", event => { if (event.target === node) closeModal(node.id); }));
    document.addEventListener("change", event => { if (event.target.closest(".action-card")) document.getElementById("rawAction").value = JSON.stringify(buildActionFromCards(), null, 2); });
    document.getElementById("rawAction").value = JSON.stringify({ commands: [] }, null, 2);
    appendLog("Start by clicking Reset Scenario.");
  </script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.post("/dashboard/session/reset")
def dashboard_session_reset(request: DashboardResetRequest = Body(default_factory=DashboardResetRequest)) -> dict[str, Any]:
    task_id = request.task_id or os.getenv("RELIEF_ROUTE_TASK", "easy")
    seed = request.seed
    session_id, observation = DASHBOARD_SESSIONS.create(task_id=task_id, seed=seed)
    return {"session_id": session_id, "observation": observation.model_dump()}


@app.post("/dashboard/session/step")
def dashboard_session_step(session_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        env = DASHBOARD_SESSIONS.get(session_id)
    except KeyError:
        return {"error": "unknown_session"}
    action_payload = request.get("action", request)
    action = ReliefRouteAction.model_validate(action_payload)
    observation = env.step(action)
    return {"session_id": session_id, "observation": observation.model_dump()}


@app.post("/dashboard/session/close")
def dashboard_session_close(session_id: str) -> dict[str, bool]:
    DASHBOARD_SESSIONS.destroy(session_id)
    return {"ok": True}


@app.get("/dashboard/replay", response_model=EpisodeTrace)
def dashboard_replay(task_id: str = "easy", policy: str = "heuristic") -> EpisodeTrace:
    return _run_policy_episode(task_id, policy)


@app.post("/dashboard/suggest", response_model=ReliefRouteAction)
def dashboard_suggest(request: SuggestTurnRequest = Body(...)) -> ReliefRouteAction:
    return _policy_action(request.policy, request.observation, Random(7))


@app.get("/dashboard/compare")
def dashboard_compare(task_id: str = "easy") -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for policy in ("heuristic", "greedy", "random"):
        trace = _run_policy_episode(task_id, policy)
        final_observation = trace.steps[-1].observation if trace.steps else trace.initial_observation
        results.append(
            {
                "policy": policy,
                "final_score": trace.final_score,
                "weighted_fulfillment": final_observation.info.weighted_fulfillment,
                "on_time_coverage": final_observation.info.on_time_coverage,
                "safety_score": final_observation.info.safety_score,
                "completion_reason": final_observation.metadata.get("completion_reason", "unknown"),
            }
        )
    _record_leaderboard(task_id, results)
    return results


@app.get("/dashboard/leaderboard")
def dashboard_leaderboard() -> dict[str, list[dict[str, Any]]]:
    return _load_leaderboard()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
