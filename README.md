# ReliefRoute OpenEnv

ReliefRoute is an OpenEnv benchmark for humanitarian aid dispatch across disaster-hit and conflict-affected civilian zones. An agent must route limited vehicles carrying water, food, and medicine while handling deadlines, blocked corridors, access windows, checkpoint delays, and route risk.

## Overview

This repository is organized in the same order as the standard validation and deployment flow:

1. OpenEnv validation
2. Docker build and run
3. Baseline inference script
4. Three graded tasks with deterministic `0.0-1.0` scoring
5. Optional visual dashboard

## Project Summary

- `openenv.yaml` at the repo root
- root-level `inference.py`
- Dockerfile at [`server/Dockerfile`](server/Dockerfile)
- three required tasks: `easy`, `medium`, `hard`
- additional stress task: `expert`
- deterministic graders with final score in `0.0-1.0`
- Hugging Face Space deploy target

## Quick Validation

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
python -m openenv.cli validate
docker build --no-cache -f server\Dockerfile -t relief-route-openenv .
```

If these pass, the project has cleared the main technical gates.

## Tasks

### `easy`

- flood-response scenario
- 1 vehicle
- 3 zones
- open routes and low-risk corridors

### `medium`

- earthquake-response scenario
- 2 vehicles
- 4 zones
- staggered route reopening and competing priorities

### `hard`

- disaster plus conflict-affected humanitarian response
- 3 vehicles
- 5 zones
- access windows, checkpoint delays, and risky corridors

### `expert`

- larger multi-front humanitarian response
- 4 vehicles
- 6 zones
- overlapping deadlines and more severe sequencing pressure

## Reward And Grader

- step reward is normalized to `0.0-1.0`
- final grader score is normalized to `0.0-1.0`
- better actions improve weighted fulfillment, timeliness, efficiency, and safety
- invalid or wasteful actions lower the score

The grader is deterministic for the same task and seed, but different actions produce different outcomes.

## Inference Script

The root inference script is [`inference.py`](inference.py).

It:

- uses the OpenAI client
- launches the environment from a local Docker image
- reads credentials from environment variables
- prints strict `[START]`, `[STEP]`, `[END]` lines

### Required environment variables

```powershell
$env:LOCAL_IMAGE_NAME="relief-route-openenv"
$env:HF_TOKEN="<your_token>"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:RELIEF_ROUTE_TASK="hard"
```

### Run inference

```powershell
python inference.py
```

Expected stdout shape:

```text
[START] task=hard env=relief-route model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.42 done=false error=null
[END] success=true steps=6 rewards=0.42,0.38,0.51,0.20,0.11,0.09
```

## Evaluation Utilities

### Policy evaluation

Run the reference policies with [`scripts/evaluate_baselines.py`](scripts/evaluate_baselines.py):

```powershell
python scripts/evaluate_baselines.py --tasks easy medium hard expert --episodes 3 --policy heuristic
python scripts/evaluate_baselines.py --tasks easy medium hard expert --episodes 3 --policy greedy
python scripts/evaluate_baselines.py --tasks easy medium hard expert --episodes 3 --policy random
```

### Trace replay

Replay a saved trace with [`scripts/replay_trace.py`](scripts/replay_trace.py):

```powershell
python scripts/evaluate_baselines.py --tasks hard --episodes 1 --policy heuristic --trace-dir traces
python scripts/replay_trace.py traces/heuristic_hard_episode_1.json
```

### Interactive terminal runner

Use [`scripts/interactive_console.py`](scripts/interactive_console.py):

```powershell
python scripts/interactive_console.py --task expert
```

Or the installed command:

```powershell
relief-route-console --task expert
```

## Local Dashboard

Start the server:

```powershell
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open:

- [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The dashboard supports:

- task switching
- manual dispatch building
- policy-assisted suggestions
- baseline comparison
- baseline replay
- live mission map and mission log

## Deployment

### Docker

```powershell
docker build --no-cache -f server\Dockerfile -t relief-route-openenv .
docker run -p 8000:8000 relief-route-openenv
```

### Hugging Face Space

1. Create a Docker Space.
2. Push this repository to the Space.
3. Confirm these routes respond:
   - `/`
   - `/dashboard`
   - `/docs`
   - `/health`

## Repository Layout

- [`openenv.yaml`](openenv.yaml): OpenEnv entrypoint metadata
- [`inference.py`](inference.py): baseline inference script
- [`relief_route_env/models.py`](relief_route_env/models.py): typed observation, action, state, and trace models
- [`relief_route_env/tasks.py`](relief_route_env/tasks.py): task definitions
- [`relief_route_env/grader.py`](relief_route_env/grader.py): deterministic scoring
- [`relief_route_env/baseline.py`](relief_route_env/baseline.py): reference policies
- [`server/relief_route_environment.py`](server/relief_route_environment.py): environment core
- [`server/app.py`](server/app.py): FastAPI app and dashboard
- [`server/Dockerfile`](server/Dockerfile): container build
- [`scripts/evaluate_baselines.py`](scripts/evaluate_baselines.py): policy evaluation script
- [`scripts/replay_trace.py`](scripts/replay_trace.py): replay utility
- [`scripts/interactive_console.py`](scripts/interactive_console.py): terminal demo
