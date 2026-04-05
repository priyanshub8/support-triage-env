---
title: Support Triage OpenEnv
emoji: 🎧
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - agent
  - rl
license: bsd-3-clause
---

# Support Triage OpenEnv

Production-style **customer-support triage** simulation for [OpenEnv](https://meta-pytorch.org/OpenEnv/): an agent manages a ticket queue (priorities, departments, sensitive replies, dispositions) through the standard **`reset` / `step` / `state`** API with typed Pydantic models and an `openenv.yaml` manifest.

This is a **realistic desk workflow** (routing + policy), not a game environment.

## Motivation

Helpdesk agents must interpret unstructured text, apply routing rules, avoid unsafe commitments in customer-visible replies, and close or escalate cases. This environment turns that into a **deterministic simulator** with:

- **Shaped rewards** during the episode (partial triage matches, penalties for invalid commands, loop detection, per-step cost).
- **Task graders** in **[0.0, 1.0]** returned as `reward_breakdown.terminal_grader` on the terminal observation (OpenEnv wire serialization omits `observation.metadata`, so graded scores are surfaced through `TriageReward.terminal_grader`).

## Action space (`SupportTriageAction`)

| Field | Description |
|--------|-------------|
| `command` | `show_inbox` · `view_ticket` · `set_triage` · `draft_reply` · `set_disposition` · `submit_episode` |
| `ticket_id` | Ticket id (e.g. `T-1001`) when required |
| `priority` | `low` · `medium` · `high` · `urgent` (for `set_triage`) |
| `department` | `billing` · `technical` · `general` (for `set_triage`) |
| `reply_body` | Customer-visible draft (for `draft_reply`) |
| `disposition` | `resolved` · `escalate_supervisor` · `pending_customer` |

Call **`submit_episode`** to end the episode and compute the final grader.

## Observation space (`SupportTriageObservation`)

| Field | Description |
|--------|-------------|
| `scenario_instruction` | Task goal and constraints |
| `inbox_summary` | Queue overview |
| `ticket_detail` | Full ticket text after `view_ticket` |
| `feedback` | Result of the last action |
| `reward_breakdown` | `TriageReward` (structured components + optional `terminal_grader`) |
| `reward`, `done` | Scalar reward and termination (OpenEnv base `Observation`) |

## Reward model (`TriageReward`)

Structured Pydantic reward (mirrors the scalar `observation.reward`):

- **`progress`**: partial credit (e.g. correct triage fields, storing drafts).
- **`penalty`**: non-positive penalties (invalid ids, policy-violating phrases, loops).
- **`terminal_grader`**: **[0.0, 1.0]** deterministic task score when `done=True`.
- **`total`**: scalar used as the step reward (clamped to **[-1, 1]**).

## State (`SupportTriageState`)

Extends OpenEnv `State` with `task`, `seed`, `max_steps`, `steps_remaining`, and `finalized`. Retrieve via the client’s **`state()`** method on the active WebSocket session.

## Tasks & difficulty

| Task | Difficulty | Objective | Grader |
|------|------------|-----------|--------|
| `easy` | Easy | Triage **one** ticket (priority + department) | 0.5 priority + 0.5 department |
| `medium` | Medium | Triage **three** tickets | Average per-ticket exact match (both fields) |
| `hard` | Hard | Policy thread: correct triage + **compliant** reply (no forbidden commitments) + correct disposition | Weighted rubric (triage, reply compliance, disposition) |

Select a task with `reset(task="easy"|"medium"|"hard", seed=...)`.

## Setup

**Python 3.10+** and [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
cd support_triage_env
uv sync
# or: pip install -e ".[baseline]"
```

### Validate (OpenEnv CLI)

```bash
openenv validate --verbose
```

### Run the server

```bash
# From this directory, with PYTHONPATH including the parent if imports fail:
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"   # monorepo layout
# or after pip install -e .:
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Client usage (sync wrapper)

```python
from support_triage_env import SupportTriageEnv, SupportTriageAction

with SupportTriageEnv(base_url="http://localhost:8000").sync() as env:
    r = env.reset(task="medium", seed=42)
    r = env.step(SupportTriageAction(command="show_inbox"))
    s = env.state()
    print(s.task, s.steps_remaining)
```

## Docker

A **`Dockerfile` at the repository root** mirrors `server/Dockerfile` so hackathon tooling (for example `validate-submission.sh` Step 2) can run `docker build` with the **environment root** as context (`COPY . /app/env` needs `models.py`, `pyproject.toml`, etc.).

```bash
cd support_triage_env
docker build -t support-triage-openenv:latest .
docker run --rm -p 8000:8000 support-triage-openenv:latest
```

You can still build explicitly from `server/Dockerfile` with the same context:

`docker build -f server/Dockerfile -t support-triage-openenv:latest .`

Requires access to `ghcr.io/meta-pytorch/openenv-base:latest` (default `BASE_IMAGE`).

## Hugging Face Spaces

1. Create a **Docker** Space.
2. Point the Space at this repository (or push with `openenv push` from this directory).
3. Ensure the Space README includes **`tags: [openenv, ...]`** (this file’s frontmatter does).
4. Default port **8000** matches `openenv.yaml`.

## Hackathon `inference.py` (mandatory stdout format)

The repository root includes **`inference.py`**, which:

- Uses the **OpenAI** Python client with **`API_BASE_URL`**, **`MODEL_NAME`**, and **`HF_TOKEN`** (or `API_KEY`), matching the hackathon template defaults for Hugging Face inference.
- Connects with **`LOCAL_IMAGE_NAME`** / **`IMAGE_NAME`** via `SupportTriageEnv.from_docker_image()`, or with **`OPENENV_BASE_URL`** / **`ENV_BASE_URL`** / **`PING_URL`** (default `http://127.0.0.1:8000`) via WebSocket.
- Runs tasks **`easy`**, **`medium`**, and **`hard`** by default (override with **`SUPPORT_TRIAGE_TASKS`**, e.g. `all` or `easy,hard`).
- Prints exactly **`[START]`**, one **`[STEP]`** per `await env.step()`, and **`[END]`** per task, with **`score`** and per-step **`reward`** in **[0, 1]** / two decimal places as specified.

```bash
export HF_TOKEN=...
export LOCAL_IMAGE_NAME=support-triage-openenv:latest   # or omit and set OPENENV_BASE_URL
export OPENENV_BASE_URL=https://your-space.hf.space
python inference.py
```

### Pre-validation script

`scripts/validate-submission.sh` runs the three checks from the hackathon (HF `/reset`, Docker build, `openenv validate`). Pass the **environment directory** as `repo_dir` if it is not the current directory:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh "https://your-space.hf.space" "$(pwd)"
```

## Baseline inference (OpenAI)

Install baseline extras, run the server, set **`OPENAI_API_KEY`**, then:

```bash
pip install -e ".[baseline]"
export OPENAI_API_KEY=sk-...
python scripts/baseline_openai.py --base-url http://127.0.0.1:8000 --model gpt-4o-mini
```

The script uses **temperature 0** and fixed seeds **42** for `easy`, `medium`, and `hard`. **Re-run** after model or prompt changes; scores are **not** guaranteed across API versions.

### Reference oracle grader scores (seed = 42)

With a perfect policy (correct triage / compliant reply / disposition), all three tasks achieve **terminal_grader = 1.0** on seed 42 (verified in automated tests). Your model’s JSON action quality will typically score lower—record those numbers as your empirical baseline.

## Layout

```
support_triage_env/
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── models.py
├── client.py
├── scripts/
│   ├── baseline_openai.py
│   └── validate-submission.sh
└── server/
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    └── support_triage_env_environment.py
```

## License

Code headers follow the BSD-style license from the OpenEnv ecosystem (see file headers).
