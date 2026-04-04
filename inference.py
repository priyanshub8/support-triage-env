#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the BSD 3-Clause License found in the LICENSE file.

"""
Hackathon inference entrypoint for ``support_triage_env``.

MANDATORY environment variables (see hackathon brief):
  API_BASE_URL     LLM API base URL (OpenAI-compatible).
  MODEL_NAME       Model id for chat completions.
  HF_TOKEN         API key (passed as ``api_key`` to OpenAI client).
  LOCAL_IMAGE_NAME Optional; if set, ``SupportTriageEnv.from_docker_image(...)`` is used.
                     Otherwise ``OPENENV_BASE_URL`` / ``ENV_BASE_URL`` (default http://127.0.0.1:8000).

STDOUT: exactly ``[START]``, one ``[STEP]`` per ``env.step()``, and a final ``[END]`` per task episode.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional

# Repo root = directory containing this file (run ``python inference.py`` from here).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

from client import SupportTriageEnv
from models import SupportTriageAction

# --- Mandatory / defaults (hackathon) ---
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or ""
OPENENV_BASE_URL = (
    os.getenv("OPENENV_BASE_URL")
    or os.getenv("ENV_BASE_URL")
    or os.getenv("PING_URL")
    or "http://127.0.0.1:8000"
)
OPENENV_BASE_URL = OPENENV_BASE_URL.rstrip("/")

BENCHMARK = os.getenv("SUPPORT_TRIAGE_BENCHMARK", "support_triage_env")
_TASKS_RAW = os.getenv("SUPPORT_TRIAGE_TASKS", "easy,medium,hard").strip().lower()
MAX_STEPS = int(os.getenv("MAX_STEPS", "28"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
BASE_SEED = int(os.getenv("SUPPORT_TRIAGE_SEED", "42"))

ACTION_SCHEMA = textwrap.dedent(
    """
    Output a single JSON object only (no markdown), with keys:
      "command": one of
        "show_inbox" | "view_ticket" | "set_triage" | "draft_reply" |
        "set_disposition" | "submit_episode"
      "ticket_id": string (e.g. "T-1001") when needed
      "priority": "low"|"medium"|"high"|"urgent" for set_triage
      "department": "billing"|"technical"|"general" for set_triage
      "reply_body": string for draft_reply
      "disposition": "resolved"|"escalate_supervisor"|"pending_customer" for set_disposition
    Call "submit_episode" when the task is complete.
    """
).strip()


def _tasks() -> list[str]:
    if _TASKS_RAW in ("", "all"):
        return ["easy", "medium", "hard"]
    out = [t.strip() for t in _TASKS_RAW.split(",") if t.strip()]
    return out or ["easy", "medium", "hard"]


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = "null" if error is None else _one_line(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()


def _obs_summary(obs) -> str:
    parts = [
        f"instruction: {obs.scenario_instruction}",
        f"inbox:\n{obs.inbox_summary}",
    ]
    if obs.ticket_detail.strip():
        parts.append(f"ticket:\n{obs.ticket_detail}")
    parts.append(f"feedback: {obs.feedback}")
    if obs.reward_breakdown is not None:
        parts.append(f"last_step_reward: {obs.reward_breakdown.total}")
    return "\n\n".join(parts)


def _action_str(action: SupportTriageAction) -> str:
    d = action.model_dump(exclude_none=True)
    d.pop("metadata", None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _parse_action(data: dict[str, Any]) -> SupportTriageAction:
    return SupportTriageAction(
        command=data["command"],
        ticket_id=str(data.get("ticket_id", "")),
        priority=str(data.get("priority", "")),
        department=str(data.get("department", "")),
        reply_body=str(data.get("reply_body", "")),
        disposition=str(data.get("disposition", "")),
    )


def _terminal_score(obs) -> float:
    rb = obs.reward_breakdown
    if rb is not None and rb.terminal_grader is not None:
        return float(max(0.0, min(1.0, rb.terminal_grader)))
    return 0.0


def get_model_action(
    client: OpenAI,
    history: list[dict[str, str]],
) -> tuple[SupportTriageAction, Optional[str]]:
    """Returns (action, error_string_or_none)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpdesk triage agent. "
                        + ACTION_SCHEMA
                    ),
                },
                *history,
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return _parse_action(data), None
    except Exception as exc:  # noqa: BLE001 — surface to stdout error field
        return SupportTriageAction(command="submit_episode"), _one_line(str(exc))


async def run_one_task(
    env: SupportTriageEnv,
    client: OpenAI,
    task: str,
    seed: int,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_action_error: Optional[str] = None
    result = None

    log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)

    chat: list[dict[str, str]] = []

    try:
        result = await env.reset(task=task, seed=seed)

        for step_idx in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user = (
                f"{ACTION_SCHEMA}\n\nCurrent observation:\n{_obs_summary(result.observation)}"
            )
            chat.append({"role": "user", "content": user})

            action, parse_err = await asyncio.to_thread(get_model_action, client, chat)
            chat.append({"role": "assistant", "content": _action_str(action)})

            if parse_err is not None:
                last_action_error = parse_err

            try:
                result = await env.step(action)
            except Exception as exc:  # noqa: BLE001
                last_action_error = _one_line(str(exc))
                reward = 0.0
                log_step(
                    step=step_idx,
                    action=_action_str(action),
                    reward=reward,
                    done=True,
                    error=last_action_error,
                )
                rewards.append(reward)
                steps_taken = step_idx
                break

            reward = float(result.reward if result.reward is not None else 0.0)
            rewards.append(reward)
            steps_taken = step_idx

            log_step(
                step=step_idx,
                action=_action_str(action),
                reward=reward,
                done=result.done,
                error=last_action_error,
            )

            last_action_error = None

            if result.done:
                break

        if result is not None:
            score = _terminal_score(result.observation)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:  # noqa: BLE001 — episode abort; [END] still emitted in finally
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def _open_env() -> SupportTriageEnv:
    if LOCAL_IMAGE_NAME.strip():
        return await SupportTriageEnv.from_docker_image(LOCAL_IMAGE_NAME.strip())
    env = SupportTriageEnv(base_url=OPENENV_BASE_URL)
    await env.connect()
    return env


async def main() -> None:
    if not HF_TOKEN:
        print(
            "HF_TOKEN (or API_KEY) must be set for OpenAI client authentication.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = await _open_env()
    try:
        for i, task in enumerate(_tasks()):
            if task not in ("easy", "medium", "hard"):
                continue
            seed = BASE_SEED + i
            await run_one_task(env, llm, task, seed)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
