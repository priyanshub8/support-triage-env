#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the BSD 3-Clause License found in the LICENSE file.

"""
Run a fixed-seed baseline against all three tasks using the OpenAI Chat Completions API.

Requires:
  - A running OpenEnv server (default http://127.0.0.1:8000)
  - OPENAI_API_KEY in the environment

Example:
  uvicorn server.app:app --host 127.0.0.1 --port 8000
  export OPENAI_API_KEY=sk-...
  python scripts/baseline_openai.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from openai import OpenAI

from support_triage_env import SupportTriageAction, SupportTriageEnv

ACTION_SCHEMA_HINT = """
Respond with a single JSON object only (no markdown), using this shape:
{
  "command": one of
    "show_inbox" | "view_ticket" | "set_triage" | "draft_reply" |
    "set_disposition" | "submit_episode",
  "ticket_id": string (e.g. "T-1001", required for view/set/draft/disposition),
  "priority": "low" | "medium" | "high" | "urgent" (for set_triage),
  "department": "billing" | "technical" | "general" (for set_triage),
  "reply_body": string (for draft_reply),
  "disposition": "resolved" | "escalate_supervisor" | "pending_customer"
}
Choose the next best action to complete the task. Call submit_episode when done.
"""


def build_user_message(obs_summary: str) -> str:
    return f"{ACTION_SCHEMA_HINT}\n\nCurrent observation:\n{obs_summary}"


def summarize_observation(obs) -> str:
    parts = [
        f"instruction: {obs.scenario_instruction}",
        f"inbox:\n{obs.inbox_summary}",
    ]
    if obs.ticket_detail.strip():
        parts.append(f"ticket:\n{obs.ticket_detail}")
    parts.append(f"feedback: {obs.feedback}")
    if obs.reward_breakdown:
        parts.append(f"last_reward: {obs.reward_breakdown.total}")
    return "\n\n".join(parts)


def parse_action(data: dict[str, Any]) -> SupportTriageAction:
    return SupportTriageAction(
        command=data["command"],
        ticket_id=str(data.get("ticket_id", "")),
        priority=str(data.get("priority", "")),
        department=str(data.get("department", "")),
        reply_body=str(data.get("reply_body", "")),
        disposition=str(data.get("disposition", "")),
    )


def run_episode(
    client: OpenTriageClient,
    env: SupportTriageEnv,
    task: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    r = env.reset(task=task, seed=seed)
    history: list[dict[str, str]] = []
    steps = 0

    while not r.done and steps < max_steps:
        user = build_user_message(summarize_observation(r.observation))
        history.append({"role": "user", "content": user})
        raw = client.complete(history)
        history.append({"role": "assistant", "content": raw})
        try:
            payload = json.loads(raw)
            action = parse_action(payload)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return {
                "task": task,
                "seed": seed,
                "error": f"bad_model_output: {exc}",
                "raw": raw[:500],
                "steps": steps,
            }

        r = env.step(action)
        steps += 1

    rb = r.observation.reward_breakdown
    final_grader = float(rb.terminal_grader) if rb and rb.terminal_grader is not None else None

    return {
        "task": task,
        "seed": seed,
        "steps": steps,
        "done": r.done,
        "grader_score": final_grader,
        "final_reward": r.observation.reward,
    }


class OpenTriageClient:
    def __init__(self, model: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        sys_msg = {
            "role": "system",
            "content": (
                "You are a helpdesk triage agent. Output only valid minified JSON "
                "for the next environment action."
            ),
        }
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[sys_msg, *messages],
        )
        choice = resp.choices[0].message.content or "{}"
        return choice.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI baseline for support_triage_env")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000"),
        help="Running OpenEnv HTTP/WebSocket base URL",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model name")
    parser.add_argument("--max-steps", type=int, default=24, help="Per-episode cap")
    args = parser.parse_args()

    seeds = {"easy": 42, "medium": 42, "hard": 42}
    llm = OpenTriageClient(args.model)
    results = []

    with SupportTriageEnv(base_url=args.base_url).sync() as env:
        for task in ("easy", "medium", "hard"):
            results.append(
                run_episode(llm, env, task, seeds[task], args.max_steps),
            )

    print(json.dumps({"baseline": results, "model": args.model}, indent=2))


if __name__ == "__main__":
    main()
