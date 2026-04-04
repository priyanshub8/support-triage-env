# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the BSD 3-Clause License found in the LICENSE file.

"""WebSocket client for the support triage OpenEnv server."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TriageReward,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TriageReward,
    )


class SupportTriageEnv(EnvClient[SupportTriageAction, SupportTriageObservation, SupportTriageState]):
    """
    Client for the support triage environment.

    Example:
        >>> with SupportTriageEnv(base_url="http://localhost:8000") as client:
        ...     r = client.reset(task="easy", seed=42)
        ...     assert r.observation.scenario_instruction
    """

    def _step_payload(self, action: SupportTriageAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportTriageObservation]:
        obs_data = payload.get("observation", {})
        rb = obs_data.get("reward_breakdown")
        reward_breakdown = TriageReward.model_validate(rb) if isinstance(rb, dict) else None

        observation = SupportTriageObservation(
            scenario_instruction=obs_data.get("scenario_instruction", ""),
            inbox_summary=obs_data.get("inbox_summary", ""),
            ticket_detail=obs_data.get("ticket_detail", ""),
            feedback=obs_data.get("feedback", ""),
            reward_breakdown=reward_breakdown,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupportTriageState:
        return SupportTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            seed=payload.get("seed", 0),
            max_steps=payload.get("max_steps", 20),
            steps_remaining=payload.get("steps_remaining", 20),
            finalized=payload.get("finalized", False),
        )
