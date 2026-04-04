# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the BSD 3-Clause License found in the LICENSE file.

"""
Typed wire models for the customer-support triage OpenEnv environment.

Simulates helpdesk triage: priority, routing, policy-compliant replies, and disposition.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class TriageReward(BaseModel):
    """
    Decomposed reward for the last transition.

    ``total`` is the scalar signal used for RL; components explain partial progress
    and penalties. Terminal episodes set ``terminal_grader`` to the task grader in [0, 1].
    """

    total: float = Field(..., ge=-1.0, le=1.0, description="Scalar reward for this step")
    progress: float = Field(
        0.0, ge=-1.0, le=1.0, description="Partial progress toward task objectives"
    )
    penalty: float = Field(
        0.0, ge=-1.0, le=0.0, description="Non-positive penalty (invalid or harmful actions)"
    )
    terminal_grader: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Deterministic task score when the episode ends",
    )


class SupportTriageAction(Action):
    """Structured helpdesk action."""

    command: Literal[
        "show_inbox",
        "view_ticket",
        "set_triage",
        "draft_reply",
        "set_disposition",
        "submit_episode",
    ] = Field(..., description="High-level command to execute")
    ticket_id: str = Field("", description="Target ticket id, when applicable")
    priority: str = Field(
        "",
        description="Suggested priority: low | medium | high | urgent",
    )
    department: str = Field(
        "",
        description="Routing department: billing | technical | general",
    )
    reply_body: str = Field("", description="Outbound customer reply draft")
    disposition: str = Field(
        "",
        description="Case disposition: resolved | escalate_supervisor | pending_customer",
    )


class SupportTriageObservation(Observation):
    """What the agent sees after each step."""

    scenario_instruction: str = Field(
        default="",
        description="Task objective and constraints for this episode",
    )
    inbox_summary: str = Field(
        default="",
        description="Human-readable view of the current queue",
    )
    ticket_detail: str = Field(
        default="",
        description="Full text of the focused ticket, if any",
    )
    feedback: str = Field(
        default="",
        description="Environment feedback about the last action",
    )
    reward_breakdown: TriageReward | None = Field(
        default=None,
        description="Structured reward matching the scalar ``reward`` field",
    )


class SupportTriageState(State):
    """Server state exposed via GET /state and WebSocket state messages."""

    task: Literal["easy", "medium", "hard"] = Field(
        "easy",
        description="Which graded scenario is active",
    )
    seed: int = Field(0, description="Scenario seed for reproducibility")
    max_steps: int = Field(20, ge=1)
    steps_remaining: int = Field(20, ge=0)
    finalized: bool = Field(False, description="Whether the episode has ended")
