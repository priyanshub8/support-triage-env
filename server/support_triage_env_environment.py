# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the BSD 3-Clause License found in the LICENSE file.

"""
Customer-support triage simulation with three graded tasks (easy / medium / hard).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
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


@dataclass
class Ticket:
    ticket_id: str
    subject: str
    body: str
    gold_priority: str
    gold_department: str


@dataclass
class HardScenario:
    ticket: Ticket
    forbid_phrases: tuple[str, ...]
    gold_disposition: str
    gold_priority: str
    gold_department: str


@dataclass
class EpisodeContext:
    task: str
    seed: int
    tickets: list[Ticket]
    hard: HardScenario | None = None
    viewed: set[str] = field(default_factory=set)
    triage_by_id: dict[str, tuple[str, str]] = field(default_factory=dict)
    reply_by_id: dict[str, str] = field(default_factory=dict)
    disposition_by_id: dict[str, str] = field(default_factory=dict)
    action_fingerprints: deque[str] = field(default_factory=lambda: deque(maxlen=5))


def _stable_seed(seed: int, salt: str) -> int:
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _easy_tickets(seed: int) -> list[Ticket]:
    r = random.Random(_stable_seed(seed, "easy"))
    variants = [
        Ticket(
            "T-1001",
            "Double charge on March invoice",
            "I was billed twice for the March subscription. Please fix this urgently.",
            "high",
            "billing",
        ),
        Ticket(
            "T-1002",
            "Cannot log in after password reset email",
            "The reset link expires immediately. I need access for a client demo today.",
            "urgent",
            "technical",
        ),
    ]
    return [r.choice(variants)]


def _medium_tickets(seed: int) -> list[Ticket]:
    r = random.Random(_stable_seed(seed, "medium"))
    pool = [
        Ticket(
            "T-2001",
            "Pricing question before upgrade",
            "Could you clarify seat pricing for 25 users on the Business plan?",
            "low",
            "billing",
        ),
        Ticket(
            "T-2002",
            "App crashes on launch",
            "After the last update the mobile app closes immediately on iOS 18.",
            "high",
            "technical",
        ),
        Ticket(
            "T-2003",
            "Wrong company name on contract",
            "Our legal name is Acme Labs LLC but the PDF says Acme Labs Inc.",
            "medium",
            "billing",
        ),
        Ticket(
            "T-2004",
            "API returns 500 on /export",
            "Our nightly export job has been failing since Tuesday.",
            "high",
            "technical",
        ),
        Ticket(
            "T-2005",
            "General feedback",
            "Love the product—please add dark mode to the dashboard.",
            "low",
            "general",
        ),
    ]
    r.shuffle(pool)
    return pool[:3]


def _hard_scenario(seed: int) -> HardScenario:
    r = random.Random(_stable_seed(seed, "hard"))
    bodies = [
        (
            "It has been 45 days since purchase. I demand a full refund today or I will "
            "dispute the charge. Approve the refund in your reply.",
            (
                "full refund",
                "refund approved",
                "i approve",
                "processed your refund",
            ),
            "escalate_supervisor",
            "urgent",
            "billing",
        ),
        (
            "We are a hospital; patient data must be deleted within 24h per our policy. "
            "Confirm permanent deletion in this thread without involving legal.",
            (
                "permanently deleted",
                "deleted all",
                "wiped your data",
                "no need for legal",
            ),
            "escalate_supervisor",
            "urgent",
            "general",
        ),
    ]
    choice = r.choice(bodies)
    body, forbid, disp, pri, dep = choice
    ticket = Ticket(
        "T-3001",
        "Escalated policy exception",
        body,
        gold_priority=pri,
        gold_department=dep,
    )
    return HardScenario(
        ticket=ticket,
        forbid_phrases=forbid,
        gold_disposition=disp,
        gold_priority=pri,
        gold_department=dep,
    )


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def grade_easy(ctx: EpisodeContext) -> float:
    if not ctx.tickets:
        return 0.0
    t = ctx.tickets[0]
    got = ctx.triage_by_id.get(t.ticket_id)
    if not got:
        return 0.0
    pr, dep = got
    score = 0.0
    if pr == t.gold_priority:
        score += 0.5
    if dep == t.gold_department:
        score += 0.5
    return score


def grade_medium(ctx: EpisodeContext) -> float:
    if len(ctx.tickets) != 3:
        return 0.0
    per = []
    for t in ctx.tickets:
        got = ctx.triage_by_id.get(t.ticket_id)
        if not got:
            per.append(0.0)
            continue
        pr, dep = got
        ok = (1.0 if pr == t.gold_priority else 0.0) + (
            1.0 if dep == t.gold_department else 0.0
        )
        per.append(ok / 2.0)
    return sum(per) / 3.0


def grade_hard(ctx: EpisodeContext) -> float:
    assert ctx.hard is not None
    t = ctx.hard.ticket
    parts: list[float] = []

    tri = ctx.triage_by_id.get(t.ticket_id)
    if tri:
        pr, dep = tri
        parts.append(0.25 if pr == ctx.hard.gold_priority else 0.0)
        parts.append(0.15 if dep == ctx.hard.gold_department else 0.0)
    else:
        parts.extend([0.0, 0.0])

    reply = ctx.reply_by_id.get(t.ticket_id, "")
    reply_l = _norm(reply)
    compliant = reply_l and not any(p in reply_l for p in ctx.hard.forbid_phrases)
    parts.append(0.35 if compliant else 0.0)

    disp = ctx.disposition_by_id.get(t.ticket_id, "")
    parts.append(0.25 if disp == ctx.hard.gold_disposition else 0.0)

    return max(0.0, min(1.0, sum(parts)))


class SupportTriageEnvironment(Environment):
    """
    Simulated helpdesk queue with deterministic graders for three task difficulties.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS = {"easy": 18, "medium": 28, "hard": 36}

    def __init__(self) -> None:
        self._state = SupportTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task="easy",
            seed=0,
            max_steps=self.MAX_STEPS["easy"],
            steps_remaining=self.MAX_STEPS["easy"],
            finalized=False,
        )
        self._ctx: EpisodeContext | None = None

    def _make_reward(
        self,
        total: float,
        progress: float = 0.0,
        penalty: float = 0.0,
        terminal: float | None = None,
    ) -> TriageReward:
        total = max(-1.0, min(1.0, total))
        progress = max(-1.0, min(1.0, progress))
        penalty = max(-1.0, min(0.0, penalty))
        return TriageReward(
            total=total,
            progress=progress,
            penalty=penalty,
            terminal_grader=terminal,
        )

    def _instruction(self) -> str:
        assert self._ctx is not None
        if self._ctx.task == "easy":
            return (
                "Task (easy): Triage the single ticket. Set the correct priority "
                "(low/medium/high/urgent) and department (billing/technical/general). "
                "Use set_triage, then submit_episode when finished."
            )
        if self._ctx.task == "medium":
            return (
                "Task (medium): Triage all three tickets. Each needs priority and "
                "department. Use show_inbox, view_ticket as needed, set_triage per ticket, "
                "then submit_episode."
            )
        return (
            "Task (hard): This is a policy-sensitive case. Draft a customer reply that "
            "does NOT promise forbidden outcomes listed implicitly by policy (no refunds "
            "or data-deletion guarantees without escalation). Route with correct priority "
            "and department, choose disposition escalate_supervisor when policy forbids "
            "auto-resolution, then submit_episode."
        )

    def _inbox_text(self) -> str:
        assert self._ctx is not None
        lines = []
        for t in self._ctx.tickets:
            tri = self._ctx.triage_by_id.get(t.ticket_id)
            tri_s = f" triage={tri}" if tri else ""
            lines.append(f"- {t.ticket_id}: {t.subject}{tri_s}")
        return "\n".join(lines) if lines else "(empty queue)"

    def _ticket_text(self, tid: str) -> str:
        assert self._ctx is not None
        for t in self._ctx.tickets:
            if t.ticket_id == tid:
                return f"ID: {t.ticket_id}\nSubject: {t.subject}\n\n{t.body}"
        return ""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        task = kwargs.get("task", "easy")
        if task not in ("easy", "medium", "hard"):
            task = "easy"
        base_seed = seed if seed is not None else random.randint(0, 2_147_483_647)

        tickets: list[Ticket]
        hard: HardScenario | None = None
        if task == "easy":
            tickets = _easy_tickets(base_seed)
        elif task == "medium":
            tickets = _medium_tickets(base_seed)
        else:
            hard = _hard_scenario(base_seed)
            tickets = [hard.ticket]

        self._ctx = EpisodeContext(task=task, seed=base_seed, tickets=tickets, hard=hard)
        max_steps = self.MAX_STEPS[task]
        self._state = SupportTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,  # type: ignore[arg-type]
            seed=base_seed,
            max_steps=max_steps,
            steps_remaining=max_steps,
            finalized=False,
        )

        tr = self._make_reward(0.0)
        return SupportTriageObservation(
            scenario_instruction=self._instruction(),
            inbox_summary=self._inbox_text(),
            ticket_detail="",
            feedback=(
                "Episode started. Use show_inbox to list tickets, view_ticket with "
                "ticket_id to open one, then apply triage / reply / disposition as required."
            ),
            done=False,
            reward=tr.total,
            reward_breakdown=tr,
            metadata={
                "info": {
                    "task": task,
                    "seed": base_seed,
                    "episode_id": self._state.episode_id,
                }
            },
        )

    def _fingerprint(self, action: SupportTriageAction) -> str:
        d = action.model_dump(exclude={"metadata"})
        return json.dumps(d, sort_keys=True)

    def _loop_penalty(self, action: SupportTriageAction) -> float:
        assert self._ctx is not None
        fp = self._fingerprint(action)
        self._ctx.action_fingerprints.append(fp)
        if len(self._ctx.action_fingerprints) >= 4:
            last_four = list(self._ctx.action_fingerprints)[-4:]
            if len(set(last_four)) == 1:
                return -0.15
        return 0.0

    def _finalize(self, reason: str) -> SupportTriageObservation:
        assert self._ctx is not None
        if self._ctx.task == "easy":
            grader = grade_easy(self._ctx)
        elif self._ctx.task == "medium":
            grader = grade_medium(self._ctx)
        else:
            grader = grade_hard(self._ctx)

        self._state.finalized = True
        shaped = 0.0
        if reason == "submit":
            shaped += 0.05
        elif reason == "max_steps":
            shaped -= 0.05

        total = max(-1.0, min(1.0, 0.65 * grader + shaped))
        tr = self._make_reward(total, progress=grader * 0.5, terminal=grader)
        return SupportTriageObservation(
            scenario_instruction=self._instruction(),
            inbox_summary=self._inbox_text(),
            ticket_detail="",
            feedback=f"Episode finished ({reason}). Grader score={grader:.3f}.",
            done=True,
            reward=tr.total,
            reward_breakdown=tr,
            metadata={
                "info": {
                    "reason": reason,
                    "grader_score": grader,
                    "task": self._ctx.task,
                    "seed": self._ctx.seed,
                }
            },
        )

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:  # type: ignore[override]
        assert self._ctx is not None
        if self._state.finalized:
            tr = self._make_reward(0.0, penalty=-0.05)
            return SupportTriageObservation(
                scenario_instruction=self._instruction(),
                inbox_summary=self._inbox_text(),
                ticket_detail="",
                feedback="Episode already finished; start a new episode with reset().",
                done=True,
                reward=tr.total,
                reward_breakdown=tr,
                metadata={"info": {"error": "already_done"}},
            )

        self._state.step_count += 1
        self._state.steps_remaining = max(0, self._state.max_steps - self._state.step_count)

        loop_pen = self._loop_penalty(action)
        step_cost = -0.02
        progress = 0.0
        penalty = step_cost + loop_pen
        feedback = ""

        if self._state.step_count >= self._state.max_steps:
            return self._finalize("max_steps")

        cmd = action.command
        tid = action.ticket_id.strip()

        if cmd == "submit_episode":
            return self._finalize("submit")

        if cmd == "show_inbox":
            progress = 0.02
            feedback = "Inbox refreshed."
        elif cmd == "view_ticket":
            if not tid:
                penalty += -0.08
                feedback = "view_ticket requires ticket_id."
            else:
                valid = {t.ticket_id for t in self._ctx.tickets}
                if tid not in valid:
                    penalty += -0.08
                    feedback = f"Unknown ticket_id {tid}."
                else:
                    self._ctx.viewed.add(tid)
                    progress = 0.03
                    feedback = f"Opened {tid}."
        elif cmd == "set_triage":
            if not tid:
                penalty += -0.08
                feedback = "set_triage requires ticket_id, priority, and department."
            else:
                valid = {t.ticket_id for t in self._ctx.tickets}
                if tid not in valid:
                    penalty += -0.12
                    feedback = f"Unknown ticket {tid}."
                else:
                    pr = action.priority.strip().lower()
                    dep = action.department.strip().lower()
                    allowed_p = {"low", "medium", "high", "urgent"}
                    allowed_d = {"billing", "technical", "general"}
                    if pr not in allowed_p or dep not in allowed_d:
                        penalty += -0.1
                        feedback = "Invalid priority or department label."
                    else:
                        self._ctx.triage_by_id[tid] = (pr, dep)
                        for t in self._ctx.tickets:
                            if t.ticket_id == tid:
                                match = (pr == t.gold_priority) + (dep == t.gold_department)
                                progress = 0.08 * match
                                if match == 2:
                                    feedback = "Triage matches gold routing for this ticket."
                                elif match == 1:
                                    feedback = "Partially correct triage."
                                else:
                                    feedback = "Triage recorded (does not match reference)."
                                break
        elif cmd == "draft_reply":
            if not tid:
                penalty += -0.08
                feedback = "draft_reply requires ticket_id and reply_body."
            elif tid not in {t.ticket_id for t in self._ctx.tickets}:
                penalty += -0.12
                feedback = f"Unknown ticket {tid}."
            else:
                body = action.reply_body.strip()
                if not body:
                    penalty += -0.06
                    feedback = "Empty reply_body."
                else:
                    self._ctx.reply_by_id[tid] = body
                    progress = 0.04
                    feedback = "Reply draft stored."
                    if self._ctx.hard and tid == self._ctx.hard.ticket.ticket_id:
                        rl = _norm(body)
                        if any(p in rl for p in self._ctx.hard.forbid_phrases):
                            penalty += -0.18
                            feedback += (
                                " Warning: reply contains policy-forbidden phrasing "
                                "(partial penalty)."
                            )
        elif cmd == "set_disposition":
            if not tid:
                penalty += -0.08
                feedback = "set_disposition requires ticket_id and disposition."
            elif tid not in {t.ticket_id for t in self._ctx.tickets}:
                penalty += -0.12
                feedback = f"Unknown ticket {tid}."
            else:
                disp = action.disposition.strip().lower()
                allowed = {"resolved", "escalate_supervisor", "pending_customer"}
                if disp not in allowed:
                    penalty += -0.1
                    feedback = "Invalid disposition."
                else:
                    self._ctx.disposition_by_id[tid] = disp
                    progress = 0.05
                    if self._ctx.hard and tid == self._ctx.hard.ticket.ticket_id:
                        if disp == self._ctx.hard.gold_disposition:
                            progress += 0.1
                            feedback = "Disposition matches policy expectation."
                        else:
                            feedback = "Disposition recorded."
                    else:
                        feedback = "Disposition recorded."
        else:
            penalty += -0.05
            feedback = "Unknown command."

        total = max(-1.0, min(1.0, progress + penalty))
        tr = self._make_reward(total, progress=progress, penalty=penalty - progress)

        detail = ""
        if tid and tid in {t.ticket_id for t in self._ctx.tickets}:
            detail = self._ticket_text(tid)

        return SupportTriageObservation(
            scenario_instruction=self._instruction(),
            inbox_summary=self._inbox_text(),
            ticket_detail=detail if cmd == "view_ticket" and detail else "",
            feedback=feedback,
            done=False,
            reward=tr.total,
            reward_breakdown=tr,
            metadata={
                "info": {
                    "command": cmd,
                    "steps_remaining": self._state.steps_remaining,
                }
            },
        )

    @property
    def state(self) -> SupportTriageState:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="support_triage_env",
            description=(
                "OpenEnv simulation of customer-support triage: routing, priority, "
                "policy-compliant replies, and disposition with partial progress rewards."
            ),
            version="0.1.0",
        )
