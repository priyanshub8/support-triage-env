"""Deterministic grader checks (no server)."""

from support_triage_env.models import SupportTriageAction
from support_triage_env.server.support_triage_env_environment import SupportTriageEnvironment


def _oracle_easy(env: SupportTriageEnvironment) -> float:
    t = env._ctx.tickets[0]
    env.step(
        SupportTriageAction(
            command="set_triage",
            ticket_id=t.ticket_id,
            priority=t.gold_priority,
            department=t.gold_department,
        )
    )
    fin = env.step(SupportTriageAction(command="submit_episode"))
    assert fin.reward_breakdown and fin.reward_breakdown.terminal_grader is not None
    return fin.reward_breakdown.terminal_grader


def test_easy_perfect_grader_seed_42() -> None:
    env = SupportTriageEnvironment()
    env.reset(seed=42, task="easy")
    assert _oracle_easy(env) == 1.0


def test_medium_perfect_grader_seed_42() -> None:
    env = SupportTriageEnvironment()
    env.reset(seed=42, task="medium")
    for t in env._ctx.tickets:
        env.step(
            SupportTriageAction(
                command="set_triage",
                ticket_id=t.ticket_id,
                priority=t.gold_priority,
                department=t.gold_department,
            )
        )
    fin = env.step(SupportTriageAction(command="submit_episode"))
    assert fin.reward_breakdown
    assert fin.reward_breakdown.terminal_grader == 1.0


def test_hard_perfect_grader_seed_42() -> None:
    env = SupportTriageEnvironment()
    env.reset(seed=42, task="hard")
    h = env._ctx.hard
    assert h is not None
    t = h.ticket
    env.step(
        SupportTriageAction(
            command="set_triage",
            ticket_id=t.ticket_id,
            priority=h.gold_priority,
            department=h.gold_department,
        )
    )
    env.step(
        SupportTriageAction(
            command="draft_reply",
            ticket_id=t.ticket_id,
            reply_body=(
                "Thank you for reaching out. I cannot commit to a refund in this thread; "
                "a specialist will follow up shortly."
            ),
        )
    )
    env.step(
        SupportTriageAction(
            command="set_disposition",
            ticket_id=t.ticket_id,
            disposition=h.gold_disposition,
        )
    )
    fin = env.step(SupportTriageAction(command="submit_episode"))
    assert fin.reward_breakdown
    assert fin.reward_breakdown.terminal_grader == 1.0
