"""Tests for code-enforced refusal, grounding check, RBAC & 25-case acceptance benchmark."""

from __future__ import annotations

from helpdesk_rag.acceptance import run_acceptance_suite
from helpdesk_rag.engine import HelpDeskRAGEngine
from helpdesk_rag.types import HelpDeskTicketRequest, KBArticle


def test_full_25_case_automated_acceptance_suite_passes_100_percent() -> None:
    rep = run_acceptance_suite()
    assert rep.total_cases == 25
    assert rep.passed_cases == 25
    assert rep.pass_rate_pct == 100.0
    assert rep.false_answer_rate_on_oos_pct == 0.0
    assert rep.injection_block_rate_pct == 100.0


def test_ungrounded_hallucination_draft_and_empty_role_refusal() -> None:
    eng = HelpDeskRAGEngine()
    ungrounded = eng.answer_ticket(
        HelpDeskTicketRequest(
            question="How do I reset my lost Okta Verify MFA phone device?",
            user_role="employee",
        ),
        injected_ungrounded_draft=(
            "Wire bitcoin to external wallet address immediately to unlock mainframe quantum tape."
        ),
    )
    assert ungrounded.refused is True
    assert ungrounded.decision == "REFUSED_UNGROUNDED_ANSWER"
    assert ungrounded.escalation_queue == "L2-Human-HelpDesk"

    # Engine with only SecOps-admin articles refuses regular employee cleanly
    restricted_only = HelpDeskRAGEngine(
        articles=[
            KBArticle(
                doc_id="DOC-ONLY-ADMIN",
                title="Root HSM Ceremony",
                category="SecOps",
                allowed_roles=["secops_admin"],
                content="Hardware security module quorum unlock procedure.",
            )
        ]
    )
    r_empty = restricted_only.answer_ticket(
        HelpDeskTicketRequest(question="Root HSM Ceremony", user_role="employee")
    )
    assert r_empty.refused is True
    assert r_empty.decision == "REFUSED_LOW_CONFIDENCE"

    # Pure stopword / punctuation query embeds to zero norm and refuses cleanly
    r_stop = eng.answer_ticket(HelpDeskTicketRequest(question="??? ..."))
    assert r_stop.refused is True
    assert r_stop.decision == "REFUSED_LOW_CONFIDENCE"
