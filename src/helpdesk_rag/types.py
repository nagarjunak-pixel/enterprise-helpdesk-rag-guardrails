"""Strict Pydantic v2 domain models for Enterprise Help-Desk RAG & Guardrails."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UserRole = Literal["employee", "it_support", "secops_admin"]


class KBArticle(BaseModel):
    """Curated enterprise help-desk knowledge base article."""

    doc_id: str
    title: str
    category: Literal["IAM", "Network", "Cloud", "SAP", "HR", "SecOps"]
    allowed_roles: list[UserRole]
    content: str
    sla_tier: str = "L1-SelfService"


class GuardrailScanResult(BaseModel):
    """C++20 / Python guardrail & PII inspection verdict."""

    has_prompt_injection: bool
    has_secret_key: bool
    has_card_or_ssn: bool
    has_email: bool
    backend: str


class RetrievedChunk(BaseModel):
    """Scored KB chunk returned by C++20 hybrid BM25 + Cosine + RRF retrieval."""

    doc_id: str
    title: str
    category: str
    cosine_similarity: float
    bm25_score: float
    hybrid_confidence: float
    snippet: str


DecisionStatus = Literal[
    "ANSWERED_GROUNDED",
    "REFUSED_LOW_CONFIDENCE",
    "BLOCKED_PROMPT_INJECTION",
    "REFUSED_UNGROUNDED_ANSWER",
]


class HelpDeskTicketRequest(BaseModel):
    """Incoming employee / engineer help-desk question with configurable thresholds."""

    question: str = Field(..., min_length=1)
    user_role: UserRole = "employee"
    min_similarity_threshold: float = Field(default=0.28, ge=0.0, le=1.0)
    min_confidence_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    min_grounding_overlap: float = Field(default=0.50, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=10)


class HelpDeskTicketResponse(BaseModel):
    """Governed RAG answer or deterministic code-enforced refusal."""

    decision: DecisionStatus
    refused: bool
    backend: str
    sanitized_question: str
    pii_redacted: bool
    top_similarity: float
    top_confidence: float
    grounding_overlap: float
    citations: list[str]
    answer: str
    escalation_queue: str | None = None
    guardrail_scan: GuardrailScanResult
    retrieved_chunks: list[RetrievedChunk]


class AcceptanceCaseResult(BaseModel):
    """Single golden acceptance test outcome."""

    case_id: str
    category: str
    expected_decision: DecisionStatus
    actual_decision: DecisionStatus
    passed: bool
    top_similarity: float
    citations: list[str]


class AcceptanceSuiteReport(BaseModel):
    """Aggregate report across the automated help-desk acceptance benchmark."""

    total_cases: int
    passed_cases: int
    pass_rate_pct: float
    false_answer_rate_on_oos_pct: float
    injection_block_rate_pct: float
    backend: str
    results: list[AcceptanceCaseResult]
