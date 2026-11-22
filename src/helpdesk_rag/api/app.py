"""FastAPI endpoints for governed enterprise help-desk RAG, refusal thresholds & acceptance suite."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from helpdesk_rag import __version__
from helpdesk_rag.acceptance import run_acceptance_suite
from helpdesk_rag.demo import run_demo_scenarios
from helpdesk_rag.engine import HelpDeskRAGEngine
from helpdesk_rag.kernels import backend_name, benchmark_kernels
from helpdesk_rag.types import (
    AcceptanceSuiteReport,
    HelpDeskTicketRequest,
    HelpDeskTicketResponse,
    KBArticle,
)


def create_app(engine: HelpDeskRAGEngine | None = None) -> FastAPI:
    rag = engine if engine is not None else HelpDeskRAGEngine()
    api = FastAPI(
        title="Enterprise Help-Desk RAG & Guardrails System",
        version=__version__,
        description="Hybrid BM25 + Cosine + RRF Help-Desk RAG with Code-Enforced Low-Confidence Refusal & Automated Acceptance Suite",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "enterprise-helpdesk-rag-guardrails",
            "version": __version__,
            "backend": backend_name(),
        }

    @api.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# HELP helpdesk_rag_queries_total Total help-desk tickets processed",
            "# TYPE helpdesk_rag_queries_total counter",
            f"helpdesk_rag_queries_total {rag.total_queries}",
            "# HELP helpdesk_rag_answered_total Total grounded answers returned",
            "# TYPE helpdesk_rag_answered_total counter",
            f"helpdesk_rag_answered_total {rag.total_answered}",
            "# HELP helpdesk_rag_refused_total Total code-enforced refusals",
            "# TYPE helpdesk_rag_refused_total counter",
            f"helpdesk_rag_refused_total {rag.total_refused}",
            "# HELP helpdesk_rag_injections_blocked_total Total prompt injections blocked",
            "# TYPE helpdesk_rag_injections_blocked_total counter",
            f"helpdesk_rag_injections_blocked_total {rag.total_injections_blocked}",
        ]
        return "\n".join(lines) + "\n"

    @api.get("/api/v1/kb", response_model=list[KBArticle])
    def list_kb() -> list[KBArticle]:
        return rag.articles

    @api.post("/api/v1/ask", response_model=HelpDeskTicketResponse)
    def ask_ep(req: HelpDeskTicketRequest) -> HelpDeskTicketResponse:
        return rag.answer_ticket(req)

    @api.get("/api/v1/acceptance", response_model=AcceptanceSuiteReport)
    def acceptance_ep() -> AcceptanceSuiteReport:
        return run_acceptance_suite(rag)

    @api.get("/api/v1/demo")
    def demo_ep() -> dict[str, Any]:
        return run_demo_scenarios()

    @api.get("/api/v1/bench")
    def bench_ep(num_docs: int = 5000, dim: int = 128) -> dict[str, float | int | str]:
        return benchmark_kernels(num_docs=num_docs, dim=dim)

    return api


app = create_app()
