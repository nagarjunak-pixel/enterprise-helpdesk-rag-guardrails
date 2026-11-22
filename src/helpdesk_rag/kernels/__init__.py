"""Native C++20 hybrid BM25 + Cosine + RRF, PII/injection & grounding kernels."""

from __future__ import annotations

from helpdesk_rag.kernels.native import (
    backend_name,
    benchmark_kernels,
    citation_token_overlap,
    hybrid_bm25_cosine_rrf,
    scan_guardrails_and_pii,
)

__all__ = [
    "backend_name",
    "benchmark_kernels",
    "citation_token_overlap",
    "hybrid_bm25_cosine_rrf",
    "scan_guardrails_and_pii",
]
