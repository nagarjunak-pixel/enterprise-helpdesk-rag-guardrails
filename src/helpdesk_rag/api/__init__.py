"""FastAPI HTTP control plane for Enterprise Help-Desk RAG & Guardrails (Port 8280)."""

from __future__ import annotations

from helpdesk_rag.api.app import app, create_app

__all__ = ["app", "create_app"]
