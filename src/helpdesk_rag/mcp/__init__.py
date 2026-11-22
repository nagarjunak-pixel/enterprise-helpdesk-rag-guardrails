"""Stdio JSON-RPC 2.0 Model Context Protocol (MCP) server for Enterprise Help-Desk RAG."""

from __future__ import annotations

from helpdesk_rag.mcp.server import MCPHelpDeskSession, run_stdio_server

__all__ = ["MCPHelpDeskSession", "run_stdio_server"]
