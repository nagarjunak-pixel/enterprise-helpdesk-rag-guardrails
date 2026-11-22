"""Stdio JSON-RPC 2.0 MCP server exposing governed help-desk RAG & acceptance tools."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from helpdesk_rag import __version__
from helpdesk_rag.acceptance import run_acceptance_suite
from helpdesk_rag.engine import HelpDeskRAGEngine
from helpdesk_rag.kernels import scan_guardrails_and_pii
from helpdesk_rag.types import HelpDeskTicketRequest

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "helpdesk_list_kb",
        "description": "List enterprise help-desk KB articles and their RBAC role scopes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "helpdesk_answer_ticket",
        "description": "Answer an enterprise help-desk ticket with C++20 hybrid RAG and code-enforced low-confidence refusal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "user_role": {"type": "string", "enum": ["employee", "it_support", "secops_admin"]},
                "min_similarity_threshold": {"type": "number"},
                "min_confidence_threshold": {"type": "number"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "helpdesk_scan_guardrails",
        "description": "Run C++20 prompt-injection & PII/secret scanner on raw ticket text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "helpdesk_run_acceptance",
        "description": "Execute the 25-case automated golden acceptance test suite.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class MCPHelpDeskSession:
    """Stateful MCP session wrapping `HelpDeskRAGEngine`."""

    def __init__(self, engine: HelpDeskRAGEngine | None = None) -> None:
        self.engine = engine if engine is not None else HelpDeskRAGEngine()

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "helpdesk_list_kb":
            return [a.model_dump(mode="json") for a in self.engine.articles]
        if name == "helpdesk_answer_ticket":
            req = HelpDeskTicketRequest.model_validate(args)
            return self.engine.answer_ticket(req).model_dump(mode="json")
        if name == "helpdesk_scan_guardrails":
            return scan_guardrails_and_pii(str(args.get("text", ""))).model_dump(mode="json")
        if name == "helpdesk_run_acceptance":
            return run_acceptance_suite(self.engine).model_dump(mode="json")
        raise ValueError(f"Unknown MCP tool: {name!r}")

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method", "")
        req_id = req.get("id")

        if method == "notifications/initialized" or req_id is None:
            return None
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "enterprise-helpdesk-rag-guardrails",
                        "version": __version__,
                    },
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}
        if method == "tools/call":
            params = req.get("params") or {}
            tname = str(params.get("name", ""))
            targs = dict(params.get("arguments") or {})
            try:
                payload = self._call_tool(tname, targs)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                        "isError": True,
                    },
                }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def run_stdio_server(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Serve newline-delimited JSON-RPC 2.0 MCP requests over stdio."""
    in_s = stdin if stdin is not None else sys.stdin
    out_s = stdout if stdout is not None else sys.stdout
    session = MCPHelpDeskSession()
    for raw_line in in_s:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            out_s.write(json.dumps(err) + "\n")
            out_s.flush()
            continue
        resp = session.handle_request(req)
        if resp is not None:
            out_s.write(json.dumps(resp) + "\n")
            out_s.flush()
