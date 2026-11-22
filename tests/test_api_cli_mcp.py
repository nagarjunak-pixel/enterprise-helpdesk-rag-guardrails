"""End-to-end tests for FastAPI REST endpoints, Typer CLI commands, and stdio MCP server."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from helpdesk_rag.api import create_app
from helpdesk_rag.cli import app as cli_app
from helpdesk_rag.mcp import MCPHelpDeskSession, run_stdio_server

runner = CliRunner()


def test_fastapi_endpoints_and_prometheus_metrics() -> None:
    client = TestClient(create_app())

    assert client.get("/health").status_code == 200
    kb = client.get("/api/v1/kb")
    assert kb.status_code == 200 and len(kb.json()) == 8

    ok_q = client.post(
        "/api/v1/ask",
        json={"question": "How do I reset my lost Okta Verify MFA phone device?"},
    )
    assert ok_q.status_code == 200 and ok_q.json()["decision"] == "ANSWERED_GROUNDED"

    ref_q = client.post(
        "/api/v1/ask",
        json={"question": "What is the cafeteria pizza menu tomorrow?"},
    )
    assert ref_q.status_code == 200 and ref_q.json()["refused"] is True

    acc = client.get("/api/v1/acceptance")
    assert acc.status_code == 200 and acc.json()["pass_rate_pct"] == 100.0

    demo = client.get("/api/v1/demo")
    assert demo.status_code == 200 and len(demo.json()["highlights"]) == 4

    bench = client.get("/api/v1/bench?num_docs=200&dim=32")
    assert bench.status_code == 200 and bench.json()["num_docs"] == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "helpdesk_rag_queries_total" in metrics.text


def test_typer_cli_commands() -> None:
    assert runner.invoke(cli_app, ["kb"]).exit_code == 0
    assert (
        runner.invoke(
            cli_app, ["ask", "How do I reset my lost Okta Verify MFA phone device?"]
        ).exit_code
        == 0
    )
    assert runner.invoke(cli_app, ["ask", "What is tomorrow's weather on Mars?"]).exit_code == 2
    assert runner.invoke(cli_app, ["ask", "test", "--role", "invalid_role"]).exit_code != 0

    assert runner.invoke(cli_app, ["acceptance"]).exit_code == 0
    assert runner.invoke(cli_app, ["acceptance", "--json"]).exit_code == 0
    assert runner.invoke(cli_app, ["demo"]).exit_code == 0
    assert runner.invoke(cli_app, ["demo", "--json"]).exit_code == 0
    assert runner.invoke(cli_app, ["bench", "--docs", "200", "--dim", "32"]).exit_code == 0

    r_mcp = runner.invoke(
        cli_app, ["mcp"], input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    )
    assert r_mcp.exit_code == 0 and '"id": 1' in r_mcp.stdout


def test_mcp_jsonrpc_session_and_stdio() -> None:
    session = MCPHelpDeskSession()
    assert session.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    init_r = session.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert (
        init_r is not None
        and init_r["result"]["serverInfo"]["name"] == "enterprise-helpdesk-rag-guardrails"
    )

    list_r = session.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_r is not None and len(list_r["result"]["tools"]) == 4

    calls: list[tuple[str, dict[str, object]]] = [
        ("helpdesk_list_kb", {}),
        (
            "helpdesk_answer_ticket",
            {"question": "How do I reset my lost Okta Verify MFA phone device?"},
        ),
        ("helpdesk_scan_guardrails", {"text": "reach alice@corp.com"}),
        ("helpdesk_run_acceptance", {}),
    ]
    for idx, (tname, targs) in enumerate(calls, start=10):
        resp = session.handle_request(
            {
                "jsonrpc": "2.0",
                "id": idx,
                "method": "tools/call",
                "params": {"name": tname, "arguments": targs},
            }
        )
        assert resp is not None and resp["result"]["isError"] is False

    bad_tool = session.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 50,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert bad_tool is not None and bad_tool["result"]["isError"] is True

    nf = session.handle_request({"jsonrpc": "2.0", "id": 51, "method": "unknown"})
    assert nf is not None and nf["error"]["code"] == -32601

    raw_in = io.StringIO(
        "\nnot-json\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"})
        + "\n"
    )
    raw_out = io.StringIO()
    run_stdio_server(stdin=raw_in, stdout=raw_out)
    lines = [json.loads(x) for x in raw_out.getvalue().strip().splitlines()]
    assert len(lines) == 2 and lines[0]["error"]["code"] == -32700
