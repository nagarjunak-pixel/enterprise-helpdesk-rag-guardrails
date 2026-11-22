"""Typer CLI entrypoint for Enterprise Help-Desk RAG & Guardrails System."""

from __future__ import annotations

import json
from typing import Literal

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from helpdesk_rag.acceptance import run_acceptance_suite
from helpdesk_rag.demo import run_demo_scenarios
from helpdesk_rag.engine import HelpDeskRAGEngine
from helpdesk_rag.kb import DEFAULT_KB_ARTICLES
from helpdesk_rag.kernels import benchmark_kernels
from helpdesk_rag.mcp import run_stdio_server
from helpdesk_rag.types import HelpDeskTicketRequest, UserRole

app = typer.Typer(
    name="helpdesk-rag",
    help="Enterprise Help-Desk Hybrid RAG & Guardrails System (C++20 + Refusal Gates + Acceptance Suite).",
    no_args_is_help=True,
)
console = Console()


@app.command("kb")
def kb_cmd() -> None:
    """List curated enterprise help-desk knowledge base articles."""
    typer.echo(json.dumps([a.model_dump(mode="json") for a in DEFAULT_KB_ARTICLES], indent=2))


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(...),
    role: str = typer.Option("employee", "--role", "-r"),
    min_sim: float = typer.Option(0.28, "--min-sim"),
    min_conf: float = typer.Option(0.34, "--min-conf"),
) -> None:
    """Query the governed Help-Desk RAG assistant with similarity & confidence thresholds."""
    if role not in {"employee", "it_support", "secops_admin"}:
        raise typer.BadParameter("role must be employee, it_support, or secops_admin")
    typed_role: Literal["employee", "it_support", "secops_admin"] = role  # type: ignore[assignment]
    user_role: UserRole = typed_role
    eng = HelpDeskRAGEngine()
    resp = eng.answer_ticket(
        HelpDeskTicketRequest(
            question=question,
            user_role=user_role,
            min_similarity_threshold=min_sim,
            min_confidence_threshold=min_conf,
        )
    )
    typer.echo(json.dumps(resp.model_dump(mode="json"), indent=2))
    if resp.refused:
        raise typer.Exit(code=2)


@app.command("acceptance")
def acceptance_cmd(json_out: bool = typer.Option(False, "--json")) -> None:
    """Execute the 25-case automated acceptance benchmark."""
    rep = run_acceptance_suite()
    if json_out:
        typer.echo(json.dumps(rep.model_dump(mode="json"), indent=2))
    else:
        console.print(
            f"[bold]Acceptance Suite:[/bold] {rep.passed_cases}/{rep.total_cases} passed "
            f"({rep.pass_rate_pct}%), OOS false-answer rate={rep.false_answer_rate_on_oos_pct}%, "
            f"injection block rate={rep.injection_block_rate_pct}% (backend={rep.backend})"
        )
    if rep.passed_cases != rep.total_cases:
        raise typer.Exit(code=1)


@app.command("demo")
def demo_cmd(json_out: bool = typer.Option(False, "--json")) -> None:
    """Run enterprise Help-Desk RAG & Guardrails scenarios + 25-case acceptance + C++20 benchmark."""
    res = run_demo_scenarios()
    if json_out:
        typer.echo(json.dumps(res, indent=2))
        return

    tbl = Table(title="Enterprise Help-Desk RAG & Guardrails — Live Verdict Summary")
    tbl.add_column("Ticket")
    tbl.add_column("Decision")
    tbl.add_column("Top Sim / Conf")
    tbl.add_column("Citations / Escalation")

    for h in res["highlights"]:
        tbl.add_row(
            h["sanitized_question"][:54],
            h["decision"],
            f"{h['top_similarity']:.3f} / {h['top_confidence']:.3f}",
            ",".join(h["citations"]) or (h["escalation_queue"] or "-"),
        )
    console.print(tbl)
    acc = res["acceptance_summary"]
    bench = res["kernel_benchmark"]
    console.print(
        f"Automated Acceptance Tests: {acc['passed_cases']}/{acc['total_cases']} ({acc['pass_rate_pct']}%) | "
        f"OOS Hallucination Rate: {acc['false_answer_rate_on_oos_pct']}% | "
        f"C++20 Hybrid Benchmark ({bench['num_docs']} chunks, dim={bench['embedding_dim']}): "
        f"C++20={bench['native_ms']} ms vs Python={bench['python_ms']} ms ({bench['speedup_x']}x speedup)"
    )


@app.command("bench")
def bench_cmd(
    docs: int = typer.Option(5000, "--docs"),
    dim: int = typer.Option(128, "--dim"),
) -> None:
    """Benchmark C++20 hybrid BM25 + Cosine + RRF & guardrail kernels vs Python."""
    typer.echo(json.dumps(benchmark_kernels(num_docs=docs, dim=dim), indent=2))


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8280, "--port"),
) -> None:
    """Start the FastAPI control plane on port 8280."""
    uvicorn.run("helpdesk_rag.api.app:app", host=host, port=port)  # pragma: no cover


@app.command("mcp")
def mcp_cmd() -> None:
    """Start the stdio JSON-RPC 2.0 Model Context Protocol (MCP) server."""
    run_stdio_server()
