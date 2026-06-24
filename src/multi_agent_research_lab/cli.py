"""Command-line entrypoint for the lab starter."""

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline_state(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    sources = SearchClient().search(query, request.max_sources)
    state.sources = sources
    source_block = "\n".join(
        f"[{index}] {source.title} - {source.url or 'no url'}\n{source.snippet}"
        for index, source in enumerate(sources, start=1)
    )
    response = LLMClient().complete(
        (
            "You are an English-only single-agent research assistant. "
            "All headings, bullets, tables, and prose must be written in English."
        ),
        (
            f"Question: {query}\n"
            f"Audience: {request.audience}\n\n"
            f"Available sources:\n{source_block}\n\n"
            "Do the full task yourself: extract relevant evidence, analyze tradeoffs, "
            "write the final answer in English unless the user explicitly requests another language, "
            "include bracket citations, and end "
            "with a short failure mode plus mitigation."
        ),
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.BASELINE,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "provider": LLMClient().settings.llm_provider,
                "model": LLMClient().model,
            },
        )
    )
    state.add_trace_event("baseline", {"model": LLMClient().model, "source_count": len(sources)})
    return state


def _run_multi_state(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    try:
        state = _run_baseline_state(query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Research query to benchmark",
        ),
    ] = "Compare single-agent and multi-agent workflows for customer support",
) -> None:
    """Run baseline and multi-agent workflows, then write a markdown report."""

    _init()
    baseline_state, baseline_metrics = run_benchmark(
        "single-agent baseline", query, _run_baseline_state
    )
    multi_state, multi_metrics = run_benchmark("multi-agent workflow", query, _run_multi_state)
    report = render_markdown_report(
        [baseline_metrics, multi_metrics],
        {
            baseline_metrics.run_name: baseline_state,
            multi_metrics.run_name: multi_state,
        },
    )
    path = LocalArtifactStore().write_text("benchmark_report.md", report)
    console.print(Panel.fit(str(path), title="Benchmark Report Written"))


if __name__ == "__main__":
    app()
