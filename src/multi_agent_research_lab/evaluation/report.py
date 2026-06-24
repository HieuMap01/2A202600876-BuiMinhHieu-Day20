"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    states: dict[str, ResearchState] | None = None,
) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "This report compares the single-agent baseline with the multi-agent workflow.",
        "",
        "## Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Failure Rate | Tokens | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        tokens = (item.input_tokens or 0) + (item.output_tokens or 0)
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{citation} | {failure} | {tokens} | {item.notes} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if len(metrics) >= 2:
        best_quality = max(metrics, key=lambda item: item.quality_score or 0)
        fastest = min(metrics, key=lambda item: item.latency_seconds)
        lines.append(
            f"Best heuristic quality: **{best_quality.run_name}**. "
            f"Fastest run: **{fastest.run_name}**. "
            "Use these numbers as operational smoke metrics; final quality should still "
            "be peer reviewed with the lab rubric."
        )
    else:
        lines.append("Only one run was recorded, so no cross-run interpretation is available.")

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| Run | Sources | Citations | Answer chars | Agent steps | Routes | Trace events | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics:
        lines.append(
            f"| {item.run_name} | {item.source_count} | {item.citation_count} | "
            f"{item.answer_chars} | {item.agent_steps} | {item.route_count} | "
            f"{item.trace_events} | {item.error_count} |"
        )

    if states:
        lines.extend(["", "## Run Details", ""])
        for item in metrics:
            state = states.get(item.run_name)
            if state is None:
                continue
            lines.extend(_render_run_details(item, state))

    lines.extend(["", "## Quality Breakdown", ""])
    for item in metrics:
        lines.append(f"### {item.run_name}")
        if not item.quality_breakdown:
            lines.append("")
            lines.append("No quality breakdown was recorded.")
            lines.append("")
            continue
        lines.extend(["", "| Component | Score |", "|---|---:|"])
        for name, value in item.quality_breakdown.items():
            label = name.replace("_", " ").title()
            lines.append(f"| {label} | {value:.1f} |")
        lines.append("")

    lines.extend(["## Comparison Notes", ""])
    if len(metrics) >= 2:
        best_quality = max(metrics, key=lambda item: item.quality_score or 0)
        fastest = min(metrics, key=lambda item: item.latency_seconds)
        most_traceable = max(metrics, key=lambda item: item.trace_events)
        lines.extend(
            [
                f"- Best heuristic quality: `{best_quality.run_name}`.",
                f"- Fastest run: `{fastest.run_name}`.",
                f"- Most traceable run: `{most_traceable.run_name}`.",
            ]
        )
    else:
        lines.append("- Only one run was recorded, so no cross-run comparison is available.")

    lines.extend(
        [
            "",
            "## Failure Mode",
            "",
            (
                "The main risk is citation hallucination: the writer may cite a source marker "
                "without the corresponding evidence supporting that exact claim. Mitigation: keep "
                "the researcher constrained to supplied sources, require bracket citations, and "
                "review citation coverage in the benchmark."
            ),
            "",
            "## Trace Explanation",
            "",
            (
                "The multi-agent run records supervisor routing decisions plus each worker span "
            "in `ResearchState.trace`, making it possible to inspect who produced research, "
            "analysis, and final synthesis."
        ),
    ]
    )
    return "\n".join(lines) + "\n"


def _render_run_details(item: BenchmarkMetrics, state: ResearchState) -> list[str]:
    status = "failed" if state.errors else "completed"
    route_history = " -> ".join(state.route_history) if state.route_history else "not routed"
    providers = _providers_from_state(state)
    sources = ", ".join(source.title for source in state.sources) if state.sources else "none"
    errors = "; ".join(state.errors) if state.errors else "none"
    final_answer = state.final_answer or "_No final answer was produced._"

    lines = [
        f"### {item.run_name}",
        "",
        f"- Status: {status}",
        f"- Query: {state.request.query}",
        f"- Route history: {route_history}",
        f"- Providers: {providers}",
        f"- Sources captured: {len(state.sources)}",
        f"- Source titles: {sources}",
        f"- Trace events: {len(state.trace)}",
        f"- Errors: {errors}",
        "",
        "#### Execution Trail",
        "",
    ]

    if state.trace:
        for index, event in enumerate(state.trace, start=1):
            lines.append(f"{index}. {_format_trace_event(event)}")
    else:
        lines.append("No trace events recorded.")

    lines.extend(["", "#### Question Answering Log", ""])
    lines.extend(_question_answering_log(state))

    lines.extend(["", "#### Agent Outputs", ""])
    if state.agent_results:
        for result in state.agent_results:
            provider = result.metadata.get("provider") or providers
            token_text = _token_text(result.metadata)
            lines.append(
                f"- {result.agent.value} via {provider}{token_text}: {_preview(result.content, 240)}"
            )
    else:
        lines.append("- No agent outputs recorded.")

    lines.extend(
        [
            "",
            "#### Full Final Answer",
            "",
            final_answer,
            "",
        ]
    )
    return lines


def _providers_from_state(state: ResearchState) -> str:
    providers = {
        str(result.metadata["provider"])
        for result in state.agent_results
        if result.metadata.get("provider")
    }
    return ", ".join(sorted(providers)) if providers else "configured LLM provider"


def _format_trace_event(event: dict[str, object]) -> str:
    name = str(event.get("name", "trace"))
    payload = event.get("payload", {})
    if isinstance(payload, dict):
        if name == "supervisor.route":
            route = payload.get("next", "unknown")
            iteration = payload.get("iteration", "?")
            flags = {
                "research": payload.get("has_research"),
                "analysis": payload.get("has_analysis"),
                "final": payload.get("has_final"),
            }
            return f"`{name}` - selected `{route}` at iteration {iteration}; state flags: {flags}"
        if "duration_seconds" in payload:
            duration = payload.get("duration_seconds")
            return f"`{name}` - duration={float(duration or 0):.3f}s; payload={payload}"
    return f"`{name}` - {_preview(str(payload), 500)}"


def _question_answering_log(state: ResearchState) -> list[str]:
    lines = [f"1. User asked: {_preview(state.request.query, 240)}"]
    next_index = 2
    if state.sources:
        source_titles = ", ".join(source.title for source in state.sources)
        lines.append(f"{next_index}. Sources collected: {source_titles}.")
        next_index += 1
    if state.research_notes:
        lines.append(f"{next_index}. Research notes: {_preview(state.research_notes, 260)}")
        next_index += 1
    if state.analysis_notes:
        lines.append(f"{next_index}. Analysis notes: {_preview(state.analysis_notes, 260)}")
        next_index += 1
    if state.final_answer:
        words = len(state.final_answer.split())
        lines.append(f"{next_index}. Final answer was produced with {words} words.")
    else:
        lines.append(f"{next_index}. Final answer was not produced.")
    return lines


def _token_text(metadata: dict[str, object]) -> str:
    input_tokens = metadata.get("input_tokens")
    output_tokens = metadata.get("output_tokens")
    if not input_tokens and not output_tokens:
        return ""
    return f" ({input_tokens or 0} input tokens, {output_tokens or 0} output tokens)"


def _preview(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."
