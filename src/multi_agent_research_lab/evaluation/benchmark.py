"""Benchmark helpers for single-agent vs multi-agent."""

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Rough blended price used only when the provider does not return explicit cost.
# Unit: USD per token. Keep this conservative and visible in the report notes.
DEFAULT_INPUT_TOKEN_PRICE = 0.10 / 1_000_000
DEFAULT_OUTPUT_TOKEN_PRICE = 0.40 / 1_000_000


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one system and compute reproducible benchmark metrics.

    The benchmark intentionally combines objective runtime metrics with simple heuristic
    quality signals. Human review should still be used for the final score, but these
    fields make the report useful before a peer reviewer reads every answer.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
    latency = perf_counter() - started

    return state, evaluate_state(run_name, state, latency)


def evaluate_state(run_name: str, state: ResearchState, latency_seconds: float) -> BenchmarkMetrics:
    """Convert a completed `ResearchState` into a benchmark metric object."""

    final_answer = state.final_answer or ""
    citation_markers = _citation_markers(final_answer)
    citation_coverage = _citation_coverage(citation_markers, len(state.sources))
    input_tokens = _sum_token_metadata(state, "input_tokens")
    output_tokens = _sum_token_metadata(state, "output_tokens")
    estimated_cost = _estimated_cost(input_tokens, output_tokens)
    quality_score, quality_breakdown = _heuristic_quality_score(state, citation_coverage)

    metrics = BenchmarkMetrics(run_name=run_name, latency_seconds=latency_seconds)
    metrics.citation_coverage = citation_coverage
    metrics.failure_rate = 1.0 if state.errors else 0.0
    metrics.input_tokens = input_tokens
    metrics.output_tokens = output_tokens
    metrics.estimated_cost_usd = estimated_cost
    metrics.quality_score = quality_score
    metrics.source_count = len(state.sources)
    metrics.citation_count = len(citation_markers)
    metrics.answer_chars = len(final_answer)
    metrics.agent_steps = len(state.agent_results)
    metrics.trace_events = len(state.trace)
    metrics.route_count = len(state.route_history)
    metrics.error_count = len(state.errors)
    metrics.quality_breakdown = quality_breakdown
    metrics.notes = "; ".join(state.errors) if state.errors else "completed"
    return metrics


def _citation_markers(answer: str) -> set[str]:
    return set(CITATION_PATTERN.findall(answer))


def _citation_coverage(citation_markers: set[str], source_count: int) -> float:
    if source_count == 0:
        return 0.0
    valid_markers = {marker for marker in citation_markers if 1 <= int(marker) <= source_count}
    return min(1.0, len(valid_markers) / source_count)


def _sum_token_metadata(state: ResearchState, key: str) -> int | None:
    total = sum(int(result.metadata.get(key) or 0) for result in state.agent_results)
    return total or None


def _estimated_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0) * DEFAULT_INPUT_TOKEN_PRICE + (
        output_tokens or 0
    ) * DEFAULT_OUTPUT_TOKEN_PRICE


def _heuristic_quality_score(
    state: ResearchState, citation_coverage: float
) -> tuple[float, dict[str, float]]:
    answer = state.final_answer or ""
    breakdown: dict[str, float] = {
        "base": 2.0,
        "answer_depth": 0.0,
        "research_handoff": 0.0,
        "analysis_handoff": 0.0,
        "citations": 0.0,
        "failure_mode": 0.0,
        "traceability": 0.0,
        "error_penalty": 0.0,
    }

    if len(answer) >= 1200:
        breakdown["answer_depth"] = 1.5
    elif len(answer) >= 600:
        breakdown["answer_depth"] = 1.0
    elif len(answer) >= 250:
        breakdown["answer_depth"] = 0.5

    if state.research_notes:
        breakdown["research_handoff"] = 1.0
    if state.analysis_notes:
        breakdown["analysis_handoff"] = 1.0
    if "failure mode" in answer.lower() or "lỗi" in answer.lower():
        breakdown["failure_mode"] = 1.0
    if state.trace:
        breakdown["traceability"] = 1.0

    breakdown["citations"] = min(1.5, citation_coverage * 1.5)
    if state.errors:
        breakdown["error_penalty"] = -2.0

    score = sum(breakdown.values())
    return max(0.0, min(10.0, score)), breakdown
