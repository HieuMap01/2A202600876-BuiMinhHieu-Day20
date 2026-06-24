# Design Template

## Problem

Build a research assistant that answers long-form technical questions by collecting source notes, analyzing tradeoffs, and producing a cited final answer. The system must compare a single-agent baseline with a multi-agent workflow.

## Why multi-agent?

A single agent is simpler and often faster, but it mixes retrieval, analysis, and writing in one prompt. The multi-agent workflow separates responsibilities so each step can be traced, debugged, retried, and scored independently.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route the workflow and stop when enough state exists | `ResearchState` | next route in `route_history` | loops forever or skips a needed worker |
| Researcher | Retrieve source documents and write cited notes | query, audience, max sources | `sources`, `research_notes` | weak sources, missing citations, LLM/provider failure |
| Analyst | Turn research notes into claims, tradeoffs, gaps, and outline | `research_notes` | `analysis_notes` | shallow analysis or unsupported claims |
| Writer | Synthesize final Vietnamese answer with citations | query, research notes, analysis notes | `final_answer` | citation hallucination or missing failure-mode discussion |
| Critic | Run basic final-answer checks | `final_answer` | critic findings in `agent_results` | only catches surface-level issues |

## Shared state

`ResearchState` is the single handoff object. It stores the original `ResearchQuery`, current `iteration`, `route_history`, retrieved `sources`, `research_notes`, `analysis_notes`, `final_answer`, `agent_results`, `trace`, and `errors`. These fields make the run debuggable because every worker writes its output to a named location instead of passing hidden context.

## Routing policy

```text
supervisor
  -> researcher if research_notes is missing
  -> analyst    if analysis_notes is missing
  -> writer     if final_answer is missing
  -> done       once final_answer exists or max_iterations is reached
```

The implemented workflow loops through the supervisor after each worker. Unknown routes and provider errors are recorded in `state.errors`.

## Guardrails

- Max iterations: loaded from `MAX_ITERATIONS`, default `6`.
- Timeout: loaded from `TIMEOUT_SECONDS`, default `60`, passed to the LLM client.
- Retry: LLM client retries transient failures with exponential backoff, but avoids retrying budget/rate-limit 429 errors.
- Fallback: workflow records errors and stops cleanly instead of crashing; benchmark converts failures into metrics.
- Validation: report tracks failure rate, citation coverage, quality heuristic, tokens, and trace availability.

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| Compare single-agent and multi-agent workflows for customer support | latency, quality, citation coverage, failure rate, token usage | single-agent should be faster; multi-agent should be more traceable and structured |
| Research GraphRAG state-of-the-art and write a 500-word summary | same metrics | multi-agent should produce clearer research and analysis handoff |
| Summarize production guardrails for LLM agents | same metrics | multi-agent should expose guardrail reasoning in trace |

Current real-provider run reached the configured LLM gateway but failed with `budget_exceeded` HTTP 429, so the generated benchmark report records 100% failure rate for that run. Re-running with a funded key should produce normal answers and token metrics.
