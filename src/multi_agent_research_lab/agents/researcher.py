"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        with trace_span(self.name, {"query": state.request.query}) as span:
            state.sources = self.search.search(state.request.query, state.request.max_sources)
            source_block = "\n".join(
                f"[{index}] {source.title} - {source.url or 'no url'}\n{source.snippet}"
                for index, source in enumerate(state.sources, start=1)
            )
            response = self.llm.complete(
                (
                    "You are a careful English-only research agent. "
                    "All research notes must be written in English."
                ),
                (
                    f"Research question: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Sources:\n{source_block}\n\n"
                    "Write 5-7 concise research notes in English. Use bracket citations like [1]. "
                    "Do not invent sources beyond the supplied list."
                ),
            )
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "source_count": len(state.sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "provider": self.llm.settings.llm_provider,
                        "model": self.llm.model,
                    },
                )
            )
            span["source_count"] = len(state.sources)
            state.add_trace_event(self.name, span)
        return state
