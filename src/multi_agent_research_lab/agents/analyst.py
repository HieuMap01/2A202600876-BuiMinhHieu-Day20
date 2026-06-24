"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        with trace_span(self.name, {"query": state.request.query}) as span:
            response = self.llm.complete(
                (
                    "You are an English-only analyst agent. "
                    "Turn research notes into structured English insights."
                ),
                (
                    f"Question: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes or ''}\n\n"
                    "Produce:\n"
                    "1. Key claims with citations.\n"
                    "2. Tradeoffs or disagreements.\n"
                    "3. Evidence gaps and failure risks.\n"
                    "4. Recommended answer outline.\n"
                    "Write in English."
                ),
            )
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "provider": self.llm.settings.llm_provider,
                        "model": self.llm.model,
                    },
                )
            )
            state.add_trace_event(self.name, span)
        return state
