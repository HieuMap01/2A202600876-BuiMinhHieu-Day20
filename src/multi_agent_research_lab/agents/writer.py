"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        with trace_span(self.name, {"query": state.request.query}) as span:
            response = self.llm.complete(
                (
                    "You are an English-only technical writer. "
                    "Synthesize a final answer with citations using English only."
                ),
                (
                    f"Question: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Research notes:\n{state.research_notes or ''}\n\n"
                    f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
                    "Write a clear final answer in English unless the user explicitly requests another language. "
                    "Include citation markers such as [1], [2] when making sourced claims. "
                    "End with a short 'Failure mode' note and mitigation."
                ),
            )
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
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
