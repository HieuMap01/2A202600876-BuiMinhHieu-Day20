"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append lightweight findings."""

        answer = state.final_answer or ""
        findings: list[str] = []
        if not answer:
            findings.append("No final answer to review.")
        if "[" not in answer or "]" not in answer:
            findings.append("Final answer has no bracket citation markers.")
        if "failure mode" not in answer.lower():
            findings.append("Final answer does not mention a failure mode.")
        if not findings:
            findings.append("Basic citation and failure-mode checks passed.")

        content = "\n".join(f"- {finding}" for finding in findings)
        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=content, metadata={"check_count": len(findings)})
        )
        state.add_trace_event("critic", {"findings": findings})
        return state
