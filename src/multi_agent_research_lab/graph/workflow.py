"""Workflow orchestration."""

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, SupervisorAgent, WriterAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        agents: dict[str, BaseAgent] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.agents = agents or {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }

    def build(self) -> object:
        """Return a lightweight graph description for debugging and documentation."""

        return {
            "nodes": ["supervisor", *self.agents.keys(), "done"],
            "edges": {
                "supervisor": ["researcher", "analyst", "writer", "done"],
                "researcher": ["supervisor"],
                "analyst": ["supervisor"],
                "writer": ["supervisor"],
            },
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        while state.iteration < self.settings.max_iterations:
            state = self.supervisor.run(state)
            route = state.route_history[-1]
            if route == "done":
                break

            agent = self.agents.get(route)
            if agent is None:
                state.errors.append(f"Unknown route: {route}")
                break

            try:
                state = agent.run(state)
            except AgentExecutionError as exc:
                state.errors.append(f"{route}: {exc}")
                state.add_trace_event("agent.error", {"agent": route, "error": str(exc)})
                if route != "writer" and state.research_notes:
                    state.record_route("writer")
                    state = self.agents["writer"].run(state)
                break

        if not state.final_answer:
            state.errors.append("Workflow ended without final_answer.")
        return state
