"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Small deterministic source retriever used when no web-search API is configured."""

    def __init__(self, corpus: list[SourceDocument] | None = None) -> None:
        self.corpus = corpus or [
            SourceDocument(
                title="Anthropic: Building effective agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Practical guidance for composing LLM workflows, choosing agentic "
                    "patterns only when needed, and adding evaluation loops."
                ),
            ),
            SourceDocument(
                title="OpenAI Agents SDK orchestration and handoffs",
                url="https://developers.openai.com/api/docs/guides/agents/orchestration",
                snippet=(
                    "Explains agent orchestration, handoffs, specialized roles, and "
                    "tool-mediated workflows for multi-agent systems."
                ),
            ),
            SourceDocument(
                title="LangGraph concepts",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "Describes graph-based control flow, state passing, conditional edges, "
                    "and durable execution patterns for agent workflows."
                ),
            ),
            SourceDocument(
                title="LangSmith tracing",
                url="https://docs.smith.langchain.com/",
                snippet=(
                    "Documentation for tracing, evaluating, and debugging LLM application "
                    "runs across steps and datasets."
                ),
            ),
            SourceDocument(
                title="Langfuse tracing and observability",
                url="https://langfuse.com/docs",
                snippet=(
                    "Observability platform for LLM traces, scores, costs, latency, and "
                    "debugging production agent systems."
                ),
            ),
        ]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return ranked local documents relevant to a query."""

        query_terms = {term.lower().strip(".,:;!?()[]") for term in query.split()}

        def score(document: SourceDocument) -> int:
            haystack = f"{document.title} {document.snippet}".lower()
            return sum(1 for term in query_terms if term and term in haystack)

        ranked = sorted(self.corpus, key=score, reverse=True)
        return ranked[:max_results]
