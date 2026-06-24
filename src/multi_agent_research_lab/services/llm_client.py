"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def model(self) -> str:
        return self.settings.llm_model or self.settings.openai_model

    @property
    def api_key(self) -> str | None:
        return self.settings.llm_api_key or self.settings.openai_api_key

    @retry(
        retry=retry_if_exception(
            lambda exc: all(
                marker not in str(exc).lower()
                for marker in ("402", "429", "budget", "insufficient credits")
            )
        ),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion using an OpenAI-compatible chat API."""

        if not self.api_key:
            raise AgentExecutionError("Missing LLM_API_KEY or OPENAI_API_KEY in environment.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise AgentExecutionError(
                "The openai package is not installed. Run `pip install -e \".[dev,llm]\"`."
            ) from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.settings.llm_base_url,
            timeout=float(self.settings.timeout_seconds),
            max_retries=0,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # pragma: no cover - provider/network specific
            raise AgentExecutionError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage: Any = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
