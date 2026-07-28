"""Contract every LLM provider (Ollama, Anthropic, OpenAI, Gemini, Groq) must
satisfy so `AIService` never has to know which one is behind it.

Structural typing (`Protocol`) rather than an ABC — providers don't need to
import this module to satisfy it, which keeps each `providers/ai/*.py` free
to stay a thin wrapper around its SDK.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AIProvider(Protocol):
    """A chat/completion-capable LLM backend."""

    @property
    def model_name(self) -> str:
        """Model identifier, used in `model_used` audit columns."""
        ...

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        """Extract structured JSON from `text` conforming to `schema`."""
        ...

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        """Generate a free-form conversational reply.

        `messages` is oldest-first, each `{"role": "user"|"assistant", "content": str}`.
        """
        ...
