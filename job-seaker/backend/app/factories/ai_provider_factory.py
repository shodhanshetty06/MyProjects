"""Picks the concrete `AIProvider` implementation from `settings.AI_PROVIDER`.

The single place that knows about every AI provider's constructor — nothing
outside this module (and `dependency_injection/providers.py`, which calls
it) should import a `providers/ai/*` class directly.
"""

from __future__ import annotations

from app.core.config import Settings
from app.interfaces.ai_provider import AIProvider


def build_ai_provider(settings: Settings) -> AIProvider:
    provider = settings.AI_PROVIDER.lower()

    if provider == "ollama":
        from app.providers.ai.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    if provider == "anthropic":
        from app.providers.ai.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if provider == "openai":
        from app.providers.ai.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if provider == "gemini":
        from app.providers.ai.gemini_provider import GeminiProvider

        return GeminiProvider(settings)
    if provider == "groq":
        from app.providers.ai.groq_provider import GroqProvider

        return GroqProvider(settings)

    raise ValueError(
        f"Unknown AI_PROVIDER '{settings.AI_PROVIDER}'. "
        "Expected one of: ollama, anthropic, openai, gemini, groq."
    )


__all__ = ["build_ai_provider"]
