"""Provider-switching tests for `build_ai_provider` — verifies `AI_PROVIDER`
alone (a `.env`-only change) selects the right concrete class. Provider
constructors don't make network calls (SDK clients connect lazily on first
use), so this needs no API keys or network access.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.factories.ai_provider_factory import build_ai_provider
from app.providers.ai.anthropic_provider import AnthropicProvider
from app.providers.ai.gemini_provider import GeminiProvider
from app.providers.ai.groq_provider import GroqProvider
from app.providers.ai.ollama_provider import OllamaProvider
from app.providers.ai.openai_provider import OpenAIProvider


@pytest.mark.parametrize(
    "provider_name,expected_class",
    [
        ("ollama", OllamaProvider),
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("gemini", GeminiProvider),
        ("groq", GroqProvider),
    ],
)
def test_build_ai_provider_selects_class_from_setting(provider_name, expected_class) -> None:
    settings = Settings(AI_PROVIDER=provider_name)

    provider = build_ai_provider(settings)

    assert isinstance(provider, expected_class)


def test_build_ai_provider_is_case_insensitive() -> None:
    settings = Settings(AI_PROVIDER="OLLAMA")

    provider = build_ai_provider(settings)

    assert isinstance(provider, OllamaProvider)


def test_build_ai_provider_rejects_unknown_provider() -> None:
    settings = Settings(AI_PROVIDER="not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown AI_PROVIDER"):
        build_ai_provider(settings)
