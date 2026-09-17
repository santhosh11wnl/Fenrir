"""Provider selection.

Imports are deferred so a project that only uses Claude never pays for
``huggingface_hub`` being installed -- and one that only uses Hugging Face
isn't forced to have an Anthropic key present.
"""

from __future__ import annotations

from ..config import ChatProvider, ProjectConfig
from .base import (
    ChatProviderBase,
    Message,
    NullExecutor,
    Role,
    ToolExecutor,
    ToolOutcome,
    ToolSpec,
)


def build_provider(config: ProjectConfig) -> ChatProviderBase:
    """Construct the provider a project's config selects.

    Adding a provider means adding a module and one arm here -- no existing
    branch changes, and nothing above this function knows the difference.
    """
    match config.model.provider:
        case ChatProvider.LOCAL:
            from .openai_compatible import OpenAICompatibleProvider

            return OpenAICompatibleProvider(config)
        case ChatProvider.HUGGINGFACE:
            from .huggingface_provider import HuggingFaceProvider

            return HuggingFaceProvider(config)
        case ChatProvider.ANTHROPIC:
            try:
                from .anthropic_provider import AnthropicProvider
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "provider 'anthropic' needs an optional dependency: "
                    "uv add 'chatbot-core[anthropic]'. The default provider is "
                    "'local', which has no external dependency."
                ) from exc

            return AnthropicProvider(config)
    raise ValueError(f"unknown chat provider: {config.model.provider}")


__all__ = [
    "ChatProviderBase",
    "Message",
    "NullExecutor",
    "Role",
    "ToolExecutor",
    "ToolOutcome",
    "ToolSpec",
    "build_provider",
]
