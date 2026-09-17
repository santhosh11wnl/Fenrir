"""Provider abstraction.

Every provider takes the same inputs (system prompt, history, a tool executor)
and emits the same :mod:`~chatbot_core.events` stream. That uniformity is what
lets one project run on a fine-tuned Hugging Face model while the rest stay on
Claude, with no change above this layer.

On owning the agent loop
------------------------
Each provider drives its own request/execute/repeat loop rather than delegating
to the Anthropic SDK's ``tool_runner``. The runner is the right default for most
applications, but it doesn't fit here for two reasons:

1. We emit ``tool_call`` *before* the tool runs and ``tool_result`` *after*, with
   a measured duration. The runner executes tools as part of its own iteration,
   so a pending-state UI would have nothing to render against.
2. The tool set is discovered at runtime from MCP and is not known statically,
   and the same loop shape has to work for a provider with no runner at all.

The tradeoff is that we own termination and error handling; ``max_iterations``
bounds the loop and every tool failure is fed back as a tool result rather than
raised, so the model can recover instead of the turn dying.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..config import ProjectConfig
from ..events import ChatEvent

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class Message:
    """One conversation turn, provider-independent.

    ``content`` is plain text. Provider-specific block structures (tool_use,
    thinking, etc.) are rebuilt inside each provider's loop and never leak into
    the conversation store -- so history stays portable if a project switches
    providers.
    """

    role: Role
    content: str


@dataclass(slots=True)
class ToolSpec:
    """A tool as advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ToolOutcome:
    """The result of running one tool.

    A failure is an outcome, not an exception: ``is_error=True`` with a message
    explaining what went wrong gets fed back to the model, which can then try a
    different approach. Raising would end the turn and strand the user.
    """

    content: str
    is_error: bool = False
    #: Retrieval tools attach the chunks they matched so the engine can emit
    #: citations without re-parsing the tool's text output.
    sources: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class ToolExecutor(Protocol):
    """Anything that can advertise and run tools.

    Implemented by the MCP client, by the retrieval tool, and by the composite
    that merges them. Providers depend on this and nothing more.
    """

    @property
    def specs(self) -> Sequence[ToolSpec]: ...

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome: ...


class NullExecutor:
    """No tools. Used when MCP is disabled and retrieval is prepended rather
    than exposed as a tool."""

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return ()

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(content=f"No such tool: {name}", is_error=True)


class ChatProviderBase(abc.ABC):
    """Base class for chat providers."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        executor: ToolExecutor,
        conversation_id: str,
    ) -> AsyncIterator[ChatEvent]:
        """Run one assistant turn, yielding events as they happen.

        Must terminate. Must not raise for recoverable problems -- yield an
        ``ErrorEvent`` and then a ``Done`` instead, so the client always sees a
        well-formed end to the stream.
        """
        ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release provider resources. Safe to call more than once.

        Deliberately concrete and empty rather than abstract: a stateless
        provider has nothing to release, and forcing every subclass to write
        an empty override is noise that hides the ones that matter.
        """


__all__ = [
    "ChatProviderBase",
    "Message",
    "NullExecutor",
    "Role",
    "ToolExecutor",
    "ToolOutcome",
    "ToolSpec",
]
