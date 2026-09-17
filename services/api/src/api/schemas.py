"""Request and response models for the HTTP surface.

Separate from ``chatbot_core.events`` on purpose: those are the *streaming*
contract, these are the request/response bodies. Keeping them apart means the
wire format for a POST body can change without touching the event protocol the
React client renders.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    #: Omit to start a new conversation; the id comes back on the first event.
    conversation_id: str | None = None
    #: Stable per-browser id the widget mints and keeps in first-party
    #: storage, so analytics can tell a returning visitor from a new one
    #: without a login. Constrained because it is client-supplied and lands in
    #: a primary key: an unbounded string here would be an unbounded row.
    #: Omitted by clients that do not track visitors; those turns are recorded
    #: against a per-conversation placeholder and simply never retain.
    visitor_id: str | None = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{8,64}$"
    )


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class ConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    project: str
    messages: list[ConversationTurn]
    created_at: float
    updated_at: float


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class HealthResponse(BaseModel):
    """Also the data source for the admin dashboard's per-project tile."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "starting"]
    project: str
    model: str
    provider: str
    ready: bool
    mcp_connected: bool
    tools: list[str]
    indexed_chunks: int


class ThemeResponse(BaseModel):
    """Project branding, so one React build can serve every project."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    name: str
    description: str
    primary: str
    accent: str
    logo_url: str | None
    greeting: str
    placeholder: str
    suggestions: list[str]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str


__all__ = [
    "ChatRequest",
    "ConversationResponse",
    "ConversationTurn",
    "ErrorResponse",
    "HealthResponse",
    "ThemeResponse",
    "ToolInfo",
]
