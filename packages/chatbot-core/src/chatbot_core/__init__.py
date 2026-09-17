"""chatbot-core -- the base every project is stamped from.

A project supplies config, tools, and a corpus. This package supplies the
engine: provider abstraction, retrieval, MCP tool use, and the streaming event
protocol shared with the API and the web client.
"""

from .config import ProjectConfig
from .engine import ChatEngine
from .events import (
    ChatEvent,
    Citations,
    Done,
    ErrorEvent,
    EventType,
    MessageStart,
    Source,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    Usage,
)
from .providers import Message
from .storage import Conversation, ConversationStore, InMemoryConversationStore

__version__ = "0.1.0"

__all__ = [
    "ChatEngine",
    "ChatEvent",
    "Citations",
    "Conversation",
    "ConversationStore",
    "Done",
    "ErrorEvent",
    "EventType",
    "InMemoryConversationStore",
    "Message",
    "MessageStart",
    "ProjectConfig",
    "Source",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "Usage",
    "__version__",
]
