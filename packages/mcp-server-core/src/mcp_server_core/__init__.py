"""mcp-server-core -- the base MCP server every project extends.

Projects contribute tools by dropping a module with a ``register(server)``
function into their ``tools/`` directory. This package supplies the server
construction, the discovery mechanism, the transports, and the built-in tools
available everywhere.
"""

from .registry import (
    ToolRegistrationError,
    register_directory,
    register_package,
)
from .server import build_server
from .settings import ServerSettings

__version__ = "0.1.0"

__all__ = [
    "ServerSettings",
    "ToolRegistrationError",
    "__version__",
    "build_server",
    "register_directory",
    "register_package",
]
