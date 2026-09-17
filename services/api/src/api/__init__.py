"""Chat API service.

A thin HTTP shell over ``chatbot_core``: it owns transport, conversation
lifetime, and rate limiting, and delegates every decision about the
assistant's behaviour to the project's configuration.
"""

from .app import AppState, create_app
from .settings import APISettings

__version__ = "0.1.0"

__all__ = ["APISettings", "AppState", "__version__", "create_app"]
