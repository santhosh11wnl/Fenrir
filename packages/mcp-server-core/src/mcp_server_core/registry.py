"""Tool registration.

The extension point that keeps seven projects from forking this package. A
project contributes tools by exposing a module with a ``register(server)``
function; the loader discovers it and calls it. Adding a project means adding a
module, never editing this file -- open for extension, closed for modification.

A project's tool module looks like::

    # projects/<name>/tools/billing.py
    from mcp.server.mcpserver import MCPServer

    def register(server: MCPServer) -> None:
        @server.tool()
        def lookup_invoice(invoice_id: str) -> str:
            '''Look up an invoice by its id.'''
            ...
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)

#: The function a tool module must expose.
REGISTER_ATTR = "register"


@runtime_checkable
class ToolModule(Protocol):
    """Structural type for a module that contributes tools."""

    def register(self, server: Any) -> None: ...


RegisterFn = Callable[[Any], None]


class ToolRegistrationError(RuntimeError):
    """A tool module was found but could not be loaded.

    Raised rather than logged: a project deployed without the tools it declares
    is a silently broken assistant, and that is worse than a failed boot.
    """


def register_package(server: Any, package: str) -> list[str]:
    """Register every tool module in an importable package.

    Args:
        server: The ``MCPServer`` to register onto.
        package: Dotted package path, e.g. ``mcp_server_core.tools``.

    Returns:
        Names of the modules that registered tools, in load order.
    """
    try:
        pkg = importlib.import_module(package)
    except ImportError as exc:
        raise ToolRegistrationError(f"cannot import tool package {package!r}: {exc}") from exc

    paths: Iterable[str] = getattr(pkg, "__path__", [])
    loaded: list[str] = []

    for info in sorted(pkgutil.iter_modules(paths), key=lambda m: m.name):
        # Leading underscore marks a helper module, not a tool module.
        if info.name.startswith("_"):
            continue
        module_path = f"{package}.{info.name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ToolRegistrationError(f"cannot import {module_path!r}: {exc}") from exc
        if _apply(server, module, module_path):
            loaded.append(module_path)

    return loaded


def register_directory(server: Any, directory: str | Path) -> list[str]:
    """Register every ``*.py`` tool module in a directory.

    Used for project tool directories, which live outside the installed package
    tree and so cannot be reached by dotted import.
    """
    directory = Path(directory)
    if not directory.is_dir():
        log.info("no_tool_directory", path=str(directory))
        return []

    loaded: list[str] = []
    for path in sorted(directory.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module = _load_from_path(path)
        if _apply(server, module, str(path)):
            loaded.append(path.stem)
    return loaded


def _load_from_path(path: Path) -> Any:
    # Namespaced by parent directory so two projects can both have `tools.py`
    # without the second silently reusing the first's cached module object.
    name = f"_project_tools_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ToolRegistrationError(f"cannot load tool module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - surfaced as a registration failure
        raise ToolRegistrationError(f"error importing {path}: {exc}") from exc
    return module


def _apply(server: Any, module: Any, label: str) -> bool:
    """Call a module's ``register`` if it has one. Returns whether it did."""
    register = getattr(module, REGISTER_ATTR, None)
    if register is None:
        log.debug("module_has_no_register", module=label)
        return False
    if not callable(register):
        raise ToolRegistrationError(f"{label}: {REGISTER_ATTR!r} is not callable")
    try:
        register(server)
    except Exception as exc:  # noqa: BLE001 - context matters more than the type
        raise ToolRegistrationError(f"{label}: register() failed: {exc}") from exc
    log.info("tools_registered", module=label)
    return True


__all__ = [
    "REGISTER_ATTR",
    "RegisterFn",
    "ToolModule",
    "ToolRegistrationError",
    "register_directory",
    "register_package",
]
