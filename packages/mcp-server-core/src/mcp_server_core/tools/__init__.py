"""Built-in tools, available to every project.

Modules here are discovered and registered automatically by
:func:`mcp_server_core.registry.register_package`. To add one, drop in a module
exposing ``register(server)`` -- no edit to this file, and no edit to the
server factory.

Modules whose names start with ``_`` are treated as helpers and skipped.
"""
