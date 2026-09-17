"""Server entry point.

Two transports, one command:

``stdio``
    The server runs as a child process, speaking MCP over stdin/stdout. This is
    how Claude Code and Claude Desktop launch it, and how you test tools
    without any HTTP in the picture::

        claude mcp add my-tools -- uv run mcp-server --transport stdio

    Nothing may be written to stdout except protocol frames -- a stray
    ``print`` corrupts the stream and the client disconnects with a parse
    error. Logging is routed to stderr for exactly this reason.

``streamable-http``
    The server is its own service, which is what deployments use so the chat
    API can scale independently of the tool runtime::

        uv run mcp-server --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog

from .server import build_server
from .settings import ServerSettings


def configure_logging(level: str, transport: str) -> None:
    """Send logs to stderr, always.

    Under stdio, stdout is the protocol channel: a single log line written
    there breaks the session. Using stderr unconditionally means the two
    transports behave identically and there is no mode-specific footgun.
    """
    logging.basicConfig(
        format="%(message)s", stream=sys.stderr, level=getattr(logging, level)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=transport != "stdio"),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mcp-server", description="Run the platform MCP server."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="streamable-http",
        help="stdio for a local subprocess; streamable-http for a deployed service",
    )
    parser.add_argument("--project", help="Project whose tools to load (overrides env)")
    parser.add_argument("--host", help="Bind address for HTTP transports")
    parser.add_argument("--port", type=int, help="Port for HTTP transports")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # CLI beats environment: the env carries the deployment default, the flag
    # is a deliberate per-invocation override.
    overrides = {
        k: v
        for k, v in {
            "project": args.project,
            "host": args.host,
            "port": args.port,
            "log_level": args.log_level,
        }.items()
        if v is not None
    }
    settings = ServerSettings(**overrides)

    configure_logging(settings.log_level, args.transport)
    log = structlog.get_logger(__name__)

    try:
        server = build_server(settings)
    except Exception:
        log.exception("server_build_failed")
        return 1

    if args.transport == "stdio":
        log.info("starting", transport="stdio")
        server.run(transport="stdio")
        return 0

    # Bind options are run() kwargs in mcp 2.x -- `server.settings` carries only
    # behavioural flags and has no host/port field. The path parameter is named
    # per transport, hence the split below.
    kwargs: dict[str, object] = {"host": settings.host, "port": settings.port}
    if args.transport == "streamable-http":
        kwargs["streamable_http_path"] = settings.mount_path
    else:
        kwargs["mount_path"] = settings.mount_path

    log.info(
        "starting",
        transport=args.transport,
        url=f"http://{settings.host}:{settings.port}{settings.mount_path}",
    )
    server.run(transport=args.transport, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
