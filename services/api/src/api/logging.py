"""Structured logging.

``console`` for a human at a terminal, ``json`` for a log aggregator. Both go
to stdout, which is where a container runtime expects application logs.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))

    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )

    # uvicorn's access log duplicates what we already record per request, and
    # at INFO it drowns the structured lines that actually carry context.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # These libraries log every HTTP request at INFO. During startup the
    # embedding model alone issues dozens of calls to huggingface.co, which
    # buries the handful of lines that say whether the service came up. Raise
    # them to WARNING; set LOG_LEVEL=DEBUG when you actually want the traffic.
    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


__all__ = ["configure_logging"]
