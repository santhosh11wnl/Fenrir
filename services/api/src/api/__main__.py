"""Entry point: ``uv run chat-api``.

Serves whichever project ``PROJECT`` names. Reload is opt-in rather than
inferred from an environment name, because reload rebuilds the engine on every
file change -- including reloading the embedding model.
"""

from __future__ import annotations

import argparse

import uvicorn

from .logging import configure_logging
from .settings import APISettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat-api", description="Run the chat API.")
    parser.add_argument("--project", help="Project to serve (overrides PROJECT)")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes (slow: rebuilds the engine each time)",
    )
    args = parser.parse_args(argv)

    overrides = {
        k: v
        for k, v in {"project": args.project, "host": args.host, "port": args.port}.items()
        if v is not None
    }
    settings = APISettings(**overrides)
    configure_logging(settings.log_level, settings.log_format)

    if not settings.config_path.is_file():
        parser.error(
            f"no config at {settings.config_path}. "
            f"Set PROJECT to a directory under {settings.projects_dir}/."
        )

    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_config=None,  # structlog owns formatting
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
