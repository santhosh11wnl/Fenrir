"""Entry point: ``uv run control-api``.

Serves the registry, project config writes, and the per-site admin proxy.
Unlike the chat API there is no ``PROJECT`` here -- this process is the one
piece of the platform that legitimately sees every site at once.
"""

from __future__ import annotations

import argparse

import uvicorn

from api.logging import configure_logging

from .settings import ControlSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="control-api", description="Run the control plane."
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--registry", help="Path to registry.yaml")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    overrides = {
        k: v
        for k, v in {
            "host": args.host,
            "port": args.port,
            "registry_file": args.registry,
        }.items()
        if v is not None
    }
    settings = ControlSettings(**overrides)
    configure_logging(settings.log_level, settings.log_format)

    if not settings.projects_dir.is_dir():
        parser.error(f"no projects directory at {settings.projects_dir}")

    uvicorn.run(
        "control.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_config=None,  # structlog owns formatting
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
