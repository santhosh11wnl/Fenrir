#!/usr/bin/env python3
"""Stamp a new project from the template.

    uv run python scripts/new_project.py support-bot --name "Support Assistant"

Creates ``projects/<id>/`` with a config, an ingest manifest, a tools module,
and a data directory -- then tells you the three things to do next.

Why a script and not "copy the folder"
--------------------------------------
Copying by hand leaves the template's ``project.id`` in place, which silently
points the new project at the template's vector collection. Two assistants then
share one corpus and neither is obviously wrong -- the answers are just subtly
about the wrong thing. This rewrites identity atomically, or refuses.

The output is a *starting point*, not a finished assistant. A stamped project
answers from an empty knowledge base until you add documents and ingest.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "projects" / "_template"
PROJECTS_DIR = REPO_ROOT / "projects"

#: Mirrors ProjectMeta.id in chatbot_core.config. Kept in sync deliberately:
#: failing here with an explanation beats failing at startup with a
#: ValidationError after the directory already exists.
ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,48}[a-z0-9]$")

#: Fetched documents and built indexes are per-project artefacts, not template
#: content. Copying them would seed a new project with the template's corpus.
SKIP = {".cache", "__pycache__", ".DS_Store"}

GREEN, YELLOW, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[90m",
    "\033[1m",
    "\033[0m",
)


def validate_id(project_id: str) -> str | None:
    """Return an error message, or None if the id is usable."""
    if not ID_PATTERN.match(project_id):
        return (
            f"{project_id!r} is not a valid project id.\n"
            "  Use lowercase letters, digits, and hyphens; start with a letter, "
            "end with a letter or digit; 3-50 characters.\n"
            "  Examples: support-bot, patient-portal, sevencells-docs"
        )
    if project_id.startswith("_"):
        return "project ids cannot start with an underscore (reserved for templates)"
    return None


def _yaml_str(value: str) -> str:
    """Render a string as an unambiguous YAML scalar.

    An empty value written bare parses as null, and unquoted text containing
    a colon, hash, or leading special character parses as something other than
    a string. Quoting sidesteps the whole category.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def copy_template(destination: Path) -> list[Path]:
    """Copy the template, skipping per-project artefacts."""
    created: list[Path] = []
    for source in sorted(TEMPLATE_DIR.rglob("*")):
        relative = source.relative_to(TEMPLATE_DIR)
        if any(part in SKIP for part in relative.parts):
            continue
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created.append(target)
    return created


def rewrite_config(path: Path, project_id: str, name: str, description: str) -> None:
    """Replace the template's identity block with the new project's.

    Line-oriented and scoped to the ``project:`` block on purpose: a blind
    string replace of "template-bot" would also hit prose in the system prompt
    and any comment mentioning it.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_project_block = False

    for line in lines:
        if line.startswith("project:"):
            in_project_block = True
            out.append(line)
            continue

        # A non-indented, non-blank line ends the block.
        if in_project_block and line and not line[0].isspace():
            in_project_block = False

        if in_project_block:
            stripped = line.strip()
            if stripped.startswith("id:"):
                out.append(f"  id: {project_id}")
                continue
            if stripped.startswith("name:"):
                out.append(f"  name: {_yaml_str(name)}")
                continue
            if stripped.startswith("description:"):
                # Quote it: an empty value written bare parses as YAML null,
                # and the field is a string. Quoting also survives any
                # punctuation a description happens to contain.
                out.append(f"  description: {_yaml_str(description)}")
                continue

        out.append(line)

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def disable_auth(path: Path) -> None:
    """Force auth off in a newly stamped project.

    A new project has no users. If it inherited `auth.enabled: true` from the
    template it would refuse to start -- correctly, since it would reject every
    request -- but that makes stamping feel broken.

    Setting it explicitly here rather than relying on the template's value means
    the template is free to ship auth *on* (so the secure path is what you see
    by default) while a new project is still runnable the moment it is created.
    Turn it on once you have added users.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    in_auth = False
    for i, line in enumerate(lines):
        if line.startswith("auth:"):
            in_auth = True
            continue
        if in_auth and line and not line[0].isspace():
            break
        if in_auth and line.strip().startswith("enabled:"):
            lines[i] = "  enabled: false  # no users yet; see scripts/manage_users.py"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_placeholder_corpus(destination: Path) -> None:
    """Remove the template's demo documents and URL sources.

    The template indexes public PEPs so a fresh checkout demonstrates
    retrieval. Inheriting those into a real project would mean a support bot
    that confidently answers questions about Python style guides.
    """
    for leftover in (destination / "data").glob("*"):
        if leftover.is_file():
            leftover.unlink()

    (destination / "data" / ".gitkeep").touch()

    manifest = destination / "ingest.yaml"
    manifest.write_text(
        f"""# What to index into this project's knowledge base.
#
# Run after editing:
#     uv run python -m chatbot_core.ingest projects/{destination.name}
#     # or, in Docker:  PROJECT={destination.name} mcpreingest
#
# Ingest is idempotent: a document's existing chunks are removed before its new
# ones are written, so re-running after an edit updates rather than duplicates.
#
# Supported: .md .txt .rst .csv .json .yaml  |  .pdf  |  .html  |  .docx
# A document that can't be extracted honestly (a scanned PDF with no text
# layer, say) is skipped and reported -- never indexed as noise.
#
# Paths are relative to this project directory.

sources:
  - type: directory
    path: ./data
    include:
      - "**/*.md"
      - "**/*.txt"
      - "**/*.pdf"
      - "**/*.html"
      - "**/*.docx"
    exclude:
      - "**/drafts/**"
    metadata:
      source_type: documentation

  # Documents fetched over HTTP are cached under .cache/, so re-ingesting does
  # not re-download. Set `refresh: true` to force a refetch.
  #
  # - type: url
  #   url: https://example.com/handbook.pdf
  #   title: Employee Handbook
  #   metadata:
  #     source_type: policy
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a new project from the template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  uv run python scripts/new_project.py support-bot \\\n"
            '      --name "Support Assistant" \\\n'
            '      --description "Answers questions about our product"'
        ),
    )
    parser.add_argument("id", help="Project id: lowercase, hyphens, 3-50 chars")
    parser.add_argument("--name", help="Human-readable name (default: derived from id)")
    parser.add_argument("--description", default="", help="One-line description")
    parser.add_argument(
        "--keep-sample-data",
        action="store_true",
        help="Keep the template's demo corpus (public PEPs) instead of starting empty",
    )
    args = parser.parse_args()

    if (error := validate_id(args.id)) is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not TEMPLATE_DIR.is_dir():
        print(f"error: template not found at {TEMPLATE_DIR}", file=sys.stderr)
        return 1

    destination = PROJECTS_DIR / args.id
    if destination.exists():
        print(
            f"error: {destination.relative_to(REPO_ROOT)} already exists.\n"
            f"  Remove it first, or choose a different id.",
            file=sys.stderr,
        )
        return 1

    name = args.name or args.id.replace("-", " ").title()

    copy_template(destination)
    rewrite_config(destination / "config.yaml", args.id, name, args.description)
    disable_auth(destination / "config.yaml")
    if not args.keep_sample_data:
        strip_placeholder_corpus(destination)

    # Prove the result is loadable rather than asserting it. A project that
    # fails validation should be caught here, not on first boot.
    try:
        sys.path.insert(0, str(REPO_ROOT / "packages" / "chatbot-core" / "src"))
        from chatbot_core.config import ProjectConfig

        config = ProjectConfig.load(destination / "config.yaml")
    except Exception as exc:  # noqa: BLE001 - report and keep the directory
        print(f"\n{YELLOW}warning: the generated config did not validate:{RESET}")
        print(f"  {exc}")
        print(f"  Files are at {destination.relative_to(REPO_ROOT)}; fix and retry.")
        return 1

    rel = destination.relative_to(REPO_ROOT)
    print(f"\n{GREEN}created{RESET} {BOLD}{config.project.name}{RESET} at {rel}/")
    print(f"{DIM}  collection: {config.collection} | model: {config.model.id}{RESET}")
    print(f"\n{BOLD}Next{RESET}")
    print(f"  1. Edit {rel}/config.yaml -- the system_prompt above all")
    print(f"  2. Add documents to {rel}/data/ (or add url sources to ingest.yaml)")
    print("  3. Ingest and run:")
    print(f"{DIM}       uv run python -m chatbot_core.ingest {rel}{RESET}")
    print(f"{DIM}       PROJECT={args.id} uv run chat-api{RESET}")
    print(f"{DIM}     or in Docker:  PROJECT={args.id} mcpreingest && PROJECT={args.id} mcpup{RESET}")
    print(
        f"\n{DIM}Until you ingest documents, it will correctly say it doesn't know"
        f" things.{RESET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
