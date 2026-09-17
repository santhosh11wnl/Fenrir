#!/usr/bin/env python3
"""Manage a project's users.

Users are per project: a key issued here works on this website's assistant and
nowhere else.

    uv run python scripts/manage_users.py roles support-bot
    uv run python scripts/manage_users.py add support-bot alice --role admin
    uv run python scripts/manage_users.py list support-bot
    uv run python scripts/manage_users.py rotate support-bot alice
    uv run python scripts/manage_users.py disable support-bot alice
    uv run python scripts/manage_users.py remove support-bot alice

Keys are shown **once**, at creation, and only their hash is stored. A lost key
cannot be recovered -- rotate to issue a new one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "chatbot-core" / "src"))

from chatbot_core.auth import UserStore, new_user  # noqa: E402
from chatbot_core.config import ProjectConfig  # noqa: E402

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[90m",
    "\033[1m",
    "\033[0m",
)


def load(project: str) -> tuple[ProjectConfig, UserStore, Path]:
    project_dir = REPO_ROOT / "projects" / project
    config_path = project_dir / "config.yaml"
    if not config_path.is_file():
        available = sorted(
            p.name for p in (REPO_ROOT / "projects").iterdir() if p.is_dir()
        )
        raise SystemExit(
            f"{RED}no such project: {project}{RESET}\navailable: {', '.join(available)}"
        )
    config = ProjectConfig.load(config_path)
    users_path = project_dir / config.auth.users_file
    return config, UserStore.load(users_path), users_path


def cmd_roles(args) -> int:
    config, store, _ = load(args.project)
    print(f"\n{BOLD}Roles defined by {config.project.name}{RESET}")
    print(f"{DIM}  (each project defines its own; these apply here only){RESET}\n")
    for name, spec in sorted(config.auth.roles.items()):
        default = f" {DIM}(default){RESET}" if name == config.auth.default_role else ""
        count = sum(1 for u in store.users if u.has_role(name))
        print(f"  {BOLD}{name}{RESET}{default}  {DIM}{count} user(s){RESET}")
        if spec.description:
            print(f"    {spec.description}")
        print(f"    {DIM}grants: {', '.join(sorted(p.value for p in spec.permissions)) or 'nothing'}{RESET}")
    print()
    return 0


def cmd_add(args) -> int:
    config, store, path = load(args.project)

    roles = set(args.role or [config.auth.default_role])
    unknown = roles - set(config.auth.roles)
    if unknown:
        print(
            f"{RED}unknown role(s) for this project: {', '.join(sorted(unknown))}{RESET}",
            file=sys.stderr,
        )
        print(f"defined: {', '.join(sorted(config.auth.roles))}", file=sys.stderr)
        return 2

    user, key = new_user(args.user_id, name=args.name or args.user_id, roles=roles)
    try:
        store.add(user)
    except ValueError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        print(f"{DIM}use 'rotate' to issue a new key for an existing user{RESET}")
        return 1

    store.save(path)
    granted = config.auth.permissions_for(frozenset(roles))

    print(f"\n{GREEN}created{RESET} {BOLD}{user.name}{RESET} on {config.project.name}")
    print(f"{DIM}  roles:       {', '.join(sorted(roles))}{RESET}")
    print(f"{DIM}  permissions: {', '.join(sorted(p.value for p in granted)) or 'none'}{RESET}")
    print(f"{DIM}  stored in:   {path.relative_to(REPO_ROOT)}{RESET}")
    _print_key(key, config.project.id)
    return 0


def cmd_rotate(args) -> int:
    config, store, path = load(args.project)
    existing = store.get(args.user_id)
    if existing is None:
        print(f"{RED}no such user: {args.user_id}{RESET}", file=sys.stderr)
        return 1

    replacement, key = new_user(
        existing.id, name=existing.name, roles=set(existing.roles)
    )
    # Preserve everything except the credential itself.
    replacement.disabled = existing.disabled
    replacement.created_at = existing.created_at
    replacement.metadata = existing.metadata

    store.remove(existing.id)
    store.add(replacement)
    store.save(path)

    print(f"\n{GREEN}rotated{RESET} key for {BOLD}{existing.name}{RESET}")
    print(f"{YELLOW}The previous key stopped working immediately.{RESET}")
    _print_key(key, config.project.id)
    return 0


def cmd_list(args) -> int:
    config, store, path = load(args.project)
    if not len(store):
        print(f"\n{YELLOW}no users{RESET} for {config.project.name}")
        print(f"{DIM}  add one: manage_users.py add {args.project} <user-id>{RESET}\n")
        return 0

    print(f"\n{BOLD}Users of {config.project.name}{RESET} {DIM}({len(store)}){RESET}")
    print(f"{DIM}  {path.relative_to(REPO_ROOT)}{RESET}\n")
    for user in sorted(store.users, key=lambda u: u.id):
        mark = f"{RED}disabled{RESET}" if user.disabled else f"{GREEN}active{RESET}"
        granted = config.auth.permissions_for(user.roles)
        print(f"  {BOLD}{user.id}{RESET} ({user.name}) -- {mark}")
        print(f"    {DIM}roles:       {', '.join(sorted(user.roles)) or 'none'}{RESET}")
        print(f"    {DIM}permissions: {', '.join(sorted(p.value for p in granted)) or 'none'}{RESET}")
        if user.created_at:
            print(f"    {DIM}created:     {user.created_at}{RESET}")
    print()
    return 0


def _set_disabled(args, disabled: bool) -> int:
    _, store, path = load(args.project)
    user = store.get(args.user_id)
    if user is None:
        print(f"{RED}no such user: {args.user_id}{RESET}", file=sys.stderr)
        return 1
    user.disabled = disabled
    store.save(path)
    word = "disabled" if disabled else "enabled"
    print(f"{GREEN}{word}{RESET} {user.id}")
    return 0


def cmd_remove(args) -> int:
    _, store, path = load(args.project)
    if not store.remove(args.user_id):
        print(f"{RED}no such user: {args.user_id}{RESET}", file=sys.stderr)
        return 1
    store.save(path)
    print(f"{GREEN}removed{RESET} {args.user_id}")
    print(f"{DIM}their key stopped working immediately{RESET}")
    return 0


def _print_key(key: str, project: str) -> None:
    print(f"\n{BOLD}API key -- shown once, not recoverable{RESET}")
    print(f"\n  {BOLD}{key}{RESET}\n")
    print(f"{DIM}  Test it:{RESET}")
    print(f'{DIM}    curl -H "Authorization: Bearer {key}" \\{RESET}')
    print(f"{DIM}      -H 'Content-Type: application/json' \\{RESET}")
    print(f"{DIM}      -d '{{\"message\":\"hello\"}}' localhost:8000/chat{RESET}")
    print(f"\n{YELLOW}  Only the hash is stored. Lost keys are rotated, not recovered.{RESET}")
    print(f"{DIM}  Valid for {project!r} only.{RESET}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage a project's users. Users are per project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("roles", help="Show the roles this project defines")
    p.add_argument("project")
    p.set_defaults(func=cmd_roles)

    p = sub.add_parser("add", help="Create a user and issue a key")
    p.add_argument("project")
    p.add_argument("user_id")
    p.add_argument("--name", help="Display name (default: the id)")
    p.add_argument(
        "--role",
        action="append",
        help="Role from this project's config; repeat for several",
    )
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("list", help="List users and their effective permissions")
    p.add_argument("project")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rotate", help="Issue a new key, invalidating the old one")
    p.add_argument("project")
    p.add_argument("user_id")
    p.set_defaults(func=cmd_rotate)

    p = sub.add_parser("disable", help="Block a user without deleting them")
    p.add_argument("project")
    p.add_argument("user_id")
    p.set_defaults(func=lambda a: _set_disabled(a, True))

    p = sub.add_parser("enable", help="Re-enable a disabled user")
    p.add_argument("project")
    p.add_argument("user_id")
    p.set_defaults(func=lambda a: _set_disabled(a, False))

    p = sub.add_parser("remove", help="Delete a user permanently")
    p.add_argument("project")
    p.add_argument("user_id")
    p.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
