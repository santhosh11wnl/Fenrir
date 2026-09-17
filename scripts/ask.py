#!/usr/bin/env python3
"""Ask the running chat API a question and render the event stream.

A terminal client for the same SSE protocol the React app consumes, which
makes it the fastest way to tell whether a problem is in the backend or the
frontend.

    uv run python scripts/ask.py "What is the maximum line length?"
    uv run python scripts/ask.py --raw "..."      # show every frame
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DIM, BOLD, GREEN, RED, CYAN, RESET = (
    "\033[90m",
    "\033[1m",
    "\033[32m",
    "\033[31m",
    "\033[36m",
    "\033[0m",
)


def frames(body):
    """Yield (event_name, payload) from an SSE response."""
    name = None
    for raw in body:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and name:
            yield name, json.loads(line[5:].strip())
            name = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the chat API a question.")
    parser.add_argument("message")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--conversation", help="Continue an existing conversation")
    parser.add_argument("--raw", action="store_true", help="Print every frame verbatim")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--key",
        default=os.environ.get("MCP_API_KEY", ""),
        help="API key, when the project has auth enabled (or set MCP_API_KEY)",
    )
    args = parser.parse_args()

    payload = {"message": args.message}
    if args.conversation:
        payload["conversation_id"] = args.conversation

    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    request = urllib.request.Request(
        f"{args.url}/chat", data=json.dumps(payload).encode(), headers=headers
    )

    answer: list[str] = []
    sources: list[dict] = []
    conversation_id = None
    stop = None
    usage = {}
    failed = False

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for name, data in frames(response):
                if args.raw:
                    print(f"{DIM}{name}{RESET} {json.dumps(data)[:160]}")
                    continue

                if name == "message_start":
                    conversation_id = data["conversation_id"]
                    print(f"{DIM}model: {data['model']}{RESET}\n")
                elif name == "text_delta":
                    answer.append(data["text"])
                    sys.stdout.write(data["text"])
                    sys.stdout.flush()
                elif name == "thinking_delta":
                    sys.stdout.write(f"{DIM}{data['text']}{RESET}")
                elif name == "tool_call":
                    argstr = json.dumps(data["input"])
                    print(f"\n{CYAN}-> {data['name']}({argstr[:80]}){RESET}")
                elif name == "tool_result":
                    mark = f"{GREEN}ok{RESET}" if data["ok"] else f"{RED}error{RESET}"
                    preview = " ".join(data["preview"].split())[:90]
                    print(f"{CYAN}   {mark} {data['duration_ms']}ms: {preview}{RESET}\n")
                elif name == "citations":
                    sources.extend(data["sources"])
                elif name == "error":
                    failed = True
                    print(f"\n{RED}error: {data['message']}{RESET}")
                elif name == "done":
                    stop = data.get("stop_reason")
                    usage = data.get("usage", {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:300]
        print(f"{RED}HTTP {exc.code}: {body}{RESET}")
        if exc.code == 401:
            print(
                f"{DIM}This project has auth enabled. Pass --key, or set "
                f"MCP_API_KEY.{RESET}"
            )
            print(f"{DIM}  uv run python scripts/manage_users.py list <project>{RESET}")
        return 1
    except urllib.error.URLError as exc:
        print(f"{RED}cannot reach {args.url}: {exc.reason}{RESET}")
        return 1

    if args.raw:
        return 0

    if sources:
        print(f"\n\n{BOLD}Sources{RESET}")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['title']} {DIM}(score {s['score']:.3f}){RESET}")
            if s.get("uri"):
                print(f"      {DIM}{s['uri']}{RESET}")

    tokens = f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
    print(f"\n{DIM}stop: {stop} | tokens: {tokens}{RESET}")
    if conversation_id:
        print(f"{DIM}continue: --conversation {conversation_id}{RESET}")

    return 1 if failed or not answer else 0


if __name__ == "__main__":
    raise SystemExit(main())
