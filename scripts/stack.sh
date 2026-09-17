#!/usr/bin/env bash
#
# Local stack control: MCP server + chat API.
#
#   ./scripts/stack.sh start [project]   start both, wait until healthy
#   ./scripts/stack.sh stop              stop both
#   ./scripts/stack.sh restart [project]
#   ./scripts/stack.sh status            what is running, and is it healthy
#   ./scripts/stack.sh logs [mcp|api]    tail a log
#
# For a server that survives reboots and restarts itself on crash, install the
# launchd agent instead: ./scripts/install-agents.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/.run"
LOG_DIR="$ROOT/.run/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

PROJECT="${2:-${PROJECT:-_template}}"
MCP_PORT="${MCP_SERVER_PORT:-8765}"
API_PORT="${API_PORT:-8000}"
MCP_URL="http://localhost:${MCP_PORT}/mcp"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
dim()   { printf '\033[90m%s\033[0m\n' "$*"; }

pid_of() { [ -f "$RUN_DIR/$1.pid" ] && cat "$RUN_DIR/$1.pid" || echo ""; }

alive() {
  local pid; pid="$(pid_of "$1")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_one() {
  local name="$1"; shift
  if alive "$name"; then
    dim "$name already running (pid $(pid_of "$name"))"
    return 0
  fi
  # Fully detach: redirect all three descriptors and close stdin.
  #
  # A background child inherits the caller's stdout, and many shells and CI
  # runners wait for that descriptor to close -- so the script appears to hang
  # long after the services are up and healthy. Redirecting stdout and stderr
  # to the log and stdin from /dev/null severs every inherited descriptor.
  # `setsid`, where available, also detaches from the controlling terminal so
  # the services survive the shell that started them.
  (
    cd "$ROOT" || exit 1
    if command -v setsid > /dev/null 2>&1; then
      setsid "$@" < /dev/null > "$LOG_DIR/$name.log" 2>&1 &
    else
      "$@" < /dev/null > "$LOG_DIR/$name.log" 2>&1 &
    fi
    echo $! > "$RUN_DIR/$name.pid"
  )
  dim "$name started (pid $(pid_of "$name"))"
}

# Poll rather than sleep a fixed amount: the API loads an embedding model at
# startup, which takes seconds on a warm cache and much longer on a cold one.
wait_healthy() {
  local url="$1" name="$2" tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" > /dev/null 2>&1; then
      green "$name healthy"
      return 0
    fi
    if ! alive "$name"; then
      red "$name exited during startup. Last lines:"
      tail -20 "$LOG_DIR/$name.log" >&2
      return 1
    fi
    sleep 1
  done
  red "$name did not become healthy in ${tries}s"
  tail -20 "$LOG_DIR/$name.log" >&2
  return 1
}

cmd_start() {
  echo "project: $PROJECT"

  start_one mcp env MCP_SERVER_PROJECT="$PROJECT" \
    uv run mcp-server --transport streamable-http --port "$MCP_PORT"

  # The MCP endpoint answers 400 to a bare GET (it expects a protocol
  # handshake), so reachability is the signal here, not a 2xx.
  for _ in $(seq 1 30); do
    curl -s -o /dev/null "http://localhost:${MCP_PORT}/mcp" && break
    alive mcp || { red "mcp exited"; tail -20 "$LOG_DIR/mcp.log" >&2; exit 1; }
    sleep 1
  done
  green "mcp listening on :$MCP_PORT"

  start_one api env PROJECT="$PROJECT" MCP_SERVER_URL="$MCP_URL" \
    uv run chat-api --port "$API_PORT"
  wait_healthy "http://localhost:${API_PORT}/health" api 120

  echo
  curl -s "http://localhost:${API_PORT}/health" | python3 -m json.tool
}

cmd_stop() {
  for name in api mcp; do
    if alive "$name"; then
      kill "$(pid_of "$name")" 2>/dev/null || true
      dim "$name stopped"
    fi
    rm -f "$RUN_DIR/$name.pid"
  done
  # Catch anything started outside this script.
  pkill -f "mcp-server --transport" 2>/dev/null || true
  pkill -f "chat-api --port" 2>/dev/null || true
}

cmd_status() {
  for name in mcp api; do
    if alive "$name"; then
      green "$name running (pid $(pid_of "$name"))"
    else
      red "$name not running"
    fi
  done
  echo
  curl -s "http://localhost:${API_PORT}/health" 2>/dev/null | python3 -m json.tool \
    || dim "api not answering on :$API_PORT"
}

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; sleep 1; cmd_start ;;
  status)  cmd_status ;;
  logs)    tail -f "$LOG_DIR/${2:-api}.log" ;;
  *)       sed -n '3,14p' "$0"; exit 1 ;;
esac
