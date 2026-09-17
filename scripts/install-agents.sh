#!/usr/bin/env bash
#
# Install launchd agents so the stack is always running -- starts at login,
# restarts on crash, no terminal held open.
#
#   ./scripts/install-agents.sh [project]     install and load
#   ./scripts/install-agents.sh --uninstall   unload and remove
#
# Agents are per-user (LaunchAgents, not LaunchDaemons): they run as you, with
# your permissions, and need no sudo. That is the right level for a dev machine
# -- a LaunchDaemon would run as root before login, which this does not need.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/.run/logs"
PREFIX="com.mcp-platform"

UV="$(command -v uv || true)"
[ -n "$UV" ] || { echo "uv not found on PATH. brew install uv" >&2; exit 1; }

uninstall() {
  for name in mcp-server chat-api; do
    plist="$AGENT_DIR/$PREFIX.$name.plist"
    if [ -f "$plist" ]; then
      launchctl bootout "gui/$(id -u)/$PREFIX.$name" 2>/dev/null || true
      rm -f "$plist"
      echo "removed $PREFIX.$name"
    fi
  done
}

[ "${1:-}" = "--uninstall" ] && { uninstall; exit 0; }

PROJECT="${1:-${PROJECT:-_template}}"
MCP_PORT="${MCP_SERVER_PORT:-8765}"
API_PORT="${API_PORT:-8000}"

mkdir -p "$AGENT_DIR" "$LOG_DIR"

# $1 label suffix, $2 log basename, $3.. program arguments
write_plist() {
  local name="$1" log="$2"; shift 2
  local plist="$AGENT_DIR/$PREFIX.$name.plist"
  {
    cat <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$PREFIX.$name</string>
  <key>ProgramArguments</key>
  <array>
XML
    for arg in "$@"; do printf '    <string>%s</string>\n' "$arg"; done
    cat <<XML
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <!-- Restart if it exits for any reason. -->
  <key>KeepAlive</key><true/>
  <!-- Back off between restarts so a config error doesn't spin the CPU. -->
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/$log.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/$log.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(dirname "$UV"):/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PROJECT</key><string>$PROJECT</string>
    <key>MCP_SERVER_PROJECT</key><string>$PROJECT</string>
    <key>MCP_SERVER_URL</key><string>http://localhost:$MCP_PORT/mcp</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
</dict>
</plist>
XML
  } > "$plist"

  # bootout first so re-running this script reloads rather than erroring.
  launchctl bootout "gui/$(id -u)/$PREFIX.$name" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "loaded $PREFIX.$name"
}

write_plist mcp-server mcp \
  "$UV" run mcp-server --transport streamable-http --port "$MCP_PORT"

write_plist chat-api api \
  "$UV" run chat-api --port "$API_PORT"

echo
echo "Agents installed for project '$PROJECT'. They start at login and restart on crash."
echo
echo "  status:    launchctl list | grep $PREFIX"
echo "  logs:      tail -f $LOG_DIR/mcp.log"
echo "  restart:   launchctl kickstart -k gui/$(id -u)/$PREFIX.mcp-server"
echo "  uninstall: ./scripts/install-agents.sh --uninstall"
