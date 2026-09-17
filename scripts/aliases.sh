# Shell shortcuts for the MCP platform.
#
# Install (once):
#     echo 'source ~/Development/mcp/scripts/aliases.sh' >> ~/.zshrc
#     source ~/.zshrc
#
# Then run `mcp` for the full list.
#
# Everything is `mcp`-prefixed so it never collides with the generic `dc*`
# aliases already in your .zshrc -- those stay bound to whatever directory you
# are standing in, while these always target this stack.
#
# Functions, not aliases, throughout. Aliases are only expanded by interactive
# shells, so a script that sourced this file would find half of it missing.
# Functions work everywhere and take arguments properly.
#
# These run from ANY directory. `--project-directory` is why: `-f` alone leaves
# compose resolving relative volume paths against your current directory, so
# `./projects` would mount whatever happened to be nearby.

export MCP_HOME="${MCP_HOME:-$HOME/Development/mcp}"

# Core wrapper. Everything else is a thin layer over this; extra arguments pass
# straight through, so `mcpup --force-recreate` works.
mcpc() {
  docker compose \
    --project-directory "$MCP_HOME" \
    -f "$MCP_HOME/docker-compose.yml" "$@"
}

# Every project at once. The base file runs one project, because one API
# process serves one project; additional sites are services in the overlay.
mcpca() {
  docker compose \
    --project-directory "$MCP_HOME" \
    -f "$MCP_HOME/docker-compose.yml" \
    -f "$MCP_HOME/docker-compose.projects.yml" "$@"
}

# Source mounted instead of baked, so a code edit does not leave the container
# running new config against old code.
mcpcd_dev() {
  docker compose \
    --project-directory "$MCP_HOME" \
    -f "$MCP_HOME/docker-compose.yml" \
    -f "$MCP_HOME/docker-compose.dev.yml" "$@"
}

# --- lifecycle --------------------------------------------------------------
mcpup()   { mcpc up -d "$@"; }              # start (detached)
mcpupb()  { mcpc up -d --build "$@"; }      # rebuild, then start
mcpdn()   { mcpc down "$@"; }               # stop, keep data
mcpb()    { mcpc build "$@"; }              # build images
mcpbn()   { mcpc build --no-cache "$@"; }   # build from scratch
mcprs()   { mcpc restart "$@"; }            # restart all
mcprsa()  { mcpc restart api; }             # restart just the api

# --- all projects -----------------------------------------------------------
mcpall()    { mcpca up -d "$@"; }           # start every project
mcpallps()  { mcpca ps; }
mcpalldn()  { mcpca down; }
mcpdev()    { mcpcd_dev up -d "$@"; }       # source-mounted, live reload

# Destructive: deletes the vector index, the model cache, and downloaded
# weights. Confirmed because `down -v` is one keystroke from `down` and the
# difference is a multi-gigabyte re-download.
mcpdnv() {
  printf 'Delete all volumes (vector index, HF cache, ollama models)? [y/N] '
  read -r reply
  case "$reply" in
    [yY]*) mcpc down -v ;;
    *)     echo "cancelled" ;;
  esac
}

# --- inspection -------------------------------------------------------------
mcpps()   { mcpc ps "$@"; }
mcpl()    { mcpc logs -f "$@"; }            # follow all logs
mcpla()   { mcpc logs -f api; }
mcplm()   { mcpc logs -f mcp-server; }
mcpstat() { mcpc stats --no-stream; }

# --- shells -----------------------------------------------------------------
mcpsh()   { mcpc exec api bash; }
mcpshm()  { mcpc exec mcp-server bash; }
mcppy()   { mcpc exec api python "$@"; }

# --- data -------------------------------------------------------------------
mcping() { mcpc run --rm ingest "$@"; }

# What you usually want: the api opens its store handle once at startup, so a
# fresh ingest is invisible until it restarts.
mcpreingest() { mcpc run --rm ingest && mcpc restart api; }

# --- app --------------------------------------------------------------------
mcphealth() {
  curl -s --max-time 15 "http://localhost:${API_PORT:-8000}/health" \
    | python3 -m json.tool 2>/dev/null \
    || echo "api not responding on :${API_PORT:-8000} (try mcpps)"
}

mcptools() {
  curl -s --max-time 15 "http://localhost:${API_PORT:-8000}/tools" \
    | python3 -m json.tool 2>/dev/null \
    || echo "api not responding on :${API_PORT:-8000}"
}

# Ask the running stack a question. Quote it: mcpask "what is PEP 8 about?"
mcpask() {
  if [ $# -eq 0 ]; then
    echo 'usage: mcpask "your question"' >&2
    return 2
  fi
  ( cd "$MCP_HOME" && uv run python scripts/ask.py "$@" )
}

# --- local (no Docker) ------------------------------------------------------
mcpstart() { "$MCP_HOME/scripts/stack.sh" start "$@"; }
mcpstop()  { "$MCP_HOME/scripts/stack.sh" stop; }

# --- dev --------------------------------------------------------------------
mcpcd()   { cd "$MCP_HOME" || return 1; }
mcptest() { ( cd "$MCP_HOME" && uv run pytest "$@" ); }
mcplint() { ( cd "$MCP_HOME" && uv run ruff check --fix packages services "$@" ); }

# Switch the project every command targets, for the rest of this shell.
mcpproject() {
  if [ $# -eq 0 ]; then
    echo "PROJECT=${PROJECT:-_template}"
    return 0
  fi
  if [ ! -d "$MCP_HOME/projects/$1" ]; then
    echo "no such project: $1" >&2
    echo "available: $(/bin/ls "$MCP_HOME/projects" 2>/dev/null | tr '\n' ' ')" >&2
    return 1
  fi
  export PROJECT="$1"
  echo "PROJECT=$PROJECT"
}

mcp() {
  cat <<'HELP'
MCP platform shortcuts

  lifecycle   mcpup      start              mcpdn   stop
              mcpupb     rebuild + start    mcpdnv  stop + DELETE volumes (asks)
              mcpb       build              mcpbn   build --no-cache
              mcprs      restart all        mcprsa  restart api

  inspect     mcpps      containers         mcpl    follow all logs
              mcpla      api logs           mcplm   mcp-server logs
              mcpstat    resource usage

  shells      mcpsh      bash in api        mcpshm  bash in mcp-server
              mcppy      python in api

  data        mcping       ingest corpus
              mcpreingest  ingest + restart api   <- usually what you want

  app         mcphealth  health json        mcptools  available tools
              mcpask "question"             ask the running stack

  local       mcpstart   run without Docker mcpstop

  dev         mcpcd      cd to repo         mcptest   pytest
              mcplint    ruff --fix         mcpproject [name]  switch project

  all sites   mcpall     start every project    mcpallps  their status
              mcpalldn   stop them              mcpdev    source-mounted dev

  raw         mcpc  <docker compose args>       one project
              mcpca <docker compose args>       every project
HELP
  echo "  MCP_HOME=$MCP_HOME"
  echo "  PROJECT=${PROJECT:-_template}"
}
