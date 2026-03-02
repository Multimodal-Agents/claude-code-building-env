#!/usr/bin/env bash
# start.sh — Start the Code Graph Monitor server. No AI needed.
#
# Usage:
#   bash scripts_and_skills/code_graph/start.sh
#   bash scripts_and_skills/code_graph/start.sh /path/to/project
#   bash scripts_and_skills/code_graph/start.sh /path/to/project 9000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_ARG="${1:-$PWD}"
PORT="${2:-8765}"
URL="http://localhost:$PORT"

# Prefer venv python
if [[ -f "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
elif [[ -f "$ROOT/venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/venv/Scripts/python.exe"
else
  PYTHON="python3"
fi

echo ""
echo "  🦀  CLAUDE CODE GRAPH MONITOR"
echo "  Project : $PATH_ARG"
echo "  URL     : $URL"
echo ""

# ── Already running? ─────────────────────────────────────────────────────────
if curl -sf "$URL/api/projects" > /dev/null 2>&1; then
  echo "  ✓ Server already live on port $PORT"
else
  # ── Check / install deps ──────────────────────────────────────────────────
  if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "  Installing requirements..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
  fi

  # ── Start server ──────────────────────────────────────────────────────────
  echo "  Starting server..."
  cd "$ROOT"
  nohup "$PYTHON" -m scripts_and_skills.code_graph.server \
    --path "$PATH_ARG" --port "$PORT" \
    > /tmp/code_graph_$PORT.log 2>&1 &
  SERVER_PID=$!
  echo "  Server PID : $SERVER_PID"

  # ── Wait up to 8s ─────────────────────────────────────────────────────────
  for i in $(seq 1 16); do
    sleep 0.5
    if curl -sf "$URL/api/projects" > /dev/null 2>&1; then
      echo "  ✓ Server live"; break
    fi
    if [[ $i -eq 16 ]]; then
      echo "  ⚠  Server didn't respond — see /tmp/code_graph_$PORT.log"; exit 1
    fi
  done
fi

# ── Open browser ──────────────────────────────────────────────────────────────
if command -v xdg-open &> /dev/null; then
  xdg-open "$URL" &
elif command -v open &> /dev/null; then
  open "$URL"
elif command -v start &> /dev/null; then
  start "$URL"
else
  echo "  Graph UI → $URL  (open manually)"
fi

echo ""
echo "  Graph UI → $URL"
echo ""
