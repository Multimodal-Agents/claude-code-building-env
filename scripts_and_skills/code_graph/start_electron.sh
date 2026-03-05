#!/usr/bin/env bash
# Launch the Code Graph Electron desktop app (Linux / macOS / WSL).
#
# Usage:
#   bash start_electron.sh [/path/to/project]
#
# If no path is given, the current directory is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELECTRON_DIR="$SCRIPT_DIR/electron_app"
PROJECT_PATH="${1:-$(pwd)}"

# ── Install deps if needed ────────────────────────────────────────────────────
if [ ! -d "$ELECTRON_DIR/node_modules" ]; then
  echo "[code-graph] Installing Electron dependencies…"
  (cd "$ELECTRON_DIR" && npm install)
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "[code-graph] Launching desktop app for: $PROJECT_PATH"
cd "$ELECTRON_DIR"
npx electron . "$PROJECT_PATH"
