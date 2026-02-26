#!/usr/bin/env bash
# run_claude.sh — WSL launcher for Claude Code
# Usage:
#   bash /mnt/m/claude_code_building_env/scripts_and_skills/claude_scripts/run_claude.sh          # Anthropic (Pro)
#   bash /mnt/m/claude_code_building_env/scripts_and_skills/claude_scripts/run_claude.sh --ollama # Local Ollama
#
# NOTE: Requires Claude Code CLI installed inside WSL:
#   npm install -g @anthropic-ai/claude-code
#
# NOTE: Ollama runs on Windows. WSL2 with mirrored networking (Windows 11 default)
#   can reach it at localhost:11434. If you get a connection error, find your
#   Windows host IP with:  cat /etc/resolv.conf | grep nameserver
#   Then set OLLAMA_HOST below to that IP instead of localhost.

REPO_DIR="/mnt/m/claude_code_building_env"
OLLAMA_HOST="localhost"
OLLAMA_PORT="11434"

cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR — is the M: drive mounted?"; exit 1; }

if [[ "$1" == "--ollama" || "$1" == "-o" ]]; then
    export ANTHROPIC_AUTH_TOKEN="ollama"
    export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
    echo "  >> Launching with Ollama (local) — ${ANTHROPIC_BASE_URL}"
    claude --model gpt-oss:20b
else
    unset ANTHROPIC_AUTH_TOKEN
    unset ANTHROPIC_BASE_URL
    echo "  >> Launching with Anthropic (Pro subscription)"
    claude
fi
