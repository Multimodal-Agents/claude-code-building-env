#!/usr/bin/env bash
# run_claude.sh — Interactive WSL launcher for Claude Code + Speech
# Usage: bash /mnt/m/claude_code_building_env/scripts_and_skills/claude_scripts/run_claude.sh
#
# NOTE: Requires Claude Code CLI installed inside WSL:
#   npm install -g @anthropic-ai/claude-code
# NOTE: Ollama runs on Windows. WSL2 mirrored networking reaches it at localhost:11434.
#   If that fails: cat /etc/resolv.conf | grep nameserver  and use that IP instead.

REPO_DIR="/mnt/m/claude_code_building_env"
OLLAMA_HOST="localhost"
OLLAMA_PORT="11434"
PYTHON="$REPO_DIR/venv/Scripts/python.exe"

cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR — is the M: drive mounted?"; exit 1; }

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║           Claude Launcher            ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
echo "  1) Claude  — Anthropic Pro"
echo "  2) Claude  — Ollama large  (gpt-oss:20b)"
echo "  3) Claude  — Ollama tiny   (qwen3:1.7b)"
echo "  4) Speech  — Anthropic Pro  [voice-to-voice]"
echo "  5) Speech  — Ollama         [voice-to-voice]"
echo ""
read -rp "  Select mode [1-5]: " choice
echo ""

case "$choice" in
    1)
        unset ANTHROPIC_AUTH_TOKEN
        unset ANTHROPIC_BASE_URL
        echo "  >> Claude — Anthropic Pro"
        claude
        ;;
    2)
        export ANTHROPIC_AUTH_TOKEN="ollama"
        export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
        echo "  >> Claude — Ollama (gpt-oss:20b)"
        claude --model gpt-oss:20b
        ;;
    3)
        export ANTHROPIC_AUTH_TOKEN="ollama"
        export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
        echo "  >> Claude — Ollama tiny (qwen3:1.7b)"
        claude --allowedTools "Bash,Read,Edit,Write,Glob,Grep" --model qwen3:1.7b
        ;;
    4)
        echo "  >> Speech — Anthropic Pro  (say 'goodbye claude' to exit)"
        "$PYTHON" -m scripts_and_skills.speech.voice_pipeline \
            --voice "en-US-AriaNeural" --log-level "WARNING"
        ;;
    5)
        echo "  >> Speech — Ollama  (say 'goodbye claude' to exit)"
        "$PYTHON" -m scripts_and_skills.speech.voice_pipeline \
            --ollama --voice "en-US-AriaNeural" --log-level "WARNING"
        ;;
    *)
        echo "  Invalid selection. Please enter 1-5."
        exit 1
        ;;
esac
