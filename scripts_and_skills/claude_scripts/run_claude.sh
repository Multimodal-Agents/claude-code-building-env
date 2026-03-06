#!/usr/bin/env bash
# run_claude.sh — Interactive WSL launcher for Claude Code + Speech
# Usage: bash /mnt/m/claude_code_building_env/scripts_and_skills/claude_scripts/run_claude.sh
#
# NOTE: Requires Claude Code CLI installed inside WSL:
#   npm install -g @anthropic-ai/claude-code
# NOTE: Ollama runs on Windows. WSL2 mirrored networking reaches it at localhost:11434.
#   If that fails: cat /etc/resolv.conf | grep nameserver  and use that IP instead.

REPO_DIR="/mnt/m/claude_code_building_env"
OLLAMA_PORT="11434"
PYTHON="$REPO_DIR/venv/Scripts/python.exe"

# --- Detect Windows host for Ollama (WSL2 NAT vs mirrored) ----------------
if curl -s --max-time 1 "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; then
    OLLAMA_HOST="localhost"
else
    # WSL2 NAT mode: Windows host IP is in /etc/resolv.conf
    _win_ip=$(grep -m1 nameserver /etc/resolv.conf 2>/dev/null | awk '{print $2}')
    if [ -n "$_win_ip" ] && curl -s --max-time 1 "http://${_win_ip}:${OLLAMA_PORT}/api/tags" &>/dev/null; then
        OLLAMA_HOST="$_win_ip"
    else
        OLLAMA_HOST="localhost"  # fallback, even if unreachable
    fi
fi
# ---------------------------------------------------------------------------

cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR — is the M: drive mounted?"; exit 1; }

# --- Auto-install gum if missing -------------------------------------------
if ! command -v gum &>/dev/null; then
    echo "  gum not found — installing..."
    if command -v apt-get &>/dev/null; then
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://repo.charm.sh/apt/gpg.key \
            | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
        echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" \
            | sudo tee /etc/apt/sources.list.d/charm.list > /dev/null
        sudo apt-get update -qq && sudo apt-get install -y gum
    else
        echo "  ERROR: apt-get not found. Install gum manually: https://github.com/charmbracelet/gum"
        exit 1
    fi
fi

# --- Ensure nvm/npm global bin is on PATH (where claude lives) -------------
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck source=/dev/null
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh" --no-use
# Fallback: add all node version bin dirs to PATH
for _d in "$NVM_DIR"/versions/node/*/bin; do
    [[ -x "$_d/claude" ]] && export PATH="$_d:$PATH" && break
done
# ---------------------------------------------------------------------------

# --- Wild startup sequence -------------------------------------------------
clear

# Matrix rain via python3
python3 - <<'PYEOF'
import sys, time, random, os

try:
    sz = os.get_terminal_size()
    rows, cols = sz.lines - 1, sz.columns
except Exception:
    rows, cols = 24, 80

chars = list('ｦｧｨｩｪｫｬｭｮｯｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾙﾚﾛﾜﾝ0123456789ABCDEF')
W = '\033[97m'; G = '\033[32m'; DG = '\033[2;32m'; RS = '\033[0m'

stream_cols = list(range(0, cols, 2))
heads  = {c: random.randint(-rows, 0)   for c in stream_cols}
speeds = {c: random.randint(1, 3)       for c in stream_cols}
lens   = {c: random.randint(6, 18)      for c in stream_cols}

sys.stdout.write('\033[2J\033[?25l')
end = time.time() + 2.2
out = []
while time.time() < end:
    out.clear()
    for col in stream_cols:
        h = heads[col]; l = lens[col]
        if 0 <= h < rows:
            out.append(f'\033[{h+1};{col+1}H{W}{random.choice(chars)}{RS}')
        for r in range(max(0, h-1), max(0, h-l), -1):
            fade = h - r
            ch = random.choice(chars)
            if   fade < 3: out.append(f'\033[{r+1};{col+1}H{G}{ch}{RS}')
            elif fade < 8: out.append(f'\033[{r+1};{col+1}H{DG}{ch}{RS}')
            else:          out.append(f'\033[{r+1};{col+1}H ')
        tail = h - l
        if 0 <= tail < rows:
            out.append(f'\033[{tail+1};{col+1}H ')
        heads[col] += speeds[col]
        if heads[col] - l > rows:
            heads[col]  = random.randint(-rows // 2, 0)
            speeds[col] = random.randint(1, 3)
            lens[col]   = random.randint(6, 18)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()
    time.sleep(0.045)

sys.stdout.write('\033[?25h\033[2J\033[H')
PYEOF

echo ""
printf "\033[32m  ____  _         _    _   _ ____  _____  \033[0m\n"
printf "\033[32m / ___|| |       / \\  | | | |  _ \\| ____|\033[0m\n"
printf "\033[32m| |    | |      / _ \\ | | | | | | |  _|  \033[0m\n"
printf "\033[32m| |___ | |___  / ___ \\| |_| | |_| | |___ \033[0m\n"
printf "\033[32m \\____||_____|/_/   \\_\\\\___/ |____/|_____|\033[0m\n"
printf "\033[32m                                           \033[0m\n"
printf "\033[32m        W  O  R  K  S  T  A  T  I  O  N  \033[0m\n"
echo ""

# Wild progress bar
_stages=(
    "Punching spacetime" "Bribing the GPU" "Summoning model weights"
    "Aligning the vibes" "Opening neural portals" "Overclocking imagination"
    "Calibrating chaos engine" "Warmup complete" "Ego dissolved. Ready"
    "LAUNCH SEQUENCE CONFIRMED"
)
_colors=('\033[35m' '\033[33m' '\033[36m' '\033[32m' '\033[35m' '\033[33m' '\033[31m' '\033[36m' '\033[32m' '\033[97m')
_barw=20
for _i in "${!_stages[@]}"; do
    _f=$(( _i ))
    _bar=$(printf '%0.s=' $(seq 1 $([[ $_f -gt 0 ]] && echo $_f || echo 0)))
    _bar+=">"
    _e=$(( _barw - _f - 1 ))
    [[ $_e -gt 0 ]] && _bar+=$(printf '%0.s.' $(seq 1 $_e))
    printf "\r${_colors[$_i]}  [%-20s] %s\033[0m" "$_bar" "${_stages[$_i]}"
    sleep 0.0$(( 6 + RANDOM % 7 ))
done
echo -e "\n"

# Flash
_flash="  >>> ALL SYSTEMS ONLINE <<<"
for _fc in '\033[90m' '\033[37m' '\033[97m' '\033[96m' '\033[97m' '\033[96m' '\033[97m'; do
    printf "\r${_fc}%s\033[0m" "$_flash"; sleep 0.07
done
echo -e "\n"

# Hype quote
_quotes=(
    '"The weights are warm. Anything is possible."'
    '"You are about to think faster than a human should."'
    '"Your GPU is trembling with anticipation."'
    '"Let us build something the laws of physics did not expect."'
    '"The model does not dream. But it is close."'
    '"Tokens in. Reality out."'
    '"Today is a good day for breakthroughs."'
    '"The code whispers back. Are you listening?"'
)
printf "  \033[36m%s\033[0m\n\n" "${_quotes[$(( RANDOM % ${#_quotes[@]} ))]}"
sleep 0.5
# ---------------------------------------------------------------------------

# --- GPU stats panel -------------------------------------------------------
GPU_RAW=$(nvidia-smi \
    --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$GPU_RAW" ]; then
    IFS=',' read -r GPU_NAME VRAM_USED VRAM_TOTAL GPU_UTIL GPU_TEMP <<< "$GPU_RAW"
    GPU_NAME=$(echo "$GPU_NAME" | xargs)
    VRAM_USED=$(echo "$VRAM_USED" | xargs)
    VRAM_TOTAL=$(echo "$VRAM_TOTAL" | xargs)
    GPU_UTIL=$(echo "$GPU_UTIL" | xargs)
    GPU_TEMP=$(echo "$GPU_TEMP" | xargs)
    _f=$(( VRAM_USED * 20 / VRAM_TOTAL ))
    _e=$(( 20 - _f ))
    VRAM_BAR=$(python3 -c "print('\u2588'*${_f}+'\u2591'*${_e})" 2>/dev/null \
        || { printf '%0.s#' $(seq 1 $_f); printf '%0.s.' $(seq 1 $_e); })
    gum style \
        --border rounded \
        --border-foreground 214 \
        --padding "0 2" \
        --margin "0 2" \
        "GPU   $GPU_NAME" \
        "VRAM  [$VRAM_BAR] ${VRAM_USED}/${VRAM_TOTAL} MB    ${GPU_UTIL}% util    ${GPU_TEMP}C"
fi

# --- RAM + Ollama panel ----------------------------------------------------
RAM_USED=$(free -m 2>/dev/null | awk '/^Mem:/{print $3}')
RAM_TOTAL=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}')
if [ -n "$RAM_USED" ] && [ "$RAM_TOTAL" -gt 0 ] 2>/dev/null; then
    _f=$(( RAM_USED * 20 / RAM_TOTAL ))
    _e=$(( 20 - _f ))
    RAM_BAR=$(python3 -c "print('\u2588'*${_f}+'\u2591'*${_e})" 2>/dev/null \
        || { printf '%0.s#' $(seq 1 $_f); printf '%0.s.' $(seq 1 $_e); })
    RAM_GB_USED=$(awk "BEGIN{printf \"%.1f\", $RAM_USED/1024}")
    RAM_GB_TOTAL=$(awk "BEGIN{printf \"%.0f\", $RAM_TOTAL/1024}")
    RAM_LINE="RAM   [$RAM_BAR] ${RAM_GB_USED}/${RAM_GB_TOTAL} GB"
else
    RAM_LINE="RAM   (unavailable)"
fi
if curl -s --max-time 2 "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" &>/dev/null; then
    LOADED=$(curl -s --max-time 2 "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null \
        || echo "?")
    OLLAMA_LINE="Ollama  ONLINE  (${LOADED} model(s) available)"
else
    OLLAMA_LINE="Ollama  OFFLINE"
fi
gum style \
    --border rounded \
    --border-foreground 63 \
    --padding "0 2" \
    --margin "0 2" \
    "$RAM_LINE" \
    "$OLLAMA_LINE"
# ---------------------------------------------------------------------------

choice=$(gum choose \
    "1) Claude — Anthropic Pro" \
    "2) Claude — Ollama large (gpt-oss:20b)" \
    "3) Claude — Ollama tiny (qwen3:1.7b)" \
    "4) Speech — Anthropic Pro [voice-to-voice]" \
    "5) Speech — Ollama [voice-to-voice]" \
    "6) NVIDIA — qwen2.5-coder-32b  [nvidia api]" \
    "7) Claude — Anthropic Pro [full auto]" \
    "8) Claude — Ollama gpt-oss:20b [full auto]" \
)

case "$choice" in
    "1) Claude — Anthropic Pro")
        unset ANTHROPIC_AUTH_TOKEN
        unset ANTHROPIC_BASE_URL
        echo "  >> Claude — Anthropic Pro"
        claude
        exec bash
        ;;
    "2) Claude — Ollama large (gpt-oss:20b)")
        export ANTHROPIC_AUTH_TOKEN="ollama"
        export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
        echo "  >> Claude — Ollama (gpt-oss:20b)"
        claude --model gpt-oss:20b
        exec bash
        ;;
    "3) Claude — Ollama tiny (qwen3:1.7b)")
        export ANTHROPIC_AUTH_TOKEN="ollama"
        export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
        echo "  >> Claude — Ollama tiny (qwen3:1.7b)"
        claude --allowedTools "Bash,Read,Edit,Write,Glob,Grep,LS" --model qwen3:1.7b
        exec bash
        ;;
    "4) Speech — Anthropic Pro [voice-to-voice]")
        echo "  >> Speech — Anthropic Pro  (say 'goodbye claude' to exit)"
        "$PYTHON" -m scripts_and_skills.speech.voice_pipeline \
            --voice "en-US-AriaNeural" --log-level "WARNING"
        exec bash
        ;;
    "5) Speech — Ollama [voice-to-voice]")
        echo "  >> Speech — Ollama  (say 'goodbye claude' to exit)"
        "$PYTHON" -m scripts_and_skills.speech.voice_pipeline \
            --ollama --voice "en-US-AriaNeural" --log-level "WARNING"
        exec bash
        ;;
    "6) NVIDIA — qwen2.5-coder-32b  [nvidia api]")
        KEY_FILE="/mnt/c/Users/ADA/Desktop/test_file.txt"
        if [ ! -f "$KEY_FILE" ]; then
            echo "  ERROR: Key file not found at $KEY_FILE"
            exit 1
        fi
        export ANTHROPIC_AUTH_TOKEN
        ANTHROPIC_AUTH_TOKEN=$(tr -d '[:space:]' < "$KEY_FILE")
        export ANTHROPIC_BASE_URL="https://integrate.api.nvidia.com/v1"
        echo "  >> NVIDIA API key loaded"
        echo "  >> Routing Claude Code -> NVIDIA (qwen2.5-coder-32b-instruct)"
        claude --model qwen/qwen2.5-coder-32b-instruct
        exec bash
        ;;
    "7) Claude — Anthropic Pro [full auto]")
        unset ANTHROPIC_AUTH_TOKEN
        unset ANTHROPIC_BASE_URL
        echo "  >> Claude — Anthropic Pro [full auto / --dangerously-skip-permissions]"
        echo "  >> All tool confirmations suppressed."
        claude --dangerously-skip-permissions
        exec bash
        ;;
    "8) Claude — Ollama gpt-oss:20b [full auto]")
        export ANTHROPIC_AUTH_TOKEN="ollama"
        export ANTHROPIC_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
        echo "  >> Claude — Ollama (gpt-oss:20b) [full auto / --dangerously-skip-permissions]"
        echo "  >> All tool confirmations suppressed."
        claude --model gpt-oss:20b --dangerously-skip-permissions
        exec bash
        ;;
    *)
        echo "  Invalid selection. Please enter 1-8."
        exit 1
        ;;
esac
