# ============================================================
#  setup.ps1  —  First-time setup for claude-code-building-env
#  Run this once after cloning the repo.
#
#  Usage:
#    .\setup.ps1
#    .\setup.ps1 -ClaudeCodeDir "C:\your\existing\claude-code"
#    .\setup.ps1 -SkipClaudeCode
# ============================================================

param(
    # Override where to clone/find the claude-code reference repo.
    # Leave blank to clone fresh into the default location.
    [string]$ClaudeCodeDir = "",

    # Skip cloning claude-code entirely (e.g. you only want the skills ref).
    [switch]$SkipClaudeCode,

    # Skip cloning the Anthropic skills reference.
    [switch]$SkipSkills
)

$REPO_ROOT    = $PSScriptRoot
$SKILLS_DEST  = "$REPO_ROOT\scripts_and_skills\claude_skills\skills"
$CC_DEST      = if ($ClaudeCodeDir -ne "") { $ClaudeCodeDir } else { "$REPO_ROOT\claude_code_custom\claude-code" }

$SKILLS_URL   = "https://github.com/anthropics/skills.git"
$CC_URL       = "https://github.com/anthropics/claude-code.git"

function Write-Step($msg) { Write-Host "  >> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK $msg" -ForegroundColor Green }
function Write-Skip($msg) { Write-Host "  -- $msg" -ForegroundColor DarkGray }
function Write-Warn($msg) { Write-Host "  !! $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "  claude-code-building-env setup" -ForegroundColor White
Write-Host "  ──────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# ── 1. Check claude CLI is installed ────────────────────────
Write-Step "Checking for claude CLI..."
$claudeBin = Get-Command claude -ErrorAction SilentlyContinue
if ($claudeBin) {
    Write-Ok "claude found at: $($claudeBin.Source)"
} else {
    Write-Warn "claude CLI not found in PATH."
    Write-Host "     Install it from: https://code.claude.com/docs/en/setup" -ForegroundColor Yellow
    Write-Host ""
}

# ── 2. Clone Anthropic skills reference ─────────────────────
if (-not $SkipSkills) {
    Write-Step "Anthropic skills reference -> $SKILLS_DEST"
    if (Test-Path "$SKILLS_DEST\.git") {
        Write-Skip "Already cloned. Pulling latest..."
        git -C $SKILLS_DEST pull --ff-only
    } else {
        New-Item -ItemType Directory -Force -Path $SKILLS_DEST | Out-Null
        git clone $SKILLS_URL $SKILLS_DEST
        Write-Ok "Skills cloned."
    }
} else {
    Write-Skip "Skipping Anthropic skills clone (-SkipSkills)."
}

Write-Host ""

# ── 3. Clone claude-code reference ──────────────────────────
if (-not $SkipClaudeCode) {
    Write-Step "Anthropic claude-code reference -> $CC_DEST"
    if (Test-Path "$CC_DEST\.git") {
        Write-Skip "Already cloned. Pulling latest..."
        git -C $CC_DEST pull --ff-only
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $CC_DEST) | Out-Null
        git clone $CC_URL $CC_DEST
        Write-Ok "claude-code cloned."
    }
} else {
    Write-Skip "Skipping claude-code clone (-SkipClaudeCode)."
}

Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  To launch: .\scripts_and_skills\claude_scripts\run_claude.ps1" -ForegroundColor DarkGray
Write-Host ""
