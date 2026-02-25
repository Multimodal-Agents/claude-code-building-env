# ============================================================
#  run_claude.ps1  —  Main entry point for the local AI stack
#  Desktop shortcut target:
#    powershell.exe -ExecutionPolicy Bypass -File "M:\claude_code_building_env\powershell_scripts\claude_scripts\run_claude.ps1"
# ============================================================

param(
    [string]$Model      = "gpt-oss:20b",
    [string]$Project    = "",
    [switch]$ListModels,
    [switch]$Debug
)

# ── Config ──────────────────────────────────────────────────
$WORKSPACE      = "M:\claude_code_building_env"
$SKILLS_ROOT    = "$WORKSPACE\powershell_scripts\claude_skills"
$PROJECTS_ROOT  = "$WORKSPACE\claude_custom_projects_1"

# ── Known local models ──────────────────────────────────────
$LOCAL_MODELS = @{
    "gpt-oss:20b"   = "14 GB  — fast, Titan XP native"
    "claude-sonnet" = "API    — Anthropic cloud (needs key)"
}

# ── Banner ───────────────────────────────────────────────────
function Show-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║      LOCAL AI DEV ENVIRONMENT  v0.1.0        ║" -ForegroundColor Cyan
    Write-Host "  ║      Abstraction Layer  •  Claude Code CLI    ║" -ForegroundColor Cyan
    Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# ── List available models ────────────────────────────────────
function Show-Models {
    Write-Host "  Available models:" -ForegroundColor Yellow
    foreach ($m in $LOCAL_MODELS.GetEnumerator()) {
        Write-Host ("    {0,-20} {1}" -f $m.Key, $m.Value) -ForegroundColor White
    }
    Write-Host ""
}

# ── Check Ollama is running ──────────────────────────────────
function Test-Ollama {
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# ── Resolve working directory ────────────────────────────────
function Resolve-WorkDir {
    if ($Project -ne "") {
        $p = Join-Path $PROJECTS_ROOT $Project
        if (Test-Path $p) { return $p }
        Write-Host "  [WARN] Project '$Project' not found — using workspace root." -ForegroundColor Yellow
    }
    return $WORKSPACE
}

# ── Build the claude CLI argument list ───────────────────────
function Build-ClaudeArgs {
    $a = @("--model", $Model)
    if ($Debug) { $a += "--debug" }
    return $a
}

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
Show-Banner

if ($ListModels) { Show-Models; exit 0 }

# Check Ollama when using a local model
if ($Model -notlike "*claude*") {
    Write-Host "  Checking Ollama..." -ForegroundColor DarkGray -NoNewline
    if (Test-Ollama) {
        Write-Host " running ✓" -ForegroundColor Green
    } else {
        Write-Host " NOT FOUND" -ForegroundColor Red
        Write-Host "  Start Ollama first:  ollama serve" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "  Press Enter to try anyway or Ctrl+C to abort"
    }
}

# Show active config
$WorkDir = Resolve-WorkDir
Write-Host ""
Write-Host "  Model    : $Model"    -ForegroundColor Cyan
Write-Host "  Work dir : $WorkDir"  -ForegroundColor Cyan
Write-Host "  Skills   : $SKILLS_ROOT" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path $WorkDir

$cliArgs = Build-ClaudeArgs
Write-Host "  Launching: claude $($cliArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

claude @cliArgs