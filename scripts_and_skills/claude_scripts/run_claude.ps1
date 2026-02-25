# ============================================================
#  run_claude.ps1  —  Main entry point for the local AI stack
#  Desktop shortcut target:
#    powershell.exe -ExecutionPolicy Bypass -File "M:\claude_code_building_env\scripts_and_skills\claude_scripts\run_claude.ps1"
# ============================================================

param(
    [string]$Model      = "gpt-oss:20b",
    [string]$Project    = "",
    [switch]$ListModels,
    [switch]$Debug
)

# ── Config ──────────────────────────────────────────────────
$WORKSPACE      = "M:\claude_code_building_env"
$SKILLS_ROOT    = "$WORKSPACE\scripts_and_skills\claude_skills"
$PROJECTS_ROOT  = "$WORKSPACE\claude_custom_projects_1"

# ── Known local models ──────────────────────────────────────
$LOCAL_MODELS = @{
    "gpt-oss:20b"   = "14 GB  — fast, Titan XP native"
    "claude-sonnet" = "API    — Anthropic cloud (needs key)"
}

# ── List available models ────────────────────────────────────
function Show-Models {
    Write-Host "Models:" -ForegroundColor Yellow
    foreach ($m in $LOCAL_MODELS.GetEnumerator()) {
        Write-Host ("  {0,-20} {1}" -f $m.Key, $m.Value)
    }
    exit 0
}

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
if ($ListModels) { Show-Models }

# Resolve working directory
$WorkDir = $WORKSPACE
if ($Project -ne "") {
    $p = Join-Path $PROJECTS_ROOT $Project
    if (Test-Path $p) { $WorkDir = $p }
    else { Write-Warning "Project '$Project' not found — using workspace root." }
}

Set-Location -Path $WorkDir

$cliArgs = @("--model", $Model)
if ($Debug) { $cliArgs += "--debug" }

claude @cliArgs