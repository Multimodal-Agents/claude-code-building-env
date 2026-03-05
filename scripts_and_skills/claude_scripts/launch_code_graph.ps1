<#
.SYNOPSIS
    Launch the Code Graph desktop app.
    Double-click this file, or create a shortcut (see below).

.PARAMETER Path
    Project folder to open. Defaults to the last used project,
    or prompts for a folder if none is saved.

.PARAMETER CreateShortcut
    Create a Desktop shortcut that runs this launcher.

.EXAMPLE
    .\launch_code_graph.ps1
    .\launch_code_graph.ps1 -Path "C:\my\project"
    .\launch_code_graph.ps1 -CreateShortcut
#>
param(
    [string]$Path         = "",
    [switch]$CreateShortcut
)

$ErrorActionPreference = "Stop"

# ── Resolve repo root robustly ────────────────────────────────────────────────
# $PSScriptRoot can be empty/wrong when launched from a desktop shortcut whose
# "Start in" field is blank. Fall back to the known absolute path.
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -or -not (Test-Path $ScriptDir)) {
    $ScriptDir = "M:\claude_code_building_env\scripts_and_skills\claude_scripts"
}
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

# ── Guard: mapped drive must exist ────────────────────────────────────────────
$DriveLetter = (Split-Path $RepoRoot -Qualifier)   # e.g. "M:"
if (-not (Test-Path "$DriveLetter\")) {
    # Drive not mounted yet -- common at login for network/subst drives
    $msg = "[code-graph] ERROR: Drive $DriveLetter is not available. Mount it first."
    [System.Windows.Forms.MessageBox]::Show($msg, "Code Graph", 0, 16) 2>$null
    exit 1
}

$ElectronDir = Join-Path $RepoRoot "scripts_and_skills\code_graph\electron_app"
$LogFile     = Join-Path $RepoRoot "local_data\code_graph_launch.log"

# ── Logging helper (critical when -WindowStyle Hidden eats all output) ────────
function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Msg"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
}

Write-Log "=== Code Graph launcher started ===" "Cyan"
Write-Log "ScriptDir : $ScriptDir"
Write-Log "RepoRoot  : $RepoRoot"
Write-Log "ElectronDir: $ElectronDir"

# ── Activate Python venv (needed by the FastAPI server spawned by Electron) ─────
$VenvActivate = Join-Path $RepoRoot "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Log "Activating venv..." "Cyan"
    & $VenvActivate
} else {
    Write-Log "WARNING: venv not found at $VenvActivate -- Python server may fail" "Yellow"
}

# ── Create Desktop shortcut ───────────────────────────────────────────────────
if ($CreateShortcut) {
    $Desktop    = [Environment]::GetFolderPath("Desktop")
    $Shortcut   = Join-Path $Desktop "Code Graph.lnk"
    $Shell      = New-Object -ComObject WScript.Shell
    $Lnk        = $Shell.CreateShortcut($Shortcut)
    $Lnk.TargetPath       = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $Lnk.Arguments        = "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$($PSCommandPath)`""
    # Working dir = script folder, so $PSScriptRoot resolves correctly
    $Lnk.WorkingDirectory = $ScriptDir
    $Lnk.WindowStyle      = 7   # 7 = Minimized
    $Lnk.Description      = "Code Graph -- live dependency visualiser"

    # Use the existing icon if available
    $IconPath = Join-Path $RepoRoot "scripts_and_skills\code_graph\claude_code_cli_icon-subject.ico"
    if (Test-Path $IconPath) { $Lnk.IconLocation = $IconPath }

    $Lnk.Save()
    Write-Log "Desktop shortcut created: $Shortcut" "Green"
    Write-Host "[code-graph] Desktop shortcut created: $Shortcut" -ForegroundColor Green
    exit 0
}

# ── Check node + npm ──────────────────────────────────────────────────────────
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Log "ERROR: npm not found. Install Node.js from https://nodejs.org" "Red"
    Read-Host "Press Enter to close"
    exit 1
}

# ── Validate ElectronDir exists ───────────────────────────────────────────────
if (-not (Test-Path $ElectronDir)) {
    Write-Log "ERROR: Electron app folder not found at $ElectronDir" "Red"
    exit 1
}

# ── Install Electron if node_modules is missing ───────────────────────────────
$NodeModules = Join-Path $ElectronDir "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Log "First run -- installing Electron..." "Cyan"
    Push-Location $ElectronDir
    npm install --silent
    Pop-Location
    Write-Log "Electron installed." "Green"
}

# ── Launch ────────────────────────────────────────────────────────────────────
$LaunchArgs = @()
if ($Path -ne "") { $LaunchArgs += $Path }

Write-Log "Starting Electron..." "Cyan"
Push-Location $ElectronDir
try {
    if ($LaunchArgs.Count -gt 0) {
        npx electron . $LaunchArgs[0]
    } else {
        npx electron .
    }
} catch {
    Write-Log "ERROR launching Electron: $_" "Red"
} finally {
    Pop-Location
}
Write-Log "=== Code Graph launcher finished ===" "Cyan"
