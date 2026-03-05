<#
.SYNOPSIS
    Launch the Code Graph Electron desktop app.

.PARAMETER Path
    Project folder to open on startup. Defaults to the current directory.

.EXAMPLE
    .\start_electron.ps1
    .\start_electron.ps1 -Path "C:\my\project"
#>
param(
    [string]$Path = ""
)

$ErrorActionPreference = "Stop"
$ElectronDir = Join-Path $PSScriptRoot "electron_app"

# ── 1. Install node_modules if missing ───────────────────────────────────────
if (-not (Test-Path (Join-Path $ElectronDir "node_modules"))) {
    Write-Host "[code-graph] Installing Electron dependencies..." -ForegroundColor Cyan
    Push-Location $ElectronDir
    npm install
    Pop-Location
    Write-Host "[code-graph] Done." -ForegroundColor Green
}

# ── 2. Resolve project path ───────────────────────────────────────────────────
if ($Path -eq "") {
    $Path = (Get-Location).Path
}

# ── 3. Launch ─────────────────────────────────────────────────────────────────
Write-Host "[code-graph] Launching desktop app for: $Path" -ForegroundColor Cyan
$npx = Get-Command npx -ErrorAction SilentlyContinue
if ($null -eq $npx) {
    Write-Error "npx not found — make sure Node.js is installed and on PATH."
    exit 1
}

Push-Location $ElectronDir
npx electron . $Path
Pop-Location
