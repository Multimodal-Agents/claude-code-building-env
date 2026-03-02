<#
.SYNOPSIS
  Start the Code Graph Monitor server and open the browser.
  Run directly -- no AI needed.

.USAGE
  # From any directory:
  .\scripts_and_skills\code_graph\start.ps1

  # Override target project:
  .\scripts_and_skills\code_graph\start.ps1 -Path C:\my\project

  # Custom port:
  .\scripts_and_skills\code_graph\start.ps1 -Port 9000
#>
param(
    [string]$Path = (Get-Location).Path,
    [int]   $Port = 8765
)

$Root    = Split-Path $PSScriptRoot -Parent | Split-Path -Parent  # repo root
$EnvPy   = Join-Path $Root "venv\Scripts\python.exe"
$Python  = if (Test-Path $EnvPy) { $EnvPy } else { "python" }
$Req     = Join-Path $PSScriptRoot "requirements.txt"
$Module  = "scripts_and_skills.code_graph.server"
$Url     = "http://localhost:$Port"

Write-Host ""
Write-Host "  +---------------------------------------+" -ForegroundColor DarkRed
Write-Host "  |     CLAUDE CODE GRAPH MONITOR     |" -ForegroundColor Red
Write-Host "  +---------------------------------------+" -ForegroundColor DarkRed
Write-Host ""
Write-Host "  Project : $Path" -ForegroundColor DarkYellow
Write-Host "  URL     : $Url"  -ForegroundColor DarkYellow
Write-Host ""

# -- 1. Check if already running ----------------------------------------------
$running = $false
try {
    $resp = Invoke-WebRequest -Uri "$Url/api/projects" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
    $running = $true
} catch {}

if ($running) {
    Write-Host "  [OK] Server already live on port $Port" -ForegroundColor Green
} else {
    # -- 2. Install requirements if missing -----------------------------------
    Write-Host "  Checking dependencies..." -ForegroundColor DarkGray
    $check = & $Python -c "import fastapi, uvicorn" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Installing requirements..." -ForegroundColor DarkYellow
        & $Python -m pip install -r $Req -q
    }

    # -- 3. Start server in background ----------------------------------------
    Write-Host "  Starting server..." -ForegroundColor DarkGray
    $proc = Start-Process -FilePath $Python `
        -ArgumentList "-m", $Module, "--path", "`"$Path`"", "--port", $Port `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "  Server PID : $($proc.Id)" -ForegroundColor DarkGray

    # -- 4. Wait for it to come up (max 8s) ----------------------------------
    $ready = $false
    for ($i = 0; $i -lt 16; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-WebRequest -Uri "$Url/api/projects" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop | Out-Null
            $ready = $true; break
        } catch {}
    }
    if (-not $ready) {
        Write-Host "  [!]  Server didn't respond in 8s -- check for errors." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  [OK] Server live" -ForegroundColor Green
}

# -- 5. Open browser ----------------------------------------------------------
Write-Host "  Opening browser..." -ForegroundColor DarkGray
Start-Process $Url
Write-Host ""
Write-Host "  Graph UI -> $Url" -ForegroundColor Cyan
Write-Host "  Stop     -> kill PID shown above (or close the hidden window)" -ForegroundColor DarkGray
Write-Host ""
