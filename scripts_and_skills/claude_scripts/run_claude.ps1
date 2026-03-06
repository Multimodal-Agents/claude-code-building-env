# run_claude.ps1 - Interactive Claude / Speech launcher
# Desktop shortcut target:
#   powershell -NoExit -File "M:\claude_code_building_env\scripts_and_skills\claude_scripts\run_claude.ps1"

Set-Location -Path "M:\claude_code_building_env"
$PythonExe = "M:\claude_code_building_env\venv\Scripts\python.exe"

# Maximize the window
try {
    $sig = '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'
    $t = Add-Type -MemberDefinition $sig -Name WinAPI -Namespace WinAPI -PassThru -ErrorAction Stop
    $hwnd = (Get-Process -Id $PID).MainWindowHandle
    if ($hwnd -ne [IntPtr]::Zero) { $null = $t::ShowWindow($hwnd, 3) }  # 3 = SW_MAXIMIZE
    Start-Sleep -Milliseconds 150
} catch {}

# --- STARTUP SEQUENCE ------------------------------------------------------
function Show-Startup {
    [Console]::CursorVisible = $false
    $rng = [System.Random]::new()

    # --- Matrix rain -------------------------------------------------------
    # Use BufferSize (not WindowSize) — SetCursorPosition is in buffer coords
    $bufW  = $Host.UI.RawUI.BufferSize.Width
    $bufH  = $Host.UI.RawUI.BufferSize.Height
    $w     = [Math]::Max(10, $bufW - 2)   # stay 2 cols inside buffer edge
    $h     = [Math]::Max(5,  [Math]::Min($bufH - 2, $Host.UI.RawUI.WindowSize.Height - 2))
    $chars = '0123456789ABCDEFabcdef!@#$%^&*<>?/\|{}[]~`+-='.ToCharArray()
    $ccount = $chars.Count

    # Only even columns, each strictly < $w
    $cols = @(0..[Math]::Floor(($w - 1) / 2) | ForEach-Object { $_ * 2 } | Where-Object { $_ -lt $w })
    $heads     = @{}; $speeds = @{}; $lens = @{}
    foreach ($c in $cols) {
        $heads[$c]  = $rng.Next(-$h, 0)
        $speeds[$c] = $rng.Next(1, 4)
        $lens[$c]   = $rng.Next(6, 20)
    }

    Clear-Host
    $matrixEnd = (Get-Date).AddMilliseconds(2200)
    while ((Get-Date) -lt $matrixEnd) {
        foreach ($col in $cols) {
            $hd = $heads[$col]; $ln = $lens[$col]
            # bright head
            if ($hd -ge 0 -and $hd -lt $h -and $col -lt $w) {
                try { [Console]::SetCursorPosition($col, $hd) } catch {}
                Write-Host $chars[$rng.Next($ccount)] -NoNewline -ForegroundColor White
            }
            # green trail
            for ($r = [Math]::Max(0,$hd-1); $r -ge [Math]::Max(0,$hd-$ln); $r--) {
                if ($col -ge $w -or $r -ge $h) { continue }
                $fade = $hd - $r
                try { [Console]::SetCursorPosition($col, $r) } catch {}
                if     ($fade -lt 3) { Write-Host $chars[$rng.Next($ccount)] -NoNewline -ForegroundColor Green }
                elseif ($fade -lt 8) { Write-Host $chars[$rng.Next($ccount)] -NoNewline -ForegroundColor DarkGreen }
                else                 { Write-Host ' ' -NoNewline }
            }
            # erase tail
            $tail = $hd - $ln
            if ($tail -ge 0 -and $tail -lt $h -and $col -lt $w) {
                try { [Console]::SetCursorPosition($col, $tail) } catch {}
                Write-Host ' ' -NoNewline
            }
            $heads[$col] += $speeds[$col]
            if ($heads[$col] - $ln -gt $h) {
                $heads[$col]  = $rng.Next(-$h, -2)
                $speeds[$col] = $rng.Next(1, 4)
                $lens[$col]   = $rng.Next(6, 20)
            }
        }
        Start-Sleep -Milliseconds 45
    }

    # --- Logo reveal -------------------------------------------------------
    Clear-Host
    Write-Host ""
    $logo = @(
        "  ____  _         _    _   _ ____  _____  ",
        " / ___|| |       / \  | | | |  _ \| ____|",
        "| |    | |      / _ \ | | | | | | |  _|  ",
        "| |___ | |___  / ___ \| |_| | |_| | |___ ",
        " \____||_____| /_/   \_\\___/ |____/|_____|",
        "                                           ",
        "         W  O  R  K  S  T  A  T  I  O  N "
    )
    foreach ($line in $logo) { Write-Host "  $line" -ForegroundColor Green }
    Write-Host ""

    # --- Progress bar ------------------------------------------------------
    $stages = @(
        @{ L="Punching spacetime...        "; C="Magenta" },
        @{ L="Bribing the GPU...           "; C="Yellow"  },
        @{ L="Summoning model weights...   "; C="Cyan"    },
        @{ L="Aligning the vibes...        "; C="Green"   },
        @{ L="Opening neural portals...    "; C="Magenta" },
        @{ L="Overclocking imagination...  "; C="Yellow"  },
        @{ L="Calibrating chaos engine...  "; C="Red"     },
        @{ L="Warmup complete.             "; C="Cyan"    },
        @{ L="Ego dissolved. Ready.        "; C="Green"   },
        @{ L="LAUNCH SEQUENCE CONFIRMED.   "; C="White"   }
    )
    $barW = 20
    for ($s = 0; $s -lt $stages.Count; $s++) {
        $bar = ('=' * $s) + '>' + ('.' * ($barW - $s - 1))
        Write-Host ("`r  [$bar] $($stages[$s].L)") -NoNewline -ForegroundColor $stages[$s].C
        Start-Sleep -Milliseconds ($rng.Next(55, 130))
    }
    Write-Host ""
    Write-Host ""

    # --- Flash ALL SYSTEMS ONLINE ------------------------------------------
    $flash = "  >>> ALL SYSTEMS ONLINE <<<"
    foreach ($fc in @('DarkGray','Gray','White','Cyan','White','Cyan','White')) {
        Write-Host "`r$flash" -NoNewline -ForegroundColor $fc
        Start-Sleep -Milliseconds 65
    }
    Write-Host ""; Write-Host ""

    # --- Hype quote --------------------------------------------------------
    $quotes = @(
        '"The weights are warm. Anything is possible."',
        '"You are about to think faster than a human should."',
        '"Your GPU is trembling with anticipation."',
        '"Let us build something the laws of physics did not expect."',
        '"The model does not dream. But it is close."',
        '"Tokens in. Reality out."',
        '"Today is a good day for breakthroughs."',
        '"The code whispers back. Are you listening?"'
    )
    Write-Host ("  " + $quotes[$rng.Next($quotes.Count)]) -ForegroundColor DarkCyan
    Write-Host ""
    Start-Sleep -Milliseconds 500

    [Console]::CursorVisible = $true
}
# ---------------------------------------------------------------------------

# --- Gather system stats (called once, results passed into menu) -----------
function Get-Dashboard {
    $lines = [System.Collections.Generic.List[hashtable]]::new()

    # GPU via nvidia-smi
    $nvsmi = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu `
        --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($nvsmi) {
        $p      = $nvsmi -split ','
        $gname  = $p[0].Trim()
        $vUsed  = [int]$p[1].Trim()
        $vTotal = [int]$p[2].Trim()
        $util   = $p[3].Trim()
        $temp   = $p[4].Trim()
        $filled = [Math]::Round($vUsed * 20 / $vTotal)
        $bar    = ('#' * $filled) + ('.' * (20 - $filled))
        $lines.Add(@{ T = "  GPU   $gname";                                              C = "Yellow" })
        $lines.Add(@{ T = "  VRAM  [$bar] ${vUsed}/${vTotal} MB  |  ${util}%  |  ${temp}C"; C = "Cyan"   })
    }

    # RAM
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os) {
        $rUsed  = [Math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
        $rTotal = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 0)
        $filled = [Math]::Round($rUsed * 20 / $rTotal)
        $bar    = ('#' * $filled) + ('.' * (20 - $filled))
        $lines.Add(@{ T = "  RAM   [$bar] ${rUsed}/${rTotal} GB"; C = "Cyan" })
    }

    # Ollama — use curl.exe (Invoke-WebRequest hangs on Ollama's root endpoint)
    $ollamaJson = curl.exe -s --max-time 2 "http://localhost:11434/api/tags" 2>$null
    if ($ollamaJson) {
        try {
            $loaded = ($ollamaJson | ConvertFrom-Json).models.Count
        } catch { $loaded = "?" }
        $lines.Add(@{ T = "  Ollama  ONLINE  ($loaded model(s) available)"; C = "Green" })
    } else {
        $lines.Add(@{ T = "  Ollama  OFFLINE"; C = "DarkRed" })
    }

    return $lines.ToArray()
}

# --- Arrow-key menu with live dashboard ------------------------------------
function Show-ArrowMenu {
    param(
        [string[]]$Items,
        [string]$Title = "Claude Launcher",
        [object[]]$Dashboard = @()
    )
    $selected = 0
    [Console]::CursorVisible = $false
    try {
        while ($true) {
            Clear-Host
            Write-Host ""
            Write-Host "  +================================================+" -ForegroundColor DarkCyan
            Write-Host ("  |   {0,-46}|" -f "  CLAUDE  WORKSTATION") -ForegroundColor Cyan
            Write-Host "  +================================================+" -ForegroundColor DarkCyan
            Write-Host ""
            if ($Dashboard.Count -gt 0) {
                foreach ($line in $Dashboard) {
                    Write-Host $line.T -ForegroundColor $line.C
                }
                Write-Host ""
                Write-Host "  +------------------------------------------------+" -ForegroundColor DarkGray
                Write-Host ""
            }
            for ($i = 0; $i -lt $Items.Count; $i++) {
                if ($i -eq $selected) {
                    Write-Host "  > $($Items[$i])" -ForegroundColor Green
                } else {
                    Write-Host "    $($Items[$i])" -ForegroundColor DarkGray
                }
            }
            Write-Host ""
            Write-Host "  [Up/Dn] Navigate   [Enter] Select   [Esc] Quit" -ForegroundColor DarkGray
            $key = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            switch ($key.VirtualKeyCode) {
                38 { $selected = if ($selected -gt 0) { $selected - 1 } else { $Items.Count - 1 } }
                40 { $selected = if ($selected -lt $Items.Count - 1) { $selected + 1 } else { 0 } }
                13 { return "$($selected + 1)" }
                27 { return $null }
            }
        }
    } finally {
        [Console]::CursorVisible = $true
    }
}
# ---------------------------------------------------------------------------

$menuItems = @(
    "1) Claude  - Anthropic Pro",
    "2) Claude  - Ollama large  (gpt-oss:20b)",
    "3) Claude  - Ollama tiny   (forge:1.7b)",
    "4) Speech  - Anthropic Pro  [voice-to-voice]",
    "5) Speech  - Ollama         [voice-to-voice]",
    "6) NVIDIA  - qwen2.5-coder-32b  [nvidia api]",
    "7) Claude  - Anthropic Pro  [full auto]",
    "8) Claude  - Ollama gpt-oss:20b  [full auto]"
)

Clear-Host
Show-Startup
$dash   = Get-Dashboard
$choice = Show-ArrowMenu -Items $menuItems -Dashboard $dash
if (-not $choice) { exit 0 }
Write-Host ""

switch ($choice) {
    "1" {
        Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:ANTHROPIC_BASE_URL   -ErrorAction SilentlyContinue
        Write-Host "  >> Claude - Anthropic Pro" -ForegroundColor Cyan
        claude
    }
    "2" {
        $env:ANTHROPIC_AUTH_TOKEN = "ollama"
        $env:ANTHROPIC_BASE_URL   = "http://localhost:11434"
        Write-Host "  >> Claude - Ollama (gpt-oss:20b)" -ForegroundColor Yellow
        claude --model gpt-oss:20b
    }
    "3" {
        $env:ANTHROPIC_AUTH_TOKEN = "ollama"
        $env:ANTHROPIC_BASE_URL   = "http://localhost:11434"
        Write-Host "  >> Claude - Ollama tiny (qwen3:1.7b)" -ForegroundColor Yellow
        claude --allowedTools "Bash,Read,Edit,Write,MultiEdit,Glob,Grep,LS" --model forge:1.7b
    }
    "4" {
        Write-Host "  >> Speech - Anthropic Pro  (say 'goodbye claude' to exit)" -ForegroundColor Magenta
        & $PythonExe -m scripts_and_skills.speech.voice_pipeline `
            --voice "en-US-AriaNeural" --log-level "WARNING"
    }
    "5" {
        Write-Host "  >> Speech - Ollama  (say 'goodbye claude' to exit)" -ForegroundColor Magenta
        & $PythonExe -m scripts_and_skills.speech.voice_pipeline `
            --ollama --voice "en-US-AriaNeural" --log-level "WARNING"
    }
    "6" {
        $keyFile = "$env:USERPROFILE\Desktop\test_file.txt"
        if (-not (Test-Path $keyFile)) {
            Write-Host "  ERROR: Key file not found at $keyFile" -ForegroundColor Red
        } else {
            $apiKey = (Get-Content $keyFile -Raw).Trim()
            $env:ANTHROPIC_AUTH_TOKEN = $apiKey
            $env:ANTHROPIC_BASE_URL   = "https://integrate.api.nvidia.com/v1"
            Write-Host "  >> NVIDIA API key loaded" -ForegroundColor Green
            Write-Host "  >> Routing Claude Code -> NVIDIA (qwen2.5-coder-32b-instruct)" -ForegroundColor Cyan
            claude --model qwen/qwen2.5-coder-32b-instruct
        }
    }
    "7" {
        Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:ANTHROPIC_BASE_URL   -ErrorAction SilentlyContinue
        Write-Host "  >> Claude - Anthropic Pro  [full auto / --dangerously-skip-permissions]" -ForegroundColor Yellow
        Write-Host "  >> All tool confirmations suppressed." -ForegroundColor DarkYellow
        claude --dangerously-skip-permissions
    }
    "8" {
        $env:ANTHROPIC_AUTH_TOKEN = "ollama"
        $env:ANTHROPIC_BASE_URL   = "http://localhost:11434"
        Write-Host "  >> Claude - Ollama (gpt-oss:20b)  [full auto / --dangerously-skip-permissions]" -ForegroundColor Yellow
        Write-Host "  >> All tool confirmations suppressed." -ForegroundColor DarkYellow
        claude --model gpt-oss:20b --dangerously-skip-permissions
    }
    default {
        Write-Host "  Invalid selection. Please enter 1-8." -ForegroundColor Red
    }
}