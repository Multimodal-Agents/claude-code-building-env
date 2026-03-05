# run_claude.ps1 - Interactive Claude / Speech launcher
# Desktop shortcut target:
#   powershell -NoExit -File "M:\claude_code_building_env\scripts_and_skills\claude_scripts\run_claude.ps1"

Set-Location -Path "M:\claude_code_building_env"
$PythonExe = "M:\claude_code_building_env\venv\Scripts\python.exe"

Write-Host ""
Write-Host "  +--------------------------------------+" -ForegroundColor Cyan
Write-Host "  |         Claude Launcher              |" -ForegroundColor Cyan
Write-Host "  +--------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1) Claude  - Anthropic Pro" -ForegroundColor White
Write-Host "  2) Claude  - Ollama large  (gpt-oss:20b)" -ForegroundColor Yellow
Write-Host "  3) Claude  - Ollama tiny   (forge:1.7b)" -ForegroundColor Yellow
Write-Host "  4) Speech  - Anthropic Pro  [voice-to-voice]" -ForegroundColor Magenta
Write-Host "  5) Speech  - Ollama         [voice-to-voice]" -ForegroundColor Magenta
Write-Host "  6) NVIDIA  - Developer API  (sets NVIDIA_API_KEY + launches Claude)" -ForegroundColor Green
Write-Host ""

$choice = Read-Host "  Select mode [1-6]"
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
            $env:NVIDIA_API_KEY = (Get-Content $keyFile -Raw).Trim()
            Write-Host "  >> NVIDIA API key loaded from desktop file" -ForegroundColor Green
            Write-Host "  >> NVIDIA_API_KEY is set for this session" -ForegroundColor Green
            Write-Host "  >> Launching Claude (Anthropic Pro) - use NVIDIA API in your scripts" -ForegroundColor Cyan
            Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
            Remove-Item Env:ANTHROPIC_BASE_URL   -ErrorAction SilentlyContinue
            claude
        }
    }
    default {
        Write-Host "  Invalid selection. Please enter 1-6." -ForegroundColor Red
    }
}