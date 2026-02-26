param(
    [switch]$Ollama  # use -Ollama flag for local, otherwise uses Anthropic
)

Set-Location -Path "M:\claude_code_building_env"

if ($Ollama) {
    $env:ANTHROPIC_AUTH_TOKEN = "ollama"
    $env:ANTHROPIC_BASE_URL = "http://localhost:11434"
    Write-Host "  >> Launching with Ollama (local)" -ForegroundColor Cyan
    claude --model gpt-oss:20b
} else {
    Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "  >> Launching with Anthropic (Pro subscription)" -ForegroundColor Cyan
    claude
}