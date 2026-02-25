<#
.SYNOPSIS
    Sets up llama.cpp for GGUF conversion and quantization.

.DESCRIPTION
    Clones the llama.cpp repository, installs Python conversion
    dependencies, and optionally builds the binaries (requires cmake + C++ compiler).

.PARAMETER LlamaCppDir
    Where to clone llama.cpp (default: M:\llama.cpp)

.PARAMETER SkipBuild
    Skip building the C++ binaries. You still get convert_hf_to_gguf.py,
    but won't have llama-quantize for quantization.

.PARAMETER SkipPython
    Skip installing the Python requirements.

.PARAMETER CudaBuild
    Build with CUDA support (requires CUDA toolkit — good for Titan XP).

.EXAMPLE
    # Basic setup (Python tools only, no compile)
    .\setup_llamacpp.ps1 -SkipBuild

    # Full build with CUDA
    .\setup_llamacpp.ps1 -CudaBuild
#>

param(
    [string]$LlamaCppDir = "M:\llama.cpp",
    [switch]$SkipBuild,
    [switch]$SkipPython,
    [switch]$CudaBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "    [OK] $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "    [WARN] $msg" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 1. Clone or update llama.cpp
# ---------------------------------------------------------------------------
Write-Step "llama.cpp repository"

if (Test-Path "$LlamaCppDir\.git") {
    Write-Host "    Found existing clone at $LlamaCppDir — pulling latest..."
    Push-Location $LlamaCppDir
    git pull --ff-only
    Pop-Location
    Write-OK "Updated"
}
else {
    Write-Host "    Cloning into $LlamaCppDir..."
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaCppDir
    Write-OK "Cloned"
}

# ---------------------------------------------------------------------------
# 2. Python dependencies for convert_hf_to_gguf.py
# ---------------------------------------------------------------------------
if (-not $SkipPython) {
    Write-Step "Python dependencies (convert_hf_to_gguf)"

    $reqFile = "$LlamaCppDir\requirements\requirements-convert_hf_to_gguf.txt"
    if (-not (Test-Path $reqFile)) {
        # Older llama.cpp layout
        $reqFile = "$LlamaCppDir\requirements.txt"
    }

    if (Test-Path $reqFile) {
        Write-Host "    Installing from $reqFile..."
        pip install -r $reqFile
        Write-OK "Python deps installed"
    }
    else {
        Write-Warn "requirements file not found at expected path. Manual install:"
        Write-Host "         pip install numpy torch transformers sentencepiece gguf" -ForegroundColor DarkGray
    }

    # Also install huggingface_hub for the hf_download helper
    Write-Host "    Installing huggingface_hub..."
    pip install huggingface_hub
    Write-OK "huggingface_hub installed"
}

# ---------------------------------------------------------------------------
# 3. Build C++ binaries (llama-quantize, llama-cli, etc.)
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Step "Building llama.cpp binaries"

    # Check for cmake
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Warn "cmake not found. Install from https://cmake.org/download/ then rerun."
        Write-Warn "Skipping build step. You can still use convert_hf_to_gguf.py."
        goto done_build
    }

    Push-Location $LlamaCppDir

    $buildDir = "$LlamaCppDir\build"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

    if ($CudaBuild) {
        Write-Host "    Configuring with CUDA support..."
        cmake -B $buildDir -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    }
    else {
        Write-Host "    Configuring CPU-only build..."
        cmake -B $buildDir -DCMAKE_BUILD_TYPE=Release
    }

    Write-Host "    Compiling (this will take a few minutes)..."
    cmake --build $buildDir --config Release -j $env:NUMBER_OF_PROCESSORS

    Pop-Location
    Write-OK "Build complete — binaries in $LlamaCppDir\build\bin\Release (or build\bin)"
}
:done_build

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  llama.cpp setup complete" -ForegroundColor Cyan
Write-Host "============================================================"

$convertScript = "$LlamaCppDir\convert_hf_to_gguf.py"
if (Test-Path $convertScript) {
    Write-OK "convert_hf_to_gguf.py: $convertScript"
}
else {
    Write-Warn "convert_hf_to_gguf.py not found. Check the repo structure."
}

$quantBin = "$LlamaCppDir\build\bin\Release\llama-quantize.exe"
$quantBin2 = "$LlamaCppDir\build\bin\llama-quantize"
if ((Test-Path $quantBin) -or (Test-Path $quantBin2)) {
    Write-OK "llama-quantize binary found"
}
elseif (-not $SkipBuild) {
    Write-Warn "llama-quantize not found — quantization will need the binary in PATH"
}

Write-Host ""
Write-Host "  Use the GGUF tools:" -ForegroundColor White
Write-Host "    # Download a GGUF from HuggingFace" -ForegroundColor DarkGray
Write-Host "    python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF --list"
Write-Host ""
Write-Host "    # Convert a HF safetensors model to GGUF" -ForegroundColor DarkGray
Write-Host "    python -m scripts_and_skills.model_manager.gguf_manager convert \"" + 'C:\models\hf\my-model' + "\" output.gguf --script $convertScript"
Write-Host ""
Write-Host "    # Quantize" -ForegroundColor DarkGray
Write-Host "    python -m scripts_and_skills.model_manager.gguf_manager quantize input-f16.gguf output-Q4_K_M.gguf --type Q4_K_M"
Write-Host ""
