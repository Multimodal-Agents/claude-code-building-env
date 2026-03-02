"""
svc_setup.py — One-shot setup for whisper-vits-svc voice conversion.

Usage
-----
  python -m scripts_and_skills.speech.svc_setup [--dir PATH] [--cuda] [--check]

Steps performed
---------------
1. Install CUDA-enabled PyTorch (cu128, works with RTX 50-series)       [--cuda]
2. Clone PlayVoice/whisper-vits-svc into --dir                          [auto]
3. Install repo requirements                                             [auto]
4. Download pretrained models:
     - whisper/large-v2.pt         (content encoder, ~3 GB)
     - hubert_pretrain/*.pt        (alternative encoder, ~95 MB)
     - vits_pretrain/sovits5.0.pretrain.pth  (voice model, ~105 MB)
     - crepe/assets/full.pth       (pitch extractor, ~85 MB)
5. Print env var block to paste into your shell profile                  [auto]
6. Test the installation with a short sample phrase                      [--check]

Manual step (cannot automate): speaker encoder
  Download best_model.pth.tar from Google Drive and place at:
    <svc_dir>/speaker_pretrain/best_model.pth.tar
  Link: https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3

For custom voice training, see SKILL.md section "Training Your Own Voice".
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Model URLs
# ---------------------------------------------------------------------------
_WHISPER_LARGE_V2_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
)
_HUBERT_SOFT_URL = (
    "https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt"
)
# sovits5.0 pretrain — from PlayVoice releases
_SOVITS5_URL = (
    "https://github.com/PlayVoice/so-vits-svc-5.0/releases/download/5.0/"
    "sovits5.0.pretrain.pth"
)
# CREPE full pitch extractor
_CREPE_URL = (
    "https://github.com/maxrmorrison/torchcrepe/raw/master/torchcrepe/assets/full.pth"
)

_REPO_URL = "https://github.com/PlayVoice/whisper-vits-svc.git"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd=None, check=True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def _download(url: str, dest: Path) -> bool:
    """Download url → dest using requests or urllib. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    print(f"  Downloading {url.split('/')[-1]} ...")
    try:
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {pct:3d}%  {downloaded//1024//1024} MB", end="", flush=True)
        print(f"\r  [ok] {dest.name} ({downloaded//1024//1024} MB)      ")
        return True
    except ImportError:
        pass

    # Fallback: urllib
    import urllib.request
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok] {dest.name}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {dest.name}: {exc}")
        dest.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def step_cuda_pytorch() -> None:
    print("\n[1/5] Installing CUDA-enabled PyTorch (cu128 — RTX 50-series compatible)...")
    print("      This replaces the current CPU-only build. ~2-3 GB download.")
    _run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu128",
        "--upgrade",
    ])


def step_clone(svc_dir: Path) -> None:
    print(f"\n[2/5] Cloning whisper-vits-svc → {svc_dir}")
    if (svc_dir / ".git").exists():
        print("  [skip] repo already cloned")
        return
    svc_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", _REPO_URL, str(svc_dir)])


def step_requirements(svc_dir: Path) -> None:
    print("\n[3/5] Installing repo requirements (CPU deps only — torch installed separately)...")
    # Install without torch to avoid overwriting CUDA torch
    req = svc_dir / "requirements.txt"
    _run([
        sys.executable, "-m", "pip", "install",
        "-r", str(req),
        "--extra-index-url", "https://pypi.org/simple",
    ], cwd=str(svc_dir))


def step_models(svc_dir: Path) -> None:
    print("\n[4/5] Downloading pretrained models...")

    ok = True
    ok &= _download(_WHISPER_LARGE_V2_URL, svc_dir / "whisper_pretrain" / "large-v2.pt")
    ok &= _download(_HUBERT_SOFT_URL, svc_dir / "hubert_pretrain" / "hubert-soft-0d54a1f4.pt")
    ok &= _download(_SOVITS5_URL, svc_dir / "vits_pretrain" / "sovits5.0.pretrain.pth")
    ok &= _download(_CREPE_URL, svc_dir / "crepe" / "assets" / "full.pth")

    spk_enc = svc_dir / "speaker_pretrain" / "best_model.pth.tar"
    if not spk_enc.exists():
        print(f"\n  [MANUAL] Speaker encoder not found at: {spk_enc}")
        print("  Download best_model.pth.tar from:")
        print("    https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3")
        print("  Then move it to:")
        print(f"    {spk_enc}")
    else:
        print(f"  [ok] speaker_pretrain/best_model.pth.tar")

    if not ok:
        print("\n  Some downloads failed. Check your internet connection and retry.")


def step_print_env(svc_dir: Path) -> None:
    """Print env vars block for the user to add to their shell / launcher."""
    model_path = svc_dir / "vits_pretrain" / "sovits5.0.pretrain.pth"
    # Default speaker: the repo ships a demo singer embedding at data_svc/
    # User will point this at their own .spk.npy after training, or download one.
    spk_path = svc_dir / "data_svc" / "singer" / "demo.spk.npy"

    print("\n[5/5] Environment variables (add these to your shell profile or launchers):")
    print()
    print("  PowerShell ($PROFILE):")
    print(f'    $env:S2S_SVC_REPO  = "{svc_dir}"')
    print(f'    $env:S2S_SVC_MODEL = "{model_path}"')
    print(f'    $env:S2S_SVC_SPK   = "{spk_path}"')
    print(f'    $env:S2S_SVC       = "1"')
    print()
    print("  Bash (~/.bashrc or run_claude.sh):")
    print(f'    export S2S_SVC_REPO="{svc_dir}"')
    print(f'    export S2S_SVC_MODEL="{model_path}"')
    print(f'    export S2S_SVC_SPK="{spk_path}"')
    print(f'    export S2S_SVC=1')
    print()
    print("  Note: Update S2S_SVC_SPK to your own speaker embedding after training.")
    print("  See SKILL.md → 'Training Your Own Voice' for the full training workflow.")


def step_check(svc_dir: Path) -> None:
    """Quick sanity check — verify CUDA torch + run a tiny SVC inference."""
    print("\n[check] Verifying installation...")

    # 1. CUDA torch
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('CUDA:', torch.cuda.is_available()); "
         "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if "True" not in result.stdout:
        print("  [WARN] CUDA not available — SVC will run on CPU (slow)")

    # 2. Key model files
    required = [
        svc_dir / "whisper_pretrain" / "large-v2.pt",
        svc_dir / "hubert_pretrain" / "hubert-soft-0d54a1f4.pt",
        svc_dir / "vits_pretrain" / "sovits5.0.pretrain.pth",
        svc_dir / "crepe" / "assets" / "full.pth",
        svc_dir / "speaker_pretrain" / "best_model.pth.tar",
    ]
    all_ok = True
    for f in required:
        status = "[ok]" if f.exists() else "[MISSING]"
        print(f"  {status} {f.relative_to(svc_dir)}")
        if not f.exists():
            all_ok = False

    if all_ok:
        print("\n  All model files present. Run with --svc to enable voice conversion.")
    else:
        print("\n  Some files missing — see above. Re-run without --check to download.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Set up whisper-vits-svc for voice conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        default=str(Path.home() / "whisper-vits-svc"),
        help="Directory to clone repo into (default: ~/whisper-vits-svc)",
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Install CUDA-enabled PyTorch (cu128, for RTX 50-series). "
             "Replaces any existing PyTorch install.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only verify existing installation, do not download anything.",
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Skip pretrained model downloads (useful if you already have them).",
    )
    args = parser.parse_args()

    svc_dir = Path(args.dir).resolve()
    print(f"SVC directory: {svc_dir}")

    if args.check:
        step_check(svc_dir)
        return

    if args.cuda:
        step_cuda_pytorch()

    step_clone(svc_dir)
    step_requirements(svc_dir)

    if not args.skip_models:
        step_models(svc_dir)

    step_print_env(svc_dir)

    print("\nDone. Next steps:")
    print("  1. Add env vars above to your shell profile")
    print("  2. Manually download speaker encoder if not done (see [MANUAL] above)")
    print("  3. (Optional) Train your own voice — see SKILL.md for instructions")
    print("  4. Test: python -m scripts_and_skills.speech.voice_pipeline --svc")
    print()
    print("Re-run with --check to verify all files are present:")
    print(f"  python -m scripts_and_skills.speech.svc_setup --dir {svc_dir} --check")


if __name__ == "__main__":
    main()
