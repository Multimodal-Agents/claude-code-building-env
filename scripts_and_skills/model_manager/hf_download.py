"""
hf_download.py — Download GGUF files from HuggingFace Hub

Downloads specific files (or whole repos) from HuggingFace into your local
GGUF directory, then auto-registers them in the GGUFManager registry.

Requires:
    pip install huggingface_hub

Usage examples:

    # Download a specific quantized GGUF (recommended — pick your quant)
    python -m scripts_and_skills.model_manager.hf_download \\
        unsloth/gpt-oss-20b-GGUF \\
        --file "gpt-oss-20b-Q4_K_M.gguf" \\
        --out "C:/models/gguf"

    # List all files in an HF GGUF repo (to see what quants are available)
    python -m scripts_and_skills.model_manager.hf_download \\
        unsloth/gpt-oss-20b-GGUF --list

    # Download ALL files in a repo (careful — can be many GB)
    python -m scripts_and_skills.model_manager.hf_download \\
        unsloth/gpt-oss-20b-GGUF --all --out "C:/models/gguf"

    # Deploy immediately to Ollama after download
    python -m scripts_and_skills.model_manager.hf_download \\
        unsloth/gpt-oss-20b-GGUF \\
        --file "gpt-oss-20b-Q4_K_M.gguf" \\
        --deploy --name "gpt-oss:20b-q4"

Quant size guide (for 20B models):
    Q2_K   ~  8 GB  — smallest, quality loss
    Q3_K_M ~ 11 GB  — workable
    Q4_K_M ~ 13 GB  — recommended balance  ← good for Titan XP
    Q5_K_M ~ 16 GB  — better quality
    Q8_0   ~ 22 GB  — near lossless
    F16    ~ 40 GB  — unquantized (too large for 12 GB VRAM)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

# Default GGUF output directory
DEFAULT_OUTPUT = Path(os.environ.get("GGUF_DIR", "M:/models/gguf"))


def _hub():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
        return hf_hub_download, list_repo_files, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)


def list_repo(repo_id: str, token: Optional[str] = None):
    """Print all files in an HF repo, highlighting GGUF files."""
    _, list_repo_files, _ = _hub()
    print(f"\nFiles in {repo_id}:")
    print("-" * 60)
    files = list(list_repo_files(repo_id, token=token))
    files.sort()
    for f in files:
        marker = "  ← GGUF" if f.endswith(".gguf") else ""
        size_hint = ""
        if "Q4_K_M" in f:  size_hint = " (recommended)"
        elif "Q5_K_M" in f: size_hint = " (high quality)"
        elif "Q8_0" in f:   size_hint = " (near lossless)"
        elif "F16" in f:    size_hint = " (unquantized)"
        print(f"  {f}{marker}{size_hint}")
    print()
    gguf_files = [f for f in files if f.endswith(".gguf")]
    print(f"Found {len(gguf_files)} GGUF file(s).")
    return files


def download_file(
    repo_id: str,
    filename: str,
    output_dir: str | Path = DEFAULT_OUTPUT,
    token: Optional[str] = None,
    auto_register: bool = True,
) -> Path:
    """
    Download a single GGUF file from an HF repo.

    Args:
        repo_id:      e.g. "unsloth/gpt-oss-20b-GGUF"
        filename:     e.g. "gpt-oss-20b-Q4_K_M.gguf"
        output_dir:   local directory to save the file
        token:        HF token for private repos
        auto_register: add to GGUFManager registry after download

    Returns:
        Path to the downloaded file.
    """
    hf_hub_download, _, _ = _hub()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")
    print(f"Destination: {output_dir}")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
        token=token,
    )
    local_path = Path(local_path)
    size_gb = local_path.stat().st_size / 1e9
    print(f"Downloaded: {local_path.name}  ({size_gb:.2f} GB)")

    if auto_register:
        from .gguf_manager import GGUFManager
        mgr = GGUFManager()
        # Infer base model name from repo (e.g. "unsloth/gpt-oss-20b-GGUF" → "gpt-oss-20b")
        base = repo_id.split("/")[-1].replace("-GGUF", "").replace("-gguf", "")
        mgr.register(local_path, base_model=base, source=f"hf:{repo_id}")

    return local_path


def download_all(
    repo_id: str,
    output_dir: str | Path = DEFAULT_OUTPUT,
    token: Optional[str] = None,
    gguf_only: bool = True,
) -> list[Path]:
    """
    Download all (GGUF) files from a repo.
    Set gguf_only=False to download everything including config files.
    WARNING: Full repos can be very large.
    """
    _, list_repo_files, snapshot_download = _hub()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gguf_only:
        files = [f for f in list_repo_files(repo_id, token=token) if f.endswith(".gguf")]
        paths = []
        for f in files:
            p = download_file(repo_id, f, output_dir, token, auto_register=True)
            paths.append(p)
        return paths
    else:
        print(f"Downloading full repo {repo_id} → {output_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(output_dir), token=token)
        # Register any GGUFs found
        from .gguf_manager import GGUFManager
        mgr = GGUFManager()
        return [Path(mgr.register(f)["path"]) for f in output_dir.rglob("*.gguf")]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GGUF files from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See what files are in a repo
  python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF --list

  # Download the Q4_K_M variant (good for Titan XP / 12 GB VRAM)
  python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF \\
      --file "gpt-oss-20b-Q4_K_M.gguf" --out M:/models/gguf

  # Download and immediately deploy to Ollama
  python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF \\
      --file "gpt-oss-20b-Q4_K_M.gguf" --out M:/models/gguf \\
      --deploy --name gpt-oss:20b
        """
    )
    parser.add_argument("repo_id", help="HuggingFace repo ID, e.g. unsloth/gpt-oss-20b-GGUF")
    parser.add_argument("--file",    default=None,          help="Specific filename to download")
    parser.add_argument("--out",     default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--token",   default=None,          help="HF token for private repos")
    parser.add_argument("--list",    action="store_true",   help="List files in the repo (no download)")
    parser.add_argument("--all",     action="store_true",   help="Download all GGUF files in the repo")
    parser.add_argument("--deploy",  action="store_true",   help="Deploy to Ollama after download")
    parser.add_argument("--name",    default=None,          help="Ollama model name (for --deploy)")
    parser.add_argument("--system",  default="",            help="System prompt for --deploy")
    parser.add_argument("--ctx",     type=int, default=None, help="Context window for --deploy")

    args = parser.parse_args()

    if args.list:
        list_repo(args.repo_id, token=args.token)
        return

    if args.all:
        paths = download_all(args.repo_id, output_dir=args.out, token=args.token)
        print(f"Downloaded {len(paths)} files.")
        return

    if args.file is None:
        print("Specify --file <filename> or use --list to see available files, or --all to download everything.")
        print(f"\nQuick tip: python -m scripts_and_skills.model_manager.hf_download {args.repo_id} --list")
        return

    local_path = download_file(args.repo_id, args.file, output_dir=args.out, token=args.token)

    if args.deploy:
        if args.name is None:
            # Derive name from filename
            args.name = Path(args.file).stem.lower().replace("_", "-").replace(".", "-")
            print(f"Auto-derived Ollama name: {args.name}")
        from .gguf_manager import GGUFManager
        GGUFManager().deploy_to_ollama(
            local_path,
            ollama_name=args.name,
            system_prompt=args.system,
            num_ctx=args.ctx,
        )


if __name__ == "__main__":
    _cli()
