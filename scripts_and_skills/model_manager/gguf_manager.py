"""
gguf_manager.py — GGUF file registry and HuggingFace → GGUF conversion

Tracks local GGUF files in a parquet registry, provides conversion from
HuggingFace safetensors via llama.cpp's convert_hf_to_gguf.py, and
quantization helpers.

Conversion source (llama.cpp):
    https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py

Usage:
    # Scan a directory and register all GGUFs
    mgr = GGUFManager()
    mgr.scan("/models/gguf")

    # Convert a HuggingFace model directory to GGUF
    mgr.convert_hf_to_gguf(
        model_dir="C:/models/hf/mistral-7b",
        output_path="C:/models/gguf/mistral-7b-f16.gguf",
        convert_script="C:/llama.cpp/convert_hf_to_gguf.py"
    )

    # Register a specific GGUF with metadata
    mgr.register("/models/gguf/mymodel.gguf", base_model="mistral-7b", quant="Q4_K_M")

    # List all known GGUFs
    mgr.list()

    # Create an Ollama model from a registered GGUF
    mgr.deploy_to_ollama("mymodel.gguf", ollama_name="my-mistral", system_prompt="You are...")

CLI:
    python -m scripts_and_skills.model_manager.gguf_manager scan <dir>
    python -m scripts_and_skills.model_manager.gguf_manager list
    python -m scripts_and_skills.model_manager.gguf_manager convert <hf_dir> <output.gguf> --script <path>
    python -m scripts_and_skills.model_manager.gguf_manager deploy <gguf_path> --name <ollama_name>
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .modelfile import ModelfileBuilder

# Default registry location
REGISTRY_DIR = Path(os.environ.get("GGUF_REGISTRY_DIR", "M:/claude_code_building_env/local_data/gguf"))
REGISTRY_FILE = REGISTRY_DIR / "registry.parquet"

# Schema columns
COLUMNS = [
    "id",            # uuid
    "filename",      # basename of the file
    "path",          # full absolute path
    "size_gb",       # float
    "base_model",    # e.g. mistral-7b, llama3.2
    "quantization",  # e.g. Q4_K_M, F16, Q8_0
    "architecture",  # e.g. llama, mistral, gemma
    "source",        # hf repo or manual
    "notes",         # free-form
    "registered_at", # ISO timestamp
    "ollama_name",   # name in Ollama if deployed
]


class GGUFManager:
    """Manages a parquet registry of local GGUF files."""

    def __init__(self, registry_dir: str | Path = REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.parquet"
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Registry I/O
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if self.registry_file.exists():
            self._df = pd.read_parquet(self.registry_file)
        else:
            self._df = pd.DataFrame(columns=COLUMNS)
        return self._df

    def _save(self, df: pd.DataFrame):
        df.to_parquet(self.registry_file, index=False)
        self._df = df

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        path: str | Path,
        base_model: str = "",
        quantization: str = "",
        architecture: str = "",
        source: str = "manual",
        notes: str = "",
    ) -> dict:
        """
        Add a GGUF file to the registry.
        If the file is already registered (same path), updates metadata.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GGUF not found: {path}")

        df = self._load()
        size_gb = path.stat().st_size / 1e9

        # Auto-detect quantization from filename if not provided
        if not quantization:
            quantization = _guess_quant(path.name)
        if not architecture:
            architecture = _guess_arch(path.name)

        existing = df[df["path"] == str(path)]
        if not existing.empty:
            idx = existing.index[0]
            df.at[idx, "base_model"]    = base_model or df.at[idx, "base_model"]
            df.at[idx, "quantization"]  = quantization or df.at[idx, "quantization"]
            df.at[idx, "architecture"]  = architecture or df.at[idx, "architecture"]
            df.at[idx, "notes"]         = notes or df.at[idx, "notes"]
            df.at[idx, "size_gb"]       = size_gb
            record = df.iloc[idx].to_dict()
        else:
            record = {
                "id":            str(uuid.uuid4()),
                "filename":      path.name,
                "path":          str(path),
                "size_gb":       round(size_gb, 3),
                "base_model":    base_model,
                "quantization":  quantization,
                "architecture":  architecture,
                "source":        source,
                "notes":         notes,
                "registered_at": datetime.utcnow().isoformat(),
                "ollama_name":   "",
            }
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

        self._save(df)
        print(f"Registered: {path.name}  ({size_gb:.2f} GB)")
        return record

    def scan(self, directory: str | Path, **kwargs) -> int:
        """Recursively scan a directory and register all .gguf files."""
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return 0
        found = list(directory.rglob("*.gguf"))
        for f in found:
            self.register(f, **kwargs)
        print(f"Scanned {len(found)} GGUF(s) in {directory}")
        return len(found)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list(self, verbose: bool = False):
        df = self._load()
        if df.empty:
            print("No GGUFs registered. Run: gguf_manager.scan('<dir>')")
            return
        cols = ["filename", "size_gb", "base_model", "quantization", "ollama_name"] if not verbose else COLUMNS
        print(df[cols].to_string(index=False))

    def get(self, filename_or_path: str) -> Optional[dict]:
        df = self._load()
        row = df[(df["filename"] == filename_or_path) | (df["path"] == filename_or_path)]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # HuggingFace → GGUF conversion (via llama.cpp)
    # ------------------------------------------------------------------

    def convert_hf_to_gguf(
        self,
        model_dir: str | Path,
        output_path: str | Path,
        convert_script: str | Path | None = None,
        outtype: str = "f16",
        extra_args: list[str] | None = None,
    ) -> Path:
        """
        Convert a HuggingFace safetensors model directory to GGUF using
        llama.cpp's convert_hf_to_gguf.py.

        Args:
            model_dir:       Path to the HF model directory (has config.json, *.safetensors)
            output_path:     Desired output .gguf file path
            convert_script:  Path to convert_hf_to_gguf.py (auto-detected if not given)
            outtype:         Output dtype: f32, f16, bf16, q8_0, auto (default: f16)
            extra_args:      Additional CLI args passed verbatim to the script

        Returns:
            Path to the created GGUF file.

        Notes:
            Requires llama.cpp cloned locally. The script is at:
            https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
            Install deps: pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
        """
        model_dir   = Path(model_dir).resolve()
        output_path = Path(output_path).resolve()

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Auto-detect convert script
        if convert_script is None:
            convert_script = _find_convert_script()
        convert_script = Path(convert_script)
        if not convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at: {convert_script}\n"
                "Clone llama.cpp and pass --script <path/to/convert_hf_to_gguf.py>"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outfile", str(output_path),
            "--outtype", outtype,
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"Converting {model_dir.name} → {output_path.name} (outtype={outtype})")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed (exit {result.returncode}). See output above.")

        print(f"Conversion complete: {output_path}")
        # Auto-register
        self.register(output_path, source=str(model_dir), quantization=outtype.upper())
        return output_path

    # ------------------------------------------------------------------
    # Quantize an existing GGUF (requires llama-quantize binary)
    # ------------------------------------------------------------------

    def quantize(
        self,
        gguf_path: str | Path,
        output_path: str | Path,
        quant_type: str = "Q4_K_M",
        llama_quantize: str | Path | None = None,
    ) -> Path:
        """
        Quantize a GGUF file using llama-quantize (or quantize.exe on Windows).

        Args:
            gguf_path:      Source F16/F32 GGUF
            output_path:    Output quantized GGUF path
            quant_type:     e.g. Q4_K_M, Q5_K_M, Q8_0, Q2_K
            llama_quantize: Path to the llama-quantize binary

        Common quant types:
            Q2_K  — smallest, lowest quality
            Q4_K_M — good balance (recommended for 7B models)
            Q5_K_M — better quality, slightly larger
            Q8_0   — near lossless, 2x the size of Q4
        """
        gguf_path   = Path(gguf_path).resolve()
        output_path = Path(output_path).resolve()

        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF not found: {gguf_path}")

        binary = llama_quantize or _find_binary("llama-quantize", "quantize")
        if binary is None:
            raise FileNotFoundError(
                "llama-quantize binary not found. Build llama.cpp and add it to PATH, "
                "or pass llama_quantize='path/to/llama-quantize'."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(binary), str(gguf_path), str(output_path), quant_type]

        print(f"Quantizing {gguf_path.name} → {output_path.name} ({quant_type})")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Quantization failed (exit {result.returncode})")

        print(f"Quantized: {output_path}")
        self.register(output_path, quantization=quant_type)
        return output_path

    # ------------------------------------------------------------------
    # Deploy to Ollama
    # ------------------------------------------------------------------

    def deploy_to_ollama(
        self,
        gguf_path: str | Path,
        ollama_name: str,
        system_prompt: str = "",
        temperature: float | None = None,
        num_ctx: int | None = None,
    ) -> None:
        """
        Create an Ollama model from a GGUF file.
        Wraps ModelfileBuilder.from_gguf().create_model().
        """
        b = ModelfileBuilder.from_gguf(str(gguf_path))
        if system_prompt:
            b.set_system(system_prompt)
        if temperature is not None:
            b.set_parameter("temperature", temperature)
        if num_ctx is not None:
            b.set_parameter("num_ctx", num_ctx)
        b.create_model(ollama_name)

        # Update registry
        df = self._load()
        mask = df["path"] == str(Path(gguf_path).resolve())
        if mask.any():
            df.loc[mask, "ollama_name"] = ollama_name
            self._save(df)
            print(f"Registry updated: ollama_name={ollama_name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_quant(filename: str) -> str:
    """Try to extract quantization from filename (e.g. model-Q4_K_M.gguf)."""
    patterns = ["Q2_K", "Q3_K", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M",
                "Q5_K_S", "Q5_0", "Q6_K", "Q8_0", "F16", "F32", "BF16"]
    upper = filename.upper()
    for p in patterns:
        if p in upper:
            return p
    return ""


def _guess_arch(filename: str) -> str:
    lower = filename.lower()
    for arch in ("llama", "mistral", "gemma", "phi", "qwen", "falcon", "mpt", "deepseek"):
        if arch in lower:
            return arch
    return ""


def _find_convert_script() -> Path:
    """Try common locations for convert_hf_to_gguf.py."""
    candidates = [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("../llama.cpp/convert_hf_to_gguf.py"),
        Path("C:/llama.cpp/convert_hf_to_gguf.py"),
        Path("M:/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not auto-detect convert_hf_to_gguf.py. "
        "Use --script <path> or clone https://github.com/ggml-org/llama.cpp"
    )


def _find_binary(*names) -> Optional[Path]:
    """Search PATH for a binary by multiple possible names."""
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
        # Windows .exe variant
        found = shutil.which(name + ".exe")
        if found:
            return Path(found)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="GGUF manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List registered GGUFs")
    sub_scan = sub.add_parser("scan", help="Scan a directory for GGUFs")
    sub_scan.add_argument("dir")

    sub_reg = sub.add_parser("register", help="Register a single GGUF")
    sub_reg.add_argument("path")
    sub_reg.add_argument("--base",  default="")
    sub_reg.add_argument("--quant", default="")
    sub_reg.add_argument("--arch",  default="")
    sub_reg.add_argument("--notes", default="")

    sub_conv = sub.add_parser("convert", help="Convert HF safetensors to GGUF")
    sub_conv.add_argument("model_dir",   help="HuggingFace model directory")
    sub_conv.add_argument("output",      help="Output .gguf path")
    sub_conv.add_argument("--script",    default=None, help="Path to convert_hf_to_gguf.py")
    sub_conv.add_argument("--outtype",   default="f16", help="f32/f16/bf16/q8_0/auto")

    sub_quant = sub.add_parser("quantize", help="Quantize a GGUF")
    sub_quant.add_argument("gguf",    help="Source F16 GGUF")
    sub_quant.add_argument("output",  help="Output GGUF path")
    sub_quant.add_argument("--type",  default="Q4_K_M", dest="quant_type")
    sub_quant.add_argument("--binary", default=None)

    sub_dep = sub.add_parser("deploy", help="Deploy GGUF to Ollama")
    sub_dep.add_argument("gguf",   help="GGUF file path")
    sub_dep.add_argument("--name", required=True, dest="ollama_name")
    sub_dep.add_argument("--system", default="")
    sub_dep.add_argument("--ctx",    type=int, default=None)
    sub_dep.add_argument("--temp",   type=float, default=None)

    args = parser.parse_args()
    mgr = GGUFManager()

    if args.cmd == "list":
        mgr.list()
    elif args.cmd == "scan":
        mgr.scan(args.dir)
    elif args.cmd == "register":
        mgr.register(args.path, base_model=args.base, quantization=args.quant,
                     architecture=args.arch, notes=args.notes)
    elif args.cmd == "convert":
        mgr.convert_hf_to_gguf(args.model_dir, args.output,
                                convert_script=args.script, outtype=args.outtype)
    elif args.cmd == "quantize":
        mgr.quantize(args.gguf, args.output, quant_type=args.quant_type,
                     llama_quantize=args.binary)
    elif args.cmd == "deploy":
        mgr.deploy_to_ollama(args.gguf, args.ollama_name,
                             system_prompt=args.system,
                             num_ctx=args.ctx, temperature=args.temp)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
