"""
lora_manager.py — LoRA adapter registry and application

Tracks local LoRA adapter files (.gguf or safetensor) in a parquet
registry, and provides helpers to apply them via Ollama Modelfiles
or llama.cpp's server.

LoRA adapters in Ollama:
  The ADAPTER instruction in a Modelfile applies a (Q)LoRA adapter on top
  of the base model. The adapter must be a GGUF file compatible with the base.
  Reference: basic_reference_documentation_library/ollama/modelfile.mdx#adapter

Usage:
    # Register an adapter
    lm = LoRAManager()
    lm.register("C:/adapters/coding-expert.gguf", base_model="llama3.2", task="code")

    # List adapters
    lm.list()

    # Build an Ollama model that uses the adapter
    lm.deploy_with_adapter(
        adapter="coding-expert.gguf",
        base_model="llama3.2",
        ollama_name="llama3-coder",
        system_prompt="You are an expert Python developer."
    )

CLI:
    python -m scripts_and_skills.model_manager.lora_manager list
    python -m scripts_and_skills.model_manager.lora_manager register <path> --base llama3.2 --task code
    python -m scripts_and_skills.model_manager.lora_manager deploy <adapter_file> --base llama3.2 --name my-model
"""

from __future__ import annotations
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .modelfile import ModelfileBuilder

LORA_DIR = Path(os.environ.get("LORA_DIR", "M:/claude_code_building_env/local_data/lora"))
REGISTRY_FILE = LORA_DIR / "registry.parquet"

COLUMNS = [
    "id",
    "filename",
    "path",
    "size_gb",
    "base_model",     # compatible base model (e.g. llama3.2)
    "task",           # e.g. code, chat, math, roleplay
    "format",         # gguf, safetensors
    "source",         # hf repo, training run, etc.
    "notes",
    "registered_at",
    "ollama_name",    # if deployed as an Ollama model
]


class LoRAManager:
    """Registry and deployment helper for LoRA adapters."""

    def __init__(self, lora_dir: str | Path = LORA_DIR):
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.lora_dir / "registry.parquet"
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
        task: str = "",
        source: str = "manual",
        notes: str = "",
    ) -> dict:
        """
        Register a LoRA adapter file.
        Supports .gguf (Ollama) and .safetensors (for llama.cpp or Unsloth training).
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {path}")

        fmt = "gguf" if path.suffix == ".gguf" else "safetensors"
        size_gb = path.stat().st_size / 1e9

        df = self._load()
        existing = df[df["path"] == str(path)]
        if not existing.empty:
            idx = existing.index[0]
            df.at[idx, "base_model"] = base_model or df.at[idx, "base_model"]
            df.at[idx, "task"]       = task       or df.at[idx, "task"]
            df.at[idx, "notes"]      = notes      or df.at[idx, "notes"]
            record = df.iloc[idx].to_dict()
        else:
            record = {
                "id":            str(uuid.uuid4()),
                "filename":      path.name,
                "path":          str(path),
                "size_gb":       round(size_gb, 3),
                "base_model":    base_model,
                "task":          task,
                "format":        fmt,
                "source":        source,
                "notes":         notes,
                "registered_at": datetime.utcnow().isoformat(),
                "ollama_name":   "",
            }
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

        self._save(df)
        print(f"Registered adapter: {path.name}  ({size_gb:.2f} GB, format={fmt})")
        return record

    def list(self, task_filter: str = ""):
        df = self._load()
        if df.empty:
            print("No LoRA adapters registered.")
            return
        if task_filter:
            df = df[df["task"].str.contains(task_filter, case=False, na=False)]
        cols = ["filename", "size_gb", "base_model", "task", "format", "ollama_name"]
        print(df[cols].to_string(index=False))

    def get(self, filename_or_path: str) -> Optional[dict]:
        df = self._load()
        row = df[(df["filename"] == filename_or_path) | (df["path"] == filename_or_path)]
        return row.iloc[0].to_dict() if not row.empty else None

    # ------------------------------------------------------------------
    # Deploy adapter via Ollama Modelfile
    # ------------------------------------------------------------------

    def deploy_with_adapter(
        self,
        adapter: str | Path,
        base_model: str,
        ollama_name: str,
        system_prompt: str = "",
        temperature: float | None = None,
        num_ctx: int | None = None,
    ) -> None:
        """
        Create an Ollama model that applies a LoRA adapter to a base model.

        Args:
            adapter:      Path to adapter .gguf file (must be GGUF for Ollama)
            base_model:   Base Ollama model name (e.g. "llama3.2")
            ollama_name:  Name for the new Ollama model
            system_prompt: Override system prompt (optional)
            temperature:  Override temperature (optional)
            num_ctx:      Override context window (optional)

        The generated Modelfile will look like:
            FROM llama3.2
            ADAPTER /path/to/adapter.gguf
            SYSTEM ...
            PARAMETER temperature 0.2
        """
        adapter = Path(adapter).resolve()
        if not adapter.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter}")
        if adapter.suffix != ".gguf":
            raise ValueError(
                f"Ollama requires a GGUF adapter, got: {adapter.suffix}. "
                "Convert with: ollama_api.py or llama.cpp's llama-gguf-split."
            )

        b = ModelfileBuilder.from_base(base_model)
        b.add_adapter(str(adapter))
        if system_prompt:
            b.set_system(system_prompt)
        if temperature is not None:
            b.set_parameter("temperature", temperature)
        if num_ctx is not None:
            b.set_parameter("num_ctx", num_ctx)

        b.create_model(ollama_name)

        # Update registry
        df = self._load()
        mask = df["path"] == str(adapter)
        if mask.any():
            df.loc[mask, "ollama_name"] = ollama_name
            self._save(df)
            print(f"Registry updated: ollama_name={ollama_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA adapter manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List registered adapters")

    p_reg = sub.add_parser("register", help="Register an adapter file")
    p_reg.add_argument("path")
    p_reg.add_argument("--base",   default="", help="Compatible base model name")
    p_reg.add_argument("--task",   default="", help="Task type: code/chat/math/etc.")
    p_reg.add_argument("--source", default="manual")
    p_reg.add_argument("--notes",  default="")

    p_dep = sub.add_parser("deploy", help="Create Ollama model with adapter")
    p_dep.add_argument("adapter",    help="Path to adapter .gguf")
    p_dep.add_argument("--base",     required=True, help="Base Ollama model name")
    p_dep.add_argument("--name",     required=True, dest="ollama_name")
    p_dep.add_argument("--system",   default="")
    p_dep.add_argument("--temp",     type=float, default=None)
    p_dep.add_argument("--ctx",      type=int,   default=None)

    args = parser.parse_args()
    mgr = LoRAManager()

    if args.cmd == "list":
        mgr.list()
    elif args.cmd == "register":
        mgr.register(args.path, base_model=args.base, task=args.task,
                     source=args.source, notes=args.notes)
    elif args.cmd == "deploy":
        mgr.deploy_with_adapter(
            adapter=args.adapter,
            base_model=args.base,
            ollama_name=args.ollama_name,
            system_prompt=args.system,
            temperature=args.temp,
            num_ctx=args.ctx,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
