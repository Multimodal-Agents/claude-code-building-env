"""
modelfile.py — Programmatic Modelfile builder and manager

Modelfiles define how Ollama runs a model: base model, system prompt,
parameters, prompt template, and LoRA adapters.

Reference: basic_reference_documentation_library/ollama/modelfile.mdx

Usage:
    # Build and deploy a custom model with a new system prompt
    builder = ModelfileBuilder.from_existing("llama3.2")
    builder.set_system("You are a Python expert who only speaks in code.")
    builder.set_parameter("temperature", 0.2)
    builder.set_parameter("num_ctx", 8192)
    builder.create_model("python-expert")

    # Build from a local GGUF
    builder = ModelfileBuilder.from_gguf("/models/mymodel.gguf")
    builder.set_system("You are CoreCoder...")
    builder.create_model("corecoder")

    # CLI
    python -m scripts_and_skills.model_manager.modelfile list
    python -m scripts_and_skills.model_manager.modelfile show llama3.2
    python -m scripts_and_skills.model_manager.modelfile create mymodel --from llama3.2 --system "..."
    python -m scripts_and_skills.model_manager.modelfile save mymodel modelfile.txt
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional

from .ollama_api import OllamaAPI

# Default manageable Modelfiles are stored here
MODELFILE_DIR = Path(os.environ.get("MODELFILE_DIR", "M:/claude_code_building_env/local_data/modelfiles"))

# Valid parameter names per Ollama docs
VALID_PARAMS = {
    "num_ctx", "repeat_last_n", "repeat_penalty", "temperature",
    "seed", "stop", "num_predict", "top_k", "top_p", "min_p",
}


class ModelfileBuilder:
    """
    Builds Ollama Modelfile content programmatically.

    A Modelfile is a plain-text blueprint:
        FROM <base>
        SYSTEM <prompt>
        PARAMETER <key> <value>
        TEMPLATE <template>
        ADAPTER <path>
        MESSAGE <role> <content>
    """

    def __init__(self):
        self._from: str = ""
        self._system: str = ""
        self._template: str = ""
        self._parameters: dict[str, list] = {}  # key → [val, val, ...] (stop can repeat)
        self._adapters: list[str] = []
        self._messages: list[tuple[str, str]] = []  # [(role, content), ...]
        self._license: str = ""
        self._api = OllamaAPI()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_base(cls, base_model: str) -> "ModelfileBuilder":
        """Start from a model name (local or pullable)."""
        b = cls()
        b._from = base_model
        return b

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "ModelfileBuilder":
        """Start from a local GGUF file path."""
        path = Path(gguf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GGUF not found: {path}")
        b = cls()
        b._from = str(path)
        return b

    @classmethod
    def from_existing(cls, model_name: str) -> "ModelfileBuilder":
        """
        Pull the existing Modelfile from Ollama and load it so you
        can modify it in place (e.g. swap the system prompt).
        """
        api = OllamaAPI()
        info = api.show(model_name)
        raw = info.get("modelfile", "")
        return cls.parse(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelfileBuilder":
        """Load a saved Modelfile from disk."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.parse(text)

    @classmethod
    def parse(cls, text: str) -> "ModelfileBuilder":
        """Parse raw Modelfile text into a builder instance."""
        b = cls()
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip comments and blanks
            if not line or line.startswith("#"):
                i += 1
                continue

            upper = line.upper()

            if upper.startswith("FROM "):
                b._from = line[5:].strip()
            elif upper.startswith("SYSTEM "):
                value = line[7:].strip()
                if value.startswith('"""'):
                    # Multi-line heredoc
                    value = value[3:]
                    parts = [value]
                    while not parts[-1].endswith('"""'):
                        i += 1
                        parts.append(lines[i])
                    full = "\n".join(parts)
                    b._system = full[: full.rfind('"""')].strip()
                else:
                    b._system = value
            elif upper.startswith("PARAMETER "):
                rest = line[10:].strip()
                key, _, val = rest.partition(" ")
                b._parameters.setdefault(key.lower(), []).append(val)
            elif upper.startswith("ADAPTER "):
                b._adapters.append(line[8:].strip())
            elif upper.startswith("TEMPLATE "):
                value = line[9:].strip()
                if value.startswith('"""'):
                    value = value[3:]
                    parts = [value]
                    while not parts[-1].endswith('"""'):
                        i += 1
                        parts.append(lines[i])
                    full = "\n".join(parts)
                    b._template = full[: full.rfind('"""')]
                else:
                    b._template = value
            elif upper.startswith("MESSAGE "):
                rest = line[8:].strip()
                role, _, content = rest.partition(" ")
                b._messages.append((role, content))
            elif upper.startswith("LICENSE "):
                b._license = line[8:].strip()

            i += 1
        return b

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_base(self, base: str) -> "ModelfileBuilder":
        self._from = base
        return self

    def set_system(self, system_prompt: str) -> "ModelfileBuilder":
        self._system = system_prompt
        return self

    def clear_system(self) -> "ModelfileBuilder":
        """
        Remove any embedded system prompt, falling back to the base model's
        own default (or no system prompt if the base has none).
        """
        self._system = ""
        return self

    def set_parameter(self, key: str, value) -> "ModelfileBuilder":
        key = key.lower()
        if key not in VALID_PARAMS:
            raise ValueError(f"Unknown parameter '{key}'. Valid: {sorted(VALID_PARAMS)}")
        self._parameters[key] = [str(value)]
        return self

    def add_stop(self, stop_token: str) -> "ModelfileBuilder":
        """Add a stop token (multiple stop tokens are allowed)."""
        self._parameters.setdefault("stop", []).append(stop_token)
        return self

    def set_template(self, template: str) -> "ModelfileBuilder":
        self._template = template
        return self

    def add_adapter(self, path: str) -> "ModelfileBuilder":
        self._adapters.append(path)
        return self

    def add_message(self, role: str, content: str) -> "ModelfileBuilder":
        assert role in ("system", "user", "assistant"), f"Invalid role: {role}"
        self._messages.append((role, content))
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Render the Modelfile as a string."""
        if not self._from:
            raise ValueError("FROM is required — call set_base() or use a factory")

        lines = [f"FROM {self._from}"]

        if self._system:
            # Wrap in triple-quotes if multi-line
            if "\n" in self._system:
                lines.append(f'SYSTEM """\n{self._system}\n"""')
            else:
                lines.append(f"SYSTEM {self._system}")

        if self._template:
            lines.append(f'TEMPLATE """\n{self._template}\n"""')

        for key, values in self._parameters.items():
            for v in values:
                lines.append(f"PARAMETER {key} {v}")

        for adapter in self._adapters:
            lines.append(f"ADAPTER {adapter}")

        for role, content in self._messages:
            lines.append(f"MESSAGE {role} {content}")

        if self._license:
            lines.append(f"LICENSE {self._license}")

        return "\n".join(lines) + "\n"

    def save(self, path: str | Path | None = None, name: str | None = None) -> Path:
        """
        Save Modelfile to disk.
        If path is omitted, saves to MODELFILE_DIR/<name>.modelfile
        """
        if path is None:
            if name is None:
                raise ValueError("Provide either path or name")
            MODELFILE_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELFILE_DIR / f"{name}.modelfile"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(), encoding="utf-8")
        print(f"Saved Modelfile → {path}")
        return path

    def create_model(self, model_name: str, stream: bool = True) -> None:
        """
        Push this Modelfile to Ollama and create the model.
        Streams progress by default.
        """
        content = self.build()
        print(f"Creating model '{model_name}'...")
        response = self._api.create(model_name, content, stream=stream)
        if stream:
            for line in response.iter_lines():
                if line:
                    obj = json.loads(line)
                    status = obj.get("status", "")
                    print(f"  {status}")
        else:
            print(response)
        print(f"Model '{model_name}' created.")


# ---------------------------------------------------------------------------
# Managed Modelfile store — list/load saved modelfiles
# ---------------------------------------------------------------------------

def list_saved() -> list[Path]:
    if not MODELFILE_DIR.exists():
        return []
    return sorted(MODELFILE_DIR.glob("*.modelfile"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Modelfile builder CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List saved local Modelfiles")

    p_show = sub.add_parser("show", help="Show Modelfile for an existing Ollama model")
    p_show.add_argument("model")

    p_create = sub.add_parser("create", help="Create an Ollama model from flags")
    p_create.add_argument("model_name")
    p_create.add_argument("--from", dest="base", required=True, help="Base model or GGUF path")
    p_create.add_argument("--system", default="", help="System prompt text")
    p_create.add_argument("--reset-system", action="store_true",
                          help="Remove any embedded system prompt (restore base model default)")
    p_create.add_argument("--temperature", type=float, default=None)
    p_create.add_argument("--ctx", type=int, default=None)
    p_create.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    p_create.add_argument("--save-only", action="store_true", help="Save Modelfile but don't push to Ollama")

    p_save = sub.add_parser("save", help="Save existing model's Modelfile to disk")
    p_save.add_argument("model")
    p_save.add_argument("output", nargs="?", help="Output file path")

    args = parser.parse_args()

    if args.cmd == "list":
        files = list_saved()
        if not files:
            print(f"No saved Modelfiles in {MODELFILE_DIR}")
        for f in files:
            print(f"  {f.stem}")

    elif args.cmd == "show":
        info = OllamaAPI().show(args.model)
        print(info.get("modelfile", json.dumps(info, indent=2)))

    elif args.cmd == "create":
        gguf_exts = (".gguf", ".bin")
        if any(args.base.endswith(e) for e in gguf_exts):
            b = ModelfileBuilder.from_gguf(args.base)
        else:
            b = ModelfileBuilder.from_base(args.base)
        if args.reset_system:
            b.clear_system()
        elif args.system:
            b.set_system(args.system)
        if args.temperature is not None:
            b.set_parameter("temperature", args.temperature)
        if args.ctx is not None:
            b.set_parameter("num_ctx", args.ctx)
        if args.adapter:
            b.add_adapter(args.adapter)
        if args.save_only:
            b.save(name=args.model_name)
        else:
            b.create_model(args.model_name)

    elif args.cmd == "save":
        b = ModelfileBuilder.from_existing(args.model)
        out = args.output or None
        b.save(path=out, name=args.model if out is None else None)

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
