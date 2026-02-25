"""
ollama_api.py — Thin REST wrapper around the Ollama HTTP API

Covers all model-management endpoints:
  - create / delete / copy / pull / push
  - list local models + running models
  - show modelfile / model info
  - generate + chat (single-shot helpers)
  - generate embeddings

Reference: local_data/../basic_reference_documentation_library/ollama/api.md

Usage:
    python -m scripts_and_skills.model_manager.ollama_api list
    python -m scripts_and_skills.model_manager.ollama_api show llama3.2
    python -m scripts_and_skills.model_manager.ollama_api delete mymodel
"""

from __future__ import annotations
import json
import os
import sys
import textwrap
from typing import Any

import requests

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))


class OllamaAPI:
    """Lightweight wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = OLLAMA_BASE, timeout: int = DEFAULT_TIMEOUT):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> Any:
        r = requests.get(f"{self.base}{path}", timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict, stream: bool = False) -> Any:
        r = requests.post(
            f"{self.base}{path}",
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} — Ollama said: {detail}",
                response=r,
            )
        if stream:
            return r  # caller iterates line by line
        return r.json()

    def _delete(self, path: str, payload: dict) -> Any:
        r = requests.delete(f"{self.base}{path}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.text else {}

    # ------------------------------------------------------------------
    # Model inventory
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return a list of locally available models."""
        data = self._get("/api/tags")
        return data.get("models", [])

    def list_running(self) -> list[dict]:
        """Return a list of models currently loaded in memory."""
        data = self._get("/api/ps")
        return data.get("models", [])

    def show(self, model: str, verbose: bool = False) -> dict:
        """Return detailed info + Modelfile for a model."""
        payload: dict = {"model": model}
        if verbose:
            payload["verbose"] = True
        return self._post("/api/show", payload)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        from_model: str = "",
        system: str = "",
        parameters: dict | None = None,
        template: str = "",
        messages: list | None = None,
        adapters: dict | None = None,
        quantize: str = "",
        stream: bool = True,
        # Legacy Modelfile string (Ollama <0.5) — kept for reference/saving only
        modelfile: str = "",
    ) -> dict | "requests.Response":
        """
        Create (or recreate) a model using the Ollama 0.17+ structured API.

        Ollama 0.17 removed the 'modelfile' string field and replaced it with
        direct structured parameters:
          - from:       base model name  (e.g. 'gpt-oss:20b')
          - system:     system prompt text
          - parameters: dict of model params  (temperature, num_ctx, stop, ...)
          - template:   prompt template string
          - adapters:   {filename: sha256} for LoRA adapters
          - quantize:   quantization type for on-the-fly quantization
        """
        payload: dict = {"model": name, "stream": stream}
        if from_model:
            payload["from"] = from_model
        if system:
            payload["system"] = system
        if parameters:
            payload["parameters"] = parameters
        if template:
            payload["template"] = template
        if messages:
            payload["messages"] = messages
        if adapters:
            payload["adapters"] = adapters
        if quantize:
            payload["quantize"] = quantize
        return self._post("/api/create", payload, stream=stream)

    def copy(self, source: str, destination: str) -> dict:
        """Copy a model to a new name."""
        return self._post("/api/copy", {"source": source, "destination": destination})

    def delete(self, model: str) -> dict:
        """Delete a local model."""
        return self._delete("/api/delete", {"model": model})

    def pull(self, model: str, stream: bool = True) -> dict | requests.Response:
        """Pull a model from the Ollama registry."""
        return self._post("/api/pull", {"model": model, "stream": stream}, stream=stream)

    # ------------------------------------------------------------------
    # Inference helpers (single-shot, non-streaming)
    # ------------------------------------------------------------------

    def generate(self, model: str, prompt: str, system: str = "", **options) -> str:
        """Simple one-shot text generation. Returns the response string."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **options,
        }
        if system:
            payload["system"] = system
        data = self._post("/api/generate", payload)
        return data.get("response", "")

    def chat(self, model: str, messages: list[dict], **options) -> str:
        """Chat completion (non-streaming). Returns assistant message content."""
        payload = {"model": model, "messages": messages, "stream": False, **options}
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, model: str, text: str) -> list[float]:
        """Generate an embedding vector for text using *model*."""
        data = self._post("/api/embeddings", {"model": model, "prompt": text})
        return data.get("embedding", [])

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------

    def version(self) -> str:
        return self._get("/api/version").get("version", "unknown")

    # ------------------------------------------------------------------
    # Convenience: print table
    # ------------------------------------------------------------------

    def print_models(self):
        models = self.list_models()
        if not models:
            print("No local models found.")
            return
        print(f"{'NAME':<40} {'SIZE (GB)':>10}  {'MODIFIED'}")
        print("-" * 65)
        for m in models:
            size_gb = m.get("size", 0) / 1e9
            print(f"{m['name']:<40} {size_gb:>9.2f}  {m.get('modified_at', '')[:10]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Ollama API wrapper CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list",    help="List local models")
    sub.add_parser("running", help="List running models")

    p_show = sub.add_parser("show", help="Show model info / Modelfile")
    p_show.add_argument("model")
    p_show.add_argument("--verbose", action="store_true")

    p_del = sub.add_parser("delete", help="Delete a model")
    p_del.add_argument("model")

    p_copy = sub.add_parser("copy", help="Copy a model")
    p_copy.add_argument("source")
    p_copy.add_argument("destination")

    p_pull = sub.add_parser("pull", help="Pull a model")
    p_pull.add_argument("model")

    p_gen = sub.add_parser("generate", help="One-shot generation")
    p_gen.add_argument("model")
    p_gen.add_argument("prompt")

    sub.add_parser("version", help="Ollama server version")

    args = parser.parse_args()
    api = OllamaAPI()

    if args.cmd == "list":
        api.print_models()
    elif args.cmd == "running":
        for m in api.list_running():
            print(m.get("name"))
    elif args.cmd == "show":
        info = api.show(args.model, verbose=args.verbose)
        print(info.get("modelfile", json.dumps(info, indent=2)))
    elif args.cmd == "delete":
        api.delete(args.model)
        print(f"Deleted {args.model}")
    elif args.cmd == "copy":
        api.copy(args.source, args.destination)
        print(f"Copied {args.source} → {args.destination}")
    elif args.cmd == "pull":
        for line in api.pull(args.model, stream=True).iter_lines():
            if line:
                obj = json.loads(line)
                print(obj.get("status", ""), obj.get("completed", ""), "/", obj.get("total", ""))
    elif args.cmd == "generate":
        print(api.generate(args.model, args.prompt))
    elif args.cmd == "version":
        print(api.version())
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
