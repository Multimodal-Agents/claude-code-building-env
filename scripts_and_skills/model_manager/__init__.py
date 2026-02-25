"""
model_manager — Ollama model lifecycle management

Modules:
    ollama_api    — REST wrapper around Ollama's HTTP API
    modelfile     — Programmatic Modelfile builder / parser
    gguf_manager  — GGUF file registry + HF → GGUF conversion
    lora_manager  — LoRA adapter tracking and application
    hf_download   — Download GGUF files from HuggingFace Hub
"""

from .ollama_api import OllamaAPI
from .modelfile import ModelfileBuilder
from .gguf_manager import GGUFManager
from .lora_manager import LoRAManager
from .hf_download import download_file as hf_download, list_repo as hf_list_repo

__all__ = ["OllamaAPI", "ModelfileBuilder", "GGUFManager", "LoRAManager", "hf_download", "hf_list_repo"]
