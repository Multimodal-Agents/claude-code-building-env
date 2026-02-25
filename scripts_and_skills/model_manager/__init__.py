"""
model_manager — Ollama model lifecycle management

Modules:
    ollama_api    — REST wrapper around Ollama's HTTP API
    modelfile     — Programmatic Modelfile builder / parser
    gguf_manager  — GGUF file registry + HF → GGUF conversion
    lora_manager  — LoRA adapter tracking and application
    hf_download   — Download GGUF files from HuggingFace Hub

Lazy imports — import directly from submodules to avoid sys.modules conflicts
when running with `python -m`:
    from scripts_and_skills.model_manager.ollama_api import OllamaAPI
"""

__all__ = ["OllamaAPI", "ModelfileBuilder", "GGUFManager", "LoRAManager"]
