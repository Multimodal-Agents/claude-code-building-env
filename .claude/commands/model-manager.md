# /model-manager — Manage Ollama models, Modelfiles, GGUFs, and LoRA adapters

You have a full model management toolkit in `scripts_and_skills/model_manager/`.

---

## List local Ollama models

```
python -m scripts_and_skills.model_manager.ollama_api list
```

## Show a model's current Modelfile

```
python -m scripts_and_skills.model_manager.ollama_api show <model-name>
```

## Create / update a model with a new system prompt

```
python -m scripts_and_skills.model_manager.modelfile create <new-model-name> \
    --from <base-model-name> \
    --system "You are CoreCoder, a highly skilled software engineer..."
```

Or do it programmatically:
```python
from scripts_and_skills.model_manager import ModelfileBuilder

b = ModelfileBuilder.from_existing("gpt-oss:20b")
b.set_system("You are CoreCoder, a precision-driven software engineer with full terminal access.")
b.set_parameter("temperature", 0.2)
b.set_parameter("num_ctx", 8192)
b.create_model("corecoder:latest")
```

## Deploy a local GGUF to Ollama

```
python -m scripts_and_skills.model_manager.gguf_manager deploy /path/to/model.gguf \
    --name mymodel:latest \
    --system "You are a helpful assistant." \
    --ctx 4096
```

## Convert HuggingFace safetensors → GGUF

```
python -m scripts_and_skills.model_manager.gguf_manager convert \
    C:/models/hf/mistral-7b \
    C:/models/gguf/mistral-7b-f16.gguf \
    --script C:/llama.cpp/convert_hf_to_gguf.py \
    --outtype f16
```

## Quantize a GGUF

```
python -m scripts_and_skills.model_manager.gguf_manager quantize \
    C:/models/gguf/mistral-7b-f16.gguf \
    C:/models/gguf/mistral-7b-Q4_K_M.gguf \
    --type Q4_K_M
```
Common quant types: `Q4_K_M` (recommended), `Q5_K_M` (better quality), `Q8_0` (near lossless), `F16` (unquantized)

## Register and track GGUFs

```
python -m scripts_and_skills.model_manager.gguf_manager scan C:/models/gguf
python -m scripts_and_skills.model_manager.gguf_manager list
```

## Apply a LoRA adapter

```
python -m scripts_and_skills.model_manager.lora_manager deploy /path/to/adapter.gguf \
    --base llama3.2 \
    --name llama3-finetuned \
    --system "You are a Python expert."
```

## List registered LoRA adapters

```
python -m scripts_and_skills.model_manager.lora_manager list
```

---

## Instructions

1. Ask the user what they want: change system prompt, deploy a new GGUF, convert HF model, apply LoRA, etc.
2. For system prompt changes on an existing model: use `ModelfileBuilder.from_existing()` to preserve other settings.
3. For new local models: use `GGUFManager.deploy_to_ollama()` or the modelfile CLI.
4. Always confirm what model name will be created/replaced before running `create_model()`.
5. For HF → GGUF: ask for the HF model dir, llama.cpp path, and desired quant type.
6. The CoreCoder system prompt from the `corecoder-vscode-copilot` dataset is a good default for developer personas.
