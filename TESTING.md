# Testing Checklist

> Run from `M:\claude_code_building_env` with venv active: `.\venv\Scripts\Activate.ps1`
>
> **Status key:** ✅ passing &nbsp; ❌ fixed (was broken) &nbsp; ⏳ not yet run

---

## 1. Environment

- [x] `python --version` → 3.11.9 ✅
- [x] `pip show pandas pyarrow requests` → all resolve ✅
- [x] `ollama list` → shows `gpt-oss:20b`, `corecoder:latest`, `nomic-embed-text:latest` ✅
- [x] `ollama list | findstr corecoder` → `corecoder:latest  13.79 GB` ✅

---

## 2. OllamaAPI module

```powershell
python -m scripts_and_skills.model_manager.ollama_api version
python -m scripts_and_skills.model_manager.ollama_api list
```

- [x] `version` returns `0.17.0` ✅
- [x] `list` shows all models cleanly (no RuntimeWarning) ✅

---

## 3. PromptStore

```powershell
python -m scripts_and_skills.data.prompt_store list
python -m scripts_and_skills.data.prompt_store stats corecoder-vscode-copilot
```

- [x] `list` shows `corecoder-vscode-copilot` (19 rows) ✅
- [x] `stats` shows `total_rows=19` ✅

---

## 4. Seed script (idempotency check)

```powershell
python -m scripts_and_skills.data.seeds.corecoder_prompts
```

- [x] Output shows `rows=19` (confirms upsert, not append) ❌→✅ *was appending (rows=38); fixed `add_prompt` → `upsert_prompt`*

---

## 5. Embeddings

```powershell
ollama pull nomic-embed-text
python -m scripts_and_skills.data.embeddings embed corecoder-vscode-copilot
python -m scripts_and_skills.data.embeddings search corecoder-vscode-copilot "terminal investigation" --top 3
```

- [x] `ollama pull nomic-embed-text` ✅
- [x] `embed` reports `Added 19 embeddings` ❌→✅ *two bugs fixed: wrong PromptStore path (`parent.parent` → `parent`); Ollama 0.17 embed endpoint `/api/embed` with `input` field*
- [x] `search` returns 3 rows with similarity scores (~0.59–0.61) ✅

---

## 6. WebSearch

```powershell
python -m scripts_and_skills.data.web_search "Claude Code local model" --top 3
```

- [x] Returns 3 results with title/url/snippet ❌→✅ *`duckduckgo_search` renamed to `ddgs`; fixed import + `pip install ddgs`*

---

## 7. ModelfileBuilder

```powershell
python -m scripts_and_skills.model_manager.modelfile show corecoder:latest
python -m scripts_and_skills.model_manager.modelfile list
```

- [x] `show` prints the CoreCoder system prompt baked in ✅
- [x] `list` shows `corecoder` ✅

---

## 8. corecoder:latest model

```powershell
ollama run corecoder:latest "Who are you and what can you do?"
ollama run corecoder:latest "List the current directory contents"
```

- [x] Responds as CoreCoder with capabilities table ✅
- [x] Tool call test ✅ *(expected: `container.exec()` not found — no tool runtime in standalone `ollama run`)*

---

## 9. HF Download (dry run — no actual download)

```powershell
python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF --list
```

- [x] Shows 16 GGUF files with labels (Q4_K_M recommended, Q8_0 near lossless, etc.) ✅

---

## 10. Slash commands (inside a `claude` session)

Launch: `claude --model corecoder:latest`

- [ ] `/prompt-list` — shows corecoder-vscode-copilot dataset, offers search/export ⏳
- [ ] `/set-system` — lets you pick a prompt and apply it to an Ollama model ⏳
- [ ] `/parse-codebase` — reads tree once, builds mental map, asks targeted questions ⏳
- [ ] `/web-search Claude Code 2026` — returns DuckDuckGo results ⏳
- [ ] `/generate-dataset ./scripts_and_skills/data` — creates ShareGPT conversations from code files ⏳
- [ ] `/model-manager` — shows model list, GGUF registry, Modelfile options ⏳

---

## Known gaps (blockers noted)

| Item | Blocker | Done |
|------|---------|------|
| GGUF conversion | Run `.\setup_llamacpp.ps1` first | [ ] |
| LoRA deployment | Needs an actual adapter `.gguf` file | [ ] |
| DatasetGenerator | Calls Ollama API — needs `gpt-oss:20b` running | [ ] |
| Slash commands | Test inside a live `claude` session | [ ] |
