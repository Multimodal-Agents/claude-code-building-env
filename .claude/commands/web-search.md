# /web-search — DuckDuckGo web search for local model sessions

Local models can't browse the web. This command bridges that gap by
running DuckDuckGo searches and returning results for you to reason over.

## Usage

```bash
python -m scripts_and_skills.data.web_search "<query>" --top 5
python -m scripts_and_skills.data.web_search "<query>" --top 10 --json
```

## Instructions

1. Extract the search query from the user's request.
2. Run the search command above.
3. Read and reason over the results — do NOT just paste them back.
4. Synthesize a useful answer, citing URLs where relevant.
5. If results are insufficient, reformulate and search again with different terms.

## Examples

User: "what are the latest Unsloth fine-tuning options?"
```bash
python -m scripts_and_skills.data.web_search "Unsloth fine-tuning 2025 options" --top 5
```

User: "find the nomic-embed-text model page"
```bash
python -m scripts_and_skills.data.web_search "nomic-embed-text ollama model" --top 3
```

## Notes

- Requires: `pip install duckduckgo-search`
- No API key needed — uses DuckDuckGo's free API
- Results are returned as plain text snippets, not full page content
- For full page content, follow up with a specific URL fetch using curl or requests
- Rate limit: avoid more than ~10 searches per minute
