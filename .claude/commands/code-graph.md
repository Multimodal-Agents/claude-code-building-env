# /code-graph — Terminal Code Graph Monitor

Start the live terminal-themed code graph server for the current project.

## Steps

1. **Install dependencies** (if needed):
   ```bash
   pip install -r M:/claude_code_building_env/scripts_and_skills/code_graph/requirements.txt
   ```

2. **Detect the project path** — use the CWD of the current Claude Code session
   (`$ARGUMENTS` can override, e.g. `/code-graph /path/to/project`).

3. **Start the server in the background**:
   ```bash
   python -m scripts_and_skills.code_graph.server --path <PROJECT_PATH> --port 8765
   ```
   The server monitors `<PROJECT_PATH>`, serves the graph UI on port 8765,
   and listens for hook notifications from this Claude Code session.

4. **Open the browser** (Windows):
   ```bash
   start http://localhost:8765
   ```
   Or on WSL/Linux:
   ```bash
   xdg-open http://localhost:8765
   ```

5. **Hook setup** (live Claude edit tracking) — the global `.claude/settings.json`
   already contains a `PostToolUse` hook that silently notifies the graph server
   whenever Claude writes or edits a file. No extra setup needed.

## Notes

- The UI is a CRT green-on-black terminal graph. Nodes glow **amber** when Claude
  edits them, **cyan** when new files appear.
- Click **"Load Semantics"** in the sidebar to compute Ollama `nomic-embed-text`
  similarity edges (requires `ollama pull nomic-embed-text`).
- Multiple projects are supported: run `/code-graph` from different terminal sessions
  and use the project switcher dropdown in the top-left of the UI.
- Server stays running after you close this session. To stop it, kill the Python process.
