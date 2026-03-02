# /code-graph

Run the launcher script. No analysis, no extra steps, no output beyond the URL.

**On Windows / PowerShell:**
```powershell
powershell -ExecutionPolicy Bypass -File "M:\claude_code_building_env\scripts_and_skills\code_graph\start.ps1" -Path "$PWD"
```

**On WSL / bash:**
```bash
bash /m/claude_code_building_env/scripts_and_skills/code_graph/start.sh "$PWD"
```

If `$ARGUMENTS` is provided, pass it as the project path instead of `$PWD`.

The script handles everything (deps, server start, browser open). Report only the URL it printed.
