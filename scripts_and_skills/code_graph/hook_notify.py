"""
hook_notify.py — PostToolUse hook for Code Graph Monitor.

Reads Claude Code hook JSON from stdin, extracts the edited file path,
and POSTs a change notification to the graph server (port 8765).

Silently exits if the server is not running — safe to use globally.
"""

import json
import os
import sys
import urllib.request
import urllib.error


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except Exception:
        return

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path:
        # Write tool uses file_path too; NotebookEdit uses notebook_path
        file_path = tool_input.get("notebook_path", "")
    if not file_path:
        return

    payload = json.dumps({
        "file": file_path,
        "by": "claude",
        "cwd": os.getcwd(),
    }).encode("utf-8")

    port = os.getenv("CGRAPH_PORT", "8765")
    url = f"http://localhost:{port}/api/hook"

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1):
            pass
    except Exception:
        pass  # Server not running — that's fine


if __name__ == "__main__":
    main()
