"""
code_parser.py — Parse Python / JS / TS files into Cytoscape graph elements.

Walks a project directory, extracts import relationships, and returns
Cytoscape-compatible nodes and edges.

Supported: .py  .js  .ts  .jsx  .tsx  .mjs
"""

import ast
import os
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", ".next", ".nuxt", ".output", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

SUPPORTED_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs"}

LANG_MAP = {
    ".py": "python",
    ".js": "js",
    ".mjs": "js",
    ".ts": "ts",
    ".jsx": "jsx",
    ".tsx": "tsx",
}

MAX_FILES = int(os.getenv("CGRAPH_MAX_FILES", "500"))

# JS/TS import patterns
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s.*?from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
    re.MULTILINE,
)


def _node_id(rel: str) -> str:
    """Normalise path → stable node ID."""
    return rel.replace("\\", "/")


def _count_lines(text: str) -> int:
    return text.count("\n") + 1


def _collect_functions_py(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names[:20]


def _parse_py_imports(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Return list of edge dicts from a Python file."""
    edges: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return edges

    src_id = _node_id(str(file_path.relative_to(root)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_py_abs(alias.name, root, all_ids)
                if target:
                    edges.append(_make_edge(src_id, target, "import"))

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.level and node.level > 0:
                # relative import
                target = _resolve_py_rel(
                    node.module, node.level, file_path, root, all_ids
                )
            else:
                target = _resolve_py_abs(node.module, root, all_ids)
            if target:
                edges.append(_make_edge(src_id, target, "import"))

    return edges


def _resolve_py_abs(module: str, root: Path, all_ids: set[str]) -> str | None:
    parts = module.split(".")
    # Try .py file
    candidate = Path(*parts).with_suffix(".py")
    cid = _node_id(str(candidate))
    if cid in all_ids:
        return cid
    # Try package __init__.py
    candidate2 = Path(*parts) / "__init__.py"
    cid2 = _node_id(str(candidate2))
    if cid2 in all_ids:
        return cid2
    return None


def _resolve_py_rel(
    module: str,
    level: int,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> str | None:
    base = file_path.parent
    for _ in range(level - 1):
        base = base.parent
    parts = module.split(".") if module else []
    candidate = base.joinpath(*parts).with_suffix(".py")
    try:
        rel = candidate.relative_to(root)
        cid = _node_id(str(rel))
        if cid in all_ids:
            return cid
    except ValueError:
        pass
    return None


def _parse_js_imports(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))

    for m in _JS_IMPORT_RE.finditer(source):
        raw = m.group(1)
        if not raw.startswith("."):
            continue  # skip npm packages
        resolved = _resolve_js(raw, file_path, root, all_ids)
        if resolved:
            edges.append(_make_edge(src_id, resolved, "import"))

    return edges


def _resolve_js(
    raw: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> str | None:
    base = file_path.parent
    candidate = (base / raw).resolve()
    # Try exact
    for ext in ("", ".js", ".ts", ".jsx", ".tsx", ".mjs", "/index.js", "/index.ts"):
        p = Path(str(candidate) + ext)
        try:
            rel = p.relative_to(root.resolve())
            cid = _node_id(str(rel))
            if cid in all_ids:
                return cid
        except ValueError:
            pass
    return None


def _make_edge(src: str, tgt: str, kind: str) -> dict[str, Any]:
    return {
        "data": {
            "id": f"{src}→{tgt}",
            "source": src,
            "target": tgt,
            "type": kind,
            "label": kind,
        }
    }


def parse_directory(root_path: str | Path) -> dict[str, Any]:
    """
    Walk root_path, parse all supported files, return Cytoscape graph dict:
      { nodes: [...], edges: [...], stats: {...} }
    """
    root = Path(root_path).resolve()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    file_data: list[tuple[Path, str, str]] = []  # (path, source, ext)

    # --- First pass: collect files and build ID set ---
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            full = Path(dirpath) / fname
            ext = full.suffix.lower()
            if ext not in SUPPORTED_EXT:
                continue
            if len(all_ids) >= MAX_FILES:
                break
            try:
                rel = _node_id(str(full.relative_to(root)))
            except ValueError:
                continue
            all_ids.add(rel)
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source = ""
            file_data.append((full, source, ext))

    # --- Second pass: build nodes ---
    for full, source, ext in file_data:
        rel = _node_id(str(full.relative_to(root)))
        lang = LANG_MAP.get(ext, "other")
        lines = _count_lines(source)
        size = len(source.encode("utf-8"))
        # group = top-level directory
        parts = rel.split("/")
        group = parts[0] if len(parts) > 1 else ""
        label = parts[-1]

        functions: list[str] = []
        if ext == ".py":
            try:
                tree = ast.parse(source, filename=str(full))
                functions = _collect_functions_py(tree)
            except SyntaxError:
                pass

        node: dict[str, Any] = {
            "data": {
                "id": rel,
                "label": label,
                "path": str(full),
                "ext": ext,
                "lang": lang,
                "lines": lines,
                "size": size,
                "group": group,
                "classes": [],
                "functions": functions,
                "changedBy": None,
                "changeTime": None,
            }
        }
        nodes.append(node)

    # --- Third pass: build edges ---
    seen_edges: set[str] = set()
    for full, source, ext in file_data:
        if ext == ".py":
            new_edges = _parse_py_imports(source, full, root, all_ids)
        else:
            new_edges = _parse_js_imports(source, full, root, all_ids)

        for e in new_edges:
            eid = e["data"]["id"]
            if eid not in seen_edges and e["data"]["source"] != e["data"]["target"]:
                seen_edges.add(eid)
                edges.append(e)

    stats = {
        "files": len(nodes),
        "edges": len(edges),
        "root": str(root),
        "languages": _lang_counts(nodes),
    }

    logger.info(
        "Parsed %d files, %d edges from %s", len(nodes), len(edges), root
    )
    return {"nodes": nodes, "edges": edges, "stats": stats}


def _lang_counts(nodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in nodes:
        lang = n["data"].get("lang", "other")
        counts[lang] = counts.get(lang, 0) + 1
    return counts
