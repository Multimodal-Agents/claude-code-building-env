"""
code_parser.py — Parse ANY file into Cytoscape graph elements.

Walks a project directory, extracts import/link/source relationships, and
returns Cytoscape-compatible nodes and edges.

Code (AST / regex imports):  .py  .js  .ts  .jsx  .tsx  .mjs  .rs  .go  .rb  .java  .cs  .lua
Markdown links:               .md  .mdx  .markdown
Shell sources:                .sh  .bash  .zsh
PowerShell dot-sources:       .ps1  .psm1  .psd1
Data / config (nodes only):   .json  .yaml  .yml  .toml  .env  .cfg  .ini  .sql  .html  .css  .scss
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
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".eggs", "htmlcov", "local_data", ".claude", ".idea",
    ".vscode", "__snapshots__", ".cargo",
}

SUPPORTED_EXT = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".rs", ".go", ".rb", ".java", ".cs", ".lua", ".r",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    # Docs
    ".md", ".mdx", ".markdown",
    # Shell
    ".sh", ".bash", ".zsh",
    # PowerShell
    ".ps1", ".psm1", ".psd1",
    # Data / config
    ".json", ".yaml", ".yml", ".toml",
    ".env", ".cfg", ".ini", ".conf",
    # Web / style
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    # Other
    ".sql", ".jl",
}

LANG_MAP = {
    ".py": "python",
    ".js": "js", ".mjs": "js",
    ".ts": "ts",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".md": "markdown", ".mdx": "markdown", ".markdown": "markdown",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".ps1": "powershell", ".psm1": "powershell", ".psd1": "powershell",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".lua": "lua",
    ".sql": "sql",
    ".r": "r",
    ".jl": "julia",
    ".env": "config", ".cfg": "config", ".ini": "config", ".conf": "config",
}

MAX_FILES = int(os.getenv("CGRAPH_MAX_FILES", "2000"))

# ── JS/TS import patterns ────────────────────────────────────────────────────
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s.*?from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
    re.MULTILINE,
)

# ── Markdown local-link patterns ─────────────────────────────────────────────
# Matches [text](path) and ![alt](path) — local relative links only
_MD_LINK_RE = re.compile(
    r"""!?\[[^\]]*\]\(([^)#\s]+)\)""",
    re.MULTILINE,
)

# ── Shell source patterns ─────────────────────────────────────────────────────
# source ./foo.sh  or  . ./foo.sh  or  bash ./foo.sh  or  sh ./foo.sh
_SH_SOURCE_RE = re.compile(
    r"""(?:^|(?<=\n))[ \t]*(?:source|\.|bash|sh)[ \t]+[\"']?([^\"'\s;#&|]+\.(?:sh|bash|zsh|ps1))""",
    re.MULTILINE,
)

# ── PowerShell dot-source / call-operator patterns ───────────────────────────
# . .\foo.ps1  or  & .\foo.ps1  or  . $PSScriptRoot\foo.ps1
_PS1_DOTSOURCE_RE = re.compile(
    r"""(?:^|(?<=\n))[ \t]*[.&][ \t]+(?:\$PSScriptRoot[/\\\\])?['"]?([^'"\s;]+\.ps[md]?1)""",
    re.MULTILINE,
)

# ── HTML src/href patterns ────────────────────────────────────────────────────
_HTML_REF_RE = re.compile(
    r"""(?:src|href)=['"]([^'"#?]+)['"]""",
    re.MULTILINE,
)

# ── CSS @import / url() patterns ─────────────────────────────────────────────
_CSS_IMPORT_RE = re.compile(
    r"""(?:@import\s+['"]([^'"]+)['"]|url\(['"]?([^'")?]+)['"]?\))""",
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


def _resolve_generic(
    raw: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> str | None:
    """Generic relative reference resolver for markdown / shell / html / css."""
    # Skip URLs and empty strings
    raw = raw.strip().strip('"\'')
    if not raw or raw.startswith(("http", "//", "#", "data:", "mailto:")):
        return None
    # Strip PowerShell variable prefixes like $PSScriptRoot\
    raw = re.sub(r"^\$[A-Za-z_]+[/\\]", "", raw)

    base = file_path.parent
    candidate = (base / raw).resolve()
    probes = ["", ".py", ".js", ".ts", ".md", ".sh", ".ps1",
               "/index.js", "/index.ts", "/README.md"]
    for ext in probes:
        p = Path(str(candidate) + ext)
        try:
            rel = p.relative_to(root.resolve())
            cid = _node_id(str(rel))
            if cid in all_ids:
                return cid
        except ValueError:
            pass
    return None


def _parse_md_links(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse markdown local file links → edges."""
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))
    for m in _MD_LINK_RE.finditer(source):
        raw = m.group(1)
        # Skip absolute URLs and anchor-only links
        if raw.startswith(("http", "//", "#", "data:")):
            continue
        resolved = _resolve_generic(raw, file_path, root, all_ids)
        if resolved and resolved != src_id:
            edges.append(_make_edge(src_id, resolved, "link"))
    return edges


def _parse_sh_sources(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse shell source/. references → edges."""
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))
    for m in _SH_SOURCE_RE.finditer(source):
        raw = m.group(1)
        resolved = _resolve_generic(raw, file_path, root, all_ids)
        if resolved and resolved != src_id:
            edges.append(_make_edge(src_id, resolved, "source"))
    return edges


def _parse_ps1_sources(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse PowerShell dot-source / call-operator references → edges."""
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))
    for m in _PS1_DOTSOURCE_RE.finditer(source):
        raw = m.group(1)
        resolved = _resolve_generic(raw, file_path, root, all_ids)
        if resolved and resolved != src_id:
            edges.append(_make_edge(src_id, resolved, "source"))
    return edges


def _parse_html_refs(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse HTML src= / href= local references → edges."""
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))
    for m in _HTML_REF_RE.finditer(source):
        raw = m.group(1)
        if raw.startswith(("http", "//", "#", "data:", "mailto:")):
            continue
        resolved = _resolve_generic(raw, file_path, root, all_ids)
        if resolved and resolved != src_id:
            edges.append(_make_edge(src_id, resolved, "ref"))
    return edges


def _parse_css_imports(
    source: str,
    file_path: Path,
    root: Path,
    all_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse CSS @import / url() local references → edges."""
    edges: list[dict[str, Any]] = []
    src_id = _node_id(str(file_path.relative_to(root)))
    for m in _CSS_IMPORT_RE.finditer(source):
        raw = m.group(1) or m.group(2)
        if not raw:
            continue
        if raw.startswith(("http", "//", "data:")):
            continue
        resolved = _resolve_generic(raw, file_path, root, all_ids)
        if resolved and resolved != src_id:
            edges.append(_make_edge(src_id, resolved, "import"))
    return edges


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
    _JS_EXTS  = {".js", ".ts", ".jsx", ".tsx", ".mjs"}
    _MD_EXTS  = {".md", ".mdx", ".markdown"}
    _SH_EXTS  = {".sh", ".bash", ".zsh"}
    _PS1_EXTS = {".ps1", ".psm1", ".psd1"}
    _HTML_EXTS = {".html", ".htm"}
    _CSS_EXTS  = {".css", ".scss", ".sass", ".less"}

    seen_edges: set[str] = set()
    for full, source, ext in file_data:
        if ext == ".py":
            new_edges = _parse_py_imports(source, full, root, all_ids)
        elif ext in _JS_EXTS:
            new_edges = _parse_js_imports(source, full, root, all_ids)
        elif ext in _MD_EXTS:
            new_edges = _parse_md_links(source, full, root, all_ids)
        elif ext in _SH_EXTS:
            new_edges = _parse_sh_sources(source, full, root, all_ids)
        elif ext in _PS1_EXTS:
            new_edges = _parse_ps1_sources(source, full, root, all_ids)
        elif ext in _HTML_EXTS:
            new_edges = _parse_html_refs(source, full, root, all_ids)
        elif ext in _CSS_EXTS:
            new_edges = _parse_css_imports(source, full, root, all_ids)
        else:
            new_edges = []

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
