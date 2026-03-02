"""
server.py — FastAPI server for the Terminal Code Graph Monitor.

Endpoints:
  GET  /          → index.html
  GET  /api/graph → current graph JSON (for active project)
  POST /api/hook  → hook notification {file, by, cwd}
  POST /api/refresh → full re-scan
  GET  /api/embeddings → semantic similarity edges via Ollama
  GET  /api/projects → list of registered project CWDs
  WS   /ws        → WebSocket broadcast channel

Usage:
  python -m scripts_and_skills.code_graph.server --path /path/to/project
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse

from scripts_and_skills.code_graph.code_parser import parse_directory

# Optional embeddings
try:
    from scripts_and_skills.data.embeddings import EmbeddingStore, cosine_similarity
    import numpy as np
    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False

# Optional watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("CGRAPH_PORT", "8765"))
DEBOUNCE_SEC = 0.5
SEMANTIC_THRESHOLD = float(os.getenv("CGRAPH_SIM_THRESHOLD", "0.75"))

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Graph state per project
# ---------------------------------------------------------------------------

class GraphState:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.nodes: dict[str, dict] = {}   # id → node dict
        self.edges: dict[str, dict] = {}   # id → edge dict
        self.stats: dict[str, Any] = {}
        self.last_parse: float = 0.0
        # Semantic drift tracking (requires HAS_EMBEDDINGS)
        self.baseline_vecs: dict[str, Any] = {}   # node_id → np.array baseline
        self.drift_scores: dict[str, float] = {}  # node_id → cosine distance from baseline

    def load(self, graph: dict) -> None:
        self.nodes = {n["data"]["id"]: n for n in graph.get("nodes", [])}
        self.edges = {e["data"]["id"]: e for e in graph.get("edges", [])}
        self.stats = graph.get("stats", {})
        self.last_parse = time.time()

    def to_graph(self) -> dict:
        return {
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values()),
            "stats": self.stats,
        }

    def update_node_change(self, file_path: str, by: str) -> str | None:
        """Mark a node as changed. Returns node id if found."""
        full = Path(file_path).resolve()
        try:
            rel = full.relative_to(self.root)
        except ValueError:
            return None
        node_id = str(rel).replace("\\", "/")
        if node_id in self.nodes:
            self.nodes[node_id]["data"]["changedBy"] = by
            self.nodes[node_id]["data"]["changeTime"] = time.time()
            return node_id
        return None


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WS client connected (total=%d)", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        self.active = [c for c in self.active if c is not ws]
        logger.info("WS client disconnected (total=%d)", len(self.active))

    async def broadcast(self, payload: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# Watchdog handler
# ---------------------------------------------------------------------------

if HAS_WATCHDOG:
    class GraphChangeHandler(FileSystemEventHandler):
        def __init__(self, cwd: str, loop: asyncio.AbstractEventLoop) -> None:
            super().__init__()
            self.cwd = cwd
            self.loop = loop
            self._pending: dict[str, float] = {}

        def on_modified(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            self._schedule(event.src_path)

        def on_created(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            self._schedule(event.src_path)

        def _schedule(self, path: str) -> None:
            ext = Path(path).suffix.lower()
            supported = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs"}
            if ext not in supported:
                return
            now = time.time()
            last = self._pending.get(path, 0)
            if now - last < DEBOUNCE_SEC:
                return
            self._pending[path] = now
            asyncio.run_coroutine_threadsafe(
                app.state.handle_file_change(path, "watcher", self.cwd),
                self.loop,
            )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Code Graph Monitor")

@app.on_event("startup")
async def _startup() -> None:
    app.state.manager = ConnectionManager()
    app.state.projects: dict[str, GraphState] = {}
    app.state.observers: dict[str, Any] = {}
    app.state.loop = asyncio.get_event_loop()

    # Pre-register startup path if provided via env (set by __main__)
    startup_path = os.getenv("_CGRAPH_STARTUP_PATH", "")
    if startup_path:
        await app.state.register_project(startup_path)


async def _embed_baseline(state: GraphState, manager: ConnectionManager) -> None:
    """Embed all project files as baseline vectors for drift detection."""
    if not HAS_EMBEDDINGS:
        return
    store = EmbeddingStore()
    count = 0
    for node_id, node in list(state.nodes.items()):
        fpath = node["data"].get("path", "")
        if not fpath:
            continue
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        try:
            vec = await asyncio.to_thread(store.embed_text, text)
            if vec is not None:
                state.baseline_vecs[node_id] = np.array(vec, dtype=np.float32)
                count += 1
        except Exception:
            continue
    if count > 0:
        logger.info("Baseline embeddings: %d files for %s", count, state.root)
        await manager.broadcast({
            "type": "baseline-ready",
            "count": count,
            "project": str(state.root),
        })


async def _compute_drift(state: GraphState, node_id: str, manager: ConnectionManager) -> None:
    """Re-embed a changed file and broadcast its cosine drift from baseline."""
    if not HAS_EMBEDDINGS or node_id not in state.baseline_vecs:
        return
    node = state.nodes.get(node_id)
    if not node:
        return
    fpath = node["data"].get("path", "")
    if not fpath:
        return
    try:
        text = Path(fpath).read_text(encoding="utf-8", errors="replace")[:2000]
        store = EmbeddingStore()
        vec = await asyncio.to_thread(store.embed_text, text)
        if vec is None:
            return
        new_vec = np.array(vec, dtype=np.float32)
        baseline = state.baseline_vecs[node_id]
        sim_val = float(cosine_similarity(baseline, new_vec))
        drift = round(1.0 - sim_val, 4)   # 0 = unchanged, 1 = completely different
        state.drift_scores[node_id] = drift
        state.baseline_vecs[node_id] = new_vec  # rolling baseline
        await manager.broadcast({
            "type": "drift-update",
            "nodeId": node_id,
            "drift": drift,
            "project": str(state.root),
        })
    except Exception as ex:
        logger.debug("Drift compute error for %s: %s", node_id, ex)


async def _register_project(path: str) -> GraphState:
    """Parse and register a project, starting its watchdog observer."""
    projects: dict[str, GraphState] = app.state.projects
    root = str(Path(path).resolve())

    if root in projects:
        return projects[root]

    logger.info("Registering project: %s", root)
    state = GraphState(root)
    graph = await asyncio.to_thread(parse_directory, root)
    state.load(graph)
    projects[root] = state

    if HAS_WATCHDOG:
        handler = GraphChangeHandler(root, app.state.loop)
        observer = Observer()
        observer.schedule(handler, root, recursive=True)
        observer.start()
        app.state.observers[root] = observer
        logger.info("Watchdog observer started for %s", root)

    # Background baseline embedding (silently skipped if Ollama unavailable)
    if HAS_EMBEDDINGS:
        asyncio.create_task(_embed_baseline(state, app.state.manager))

    return state


app.state.register_project = _register_project


async def _handle_file_change(file_path: str, by: str, cwd: str) -> None:
    """Process a file change: re-parse file, update state, broadcast."""
    root = str(Path(cwd).resolve())
    projects: dict[str, GraphState] = app.state.projects

    # Auto-register new project if unknown
    if root not in projects:
        await _register_project(root)

    state = projects.get(root)
    if not state:
        return

    # Re-parse the specific file and update node
    full = Path(file_path).resolve()
    try:
        rel = str(full.relative_to(state.root)).replace("\\", "/")
    except ValueError:
        rel = file_path

    # Update change metadata
    node_id = state.update_node_change(file_path, by)

    # Broadcast change event
    if node_id:
        await app.state.manager.broadcast({
            "type": "change",
            "nodeId": node_id,
            "by": by,
            "changeType": "edit",
            "project": root,
        })
        logger.info("Change broadcast: %s by %s", node_id, by)
        # Background drift computation (no-op if Ollama unavailable)
        if HAS_EMBEDDINGS:
            asyncio.create_task(_compute_drift(state, node_id, app.state.manager))
    else:
        # New file — do a full refresh of the sub-graph
        graph = await asyncio.to_thread(parse_directory, root)
        state.load(graph)
        await app.state.manager.broadcast({
            "type": "graph",
            "data": state.to_graph(),
            "project": root,
        })


app.state.handle_file_change = _handle_file_change


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html", media_type="text/html")


@app.get("/api/projects")
async def get_projects() -> JSONResponse:
    projects: dict[str, GraphState] = app.state.projects
    return JSONResponse({
        "projects": [
            {"path": k, "name": Path(k).name, "files": len(v.nodes)}
            for k, v in projects.items()
        ]
    })


@app.get("/api/graph")
async def get_graph(project: str | None = None) -> JSONResponse:
    projects: dict[str, GraphState] = app.state.projects
    if not projects:
        return JSONResponse({"nodes": [], "edges": [], "stats": {}})

    if project and project in projects:
        state = projects[project]
    else:
        state = next(iter(projects.values()))

    return JSONResponse(state.to_graph())


@app.post("/api/hook")
async def hook(request: Request) -> JSONResponse:
    body = await request.json()
    file_path = body.get("file", "")
    by = body.get("by", "unknown")
    cwd = body.get("cwd", os.getcwd())

    if not file_path:
        return JSONResponse({"ok": False, "reason": "no file"})

    asyncio.create_task(_handle_file_change(file_path, by, cwd))
    return JSONResponse({"ok": True})


@app.post("/api/refresh")
async def refresh(request: Request) -> JSONResponse:
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    project = body.get("project", "")
    projects: dict[str, GraphState] = app.state.projects

    if project and project in projects:
        state = projects[project]
        root = state.root
    elif projects:
        state = next(iter(projects.values()))
        root = state.root
    else:
        return JSONResponse({"ok": False, "reason": "no project registered"})

    graph = await asyncio.to_thread(parse_directory, root)
    state.load(graph)
    await app.state.manager.broadcast({
        "type": "graph",
        "data": state.to_graph(),
        "project": str(root),
    })
    return JSONResponse({"ok": True, "files": state.stats.get("files", 0)})


@app.get("/api/embeddings")
async def get_embeddings(project: str | None = None) -> JSONResponse:
    if not HAS_EMBEDDINGS:
        return JSONResponse({"ok": False, "reason": "embeddings not available"})

    projects: dict[str, GraphState] = app.state.projects
    if not projects:
        return JSONResponse({"edges": []})

    if project and project in projects:
        state = projects[project]
    else:
        state = next(iter(projects.values()))

    # Build {id: text} map (first 2000 chars of each file)
    texts: dict[str, str] = {}
    for node_id, node in state.nodes.items():
        fpath = node["data"].get("path", "")
        if fpath:
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")[:2000]
                texts[node_id] = text
            except OSError:
                pass

    if len(texts) < 2:
        return JSONResponse({"edges": []})

    # Embed all texts
    store = EmbeddingStore()
    vectors: dict[str, Any] = {}
    for nid, text in texts.items():
        vec = await asyncio.to_thread(store.embed_text, text)
        if vec is not None:
            vectors[nid] = np.array(vec, dtype=np.float32)

    # Compute pairwise cosine similarity
    ids = list(vectors.keys())
    existing_edge_pairs = {
        (e["data"]["source"], e["data"]["target"])
        for e in state.edges.values()
    }

    sem_edges: list[dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (a, b) in existing_edge_pairs or (b, a) in existing_edge_pairs:
                continue
            sim = cosine_similarity(vectors[a], vectors[b])
            if sim >= SEMANTIC_THRESHOLD:
                edge = {
                    "data": {
                        "id": f"sem:{a}→{b}",
                        "source": a,
                        "target": b,
                        "type": "semantic",
                        "label": f"{sim:.2f}",
                    }
                }
                sem_edges.append(edge)

    # Broadcast to clients
    if sem_edges:
        await app.state.manager.broadcast({
            "type": "semantic-edges",
            "edges": sem_edges,
            "project": str(state.root),
        })

    return JSONResponse({"edges": sem_edges, "count": len(sem_edges)})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    manager: ConnectionManager = app.state.manager
    await manager.connect(ws)
    try:
        # Send current graph on connect
        projects: dict[str, GraphState] = app.state.projects
        if projects:
            state = next(iter(projects.values()))
            await ws.send_text(json.dumps({
                "type": "graph",
                "data": state.to_graph(),
                "project": str(state.root),
            }))
        while True:
            await ws.receive_text()  # keep alive; ignore client messages
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Code Graph Monitor server")
    parser.add_argument("--path", default=os.getcwd(), help="Project root to monitor")
    parser.add_argument("--port", type=int, default=PORT, help="HTTP port (default 8765)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    args = parser.parse_args()

    os.environ["_CGRAPH_STARTUP_PATH"] = str(Path(args.path).resolve())
    os.environ["CGRAPH_PORT"] = str(args.port)

    logger.info("Starting Code Graph Monitor on http://localhost:%d", args.port)
    logger.info("Monitoring: %s", args.path)

    uvicorn.run(
        "scripts_and_skills.code_graph.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
