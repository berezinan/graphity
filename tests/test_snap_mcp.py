"""MCP server on the snapshot store: one stale project is a tool error, the rest keep answering."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("starlette")
pytest.importorskip("graphify_snap", reason="graphify-snap wheel not installed")

from starlette.testclient import TestClient  # noqa: E402

from graphify import serve as serve_mod  # noqa: E402
from graphify.snap_writer import build_snapshot  # noqa: E402

_INIT_BODY = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "test", "version": "0"}},
}
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _write_graph(path: Path, node_count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [{"id": f"n{i}", "label": f"Node{i}", "source_file": f"src/m{i}.py", "community": 0}
             for i in range(node_count)]
    links = [{"source": f"n{i}", "target": f"n{i + 1}", "relation": "calls", "confidence": "EXTRACTED"}
             for i in range(node_count - 1)]
    data = {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _project(tmp_path: Path, name: str, node_count: int) -> str:
    root = tmp_path / name
    build_snapshot(_write_graph(root / "graphify-out" / "graph.json", node_count))
    return str(root)


def _client(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _init_session(client) -> dict:
    init = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
    assert init.status_code == 200
    headers = {**_MCP_HEADERS, "mcp-session-id": init.headers.get("mcp-session-id")}
    client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


def _call_tool(client, headers, name, arguments, rid) -> str:
    resp = client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": rid, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    assert resp.status_code == 200
    return resp.json()["result"]["content"][0]["text"]


def test_stale_project_is_a_tool_error_and_others_keep_answering(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "snap")
    default = _project(tmp_path, "default", 2)
    fresh = _project(tmp_path, "fresh", 3)
    stale = _project(tmp_path, "stale", 4)
    stale_graph = Path(stale) / "graphify-out" / "graph.json"
    st = stale_graph.stat()
    os.utime(stale_graph, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    app = serve_mod._build_http_app(str(Path(default) / "graphify-out" / "graph.json"), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=2)
        bad = _call_tool(client, headers, "graph_stats", {"project_path": stale}, rid=3)
        assert "stale" in bad and "graphify snap build" in bad
        assert "Nodes: 3" in _call_tool(client, headers, "graph_stats", {"project_path": fresh}, rid=4)
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=5)


def test_mcp_on_snap_does_not_load_networkx_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "snap")
    default = _project(tmp_path, "default", 2)

    def _boom(*a, **k):
        raise AssertionError("graph.json must not be loaded on the snapshot store")

    monkeypatch.setattr(serve_mod, "_load_graph", _boom)
    app = serve_mod._build_http_app(str(Path(default) / "graphify-out" / "graph.json"), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=2)
