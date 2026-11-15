"""Fork hook in the `cluster-only` branch of cli.py (logic: graphify/recluster.py).

Upstream knows nothing about it, so a merge can drop the call site silently;
these tests go red when it does.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import graphify.query_backend as qb
from graphify.recluster import sync_community_changes

FIXTURES = Path(__file__).parent / "fixtures"


def _run(args, cwd):
    return subprocess.run([sys.executable, "-m", "graphify"] + args, cwd=cwd,
                          capture_output=True, text=True)


def _settled_project(tmp_path) -> Path:
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json
    out = tmp_path / "graphify-out"
    out.mkdir()
    G = build_from_json(json.loads((FIXTURES / "extraction.json").read_text()))
    to_json(G, cluster(G), str(out / "graph.json"))
    # First run settles whatever cluster-only itself adds (names, stamp).
    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)
    assert r.returncode == 0, r.stderr
    return out / "graph.json"


def test_idle_recluster_leaves_graph_json_alone(tmp_path):
    graph = _settled_project(tmp_path)
    before, mtime = graph.read_bytes(), graph.stat().st_mtime_ns
    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "partition unchanged" in r.stdout
    assert graph.stat().st_mtime_ns == mtime
    assert graph.read_bytes() == before
    assert (tmp_path / "graphify-out" / "GRAPH_REPORT.md").exists()


def test_changed_partition_is_still_written(tmp_path):
    graph = _settled_project(tmp_path)
    settled = graph.read_bytes()
    data = json.loads(settled)
    data["nodes"][0]["community"] = 987654          # stored partition is now stale
    graph.write_text(json.dumps(data, indent=2), encoding="utf-8")
    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "partition unchanged" not in r.stdout
    assert all(n["community"] != 987654
               for n in json.loads(graph.read_text(encoding="utf-8"))["nodes"])


def test_json_store_is_a_noop(monkeypatch):
    """On the JSON store there is nothing to republish after a write."""
    published = []
    import graphify.snap_store as snap_store
    monkeypatch.setattr(snap_store, "publish_snapshot_for", lambda p: published.append(p))
    monkeypatch.setattr(qb, "resolve_backend_config", lambda p=None: {"kind": "json", "graph_path": p})
    sync_community_changes("g.json", {"nodes": []}, {"a": (1, "B")}, {1: ["a"]})
    assert published == []
