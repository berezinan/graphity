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


class _FakeDb:
    def __init__(self, populated=True, shortfall=False, reconciled=True):
        self.populated, self.shortfall, self.reconciled = populated, shortfall, reconciled
        self.calls = []

    def is_populated(self):
        return self.populated

    def has_recorded_shortfall(self):
        return self.shortfall

    def update_communities(self, node_community, expected_sizes, **kw):
        self.calls.append((node_community, expected_sizes))
        return {"updated_nodes": len(node_community), "reconciled": self.reconciled}


RAW = {"nodes": [{"id": "a", "community": 0, "community_name": "A"},
                 {"id": "узел_б", "community": 0, "community_name": "A"}]}
COMMUNITIES = {0: ["a"], 1: ["узел_б"]}


def _patch(monkeypatch, db, kind="arcadedb"):
    monkeypatch.setattr(qb, "resolve_backend_config", lambda p=None: {"kind": kind, "database": "proj_t"})
    monkeypatch.setattr(qb, "require_backend", lambda **kw: db)


def test_moved_nodes_go_to_the_database(monkeypatch):
    db = _FakeDb()
    _patch(monkeypatch, db)
    sync_community_changes("g.json", RAW, {"узел_б": (1, "B")}, COMMUNITIES)
    assert db.calls == [({"узел_б": 1}, {0: 1, 1: 1})]


def test_name_only_change_sends_nothing(monkeypatch):
    db = _FakeDb()
    _patch(monkeypatch, db)
    sync_community_changes("g.json", RAW, {"a": (0, "Renamed")}, {0: ["a", "узел_б"]})
    assert db.calls == []


def test_json_backend_is_a_noop(monkeypatch):
    db = _FakeDb()
    _patch(monkeypatch, db, kind="json")
    sync_community_changes("g.json", RAW, {"узел_б": (1, "B")}, COMMUNITIES)
    assert db.calls == []


def test_empty_short_or_reshaped_database_gets_a_reload_hint(monkeypatch, capsys):
    for db, changed in ((_FakeDb(populated=False), {"узел_б": (1, "B")}),
                        (_FakeDb(shortfall=True), {"узел_б": (1, "B")}),
                        (_FakeDb(), None)):
        _patch(monkeypatch, db)
        sync_community_changes("g.json", RAW, changed, COMMUNITIES)
        assert db.calls == []
        assert "arcade reload" in capsys.readouterr().out


def test_unreconciled_update_fails_the_command(monkeypatch):
    import pytest
    _patch(monkeypatch, _FakeDb(reconciled=False))
    with pytest.raises(RuntimeError, match="did not reconcile"):
        sync_community_changes("g.json", RAW, {"узел_б": (1, "B")}, COMMUNITIES)
