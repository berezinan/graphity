"""Characterization + parity tests for the `graphify path` / `explain` CLI.

These commands had no CLI coverage before they were routed through GraphBackend,
so this file first pins their exact stdout/stderr/exit behaviour and then keeps
it identical after the refactor.
"""
from __future__ import annotations

import json

import networkx as nx
import pytest
from networkx.readwrite import json_graph

import graphify.__main__ as mainmod


def _write_graph(tmp_path):
    G = nx.DiGraph()
    G.add_node("n1", label="extract", source_file="extract.py", source_location="L10", community=0)
    G.add_node("n2", label="cluster", source_file="cluster.py", source_location="L5", community=0)
    G.add_node("n3", label="build", source_file="build.py", source_location="L1", community=1)
    G.add_edge("n1", "n2", relation="calls", confidence="EXTRACTED", context="call")
    G.add_edge("n2", "n3", relation="imports", confidence="EXTRACTED", context="import")
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")))
    return graph_path


def _run(monkeypatch, argv):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", argv)
    mainmod.main()


def test_path_found(monkeypatch, tmp_path, capsys):
    gp = _write_graph(tmp_path)
    _run(monkeypatch, ["graphify", "path", "extract", "build", "--graph", str(gp)])
    out = capsys.readouterr().out
    assert out == (
        "Shortest path (2 hops):\n"
        "  extract --calls [EXTRACTED]--> cluster --imports [EXTRACTED]--> build\n"
    )


def test_path_no_match_source_exits(monkeypatch, tmp_path, capsys):
    gp = _write_graph(tmp_path)
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["graphify", "path", "zzznope", "build", "--graph", str(gp)])
    assert "No node matching 'zzznope' found." in capsys.readouterr().err


def test_path_no_path(monkeypatch, tmp_path, capsys):
    gp = _write_graph(tmp_path)
    # n3 -> nothing; reverse direction has a path (undirected), so use a real gap:
    # add an isolated node and query toward it.
    data = json.loads(gp.read_text())
    data["nodes"].append({"id": "n9", "label": "island", "source_file": "x.py", "community": 5})
    gp.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["graphify", "path", "extract", "island", "--graph", str(gp)])
    assert exc.value.code == 0
    assert capsys.readouterr().out == "No path found between 'extract' and 'island'.\n"


def test_explain(monkeypatch, tmp_path, capsys):
    gp = _write_graph(tmp_path)
    _run(monkeypatch, ["graphify", "explain", "cluster", "--graph", str(gp)])
    out = capsys.readouterr().out
    assert "Node: cluster" in out
    assert "  ID:        n2" in out
    assert "  Source:    cluster.py L5" in out
    assert "  Community: 0" in out
    assert "  Degree:    2" in out
    assert "Connections (2):" in out
    assert "  --> build [imports] [EXTRACTED]" in out
    assert "  <-- extract [calls] [EXTRACTED]" in out


def test_explain_no_match(monkeypatch, tmp_path, capsys):
    gp = _write_graph(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["graphify", "explain", "zzznope", "--graph", str(gp)])
    assert exc.value.code == 0
    assert "No node matching 'zzznope' found." in capsys.readouterr().out
