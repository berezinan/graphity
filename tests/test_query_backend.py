"""Tests for query_backend.py.

The headline guarantee: JsonBackend + render_query reproduces the existing
serve._query_graph_text output byte-for-byte. That locks the abstraction so the
later serve.py/CLI rewiring (and the snapshot backend's golden parity) have a
trustworthy oracle.
"""
import re

import networkx as nx
import pytest

from graphify.serve import _query_graph_text
from graphify.query_backend import (
    JsonBackend,
    open_backend,
    resolve_backend_config,
    render_query,
    render_node,
    render_neighbors,
    render_community,
    render_god_nodes,
    render_stats,
    render_path,
)


def _make_digraph() -> nx.DiGraph:
    # Directed, mirroring production (serve._load_graph forces directed=True).
    G = nx.DiGraph()
    G.add_node("n1", label="extract", source_file="extract.py", source_location="L10", community=0)
    G.add_node("n2", label="cluster", source_file="cluster.py", source_location="L5", community=0)
    G.add_node("n3", label="build", source_file="build.py", source_location="L1", community=1)
    G.add_node("n4", label="report", source_file="report.py", source_location="L1", community=1)
    G.add_node("n5", label="isolated", source_file="other.py", source_location="L1", community=2)
    G.add_edge("n1", "n2", relation="calls", confidence="INFERRED", context="call")
    G.add_edge("n2", "n3", relation="imports", confidence="EXTRACTED", context="import")
    G.add_edge("n3", "n4", relation="uses", confidence="EXTRACTED")
    return G


# --- the core parity guarantee --------------------------------------------- #

@pytest.mark.parametrize("question", ["extract", "build", "cluster calls", "xyzzy", "report uses"])
@pytest.mark.parametrize("mode", ["bfs", "dfs"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("budget", [2000, 5])
def test_query_matches_legacy_text(question, mode, depth, budget):
    G = _make_digraph()
    jb = JsonBackend(G)
    got = render_query(jb.query(question, mode=mode, depth=depth), token_budget=budget)
    want = _query_graph_text(G, question, mode=mode, depth=depth, token_budget=budget)
    assert got == want


def test_query_with_explicit_context_filter_matches_legacy():
    G = _make_digraph()
    jb = JsonBackend(G)
    got = render_query(jb.query("extract", context_filters=["call"]), token_budget=2000)
    want = _query_graph_text(G, "extract", context_filters=["call"], token_budget=2000)
    assert got == want


def test_query_no_match():
    jb = JsonBackend(_make_digraph())
    assert render_query(jb.query("zzzznope")) == "No matching nodes found."


# --- golden output for the inline-formatted tools -------------------------- #

def test_get_node():
    jb = JsonBackend(_make_digraph())
    assert render_node(jb.get_node("extract"), "extract") == (
        "Node: extract\n  ID: n1\n  Source: extract.py L10\n  Type: \n  Community: 0\n  Degree: 1"
    )


def test_get_node_missing():
    jb = JsonBackend(_make_digraph())
    assert render_node(jb.get_node("nope"), "nope") == "No node matching 'nope' found."


def test_get_neighbors():
    jb = JsonBackend(_make_digraph())
    assert render_neighbors(jb.get_neighbors("build"), "build") == (
        "Neighbors of build:\n"
        "  --> report [uses] [EXTRACTED]\n"
        "  <-- cluster [imports] [EXTRACTED]"
    )


def test_get_neighbors_relation_filter():
    jb = JsonBackend(_make_digraph())
    out = render_neighbors(jb.get_neighbors("build", relation_filter="uses"), "build")
    assert out == "Neighbors of build:\n  --> report [uses] [EXTRACTED]"


def test_get_community():
    jb = JsonBackend(_make_digraph())
    assert render_community(jb.get_community(0), 0) == (
        "Community 0 (2 nodes):\n  extract [extract.py]\n  cluster [cluster.py]"
    )


def test_get_community_missing():
    jb = JsonBackend(_make_digraph())
    assert render_community(jb.get_community(99), 99) == "Community 99 not found."


def test_graph_stats():
    jb = JsonBackend(_make_digraph())
    assert render_stats(jb.graph_stats()) == (
        "Nodes: 5\nEdges: 3\nCommunities: 3\nEXTRACTED: 67%\nINFERRED: 33%\nAMBIGUOUS: 0%\n"
    )


def test_shortest_path():
    jb = JsonBackend(_make_digraph())
    out = render_path(jb.shortest_path("extract", "report"), "extract", "report")
    assert out == (
        "Shortest path (3 hops):\n"
        "  extract --calls [INFERRED]--> cluster --imports [EXTRACTED]--> build --uses [EXTRACTED]--> report"
    )


def test_shortest_path_same_node():
    jb = JsonBackend(_make_digraph())
    out = render_path(jb.shortest_path("extract", "extract"), "extract", "extract")
    assert "both resolved to the same node" in out


def test_god_nodes_renders():
    jb = JsonBackend(_make_digraph())
    out = render_god_nodes(jb.god_nodes(top_n=3))
    assert out.startswith("God nodes (most connected):")


# --- backend selection factory --------------------------------------------- #

def test_resolve_config_default_backend(monkeypatch):
    """The default backend with no GRAPHIFY_GRAPH_STORE set is the snapshot.

    Guarded on purpose: upstream knows nothing about the snapshot, so a merge can
    put ``json`` back as the default as silently as it can drop a call site —
    and graphify must never quietly answer from graph.json.
    """
    monkeypatch.delenv("GRAPHIFY_GRAPH_STORE", raising=False)
    cfg = resolve_backend_config("g.json")
    assert cfg["kind"] == "snap"
    assert cfg["snapshot_path"].endswith("g.snap")


def test_resolve_config_empty_value_is_the_default(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "  ")
    assert resolve_backend_config("g.json")["kind"] == "snap"


def test_resolve_config_unknown_value_is_an_error(monkeypatch):
    """A typo must not silently select JSON (loading a whole graph.json)."""
    import pytest
    from graphify.query_backend import GraphDBUnavailable
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "snapshot")
    with pytest.raises(GraphDBUnavailable) as exc:
        resolve_backend_config("g.json")
    msg = str(exc.value)
    assert "snapshot" in msg and "snap" in msg and "json" in msg
    assert "arcadedb" not in msg


def test_resolve_config_ignores_graphify_backend(monkeypatch):
    """GRAPHIFY_BACKEND is upstream's LLM selector (#4361) and never picks the store."""
    monkeypatch.delenv("GRAPHIFY_GRAPH_STORE", raising=False)
    monkeypatch.setenv("GRAPHIFY_BACKEND", "json")
    assert resolve_backend_config("g.json")["kind"] == "snap"


def test_resolve_config_json_is_the_explicit_opt_out(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
    assert resolve_backend_config("g.json") == {"kind": "json", "graph_path": "g.json"}


@pytest.mark.parametrize("value", ["arcadedb", "arcade", "ArcadeDB"])
def test_resolve_config_retired_arcadedb_is_an_error(monkeypatch, value):
    """ArcadeDB was retired (retire-arcadedb): an explicit choice is refused with
    a message of its own, never quietly served from the snapshot."""
    from graphify.query_backend import GraphDBUnavailable
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", value)
    with pytest.raises(GraphDBUnavailable) as exc:
        resolve_backend_config("g.json")
    msg = str(exc.value)
    assert "ArcadeDB was retired" in msg and "snap" in msg


def test_retired_store_exits_nonzero_from_the_cli(monkeypatch, tmp_path, capsys):
    import sys
    import graphify.__main__ as mainmod
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "arcadedb")
    monkeypatch.setattr(sys, "argv", ["graphify", "query", "x", "--graph", str(tmp_path / "graph.json")])
    with pytest.raises(SystemExit) as exc:
        mainmod.main()
    assert exc.value.code == 1
    assert "ArcadeDB was retired" in capsys.readouterr().err


def test_arcade_command_reports_the_retirement(monkeypatch, capsys):
    import sys
    from graphify import cli
    monkeypatch.setattr(sys, "argv", ["graphify", "arcade", "status"])
    with pytest.raises(SystemExit) as exc:
        cli.dispatch_command("arcade")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "ArcadeDB was retired" in err and "uninstall" in err


def test_open_backend_json_from_preloaded_graph(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
    be = open_backend(graph=_make_digraph())
    assert isinstance(be, JsonBackend)
