"""Tests for query_backend.py.

The headline guarantee: JsonBackend + render_query reproduces the existing
serve._query_graph_text output byte-for-byte. That locks the abstraction so the
later serve.py/CLI rewiring (and the ArcadeDB backend's golden parity) have a
trustworthy oracle.
"""
import networkx as nx
import pytest

from graphify.serve import _query_graph_text
from graphify.query_backend import (
    ArcadeDBBackend,
    JsonBackend,
    derive_db_name,
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

def test_resolve_config_defaults_to_json(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_BACKEND", raising=False)
    assert resolve_backend_config("g.json") == {"kind": "json", "graph_path": "g.json"}


def test_resolve_config_arcadedb_from_env(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "arcadedb")
    monkeypatch.setenv("GRAPHIFY_ARCADE_URL", "http://db.host:9999")
    monkeypatch.setenv("GRAPHIFY_ARCADE_DB", "proj_x")
    cfg = resolve_backend_config()
    assert cfg["kind"] == "arcadedb"
    assert cfg["host"] == "db.host" and cfg["port"] == 9999
    assert cfg["database"] == "proj_x"


def test_open_backend_json_from_preloaded_graph(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_BACKEND", raising=False)
    be = open_backend(graph=_make_digraph())
    assert isinstance(be, JsonBackend)


def test_derive_db_name_from_project_root():
    # graph at <root>/graphify-out/graph.json -> proj_<root>
    assert derive_db_name("/home/me/My-Repo/graphify-out/graph.json") == "proj_My_Repo"
    assert derive_db_name(None) == "graphify"


def test_resolve_config_arcadedb_derives_db_when_unset(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "arcadedb")
    monkeypatch.delenv("GRAPHIFY_ARCADE_DB", raising=False)
    cfg = resolve_backend_config("/x/CoolProj/graphify-out/graph.json")
    assert cfg["database"] == "proj_CoolProj"


def test_open_backend_arcadedb_constructs_without_connecting(monkeypatch):
    # __init__ only builds an HTTP session - no server needed to select it.
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "proj_x", "user": "root", "password": "x"})
    assert isinstance(be, ArcadeDBBackend) and be.db == "proj_x"
