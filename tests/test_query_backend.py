"""Tests for query_backend.py.

The headline guarantee: JsonBackend + render_query reproduces the existing
serve._query_graph_text output byte-for-byte. That locks the abstraction so the
later serve.py/CLI rewiring (and the ArcadeDB backend's golden parity) have a
trustworthy oracle.
"""
import re

import networkx as nx
import pytest

from graphify.serve import _query_graph_text
from graphify.query_backend import (
    ArcadeDBBackend,
    JsonBackend,
    _id_key,
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

def test_resolve_config_default_backend(monkeypatch):
    """The default backend with no GRAPHIFY_BACKEND set is the graph database.

    Guarded on purpose: upstream knows nothing about ArcadeDB, so a merge can
    put ``json`` back as the default as silently as it can drop a call site —
    and the whole point of this change is that graphify never quietly answers
    from graph.json.
    """
    monkeypatch.delenv("GRAPHIFY_BACKEND", raising=False)
    assert resolve_backend_config("g.json")["kind"] == "arcadedb"


def test_resolve_config_json_is_the_explicit_opt_out(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "json")
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
    monkeypatch.setenv("GRAPHIFY_BACKEND", "json")
    be = open_backend(graph=_make_digraph())
    assert isinstance(be, JsonBackend)


def _graph_json(tmp_path, name):
    root = tmp_path / name
    (root / "graphify-out").mkdir(parents=True)
    return root / "graphify-out" / "graph.json"


def test_derive_db_name_from_project_root(tmp_path):
    # graph at <root>/graphify-out/graph.json -> proj_<root>_<hash8>
    name = derive_db_name(str(_graph_json(tmp_path, "My-Repo")))
    assert re.fullmatch(r"proj_my_repo_[0-9a-f]{8}", name), name
    assert derive_db_name(None) == "graphify"


def test_derive_db_name_same_project_from_different_cwd(tmp_path, monkeypatch):
    """One project addressed relatively and absolutely -> one database."""
    gj = _graph_json(tmp_path, "Hotel")
    monkeypatch.chdir(gj.parent.parent)
    relative = derive_db_name("graphify-out/graph.json")
    monkeypatch.chdir(tmp_path)
    absolute = derive_db_name(str(gj))
    assert relative == absolute


def test_derive_db_name_ignores_path_case(tmp_path):
    """Same directory, different spelling of the path -> one database."""
    gj = _graph_json(tmp_path, "Hotel")
    assert derive_db_name(str(gj)) == derive_db_name(str(gj).upper())


def test_derive_db_name_separates_same_named_directories(tmp_path):
    """Two unrelated projects whose directories share a name -> two databases.

    This is the case a name built from the directory alone merged silently once
    every user of the machine shares one hive.
    """
    a = derive_db_name(str(_graph_json(tmp_path / "work", "Hotel")))
    b = derive_db_name(str(_graph_json(tmp_path / "arch", "Hotel")))
    assert a != b
    assert a.startswith("proj_hotel_") and b.startswith("proj_hotel_")


def test_resolve_config_arcadedb_derives_db_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "arcadedb")
    monkeypatch.delenv("GRAPHIFY_ARCADE_DB", raising=False)
    gj = _graph_json(tmp_path, "CoolProj")
    cfg = resolve_backend_config(str(gj))
    assert cfg["database"] == derive_db_name(str(gj))
    assert cfg["database"].startswith("proj_coolproj_")


def test_open_backend_arcadedb_constructs_without_connecting(monkeypatch):
    # __init__ only builds an HTTP session - no server needed to select it.
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "proj_x", "user": "root", "password": "x"})
    assert isinstance(be, ArcadeDBBackend) and be.db == "proj_x"


def test_claim_project_refuses_a_digest_collision():
    """A database already owned by another path is refused, not shared."""
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "proj_hotel_deadbeef", "user": "root", "password": "x",
                              "project_path": r"D:\arch\Hotel"})
    be._run = lambda *a, **kw: [{"value": r"C:\work\Hotel"}]
    with pytest.raises(RuntimeError) as exc:
        be.claim_project()
    assert r"C:\work\Hotel" in str(exc.value) and r"D:\arch\Hotel" in str(exc.value)
    assert "GRAPHIFY_ARCADE_DB" in str(exc.value)


def test_claim_project_records_the_owner_on_first_write():
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "proj_hotel_deadbeef", "user": "root", "password": "x",
                              "project_path": r"C:\work\Hotel"})
    calls = []

    def fake_run(q, **kw):
        calls.append(q)
        return []  # no Meta row yet

    be._run = fake_run
    be.claim_project()
    assert any(q.startswith("INSERT INTO Meta") and r"C:\\work\\Hotel" in q for q in calls), calls


def test_claim_project_is_a_noop_without_a_project():
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "graphify_global", "user": "root", "password": "x",
                              "project_path": ""})

    def fail(*a, **kw):
        raise AssertionError("claim_project must not touch the DB without a project")

    be._run = fail
    be.claim_project()


def test_sync_graph_deletes_changed_ids_before_reinsert(monkeypatch):
    # Regression: a node id is global to the project, but the dirty-source DELETE
    # only removes nodes whose stored source_file is dirty. A shared node whose
    # representative source_file drifted to a changed file across runs still lives
    # in the DB under its old (unchanged) source, so re-INSERT by id would hit the
    # UNIQUE Node(id_key) index (ArcadeDB 409 DuplicatedKeyException). sync_graph
    # must delete the changed ids directly, before re-inserting them.
    be = open_backend(config={"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
                              "database": "proj_x", "user": "root", "password": "x"})
    calls = []

    def fake_run(q, **kw):
        calls.append((q, kw))
        # sync_graph refuses a database that predates the ASCII index key, so the
        # schema probe has to answer the way a current-schema database would.
        if "schema:types" in q:
            return [{"properties": [{"name": "id_key"}]}]
        return []

    monkeypatch.setattr(be, "_run", fake_run)

    G = nx.DiGraph()
    G.add_node("shared_id", label="x", source_file="changed.py")
    G.add_node("n2", label="y", source_file="changed.py")
    G.add_edge("shared_id", "n2", relation="calls", confidence="EXTRACTED")
    be.sync_graph(G, changed_sources={"changed.py"}, pruned_sources=set())

    id_deletes = [(i, kw["params"]["ids"]) for i, (q, kw) in enumerate(calls)
                  if q.startswith("DELETE VERTEX FROM Node WHERE id_key IN")]
    assert id_deletes, "sync_graph must delete the changed node ids by their index key"
    assert {x for _, ids in id_deletes for x in ids} == {_id_key("shared_id"), _id_key("n2")}

    first_insert = next((i for i, (q, _) in enumerate(calls)
                         if q.startswith("INSERT INTO Node")), None)
    assert first_insert is not None, "expected an INSERT INTO Node for the changed nodes"
    assert all(i < first_insert for i, _ in id_deletes), \
        "id-delete must precede the re-insert"
