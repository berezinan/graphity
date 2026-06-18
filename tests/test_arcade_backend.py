"""Golden parity tests: ArcadeDBBackend vs JsonBackend.

Loads one fixture graph into a live ArcadeDB and asserts the DB backend produces
the same answers as the in-memory backend. Skipped entirely when no ArcadeDB is
reachable (CI / dev machines without the server), so the suite stays green
without the dependency.

Point at a server with GRAPHIFY_ARCADE_URL=http://host:port (default
127.0.0.1:2480) and GRAPHIFY_ARCADE_PASSWORD (default playwithdata).
"""
import json
import os

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from graphify.query_backend import (
    ArcadeDBBackend,
    JsonBackend,
    render_explain,
    render_node,
    render_neighbors,
    render_path,
    render_stats,
)

_URL = os.environ.get("GRAPHIFY_ARCADE_URL", "http://127.0.0.1:2480")
_PW = os.environ.get("GRAPHIFY_ARCADE_PASSWORD", "playwithdata")
_DB = "graphify_parity_test"


def _make_digraph() -> nx.DiGraph:
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


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host or "127.0.0.1", int(port or "2480")


@pytest.fixture(scope="module")
def backends(tmp_path_factory):
    host, port = _parse_host_port(_URL)
    arcade = ArcadeDBBackend(_DB, host=host, port=port, password=_PW)
    if not arcade.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    G = _make_digraph()
    path = tmp_path_factory.mktemp("graph") / "graph.json"
    path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    arcade.ensure_database(drop=True)
    arcade.load_from_graph_json(str(path))
    yield JsonBackend(G), arcade
    try:
        arcade._server(f"drop database {_DB}")
    except Exception:
        pass


def _query_sets(r):
    nodes = {n.label for n in r.nodes}
    edges = {(e.source_label, e.relation, e.target_label) for e in r.edges}
    return r.seeds, nodes, edges


@pytest.mark.parametrize("q", ["extract", "build", "cluster", "report uses", "build cluster"])
def test_query_parity(backends, q):
    jb, arc = backends
    js, jn, je = _query_sets(jb.query(q, depth=3))
    as_, an, ae = _query_sets(arc.query(q, depth=3))
    assert js == as_          # identical seed resolution (the correctness-critical part)
    assert jn == an           # same node set
    assert je == ae           # same edges


@pytest.mark.parametrize("q,ctx", [("extract", ["call"]), ("cluster", ["import"]), ("build", ["call", "import"])])
def test_query_context_filter_parity(backends, q, ctx):
    jb, arc = backends
    js, jn, je = _query_sets(jb.query(q, depth=3, context_filters=ctx))
    as_, an, ae = _query_sets(arc.query(q, depth=3, context_filters=ctx))
    assert js == as_
    assert jn == an
    assert je == ae


def test_query_no_match_parity(backends):
    jb, arc = backends
    assert jb.query("zzznope").seeds == arc.query("zzznope").seeds == []


@pytest.mark.parametrize("label", ["extract", "build", "report", "nope"])
def test_get_node_parity(backends, label):
    jb, arc = backends
    assert render_node(jb.get_node(label), label.lower()) == render_node(arc.get_node(label), label.lower())


def test_get_neighbors_parity(backends):
    jb, arc = backends
    def norm(n):
        return None if n is None else (n.label, sorted((x.outgoing, x.label, x.relation, x.confidence) for x in n.neighbors))
    assert norm(jb.get_neighbors("build")) == norm(arc.get_neighbors("build"))


def test_get_community_parity(backends):
    jb, arc = backends
    for cid in (0, 1, 2):
        jc, ac = jb.get_community(cid), arc.get_community(cid)
        assert {m.label for m in jc.members} == {m.label for m in ac.members}


def test_graph_stats_parity(backends):
    jb, arc = backends
    assert render_stats(jb.graph_stats()) == render_stats(arc.graph_stats())


def test_shortest_path_parity(backends):
    jb, arc = backends
    j = render_path(jb.shortest_path("extract", "report"), "extract", "report")
    a = render_path(arc.shortest_path("extract", "report"), "extract", "report")
    assert j == a
    assert "3 hops" in a


def test_shortest_path_no_path_parity(backends):
    jb, arc = backends
    # n5 (isolated) has no path from n1.
    j = render_path(jb.shortest_path("extract", "isolated"), "extract", "isolated")
    a = render_path(arc.shortest_path("extract", "isolated"), "extract", "isolated")
    assert j == a
    assert "No path found" in a


@pytest.mark.parametrize("label", ["build", "cluster", "extract"])
def test_explain_parity(backends, label):
    jb, arc = backends
    def norm(r):
        d = r.detail
        return (
            (d.label, d.id, d.source_file, d.file_type, d.community, d.degree),
            sorted((c.outgoing, c.label, c.relation, c.confidence, c.degree) for c in r.connections),
        )
    assert norm(jb.explain(label)) == norm(arc.explain(label))
    # render parity too (degree-ranked, but ties -> compare as the renderer emits)
    assert render_explain(jb.explain(label), label).splitlines()[:6] == \
           render_explain(arc.explain(label), label).splitlines()[:6]


def test_god_nodes_parity(backends):
    jb, arc = backends
    assert [(g.label, g.degree) for g in jb.god_nodes(5)] == [(g.label, g.degree) for g in arc.god_nodes(5)]


def test_pr_impact_parity(backends):
    jb, arc = backends
    files = ["extract.py", "build.py"]
    assert jb.pr_impact(files) == arc.pr_impact(files)


def test_global_add_syncs_arcadedb(tmp_path, monkeypatch):
    """global_add reloads the separate global ArcadeDB database."""
    host, port = _parse_host_port(_URL)
    probe = ArcadeDBBackend("graphify_global_test", host=host, port=port, password=_PW)
    if not probe.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    gdir = tmp_path / "g"
    gdir.mkdir()
    gpath = gdir / "global-graph.json"
    monkeypatch.setattr("graphify.global_graph._GLOBAL_DIR", gdir)
    monkeypatch.setattr("graphify.global_graph._GLOBAL_GRAPH", gpath)
    monkeypatch.setattr("graphify.global_graph._GLOBAL_MANIFEST", gdir / "global-manifest.json")
    monkeypatch.setenv("GRAPHIFY_BACKEND", "arcadedb")
    monkeypatch.setenv("GRAPHIFY_ARCADE_URL", _URL)
    monkeypatch.setenv("GRAPHIFY_ARCADE_PASSWORD", _PW)
    monkeypatch.setenv("GRAPHIFY_ARCADE_GLOBAL_DB", "graphify_global_test")

    sp = tmp_path / "src.json"
    sp.write_text(json.dumps(json_graph.node_link_data(_make_digraph(), edges="links")), encoding="utf-8")
    from graphify.global_graph import global_add
    try:
        global_add(sp, "repoA")
        graw = json.loads(gpath.read_text(encoding="utf-8"))
        if "links" not in graw and "edges" in graw:
            graw = dict(graw, links=graw["edges"])
        jb = JsonBackend(json_graph.node_link_graph({**graw, "directed": True}, edges="links"))
        arc = ArcadeDBBackend("graphify_global_test", host=host, port=port, password=_PW)
        assert jb.graph_stats().nodes == arc.graph_stats().nodes
        assert jb.graph_stats().edges == arc.graph_stats().edges
    finally:
        try:
            probe._server("drop database graphify_global_test")
        except Exception:
            pass


def test_sync_graph_incremental(tmp_path_factory):
    """sync_graph brings the DB to the updated graph touching only dirty files."""
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_sync_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    # Initial load of G1.
    g1 = _make_digraph()
    p = tmp_path_factory.mktemp("sync") / "g1.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        # G2: extract.py changed (relabel n1, add n6->n2); other.py pruned (drop n5).
        g2 = _make_digraph()
        g2.remove_node("n5")
        g2.nodes["n1"]["label"] = "extractor"
        g2.add_node("n6", label="helper", source_file="extract.py", source_location="L99", community=0)
        g2.add_edge("n6", "n2", relation="calls", confidence="INFERRED")
        arc.sync_graph(g2, changed_sources={"extract.py"}, pruned_sources={"other.py"})

        jb = JsonBackend(g2)
        # counts match the updated graph
        assert render_stats(jb.graph_stats()) == render_stats(arc.graph_stats())
        # relabeled node present, pruned node gone
        assert render_node(arc.get_node("extractor"), "extractor") == render_node(jb.get_node("extractor"), "extractor")
        assert arc.get_node("isolated") is None
        # new edge reachable: query around build still parity on node set
        jn = {n.label for n in jb.query("build", depth=3).nodes}
        an = {n.label for n in arc.query("build", depth=3).nodes}
        assert jn == an
    finally:
        try:
            arc._server("drop database graphify_sync_test")
        except Exception:
            pass
