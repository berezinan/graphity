"""Golden parity tests: ArcadeDBBackend vs JsonBackend.

Loads one fixture graph into a live ArcadeDB and asserts the DB backend produces
the same answers as the in-memory backend. Skipped entirely when no ArcadeDB is
reachable (CI / dev machines without the server), so the suite stays green
without the dependency.

The server is found the way the product finds it: GRAPHIFY_ARCADE_URL /
GRAPHIFY_ARCADE_PASSWORD first, then the machine config the installed service
wrote. A fixed password here would skip every one of these tests against a
normally installed service — a silent loss of the whole parity suite.
"""
import json
import os

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from graphify import arcade_server
from graphify.query_backend import (
    ArcadeDBBackend,
    JsonBackend,
    render_explain,
    render_node,
    render_neighbors,
    render_path,
    render_stats,
    shortfall_warning,
)

_MACHINE = arcade_server.read_machine_config()
_URL = os.environ.get("GRAPHIFY_ARCADE_URL") or _MACHINE.get("url") or "http://127.0.0.1:2480"
_PW = os.environ.get("GRAPHIFY_ARCADE_PASSWORD") or _MACHINE.get("password") or ""
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


def _make_mixed_alphabet_graph() -> nx.DiGraph:
    """Ids that mix ASCII with bytes >= 0x80 at the first differing position.

    This is what any non-English project produces, and it is the exact shape
    that desynchronizes ArcadeDB's LSM string index (see _id_key).
    """
    G = nx.DiGraph()
    G.add_node("doc_zakaz_klienta", label="ЗаказКлиента", source_file="a.bsl",
               source_location="L1", community=0)
    G.add_node("Справочники.Номенклатура", label="Номенклатура", source_file="b.bsl",
               source_location="L2", community=0)
    G.add_node("ЯдроОбработки", label="ЯдроОбработки", source_file="c.bsl",
               source_location="L3", community=1)
    G.add_node("zzz_tail_ascii", label="tail", source_file="d.bsl",
               source_location="L4", community=1)
    G.add_edge("doc_zakaz_klienta", "Справочники.Номенклатура", relation="uses",
               confidence="EXTRACTED", context="call")
    G.add_edge("Справочники.Номенклатура", "ЯдроОбработки", relation="calls",
               confidence="EXTRACTED", context="call")
    G.add_edge("ЯдроОбработки", "zzz_tail_ascii", relation="imports", confidence="INFERRED")
    return G


def test_mixed_alphabet_ids_resolve(tmp_path_factory):
    """Every point lookup still finds its node when ids mix alphabets.

    Regression for the shortfall measured on ERP2: with the unique index built
    over `id`, 23.35% of nodes stopped resolving after compaction and took 30.1%
    of the edges with them, because CREATE EDGE silently creates nothing when its
    endpoint subquery is empty.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_mixed_key_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    G = _make_mixed_alphabet_graph()
    p = tmp_path_factory.mktemp("mixed") / "g.json"
    p.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        jb = JsonBackend(G)
        # No edge was silently dropped: the endpoint subqueries all resolved.
        assert render_stats(jb.graph_stats()) == render_stats(arc.graph_stats())
        for label in ("ЗаказКлиента", "Номенклатура", "ЯдроОбработки", "tail"):
            assert render_node(arc.get_node(label), label) == render_node(jb.get_node(label), label)
            assert render_explain(arc.explain(label), label) == render_explain(jb.explain(label), label)
        assert render_path(arc.shortest_path("ЗаказКлиента", "tail"), "ЗаказКлиента", "tail") == \
            render_path(jb.shortest_path("ЗаказКлиента", "tail"), "ЗаказКлиента", "tail")
    finally:
        try:
            arc._server("drop database graphify_mixed_key_test")
        except Exception:
            pass


def test_duplicate_index_key_fails_loudly(tmp_path_factory, monkeypatch):
    """Two ids colliding on the index key must fail, never merge two nodes.

    blake2b-128 makes a real collision practically unreachable, so the guard is
    forced here: the point is that the UNIQUE index -- not luck -- is what stops
    two distinct nodes from becoming one.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_collision_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    monkeypatch.setattr("graphify.query_backend._id_key", lambda _id: "collide")
    G = nx.DiGraph()
    G.add_node("first", label="first", source_file="a.py", source_location="L1", community=0)
    G.add_node("second", label="second", source_file="b.py", source_location="L2", community=0)
    p = tmp_path_factory.mktemp("collide") / "g.json"
    p.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    try:
        with pytest.raises(RuntimeError):
            arc.load_from_graph_json(str(p))
    finally:
        try:
            arc._server("drop database graphify_collision_test")
        except Exception:
            pass


def test_load_reports_shortfall(tmp_path_factory):
    """A load that does not fully arrive says so in its result, not just a log.

    The shortfall is produced the way the real one was: an edge whose endpoint
    the loader cannot resolve. CREATE EDGE builds each end from a subquery and an
    empty one makes it a silent no-op (CreateEdgesStep), which is how 30.1% of
    ERP2's edges vanished while the loader reported success. Here the endpoint is
    simply absent from the node list, so the case is reproduced without depending
    on the engine defect that first exposed it.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_shortfall_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    raw = json_graph.node_link_data(_make_digraph(), edges="links")
    raw["links"].append({"source": "n1", "target": "phantom_never_inserted",
                         "relation": "calls", "confidence": "INFERRED"})
    p = tmp_path_factory.mktemp("short") / "g.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    arc.ensure_database(drop=True)
    try:
        st = arc.load_from_graph_json(str(p))
        assert st["expected_edges"] == len(raw["links"])
        assert st["edges"] == len(raw["links"]) - 1, "the dangling edge must not be counted as written"
        assert st["missing_edges"] == 1
        assert st["missing_nodes"] == 0, "the nodes themselves all landed"
        assert st["unresolved_endpoint_edges"] == 1, \
            "the loss must be attributed to the unresolvable endpoint"
        warn = shortfall_warning(st)
        assert warn is not None and "1 of 4 edges" in warn and "endpoint" in warn
    finally:
        try:
            arc._server("drop database graphify_shortfall_test")
        except Exception:
            pass


def test_full_load_reports_no_shortfall(backends):
    """A load that fully arrives must not raise a false alarm."""
    _, arc = backends
    G = _make_digraph()
    st = arc._reconcile(G.number_of_nodes(), G.number_of_edges(),
                        node_ids=list(G.nodes()),
                        edges=[(u, v) for u, v in G.edges()])
    assert st["missing_nodes"] == 0 and st["missing_edges"] == 0
    assert st["nodes"] == G.number_of_nodes() and st["edges"] == G.number_of_edges()
    assert "unresolved_endpoint_edges" not in st, "attribution must not run when nothing is missing"
    assert shortfall_warning(st) is None


def test_sync_graph_reports_shortfall_the_same_way(tmp_path_factory):
    """sync_graph reconciles against G exactly as a full load does."""
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_sync_short_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g1 = _make_digraph()
    p = tmp_path_factory.mktemp("syncshort") / "g1.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        g2 = _make_digraph()
        g2.add_node("n6", label="helper", source_file="extract.py", source_location="L99", community=0)
        g2.add_edge("n6", "n2", relation="calls", confidence="INFERRED")
        st = arc.sync_graph(g2, changed_sources={"extract.py"}, pruned_sources=set())
        # the incremental keys the callers print stay in place
        assert {"deleted_sources", "upserted_nodes", "upserted_edges"} <= set(st)
        # and the reconciliation is present and clean for a sync that fully applied
        assert st["expected_nodes"] == g2.number_of_nodes()
        assert st["expected_edges"] == g2.number_of_edges()
        assert st["missing_nodes"] == 0 and st["missing_edges"] == 0
        assert shortfall_warning(st) is None
    finally:
        try:
            arc._server("drop database graphify_sync_short_test")
        except Exception:
            pass


def test_sync_refuses_a_database_predating_the_index_key(tmp_path_factory):
    """An incremental pass into an old-schema database must not run.

    Such a DB has no id_key, so sync would DELETE nodes by one key and INSERT
    them under another and leave the database half-converted - the shape that
    does not self-heal, because is_populated() stays true and the next run only
    patches the newly-changed files.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_oldschema_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    arc.ensure_database(drop=True)
    try:
        # The schema exactly as it was written before this change.
        for ddl in ("CREATE VERTEX TYPE Node", "CREATE EDGE TYPE Rel",
                    "CREATE PROPERTY Node.id STRING", "CREATE PROPERTY Node.norm_label STRING",
                    "CREATE PROPERTY Node.degree INTEGER", "CREATE INDEX ON Node (id) UNIQUE"):
            arc._run(ddl)
        arc._run("INSERT INTO Node CONTENT " + json.dumps(
            {"id": "n1", "label": "extract", "source_file": "extract.py", "degree": 0}))

        assert arc.schema_has_id_key() is False
        assert arc.is_populated() is True, "the old DB looks populated, which is why sync would run"

        with pytest.raises(RuntimeError) as exc:
            arc.sync_graph(_make_digraph(), changed_sources={"extract.py"}, pruned_sources=set())
        assert "arcade reload" in str(exc.value), "the message must name the way out"
    finally:
        try:
            arc._server("drop database graphify_oldschema_test")
        except Exception:
            pass


def test_current_schema_database_syncs_normally(tmp_path_factory):
    """A database written by this backend reports the key and syncs as before."""
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_newschema_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g1 = _make_digraph()
    p = tmp_path_factory.mktemp("newschema") / "g.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        assert arc.schema_has_id_key() is True
        g2 = _make_digraph()
        g2.nodes["n1"]["label"] = "extractor"
        st = arc.sync_graph(g2, changed_sources={"extract.py"}, pruned_sources=set())
        assert st["missing_nodes"] == 0 and st["missing_edges"] == 0
        assert render_node(arc.get_node("extractor"), "extractor") == \
            render_node(JsonBackend(g2).get_node("extractor"), "extractor")
    finally:
        try:
            arc._server("drop database graphify_newschema_test")
        except Exception:
            pass


def _make_unattributed_graph() -> nx.DiGraph:
    """A graph carrying both spellings of "this node belongs to no file".

    The semantic extractor emits concept/shared-symbol nodes with no source of
    their own: `source_file` is either the empty string or absent entirely. Both
    appear in the same real graph, so a fixture with only one of them lets half
    the defect through.
    """
    G = _make_digraph()
    G.add_node("c1", label="Exception", source_file="", source_location="", community=0)
    G.add_node("c2", label="TestCase", source_location="", community=0)  # no source_file at all
    G.add_edge("n1", "c1", relation="uses", confidence="INFERRED")
    G.add_edge("c2", "n2", relation="uses", confidence="INFERRED")
    return G


def test_sync_graph_inserts_nodes_that_belong_to_no_file(tmp_path_factory):
    """A node with no source_file must arrive through an incremental sync.

    File dirtiness cannot express such a node: the dirty-source DELETE does not
    remove it and the changed-source INSERT does not add it, so before this it
    could only ever reach the DB through a full load. Measured on a real corpus,
    that was 23.2% of the nodes and every edge with such an endpoint after it.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_unattributed_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g1 = _make_digraph()
    p = tmp_path_factory.mktemp("unattr") / "g1.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        g2 = _make_unattributed_graph()
        g2.nodes["n1"]["label"] = "extractor"
        st = arc.sync_graph(g2, changed_sources={"extract.py"}, pruned_sources=set())

        assert st["missing_nodes"] == 0, "nodes without a source_file must arrive"
        assert st["missing_edges"] == 0, "and the edges that end on them with them"
        assert shortfall_warning(st) is None
        # both spellings, not just the empty string
        assert arc.get_node("Exception") is not None
        assert arc.get_node("TestCase") is not None
        # the edge whose only unresolved end was such a node exists now
        jb = JsonBackend(g2)
        assert {n.label for n in jb.query("Exception", depth=2).nodes} == \
               {n.label for n in arc.query("Exception", depth=2).nodes}
    finally:
        try:
            arc._server("drop database graphify_unattributed_test")
        except Exception:
            pass


def test_sync_graph_removes_a_fileless_node_the_graph_dropped(tmp_path_factory):
    """The other direction: such a node must leave when the graph loses it.

    Until now the incremental path could not delete one either, so a concept that
    disappeared stayed in the database forever. Deleting is the risky half of the
    fix - it is the first time this path can remove a node no file names - so the
    edges incident to it are asserted gone too, not just the node.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_unattributed_drop_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g1 = _make_unattributed_graph()
    p = tmp_path_factory.mktemp("unattrdrop") / "g1.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        assert arc.get_node("Exception") is not None, "precondition: it was loaded"
        g2 = _make_unattributed_graph()
        g2.remove_node("c1")
        st = arc.sync_graph(g2, changed_sources={"extract.py"}, pruned_sources=set())

        assert arc.get_node("Exception") is None
        assert st["missing_nodes"] == 0 and st["missing_edges"] == 0, \
            "the DB must equal the graph, neither short nor over"
        assert render_stats(arc.graph_stats()) == render_stats(JsonBackend(g2).graph_stats())
    finally:
        try:
            arc._server("drop database graphify_unattributed_drop_test")
        except Exception:
            pass


def test_sync_reconciles_fileless_nodes_with_no_dirty_file(tmp_path_factory):
    """No file changed is not the same as nothing to do.

    The early return on an empty dirty set predates fileless nodes; with them a
    sync still has a set to reconcile, and returning early leaves the shortfall
    in place while reporting a clean run.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_unattributed_nodirty_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g1 = _make_digraph()
    p = tmp_path_factory.mktemp("unattrnodirty") / "g1.json"
    p.write_text(json.dumps(json_graph.node_link_data(g1, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        g2 = _make_unattributed_graph()
        st = arc.sync_graph(g2, changed_sources=set(), pruned_sources=set())
        assert arc.get_node("Exception") is not None
        assert st.get("missing_nodes") == 0 and st.get("missing_edges") == 0
    finally:
        try:
            arc._server("drop database graphify_unattributed_nodirty_test")
        except Exception:
            pass


def test_recorded_shortfall_survives_and_clears(tmp_path_factory):
    """A short load must be remembered, and a clean one must forget it.

    The gate that used to decide "patch or reload" was non-emptiness, which a
    database missing most of its nodes still satisfies - so the hole was patched
    around, run after run. Remembering the reconciliation closes that; forgetting
    it again matters just as much, or one shortfall condemns the project to a
    full load forever.
    """
    host, port = _parse_host_port(_URL)
    arc = ArcadeDBBackend("graphify_shortfall_flag_test", host=host, port=port, password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    g = _make_digraph()
    p = tmp_path_factory.mktemp("shortflag") / "g.json"
    p.write_text(json.dumps(json_graph.node_link_data(g, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(p))
    try:
        assert arc.has_recorded_shortfall() is False, "a load that added up records nothing"

        arc.record_reconciliation({"missing_nodes": 15229, "missing_edges": 33391})
        assert arc.has_recorded_shortfall() is True

        # a reconciliation that could not run at all counts as a shortfall too
        arc.record_reconciliation({"reconciled": False})
        assert arc.has_recorded_shortfall() is True

        arc.record_reconciliation({"missing_nodes": 0, "missing_edges": 0})
        assert arc.has_recorded_shortfall() is False, "the marker must not stick"

        # and a real sync clears it the same way
        arc.record_reconciliation({"missing_nodes": 1, "missing_edges": 0})
        arc.sync_graph(_make_digraph(), changed_sources={"extract.py"}, pruned_sources=set())
        assert arc.has_recorded_shortfall() is False
    finally:
        try:
            arc._server("drop database graphify_shortfall_flag_test")
        except Exception:
            pass


def test_shortfall_warning_names_the_right_cause():
    """Missing nodes and an index miss wear the same symptom but want opposite fixes."""
    absent = shortfall_warning({"expected_nodes": 65397, "expected_edges": 130523,
                                "missing_nodes": 15229, "missing_edges": 33391,
                                "unresolved_endpoint_edges": 33392})
    assert "not in the database" in absent
    assert "index" not in absent, "naming the index sends the reader to the wrong defect"

    index_miss = shortfall_warning({"expected_nodes": 4, "expected_edges": 4,
                                    "missing_nodes": 0, "missing_edges": 1,
                                    "unresolved_endpoint_edges": 1})
    assert "index does not find" in index_miss
    assert "node count matches" in index_miss
