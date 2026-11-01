"""Upstream 0.9.97 lookup rules answer the same on ArcadeDB as on JSON.

Each lookup kind here is a case `serve._resolve_single_node` /
`_resolve_path_endpoint` refuse or resolve on the networkx graph: same-file and
cross-file name ties, source-less stubs, a substring near miss (#3917), dotted
Python symbols (#4366), same-named files of two repos (#4512), `path::symbol`
and node-id endpoints (#3913), a `file::symbol` miss (#4264) and the full-token
endpoint pick (#1785). Skipped when no ArcadeDB answers.
"""
import json
import os

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from graphify import arcade_server
from graphify.query_backend import (
    ArcadeDBBackend, JsonBackend, Neighbors, NodeDetail,
    Refusal, render_explain, render_neighbors, render_node, render_path, render_query,
)

_MACHINE = arcade_server.read_machine_config()
_URL = os.environ.get("GRAPHIFY_ARCADE_URL") or _MACHINE.get("url") or "http://127.0.0.1:2480"
_PW = os.environ.get("GRAPHIFY_ARCADE_PASSWORD") or _MACHINE.get("password") or ""
_DB = "graphify_lookup_parity_test"


def _graph() -> nx.MultiDiGraph:
    # Multi, as the `path` CLI loads it: a pair may carry several relations.
    G = nx.MultiDiGraph()

    def node(nid, label, sf="", loc="", **kw):
        G.add_node(nid, label=label, source_file=sf, source_location=loc, **kw)

    def edge(u, v, rel):
        G.add_edge(u, v, relation=rel, confidence="EXTRACTED")

    # One file, two classes, a `.nonce()` on each: a same-file tie.
    node("src_x", "x.ts", "src/x.ts", "L1")
    node("src_x_run", "run()", "src/x.ts", "L3")
    node("src_x_alphaclient", "AlphaClient", "src/x.ts", "L5")
    node("src_x_alphaclient_nonce", ".nonce()", "src/x.ts", "L6")
    node("src_x_betaclient", "BetaClient", "src/x.ts", "L9")
    node("src_x_betaclient_nonce", ".nonce()", "src/x.ts", "L10")
    edge("src_x", "src_x_run", "contains")
    edge("src_x_run", "src_x_alphaclient", "calls")
    edge("src_x_run", "src_x_betaclient", "calls")
    edge("src_x_alphaclient", "src_x_alphaclient_nonce", "method")
    edge("src_x_betaclient", "src_x_betaclient_nonce", "method")
    # The same name in two files: a cross-file tie.
    node("src_a_close", ".close()", "src/a.ts", "L2")
    node("src_b_close", ".close()", "src/b.ts", "L2")
    # Two source-less stubs of one external interface.
    node("tests_a_php_emi", "EntityManagerInterface")
    node("tests_b_php_emi", "EntityManagerInterface")
    # A substring hit only: `mailcontroller` sits inside this label.
    node("app_emailcontroller", "EmailController", "app/email.py", "L4")
    # Dotted Python: a function and a same-named method of one module tie; a
    # src-layout module resolves by suffix.
    node("pkg_mod_start", "start()", "pkg/mod.py", "L1")
    node("pkg_mod_cls_start", ".start()", "pkg/mod.py", "L9")
    node("src_pkg2_util_helper", "helper()", "src/pkg2/util.py", "L3")
    # The same path in two repos of a merged graph.
    node("r1::dup_f", "f()", "dup.py", "L1", repo="r1")
    node("r2::dup_f", "f()", "dup.py", "L1", repo="r2")
    # `readRoute` in two files, none in admin/route.ts (#4264).
    node("src_billing_route", "route.ts", "src/billing/route.ts", "L1")
    node("src_billing_route_readroute", "readRoute()", "src/billing/route.ts", "L3")
    node("src_seats_route", "route.ts", "src/seats/route.ts", "L1")
    node("src_seats_route_readroute", "readRoute()", "src/seats/route.ts", "L3")
    node("src_admin_route", "route.ts", "src/admin/route.ts", "L1")
    node("src_admin_route_listadmins", "listAdmins()", "src/admin/route.ts", "L3")
    node("src_billing_charge_chargecard", "chargeCard()", "src/billing/charge.ts", "L2")
    edge("src_billing_route", "src_billing_route_readroute", "contains")
    edge("src_seats_route", "src_seats_route_readroute", "contains")
    edge("src_admin_route", "src_admin_route_listadmins", "contains")
    edge("src_billing_route_readroute", "src_billing_charge_chargecard", "calls")
    # #3878: alpha.py imports a symbol that beta.py contains; no file-to-file edge.
    node("f_alpha", "alpha.py", "alpha.py", "L1")
    node("f_beta", "beta.py", "beta.py", "L1")
    node("f_beta_shared", "sharedThing()", "beta.py", "L3")
    edge("f_alpha", "f_beta_shared", "imports")
    edge("f_beta", "f_beta_shared", "contains")
    # #2074: two routes of equal length, the larger id inserted first so graph
    # order and id order disagree; both backends must print d_mid_a.
    node("d_start", "diamondStart()", "d.py", "L1")
    node("d_mid_b", "midB()", "d.py", "L2")
    node("d_mid_a", "midA()", "d.py", "L3")
    node("d_end", "diamondEnd()", "d.py", "L4")
    edge("d_start", "d_mid_b", "calls")
    edge("d_mid_b", "d_end", "calls")
    edge("d_start", "d_mid_a", "calls")
    edge("d_mid_a", "d_end", "calls")
    # Two relations on one pair: the segment names both.
    node("p_caller", "pairCaller()", "p.py", "L1")
    node("p_callee", "pairCallee()", "p.py", "L2")
    edge("p_caller", "p_callee", "calls")
    G.add_edge("p_caller", "p_callee", relation="references", confidence="INFERRED")
    # #1785: IDF ranks the decoy above the full-token match.
    node("target", "Degenerate Reject-Everything Judge")
    node("decoy", "Rejection Summary")
    for i in range(30):
        node(f"j{i}", f"Judge Helper {i}")
        node(f"e{i}", f"Everything Widget {i}")
    edge("target", "j0", "verified_by")
    edge("decoy", "e0", "mentions")
    return G


@pytest.fixture(scope="module")
def backends(tmp_path_factory):
    rest = _URL.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    arc = ArcadeDBBackend(_DB, host=host or "127.0.0.1", port=int(port or 2480), password=_PW)
    if not arc.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    G = _graph()
    path = tmp_path_factory.mktemp("graph") / "graph.json"
    path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    arc.ensure_database(drop=True)
    arc.load_from_graph_json(str(path))
    yield JsonBackend(G), arc
    try:
        arc._server(f"drop database {_DB}")
    except Exception:
        pass


def _shape(result, render):
    """What both backends must agree on.

    A refusal that lists candidates is compared by kind and candidate set: the
    order follows graph iteration on JSON and id order on ArcadeDB. Any other
    refusal, and every answer, is compared as rendered text.
    """
    if isinstance(result, Refusal):
        lines = [l.strip() for l in result.message.splitlines()]
        if "in different files" in lines[0]:
            # One representative node per file: the first in tier order, which is
            # graph order on JSON and id order on ArcadeDB. The files must agree.
            listed = sorted(l for l in lines[1:-1] if not l.startswith("id:"))
        else:
            listed = sorted(l for l in lines if l.startswith("id:"))
        return ("refused", result.ambiguous, listed or result.message)
    return render(result)


_LABELS = [
    ".nonce()", "src/x.ts::.nonce()", "src/x.ts", "src_x_betaclient_nonce",
    ".close()", "EntityManagerInterface", "mailcontroller", "EmailController",
    "pkg.mod.start", "pkg2.util.helper", "pkg2.util:helper", "f()",
    "NoSuchSymbolXyz", "src/a.ts::NotThere",
]

_EXPECTED_KIND = {
    ".nonce()": "ambiguous", "src/x.ts::.nonce()": "ambiguous", ".close()": "ambiguous",
    "EntityManagerInterface": "ambiguous", "pkg.mod.start": "ambiguous", "f()": "ambiguous",
    "mailcontroller": "miss", "NoSuchSymbolXyz": "miss", "src/a.ts::NotThere": "miss",
}


@pytest.mark.parametrize("label", _LABELS)
def test_explain_and_get_node_agree(backends, label):
    jb, arc = backends
    j, a = jb.explain(label), arc.explain(label)
    assert _shape(j, lambda r: render_explain(r, label)) == _shape(a, lambda r: render_explain(r, label))
    kind = _EXPECTED_KIND.get(label, "answer")
    assert kind == ("answer" if not isinstance(a, Refusal) else "ambiguous" if a.ambiguous else "miss")
    j, a = jb.get_node(label), arc.get_node(label)
    assert _shape(j, lambda r: render_node(r, label)) == _shape(a, lambda r: render_node(r, label))


@pytest.mark.parametrize("label", [".nonce()", "src/x.ts::.nonce()", "mailcontroller", "src_x_betaclient"])
def test_get_neighbors_agree(backends, label):
    jb, arc = backends
    def render(n):
        return sorted(render_neighbors(n, label).splitlines())
    assert _shape(jb.get_neighbors(label), render) == _shape(arc.get_neighbors(label), render)


def test_near_miss_names_the_substring_hit(backends):
    _, arc = backends
    r = arc.get_node("mailcontroller")
    assert isinstance(r, Refusal) and not r.ambiguous
    assert r.message.startswith("No exact match for 'mailcontroller'.")
    assert "EmailController" in r.message


def test_repo_names_the_two_same_path_files(backends):
    _, arc = backends
    r = arc.explain("f()")
    assert isinstance(r, Refusal) and r.ambiguous
    assert "r1/dup.py" in r.message and "r2/dup.py" in r.message


_PATHS = [
    ("run()", ".nonce()"),
    (".nonce()", "run()"),
    ("run()", "src_x_betaclient_nonce"),
    ("run()", "src/x.ts::.nonce()"),
    ("src/x.ts", "src_x_betaclient_nonce"),
    ("src/admin/route.ts::readRoute", "chargeCard"),
    ("src/billing/route.ts::readRoute", "chargeCard"),
    ("Reject-everything judge", "Judge Helper 0"),
    ("pkg.mod.start", "run()"),
    ("alpha.py", "beta.py"),
    ("beta.py", "alpha.py"),
    ("diamondStart()", "diamondEnd()"),
    ("diamondEnd()", "diamondStart()"),
]


@pytest.mark.parametrize("ends", _PATHS)
def test_shortest_path_agrees(backends, ends):
    jb, arc = backends
    src, tgt = ends
    def render(p):
        return render_path(p, src, tgt)
    j = jb.shortest_path(src, tgt, max_hops=10**9)
    a = arc.shortest_path(src, tgt, max_hops=10**9)
    assert _shape(j.refusal or j, render) == _shape(a.refusal or a, render)


@pytest.mark.parametrize("question,refused", [
    (".nonce()", True), (".close()", True), ("EntityManagerInterface", True),
    # A keyword that only prefix/substring-matches labels is a search (#4517).
    # `nonce` is refused: it equals the label tokens of `.nonce()`, an exact tier.
    ("client", False), ("alpha", False), ("nonce", True), ("readRoute", True),
])
def test_query_refuses_an_exact_name_tie_like_json(backends, question, refused):
    jb, arc = backends
    j, a = jb.query(question, depth=1), arc.query(question, depth=1)
    assert (j.refusal is not None, a.refusal is not None) == (refused, refused)
    if refused:
        assert _shape(j.refusal, None) == _shape(a.refusal, None)
        assert j.nodes == a.nodes == []


def test_path_walks_contains_back_to_the_file(backends):
    """#3878 on ArcadeDB: the route closes through the file's `contains` edge."""
    _, arc = backends
    out = render_path(arc.shortest_path("alpha.py", "beta.py"), "alpha.py", "beta.py")
    assert "alpha.py --imports [EXTRACTED]--> sharedThing() <--contains [EXTRACTED]-- beta.py" in out


@pytest.mark.parametrize("undirected", [False, True])
def test_equal_routes_pick_the_smallest_ids(backends, undirected):
    jb, arc = backends
    for b in backends:
        p = b.shortest_path("diamondStart()", "diamondEnd()", undirected=undirected)
        assert p.found and [s.label for s in p.segments] == ["midA()", "diamondEnd()"]


@pytest.mark.parametrize("undirected", [False, True])
def test_parallel_relations_share_one_segment(backends, undirected):
    for b in backends:
        p = b.shortest_path("pairCaller()", "pairCallee()", undirected=undirected)
        assert p.found and p.segments[0].relation == "calls/references"
        assert p.segments[0].confidence == "EXTRACTED/INFERRED" and p.segments[0].outgoing


def test_query_edge_at_suffix_parity(backends):
    """EDGE lines carry `at=<file>:<loc>` when the edge has source_location."""
    jb, arc = backends
    def edge_lines(r):
        return sorted(l for l in render_query(r).splitlines() if l.startswith("EDGE "))
    j = jb.query("run()", depth=2)
    a = arc.query("run()", depth=2)
    assert edge_lines(j) == edge_lines(a)


def test_neighbor_at_suffix_parity(backends):
    """Neighbor lines carry `at=` when the edge has source_location."""
    jb, arc = backends
    def neighbor_lines(r):
        return sorted(l.strip() for l in render_neighbors(r, "run()").splitlines()
                      if l.strip().startswith(("-->", "<--")))
    j = jb.get_neighbors("run()")
    a = arc.get_neighbors("run()")
    assert isinstance(j, Neighbors) and isinstance(a, Neighbors)
    assert neighbor_lines(j) == neighbor_lines(a)


def test_parallel_edge_renders_two_lines(backends):
    """pairCaller–pairCallee carries `calls` and `references`; both must appear."""
    jb, arc = backends
    for label, b in (("pairCaller()", jb), ("pairCaller()", arc)):
        n = b.get_neighbors(label)
        assert isinstance(n, Neighbors)
        rels = {r.relation for r in n.neighbors}
        assert "calls" in rels and "references" in rels, f"{label} on {b!r}: {rels}"


def test_full_token_endpoint_wins_on_arcadedb(backends):
    """#1785 on ArcadeDB: scored[0] alone anchored on the disconnected decoy."""
    _, arc = backends
    p = arc.shortest_path("Reject-everything judge", "Judge Helper 0")
    assert p.found and p.hops == 1


def test_scoped_miss_lists_the_defining_files(backends):
    _, arc = backends
    p = arc.shortest_path("src/admin/route.ts::readRoute", "chargeCard")
    assert p.refusal is not None and not p.refusal.ambiguous
    assert "src/billing/route.ts" in p.refusal.message
    assert "src/seats/route.ts" in p.refusal.message
