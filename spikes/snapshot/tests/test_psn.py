"""Tasks 2.2-2.6 and 4.5 on psn_diadoc (40 988 nodes, 96 315 links)."""
import json

import pytest
from conftest import PSN

import query_set
import ref_loaders
from graphify.query_backend import JsonBackend, render_explain, render_path, render_query
from graphify.serve import _filter_graph_by_context, _query_terms, _score_query


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


@pytest.fixture(scope="module")
def graphs(psn_data):
    return {
        "query": ref_loaders.query_graph(psn_data),
        "explain": ref_loaders.explain_graph(psn_data),
        "path": ref_loaders.path_graph(psn_data),
        "path_u": ref_loaders.path_graph(psn_data, undirected=True),
        "mcp": ref_loaders.mcp_graph(psn_data),
    }


@pytest.fixture(scope="module")
def qset(psn_data, graphs):
    gods = [g.label for g in JsonBackend(graphs["mcp"]).god_nodes(top_n=10)]
    return query_set.build(psn_data, PSN, gods)


# ---- 2.2 / 2.3: lossless round trip ---------------------------------------- #

def test_every_node_round_trips(psn_data, psn_snap):
    sb = psn_snap
    assert sb.snap.node_count == len(psn_data["nodes"])
    for i, d in enumerate(psn_data["nodes"]):
        assert sb._id(i) == d["id"]
        assert canon(sb._node(i)) == canon({k: v for k, v in d.items() if k != "id"}), d["id"]


def test_every_link_round_trips(psn_data, psn_snap):
    sb = psn_snap
    links = psn_data["links"]
    assert sb.snap.link_count == len(links)
    for j, l in enumerate(links):
        s, t = sb.snap.link_endpoints(j)
        assert (sb._id(s), sb._id(t)) == (l["source"], l["target"])
        want = {k: v for k, v in l.items() if k not in ("source", "target")}
        assert canon(sb._link(j, markers=False)) == canon(want), j


def test_graph_level_keys_round_trip(psn_data, psn_snap):
    stored = json.loads(psn_snap.snap.block_text("graph.json"))
    assert canon(stored) == canon({k: v for k, v in psn_data.items() if k not in ("nodes", "links")})


# ---- 2.4: networkx order and degree ---------------------------------------- #

def test_traversal_neighbours_match_networkx(graphs, psn_snap):
    G, sb = graphs["query"], psn_snap
    for i, nid in enumerate(G.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i)] == list(G.neighbors(nid)), nid
        assert sb.snap.tdegree(i) == G.degree(nid), nid


@pytest.mark.parametrize("filters", [["call"], ["new"], ["call", "new"], ["nope"]])
def test_filtered_neighbours_match_networkx(graphs, psn_snap, filters):
    H = _filter_graph_by_context(graphs["query"], filters)
    sb = psn_snap
    codes = [sb._ctx[f] for f in filters if f in sb._ctx]
    for i, nid in enumerate(H.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i, codes)] == list(H.neighbors(nid)), nid
        assert sb.snap.tdegree(i, codes) == H.degree(nid), nid
    from graphify.serve import _hub_threshold
    assert sb.snap.hub_threshold(codes) == _hub_threshold([H.degree(n) for n in H.nodes()])


def test_directed_view_matches_networkx(graphs, psn_snap):
    G, sb = graphs["explain"], psn_snap
    for i, nid in enumerate(G.nodes()):
        assert [sb._id(sb.snap.link_endpoints(j)[1]) for j in sb.snap.out_links(i)] == list(G.successors(nid))
        assert [sb._id(sb.snap.link_endpoints(j)[0]) for j in sb.snap.in_links(i)] == list(G.predecessors(nid))
        assert sb.snap.deg_dir(i) == G.degree(nid)


# ---- 2.5: candidates and scores -------------------------------------------- #

def test_scores_match_serve(graphs, psn_snap, qset):
    G, sb = graphs["query"], psn_snap
    probes = qset["query"] + qset["explain"] + ["form", "модуль", "контрагент", "получить", "xyzzy"]
    for q in probes:
        terms = _query_terms(q)
        want = _score_query(G, terms, collect_per_term_seeds=True)
        got = sb._score_query(terms, collect_per_term_seeds=True)
        assert list(got.ranked) == want.ranked, q
        assert got.best_seed_by_term == want.best_seed_by_term, q
        assert got.term_order == want.term_order, q


# ---- 2.6: stats -------------------------------------------------------------- #

def test_stats_and_god_nodes(graphs, psn_snap):
    jb = JsonBackend(graphs["mcp"])
    assert psn_snap.graph_stats() == jb.graph_stats()
    for k in (1, 10, 100):
        assert psn_snap.god_nodes(top_n=k) == jb.god_nodes(top_n=k)


# ---- 4.5: real queries, byte for byte ---------------------------------------- #

def test_real_queries(graphs, psn_snap, qset):
    jb = JsonBackend(graphs["query"])
    for q in qset["query"]:
        for mode in ("bfs", "dfs"):
            want = render_query(jb.query(q, mode=mode, depth=2), token_budget=2000)
            assert render_query(psn_snap.query(q, mode=mode, depth=2), token_budget=2000) == want, (q, mode)


def test_real_context_filtered_queries(graphs, psn_snap, qset):
    jb = JsonBackend(graphs["query"])
    for q, f in qset["context"]:
        want = render_query(jb.query(q, depth=2, context_filters=f), token_budget=2000)
        assert render_query(psn_snap.query(q, depth=2, context_filters=f), token_budget=2000) == want, (q, f)


def test_real_paths(graphs, psn_snap, qset):
    for key, undirected in (("path", False), ("path_u", True)):
        jb = JsonBackend(graphs[key])
        for a, b in qset["path"]:
            want = render_path(jb.shortest_path(a, b, max_hops=10**9, undirected=undirected), a, b)
            got = render_path(psn_snap.shortest_path(a, b, max_hops=10**9, undirected=undirected), a, b)
            assert got == want, (a, b, undirected)


def test_real_explains(graphs, psn_snap, qset):
    jb = JsonBackend(graphs["explain"])
    for label in qset["explain"] + qset["query"][:10]:
        assert render_explain(psn_snap.explain(label), label) == render_explain(jb.explain(label), label), label
