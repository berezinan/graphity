'''Parity on psn_diadoc (40 988 nodes, 96 315 links): SnapBackend answers byte for byte like JsonBackend (6.4).'''
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from networkx.readwrite import json_graph

PSN = Path('C:/_dev/ANB/AI/psn_diadoc_4_67_3_Pravka/graphify-out/graph.json')
pytestmark = pytest.mark.skipif(not PSN.exists(), reason='psn_diadoc graph.json not present')
pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

from graphify.build import has_parallel_links  # noqa: E402
from graphify.query_backend import JsonBackend, render_explain, render_path, render_query  # noqa: E402
from graphify.serve import _filter_graph_by_context, _query_terms, _score_query  # noqa: E402
from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_writer import publish_snapshot  # noqa: E402

SEED = 20261103


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


# The networkx graphs the CLI builds from graph.json, per command.
def _links(raw: dict) -> dict:
    raw = dict(raw, graph=dict(raw.get('graph') or {}))
    if 'links' not in raw and 'edges' in raw:
        raw = dict(raw, links=raw['edges'])
    return raw


def _markers(raw: dict) -> dict:
    return dict(raw, links=[{**l, '_src': l.get('_src', l.get('source')), '_tgt': l.get('_tgt', l.get('target'))}
                            for l in raw.get('links', [])])


def query_graph(raw):
    return json_graph.node_link_graph(_markers(_links(raw)), edges='links')


def path_graph(raw, undirected=False):
    raw = _markers(_links(raw))
    return json_graph.node_link_graph({**raw, 'directed': not undirected, 'multigraph': True}, edges='links')


def explain_graph(raw):
    raw = _links(raw)
    multi = has_parallel_links(raw.get('links'), directed=True)
    return json_graph.node_link_graph({**raw, 'directed': True, **({'multigraph': True} if multi else {})},
                                      edges='links')


def mcp_graph(raw):
    '''serve._read_graph without the file I/O.'''
    raw = _links(raw)
    data = {**raw, 'directed': True}
    if has_parallel_links(data.get('links'), directed=True):
        data = {**data, 'multigraph': True}
    G = json_graph.node_link_graph(data, edges='links')
    commit = raw.get('built_at_commit')
    if isinstance(commit, str) and commit.strip():
        G.graph['_built_at_commit'] = commit.strip()
    return G


def build_query_set(labels: list, god_labels: list) -> dict:
    '''Fixed, seeded query set.'''
    rnd = random.Random(SEED)
    labels = [l for l in labels if isinstance(l, str) and l.strip()]
    pick = lambda k: rnd.sample(labels, min(k, len(labels)))
    flt = (['call'], ['new'], ['call', 'new'], ['form-handler'], ['metadata'])
    return {
        'query': god_labels + pick(30),
        'path': [tuple(pick(2)) for _ in range(30)],
        'explain': pick(30),
        'context': list(zip(pick(5), flt)),
    }


@pytest.fixture(scope='module')
def psn_data():
    with open(PSN, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def psn_snap(psn_data, tmp_path_factory):
    # Built into a temporary directory: the project's graphify-out is never touched.
    d = tmp_path_factory.mktemp('psn')
    publish_snapshot(psn_data, d)
    return SnapBackend(d / 'graph.snap')


@pytest.fixture(scope='module')
def graphs(psn_data):
    return {
        'query': query_graph(psn_data),
        'explain': explain_graph(psn_data),
        'path': path_graph(psn_data),
        'path_u': path_graph(psn_data, undirected=True),
        'mcp': mcp_graph(psn_data),
    }


@pytest.fixture(scope='module')
def qset(psn_data, graphs):
    gods = [g.label for g in JsonBackend(graphs['mcp']).god_nodes(top_n=10)]
    return build_query_set([n.get('label') for n in psn_data['nodes']], gods)


# ---- lossless round trip --------------------------------------------------- #

def test_every_node_round_trips(psn_data, psn_snap):
    sb = psn_snap
    assert sb.snap.node_count == len(psn_data['nodes'])
    for i, d in enumerate(psn_data['nodes']):
        assert sb._id(i) == d['id']
        assert canon(sb._node(i)) == canon({k: v for k, v in d.items() if k != 'id'}), d['id']


def test_every_link_round_trips(psn_data, psn_snap):
    sb = psn_snap
    links = psn_data['links']
    assert sb.snap.link_count == len(links)
    for j, l in enumerate(links):
        s, t = sb.snap.link_endpoints(j)
        assert (sb._id(s), sb._id(t)) == (l['source'], l['target'])
        want = {k: v for k, v in l.items() if k not in ('source', 'target')}
        assert canon(sb._link(j, markers=False)) == canon(want), j


def test_graph_level_keys_round_trip(psn_data, psn_snap):
    stored = json.loads(psn_snap.snap.block_text('graph.json'))
    assert canon(stored) == canon({k: v for k, v in psn_data.items() if k not in ('nodes', 'links')})


# ---- networkx order and degree --------------------------------------------- #

def test_traversal_neighbours_match_networkx(graphs, psn_snap):
    G, sb = graphs['query'], psn_snap
    for i, nid in enumerate(G.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i)] == list(G.neighbors(nid)), nid
        assert sb.snap.tdegree(i) == G.degree(nid), nid


@pytest.mark.parametrize('filters', [['call'], ['new'], ['call', 'new'], ['nope']])
def test_filtered_neighbours_match_networkx(graphs, psn_snap, filters):
    from graphify.serve import _hub_threshold
    H = _filter_graph_by_context(graphs['query'], filters)
    sb = psn_snap
    codes = [sb._ctx[f] for f in filters if f in sb._ctx]
    for i, nid in enumerate(H.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i, codes)] == list(H.neighbors(nid)), nid
        assert sb.snap.tdegree(i, codes) == H.degree(nid), nid
    assert sb.snap.hub_threshold(codes) == _hub_threshold([H.degree(n) for n in H.nodes()])


def test_directed_view_matches_networkx(graphs, psn_snap):
    G, sb = graphs['explain'], psn_snap
    for i, nid in enumerate(G.nodes()):
        assert [sb._id(sb.snap.link_endpoints(j)[1]) for j in sb.snap.out_links(i)] == list(G.successors(nid))
        assert [sb._id(sb.snap.link_endpoints(j)[0]) for j in sb.snap.in_links(i)] == list(G.predecessors(nid))
        assert sb.snap.deg_dir(i) == G.degree(nid)


# ---- candidates, scores, tiers --------------------------------------------- #

def test_scores_match_serve(graphs, psn_snap, qset):
    G, sb = graphs['query'], psn_snap
    probes = qset['query'] + qset['explain'] + ['form', 'модуль', 'контрагент', 'получить', 'xyzzy']
    for q in probes:
        terms = _query_terms(q)
        want = _score_query(G, terms, collect_per_term_seeds=True)
        got = sb._score_query(terms, collect_per_term_seeds=True)
        assert list(got.ranked) == want.ranked, q
        assert got.best_seed_by_term == want.best_seed_by_term, q
        assert got.term_order == want.term_order, q


def test_tiers_match_find_node_tiers(graphs, psn_snap, qset):
    from graphify.serve import _find_node_tiers
    G, sb = graphs['query'], psn_snap
    labels = qset['explain'] + qset['query'] + [
        'Module.bsl', 'Form', 'ФормаУправляемая', 'x', '()', 'ПолучитьЗначениеСвойства',
        'Catalog.ДополнительныеОтчетыИОбработки', 'form.form', 'src']
    for label in labels:
        if sb._special(label):
            continue
        want = [list(t) for t in _find_node_tiers(G, label)]
        got = [sb._ids_of(t) for t in sb._tiers(label, 2)]
        assert got == want, label


# ---- stats ------------------------------------------------------------------ #

def test_stats_and_god_nodes(graphs, psn_snap):
    jb = JsonBackend(graphs['mcp'])
    assert psn_snap.graph_stats() == jb.graph_stats()
    # 500 is past the stored list: the walk over the nodes by degree (D9) must agree.
    for k in (1, 10, 100, 500):
        assert psn_snap.god_nodes(top_n=k) == jb.god_nodes(top_n=k), k


# ---- real queries, byte for byte ------------------------------------------- #

def test_real_queries(graphs, psn_snap, qset):
    jb = JsonBackend(graphs['query'])
    for q in qset['query']:
        for mode in ('bfs', 'dfs'):
            want = render_query(jb.query(q, mode=mode, depth=2), token_budget=2000)
            assert render_query(psn_snap.query(q, mode=mode, depth=2), token_budget=2000) == want, (q, mode)


def test_real_context_filtered_queries(graphs, psn_snap, qset):
    jb = JsonBackend(graphs['query'])
    for q, f in qset['context']:
        want = render_query(jb.query(q, depth=2, context_filters=f), token_budget=2000)
        assert render_query(psn_snap.query(q, depth=2, context_filters=f), token_budget=2000) == want, (q, f)


def test_real_paths(graphs, psn_snap, qset):
    for key, undirected in (('path', False), ('path_u', True)):
        jb = JsonBackend(graphs[key])
        for a, b in qset['path']:
            want = render_path(jb.shortest_path(a, b, max_hops=10**9, undirected=undirected), a, b)
            got = render_path(psn_snap.shortest_path(a, b, max_hops=10**9, undirected=undirected), a, b)
            assert got == want, (a, b, undirected)


def test_real_explains(graphs, psn_snap, qset):
    jb = JsonBackend(graphs['explain'])
    for label in qset['explain'] + qset['query'][:10]:
        assert render_explain(psn_snap.explain(label), label) == render_explain(jb.explain(label), label), label
