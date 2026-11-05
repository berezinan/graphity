'''Golden cases of tests/test_query_backend.py on JsonBackend and SnapBackend: byte-equal output (6.1).'''
from __future__ import annotations

import json

import networkx as nx
import pytest
from networkx.readwrite import json_graph

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

from graphify.query_backend import (  # noqa: E402
    JsonBackend,
    render_community,
    render_explain,
    render_god_nodes,
    render_neighbors,
    render_node,
    render_path,
    render_query,
    render_stats,
)
from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_store import snapshot_path_for  # noqa: E402
from graphify.snap_writer import build_snapshot  # noqa: E402


def _make_digraph() -> nx.DiGraph:
    # Copy of tests/test_query_backend.py::_make_digraph.
    G = nx.DiGraph()
    G.add_node('n1', label='extract', source_file='extract.py', source_location='L10', community=0)
    G.add_node('n2', label='cluster', source_file='cluster.py', source_location='L5', community=0)
    G.add_node('n3', label='build', source_file='build.py', source_location='L1', community=1)
    G.add_node('n4', label='report', source_file='report.py', source_location='L1', community=1)
    G.add_node('n5', label='isolated', source_file='other.py', source_location='L1', community=2)
    G.add_edge('n1', 'n2', relation='calls', confidence='INFERRED', context='call')
    G.add_edge('n2', 'n3', relation='imports', confidence='EXTRACTED', context='import')
    G.add_edge('n3', 'n4', relation='uses', confidence='EXTRACTED')
    return G


def snap_of(G, tmp_path):
    '''Write graph.json of G, publish its snapshot and open it (fresh by size and mtime).'''
    data = json_graph.node_link_data(G, edges='links')
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    build_snapshot(p)
    return SnapBackend(snapshot_path_for(p), graph_json=p)


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    G = _make_digraph()
    return JsonBackend(G), snap_of(G, tmp_path_factory.mktemp('golden'))


@pytest.mark.parametrize('question', ['extract', 'build', 'cluster calls', 'xyzzy', 'report uses'])
@pytest.mark.parametrize('mode', ['bfs', 'dfs'])
@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('budget', [2000, 5])
def test_query(pair, question, mode, depth, budget):
    jb, sb = pair
    want = render_query(jb.query(question, mode=mode, depth=depth), token_budget=budget)
    assert render_query(sb.query(question, mode=mode, depth=depth), token_budget=budget) == want


def test_query_explicit_context_filter(pair):
    jb, sb = pair
    want = render_query(jb.query('extract', context_filters=['call']), token_budget=2000)
    assert render_query(sb.query('extract', context_filters=['call']), token_budget=2000) == want


@pytest.mark.parametrize('seeds', [['n1'], ['extract'], ['extract.py'], ['clust'], ['n1', 'build'], ['zzznope']])
def test_query_with_seeds(pair, seeds):
    jb, sb = pair
    want = render_query(jb.query('x', seeds=seeds), token_budget=2000)
    assert render_query(sb.query('x', seeds=seeds), token_budget=2000) == want


@pytest.mark.parametrize('call', [
    lambda b: render_query(b.query('zzzznope')),
    lambda b: render_node(b.get_node('extract'), 'extract'),
    lambda b: render_node(b.get_node('nope'), 'nope'),
    lambda b: render_neighbors(b.get_neighbors('build'), 'build'),
    lambda b: render_neighbors(b.get_neighbors('build', relation_filter='uses'), 'build'),
    lambda b: render_community(b.get_community(0), 0),
    lambda b: render_community(b.get_community(99), 99),
    lambda b: render_stats(b.graph_stats()),
    lambda b: render_path(b.shortest_path('extract', 'report'), 'extract', 'report'),
    lambda b: render_path(b.shortest_path('report', 'extract'), 'report', 'extract'),
    lambda b: render_path(b.shortest_path('report', 'extract', undirected=True), 'report', 'extract'),
    lambda b: render_path(b.shortest_path('extract', 'extract'), 'extract', 'extract'),
    lambda b: render_path(b.shortest_path('extract', 'isolated'), 'extract', 'isolated'),
    lambda b: render_god_nodes(b.god_nodes(top_n=3)),
    lambda b: render_god_nodes(b.god_nodes(top_n=500)),
    lambda b: render_explain(b.explain('cluster'), 'cluster'),
    lambda b: render_explain(b.explain('zzznope'), 'zzznope'),
    lambda b: b.pr_impact(['build.py', 'cluster.py']),
], ids=['no-match', 'get-node', 'get-node-missing', 'neighbors', 'neighbors-filter',
        'community', 'community-missing', 'stats', 'path', 'path-reverse', 'path-undirected',
        'path-same', 'path-none', 'god-nodes', 'god-nodes-all', 'explain', 'explain-miss', 'pr-impact'])
def test_lookup(pair, call):
    jb, sb = pair
    assert call(sb) == call(jb)
