'''SnapBackend: --seed through affected.resolve_seed (D8) and god_nodes beyond the stored list (D9).'''
from __future__ import annotations

import json

import networkx as nx
import pytest

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

from graphify.analyze import god_nodes  # noqa: E402
from graphify.paths import load_node_link_graph  # noqa: E402
from graphify.query_backend import JsonBackend  # noqa: E402
from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_store import snapshot_path_for  # noqa: E402
from graphify.snap_writer import build_snapshot  # noqa: E402


def _node(nid, label, src, loc='L1', community=0):
    return {'id': nid, 'label': label, 'source_file': src, 'source_location': loc, 'community': community}


def _link(a, b):
    return {'source': a, 'target': b, 'relation': 'calls', 'confidence': 'EXTRACTED'}


def _snap(tmp_path, nodes, links):
    data = {'directed': False, 'multigraph': False, 'graph': {}, 'nodes': nodes, 'links': links}
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    build_snapshot(p)
    return SnapBackend(snapshot_path_for(p), graph_json=p), data


def _small():
    nodes = [
        _node('a', 'alpha()', 'src/a.py', 'L3'),
        _node('b', 'beta', 'src/b.py', 'L1'),
        _node('c', 'gamma', 'src/b.py', 'L9', 1),
        _node('d', 'delta', 'src/d.py', 'L2', 1),
    ]
    return nodes, [_link('a', 'b'), _link('b', 'c'), _link('c', 'd')]


@pytest.mark.parametrize('seed', ['alpha', 'a', 'src/a.py', 'gam', 'zzz', 'src/b.py'])
def test_seed_resolution_matches_json_backend(tmp_path, seed):
    sb, data = _snap(tmp_path, *_small())
    jb = JsonBackend(load_node_link_graph(data))
    got = sb.query('what', seeds=[seed])
    want = jb.query('what', seeds=[seed])
    assert got.seeds == want.seeds
    assert got.unresolved == want.unresolved
    assert (got.refusal.message if got.refusal else None) == (want.refusal.message if want.refusal else None)


@pytest.mark.xfail(strict=True, reason='resolve_seed compares by casefold, the snapshot by norm_label: a known gap (D8)')
def test_seed_sharp_s_gap(tmp_path):
    nodes = [_node('s1', 'Straße', 'src/s1.py'), _node('s2', 'Strasse', 'src/s2.py')]
    sb, data = _snap(tmp_path, nodes, [_link('s1', 's2')])
    jb = JsonBackend(load_node_link_graph(data))
    got = sb.query('what', seeds=['Strasse'])
    want = jb.query('what', seeds=['Strasse'])
    assert got.seeds == want.seeds


def _ring_with_chords(n=130):
    nodes = [_node(f'n{i}', f'Thing{i}', f'src/m{i}.py', 'L5', i % 4) for i in range(n)]
    pairs, links = set(), []
    for i in range(n):
        j = (i + 1) % n
        pairs.add(frozenset((i, j)))
        links.append(_link(f'n{i}', f'n{j}'))
    for i in range(0, n, 3):
        j = (i * 7 + 3) % n
        if i != j and frozenset((i, j)) not in pairs:
            pairs.add(frozenset((i, j)))
            links.append(_link(f'n{i}', f'n{j}'))
    return nodes, links


def _digraph(data):
    G = nx.DiGraph()
    for nd in data['nodes']:
        G.add_node(nd['id'], **{k: v for k, v in nd.items() if k != 'id'})
    for l in data['links']:
        G.add_edge(l['source'], l['target'])
    return G


@pytest.mark.parametrize('top_n', [10, 100, 101, 500])
def test_god_nodes_match_analyze_for_any_top_n(tmp_path, top_n):
    sb, data = _snap(tmp_path, *_ring_with_chords())
    got = [(g.label, g.degree) for g in sb.god_nodes(top_n=top_n)]
    want = [(g['label'], g['degree']) for g in god_nodes(_digraph(data), top_n=top_n)]
    assert got == want
    degrees = [d for _, d in got]
    assert len(set(degrees)) < len(degrees), 'the fixture must contain equal degrees'
