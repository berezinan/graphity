'''Graph shapes, refusals and source immutability of the snapshot writer (6.3).'''
from __future__ import annotations

import json

import pytest
from networkx.readwrite import json_graph

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

from graphify.build import has_parallel_links  # noqa: E402
from graphify.query_backend import JsonBackend, render_explain, render_path, render_query  # noqa: E402
from graphify.snap_backend import SnapBackend, _TG  # noqa: E402
from graphify.snap_store import Unsupported, snapshot_path_for  # noqa: E402
from graphify.snap_writer import build_snapshot, file_info  # noqa: E402


def _data(directed: bool) -> dict:
    nodes = [{'id': i, 'label': lbl, 'source_file': f'{lbl}.py', 'community': c}
             for i, lbl, c in (('a', 'alpha', 0), ('b', 'beta', 0), ('c', 'gamma', 1), ('d', 'delta', 1))]
    links = [
        {'source': 'a', 'target': 'a', 'relation': 'calls', 'confidence': 'EXTRACTED', 'context': 'call'},
        {'source': 'a', 'target': 'b', 'relation': 'calls', 'confidence': 'EXTRACTED', 'context': 'call', 'weight': 1.0},
        {'source': 'c', 'target': 'b', 'relation': 'uses', 'confidence': 'INFERRED'},
        # the same pair the other way, with other attributes and no context
        {'source': 'b', 'target': 'a', 'relation': 'imports', 'confidence': 'AMBIGUOUS', 'extra': [1, {'k': None}]},
        {'source': 'd', 'target': 'c', 'relation': 'contains', 'confidence': 'EXTRACTED', 'context': 'new'},
        {'source': 'c', 'target': 'a', 'relation': 'calls', 'confidence': 'EXTRACTED', 'context': 'call'},
    ]
    return {'directed': directed, 'multigraph': False, 'graph': {}, 'nodes': nodes, 'links': links}


def _write(tmp_path, data):
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return p


def _snap_of(data, tmp_path):
    p = _write(tmp_path, data)
    build_snapshot(p)
    return SnapBackend(snapshot_path_for(p), graph_json=p)


# The networkx graphs the CLI builds from graph.json, per command.
def _links(raw):
    raw = dict(raw, graph=dict(raw.get('graph') or {}))
    if 'links' not in raw and 'edges' in raw:
        raw = dict(raw, links=raw['edges'])
    return raw


def _markers(raw):
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


@pytest.mark.parametrize('directed', [False, True])
def test_loop_and_repeated_pair(tmp_path, directed):
    from graphify.serve import _traversal_view
    data = _data(directed)
    sb = _snap_of(data, tmp_path)
    G = query_graph(data)
    TG = _traversal_view(G)
    for i, nid in enumerate(TG.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i)] == list(TG.neighbors(nid)), nid
        assert sb.snap.tdegree(i) == TG.degree(nid), nid
        assert sb.snap.deg_q(i) == G.degree(nid), nid
    # one pair a-b, at the position of its first appearance, attributes merged in order
    visible = lambda d: {k: v for k, v in d.items() if not k.startswith('_')}
    assert visible(_TG(sb, None).pair_data('a', 'b')) == visible(TG['a']['b'])
    jq = JsonBackend(G)
    for q in ('alpha', 'beta', 'who calls gamma', 'delta'):
        for mode in ('bfs', 'dfs'):
            assert render_query(sb.query(q, mode=mode, depth=3)) == render_query(jq.query(q, mode=mode, depth=3)), (q, mode)
    je = JsonBackend(explain_graph(data))
    for label in ('alpha', 'beta', 'gamma'):
        assert render_explain(sb.explain(label), label) == render_explain(je.explain(label), label)
    for undirected in (False, True):
        jp = JsonBackend(path_graph(data, undirected))
        for a, b in (('delta', 'alpha'), ('alpha', 'delta'), ('beta', 'delta')):
            assert render_path(sb.shortest_path(a, b, undirected=undirected), a, b) == \
                render_path(jp.shortest_path(a, b, undirected=undirected), a, b), (a, b, undirected)


@pytest.mark.parametrize('mutate, why', [
    (lambda d: d.update(multigraph=True), 'multigraph'),
    (lambda d: d.pop('multigraph'), 'multigraph'),
    (lambda d: d['nodes'].append(dict(d['nodes'][0])), 'duplicate node id'),
    (lambda d: d['links'].append({'source': 'a', 'target': 'zz'}), 'endpoint'),
    (lambda d: d['links'].append(dict(d['links'][1])), 'parallel'),
    (lambda d: d['links'][0].update(_src='a'), '_src'),
    (lambda d: d['nodes'][0].update(label='a\0b'), 'NUL'),
])
def test_refuses_unsupported_shapes(tmp_path, mutate, why):
    data = _data(False)
    mutate(data)
    p = _write(tmp_path, data)
    with pytest.raises(Unsupported, match=why) as exc:
        build_snapshot(p)
    assert 'GRAPHIFY_GRAPH_STORE=json' in str(exc.value)
    assert not snapshot_path_for(p).exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_snap_build_command_reports_the_shape_and_the_json_store(tmp_path, capsys):
    from graphify.snap_writer import snap_build_command
    data = _data(False)
    data['links'].append(dict(data['links'][1]))
    p = _write(tmp_path, data)
    assert snap_build_command(['build', str(p)]) == 1
    err = capsys.readouterr().err
    assert 'parallel' in err and 'GRAPHIFY_GRAPH_STORE=json' in err
    assert not snapshot_path_for(p).exists()


def test_writer_leaves_graph_json_untouched(tmp_path):
    p = _write(tmp_path, _data(False))
    before = file_info(p)
    build_snapshot(p)
    assert file_info(p) == before
    assert snapshot_path_for(p).exists()


def test_attributes_outside_the_columns_survive(tmp_path):
    data = _data(False)
    data['nodes'][0]['custom'] = {'k': [1, None, 2.5]}
    sb = _snap_of(data, tmp_path)
    assert sb._node(0)['custom'] == {'k': [1, None, 2.5]}
    assert sb._link(3, markers=False)['extra'] == [1, {'k': None}]
