'''CLI level (6.1): the cases of tests/test_path_explain_cli.py (plus `query`) run through the real
`graphify` CLI twice, once as is and once with the CLI's JsonBackend swapped for SnapBackend over a
snapshot of the same --graph file. stdout, stderr and the exit code must be byte-equal.'''
from __future__ import annotations

import json

import networkx as nx
import pytest
from networkx.readwrite import json_graph

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

import graphify.__main__ as mainmod  # noqa: E402
import graphify.query_backend as qb  # noqa: E402
from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_store import snapshot_path_for  # noqa: E402
from graphify.snap_writer import build_snapshot  # noqa: E402


def _graph(tmp_path, island=False):
    # Copy of tests/test_path_explain_cli.py::_write_graph (+ its island variant).
    G = nx.DiGraph()
    G.add_node('n1', label='extract', source_file='extract.py', source_location='L10', community=0)
    G.add_node('n2', label='cluster', source_file='cluster.py', source_location='L5', community=0)
    G.add_node('n3', label='build', source_file='build.py', source_location='L1', community=1)
    G.add_edge('n1', 'n2', relation='calls', confidence='EXTRACTED', context='call')
    G.add_edge('n2', 'n3', relation='imports', confidence='EXTRACTED', context='import')
    data = json_graph.node_link_data(G, edges='links')
    if island:
        data['nodes'].append({'id': 'n9', 'label': 'island', 'source_file': 'x.py', 'community': 5})
    gp = tmp_path / 'graph.json'
    gp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return gp


OPENED: list = []


def _run(monkeypatch, capsys, argv, snap_gp=None):
    monkeypatch.setattr(mainmod, '_check_skill_version', lambda _: None)
    monkeypatch.setattr(mainmod.sys, 'argv', argv)
    if snap_gp is not None:
        # Opened BEFORE JsonBackend is swapped: the port guard hashes
        # query_backend.JsonBackend.<method>, which the factory below replaces.
        snap_backend = SnapBackend(snapshot_path_for(snap_gp), graph_json=snap_gp)

        def _factory(G, *a, **k):
            OPENED.append(snap_gp)
            return snap_backend
        monkeypatch.setattr(qb, 'JsonBackend', _factory)
    code = 0
    try:
        mainmod.main()
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    monkeypatch.undo()
    return out.out, out.err, code


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp('home')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.setenv('GRAPHIFY_GRAPH_STORE', 'json')


CASES = [
    (False, ['path', 'extract', 'build']),
    (False, ['path', 'zzznope', 'build']),
    (False, ['path', 'build', 'extract']),
    (False, ['path', 'build', 'extract', '--undirected']),
    (True, ['path', 'extract', 'island']),
    (False, ['explain', 'cluster']),
    (False, ['explain', 'zzznope']),
    (False, ['explain', 'clust']),
    (False, ['query', 'extract']),
    (False, ['query', 'who calls cluster']),
    (False, ['query', 'build', '--dfs']),
    (False, ['query', 'zzznope']),
    (False, ['query', 'what', '--seed', 'extract']),
]


@pytest.mark.parametrize('island,args', CASES, ids=[' '.join(a) for _, a in CASES])
def test_cli_parity(monkeypatch, capsys, tmp_path, island, args):
    gp = _graph(tmp_path, island)
    build_snapshot(gp)
    argv = ['graphify', *args, '--graph', str(gp)]
    want = _run(monkeypatch, capsys, argv)
    # monkeypatch.undo() above cleared the env; re-apply it for the snapshot run.
    monkeypatch.setenv('GRAPHIFY_GRAPH_STORE', 'json')
    OPENED.clear()
    got = _run(monkeypatch, capsys, argv, snap_gp=gp)
    assert OPENED, 'the CLI did not go through SnapBackend'
    assert got == want
