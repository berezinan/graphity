'''Call-site guards of the snapshot store (5.10).

Every command that writes graph.json publishes a snapshot, never touches ArcadeDB
on the snap store, and the CLI read commands run to the END on the snap branch
(the c147cde class of breakage: a variable bound in one branch only).
None of these tests needs the graphify-snap wheel.
'''
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

from graphify import cli
from graphify import query_backend as qb
from graphify import snap_store


@pytest.fixture
def snap_env(monkeypatch):
    '''Store = snap, module check a no-op, ArcadeDB booby-trapped. Returns the list of published paths.'''
    monkeypatch.setenv('GRAPHIFY_GRAPH_STORE', 'snap')
    monkeypatch.setattr(snap_store, 'check_api', lambda: None)

    def _boom(*a, **k):
        raise AssertionError('ArcadeDB must not be touched on the snapshot store')

    monkeypatch.setattr(qb, '_await_ready', _boom)
    monkeypatch.setattr(qb, 'ArcadeDBBackend', _boom)
    published: list[Path] = []
    monkeypatch.setattr(snap_store, 'publish_snapshot_for', lambda p: published.append(Path(p)))
    return published


def test_snapshot_writer_child_does_not_import_graphify_from_cwd(monkeypatch):
    '''`python -m graphify` puts the cwd first on sys.path: a `graphify/` folder
    without `__init__.py` there (a sibling clone, a build/ leftover) shadowed the
    installed submodules in the writer child -- "cannot import name 'edge_data'
    from 'graphify.build' (unknown location)".'''
    import subprocess
    seen = {}

    class _Done:
        returncode, stdout, stderr = 0, '', ''

    def _run(argv, env=None, **kw):
        seen['env'] = env
        return _Done()

    monkeypatch.delenv('PYTHONSAFEPATH', raising=False)
    monkeypatch.setattr(subprocess, 'run', _run)
    snap_store.publish_snapshot_for(Path('graph.json'))
    assert seen['env'] is not None and seen['env'].get('PYTHONSAFEPATH') == '1'


# --- read commands run to the end on the snap branch -------------------------- #

@pytest.fixture
def snap_cli(monkeypatch, tmp_path):
    G = nx.DiGraph()
    G.add_node('alpha_node', label='Alpha()', source_file='a.py', file_type='code')
    G.add_node('beta_node', label='Beta()', source_file='b.py', file_type='code')
    G.add_edge('alpha_node', 'beta_node', relation='calls', confidence='EXTRACTED')
    backend = qb.JsonBackend(G)
    cfg = {'kind': 'snap', 'graph_path': None,
           'snapshot_path': str(tmp_path / 'graphify-out' / 'graph.snap')}
    monkeypatch.setattr(qb, 'resolve_backend_config', lambda gp=None: dict(cfg))
    monkeypatch.setattr(qb, 'require_backend', lambda **kw: backend)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, 'argv', ['graphify', *argv])
        cli.dispatch_command(argv[0])

    return run, tmp_path


def _stamp(tmp_path):
    return tmp_path / 'graphify-out' / 'cache' / 'last_query_stamp'


def test_query_snap_branch_runs_to_completion(snap_cli, capsys):
    run, tmp_path = snap_cli
    run('query', 'Alpha')
    assert 'Alpha()' in capsys.readouterr().out
    assert _stamp(tmp_path).exists()


def test_path_snap_branch_runs_to_completion(snap_cli, capsys):
    run, tmp_path = snap_cli
    run('path', 'Alpha', 'Beta')
    assert 'Shortest path' in capsys.readouterr().out
    assert _stamp(tmp_path).exists()


def test_explain_snap_branch_runs_to_completion(snap_cli, capsys):
    run, tmp_path = snap_cli
    run('explain', 'Alpha')
    assert 'Alpha()' in capsys.readouterr().out
    assert _stamp(tmp_path).exists()


# --- writers publish a snapshot and leave ArcadeDB alone ---------------------- #

def test_update_on_snap_rebuilds_without_probing_arcadedb(snap_env, monkeypatch, tmp_path):
    from graphify import watch

    rebuilt = []
    monkeypatch.setattr(watch, '_rebuild_code', lambda *a, **kw: rebuilt.append('rebuild') or True)
    (tmp_path / 'a.py').write_text('def f(): pass\n', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['graphify', 'update', str(tmp_path)])
    cli.dispatch_command('update')
    assert rebuilt == ['rebuild']


def _corpus(tmp_path, monkeypatch):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'lib.py').write_text('def f(): pass\n', encoding='utf-8')
    monkeypatch.chdir(corpus)
    return corpus


def test_watch_rebuild_publishes_the_snapshot(snap_env, tmp_path, monkeypatch):
    from graphify.watch import _rebuild_code

    _corpus(tmp_path, monkeypatch)
    assert _rebuild_code(Path('.'), acquire_lock=False) is True
    assert len(snap_env) == 1 and snap_env[0].name == 'graph.json'


def test_watch_rebuild_fails_when_the_writer_fails(snap_env, tmp_path, monkeypatch):
    from graphify.watch import _rebuild_code

    def _fail(p):
        raise snap_store.SnapshotUnavailable('the snapshot writer failed')

    monkeypatch.setattr(snap_store, 'publish_snapshot_for', _fail)
    _corpus(tmp_path, monkeypatch)
    assert _rebuild_code(Path('.'), acquire_lock=False) is False


def test_global_sync_publishes_next_to_the_global_graph(snap_env):
    from graphify import global_graph

    global_graph._sync_global_db()
    assert snap_env == [Path(global_graph._GLOBAL_GRAPH)]


def test_recluster_sync_publishes_after_a_graph_json_write(snap_env):
    from graphify.recluster import sync_community_changes

    sync_community_changes('proj/graphify-out/graph.json', {}, {'n1': (1, 'C1')}, {1: ['n1']}, None)
    assert snap_env == [Path('proj/graphify-out/graph.json')]


def test_idle_reclustering_needs_no_write():
    '''cluster-only skips the write (and so the snapshot) when needs_write is False.'''
    from graphify.recluster import needs_write

    assert needs_write({'built_at_commit': 'abc'}, {}, 'abc') is False


# --- structural guard: an upstream merge must not sever the hooks ------------- #

def test_churn_files_keep_the_snapshot_hooks():
    import graphify

    src_dir = Path(graphify.__file__).parent
    for mod in ('cli.py', 'watch.py', 'recluster.py', 'global_graph.py'):
        src = (src_dir / mod).read_text(encoding='utf-8')
        assert 'publish_snapshot_for' in src, f'{mod}: the snapshot hook is gone'
    cli_src = (src_dir / 'cli.py').read_text(encoding='utf-8')
    assert "cmd == 'snap'" in cli_src.replace('"', "'"), 'cli.py: the snap build command is gone'
