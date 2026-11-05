'''Generations (6.3): rebuilds under live readers, a writer that dies, staleness, incompatible snapshots.'''
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')

import graphify_snap  # noqa: E402
from graphify import snap_store  # noqa: E402
from graphify.query_backend import render_query  # noqa: E402
from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_store import SnapshotUnavailable, snapshot_path_for  # noqa: E402
from graphify.snap_writer import build_snapshot, publish_snapshot  # noqa: E402


def _data() -> dict:
    nodes = [{'id': i, 'label': lbl, 'source_file': f'{lbl}.py', 'community': c}
             for i, lbl, c in (('a', 'alpha', 0), ('b', 'beta', 0), ('c', 'gamma', 1), ('d', 'delta', 1))]
    links = [
        {'source': 'a', 'target': 'b', 'relation': 'calls', 'confidence': 'EXTRACTED', 'context': 'call'},
        {'source': 'c', 'target': 'b', 'relation': 'uses', 'confidence': 'INFERRED'},
        {'source': 'd', 'target': 'c', 'relation': 'contains', 'confidence': 'EXTRACTED'},
        {'source': 'c', 'target': 'a', 'relation': 'calls', 'confidence': 'EXTRACTED', 'context': 'call'},
    ]
    return {'directed': False, 'multigraph': False, 'graph': {}, 'nodes': nodes, 'links': links}


def _relabel(data: dict, suffix: str) -> dict:
    data = json.loads(json.dumps(data))
    for n in data['nodes']:
        n['label'] += suffix
        n.pop('norm_label', None)
    return data


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return p


def _env() -> dict:
    # BFS output depends on the hash seed (the frontier is a set): one fixed seed everywhere.
    return {**os.environ, 'PYTHONHASHSEED': '0'}


_ANSWERS = '''
import json, sys
from graphify.query_backend import render_query
from graphify.snap_backend import SnapBackend
sb = SnapBackend(sys.argv[1])
print(json.dumps([render_query(sb.query(q, depth=2)) for q in json.loads(sys.argv[2])]))
'''

_READER = '''
import json, sys
from pathlib import Path
from graphify.query_backend import render_query
from graphify.snap_backend import SnapBackend
live, queries_s, ref_a_s, ref_b_s, stop, out = sys.argv[1:7]
queries, ref_a, ref_b = json.loads(queries_s), json.loads(ref_a_s), json.loads(ref_b_s)
stats = {'ok': 0, 'mismatch': 0, 'errors': {}, 'generations': []}
sb = None
i = 0
while not Path(stop).exists():
    q = i % len(queries)
    i += 1
    try:
        if sb is None:
            sb = SnapBackend(live)
        got = render_query(sb.query(queries[q], depth=2))
        if sb.snap.generation not in stats['generations']:
            stats['generations'].append(sb.snap.generation)
        if got in (ref_a[q], ref_b[q]):
            stats['ok'] += 1
        else:
            stats['mismatch'] += 1
        if stats['ok'] == 1:
            Path(out + '.ready').write_text('1')
    except Exception as exc:
        key = type(exc).__name__ + ': ' + str(exc)[:200]
        stats['errors'][key] = stats['errors'].get(key, 0) + 1
Path(out).write_text(json.dumps(stats), encoding='utf-8')
'''

_CRASH = '''
import json, sys
from graphify.snap_writer import publish_snapshot
publish_snapshot(json.load(open(sys.argv[1], encoding='utf-8')), sys.argv[2], crash_before_publish=True)
'''


def test_reader_follows_a_new_generation(tmp_path):
    a = _data()
    publish_snapshot(a, tmp_path)
    sb = SnapBackend(tmp_path / 'graph.snap')
    assert sb.snap.generation == 1
    first = render_query(sb.query('alpha'))
    publish_snapshot(_relabel(a, 'x'), tmp_path)
    second = render_query(sb.query('alphax'))
    assert sb.snap.generation == 2
    assert 'alphax' in second and first != second
    assert sorted(p.name for p in tmp_path.iterdir()) == ['graph.snap', 'graph.snap.lock']


def test_rebuilds_under_live_readers(tmp_path):
    a = _data()
    b = _relabel(a, ' v2')
    queries = ['alpha', 'beta', 'gamma', 'delta']
    for name, data in (('a', a), ('b', b)):
        publish_snapshot(data, tmp_path / ('ref_' + name))

    def answers(name):
        snap = str(tmp_path / ('ref_' + name) / 'graph.snap')
        proc = subprocess.run([sys.executable, '-c', _ANSWERS, snap, json.dumps(queries)],
                              capture_output=True, text=True, env=_env(), check=True)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    ref_a, ref_b = answers('a'), answers('b')
    assert all(x != y for x, y in zip(ref_a, ref_b)), 'graphs A and B must answer differently'

    live = tmp_path / 'live'
    publish_snapshot(a, live)
    stop = tmp_path / 'STOP'
    outs = [tmp_path / f'reader{k}.json' for k in range(4)]
    procs = []
    for k, out in enumerate(outs):
        cmd = [sys.executable, '-c', _READER, str(live / 'graph.snap'), json.dumps(queries),
               json.dumps(ref_a), json.dumps(ref_b), str(stop), str(out)]
        procs.append(subprocess.Popen(cmd, env=_env(), stderr=open(tmp_path / f'reader{k}.err', 'wb')))
    deadline = time.monotonic() + 120
    while not all(Path(str(o) + '.ready').exists() for o in outs):
        assert time.monotonic() < deadline, 'readers did not start answering'
        time.sleep(0.2)
    for r in range(20):
        publish_snapshot(b if r % 2 == 0 else a, live)
        time.sleep(0.05)
    time.sleep(1)
    stop.write_text('stop')
    for p in procs:
        assert p.wait(timeout=120) == 0
    stats = [json.loads(o.read_text(encoding='utf-8')) for o in outs]
    assert [s['errors'] for s in stats] == [{}, {}, {}, {}]
    assert sum(s['mismatch'] for s in stats) == 0
    assert all(s['ok'] > 0 for s in stats)
    assert max(len(s['generations']) for s in stats) > 1, 'no reader followed a rebuild'
    assert graphify_snap.peek_generation(str(live / 'graph.snap')) == 21


def test_crashed_writer_leaves_the_published_generation_and_the_next_run_cleans(tmp_path):
    src = _write(tmp_path, _data())
    live = tmp_path / 'live'
    publish_snapshot(_data(), live)
    sb = SnapBackend(live / 'graph.snap')
    proc = subprocess.run([sys.executable, '-c', _CRASH, str(src), str(live)], env=_env())
    assert proc.returncode == 3
    left = list(live.glob('*.tmp'))
    assert len(left) == 1
    assert graphify_snap.peek_generation(str(live / 'graph.snap')) == 1
    assert render_query(sb.query('alpha'))
    rep = publish_snapshot(_data(), live)
    assert rep['removed_tmp'] == [left[0].name] and not list(live.glob('*.tmp'))
    assert rep['generation'] == 2


def test_stale_graph_json_is_refused_with_the_rebuild_command(tmp_path):
    src = _write(tmp_path, _data())
    build_snapshot(src)
    sb = SnapBackend(snapshot_path_for(src), graph_json=src)
    assert render_query(sb.query('alpha'))
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    with pytest.raises(SnapshotUnavailable) as exc:
        sb.query('alpha')
    assert 'stale' in str(exc.value) and 'graphify snap build' in str(exc.value)
    build_snapshot(src)
    assert render_query(sb.query('alpha'))


@pytest.mark.parametrize('damage', ['version', 'fingerprint', 'truncate'])
def test_incompatible_snapshot_is_refused(tmp_path, monkeypatch, damage):
    publish_snapshot(_data(), tmp_path)
    p = tmp_path / 'graph.snap'
    if damage == 'version':
        with open(p, 'r+b') as f:
            f.seek(8)
            f.write((99).to_bytes(4, 'little'))
    elif damage == 'fingerprint':
        monkeypatch.setattr(snap_store, 'norm_fingerprint', lambda: bytes(32))
    else:
        cut = tmp_path / 'cut.snap'
        cut.write_bytes(p.read_bytes()[:-10])
        p = cut
    with pytest.raises(SnapshotUnavailable) as exc:
        snap_store.open_snapshot(p)
    assert 'graphify snap build' in str(exc.value)
