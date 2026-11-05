'''Port guard (6.5): upstream code that SnapBackend repeats is hashed; a change must be noticed.

After an upstream merge that changes any function of snap_store.PORT_FUNCS, test_port_hash_matches_recorded
turns red until the port is re-checked (diff read, change carried into SnapBackend or shown not to change
an answer, parity green) and PORT_HASH is updated. None of these tests needs the graphify-snap wheel,
except the one that opens a real snapshot.
'''
from __future__ import annotations

import importlib.metadata
import json

import pytest

from graphify import affected, serve, snap_store
from graphify.snap_store import (
    PORT_FUNCS,
    PORT_HASH,
    PortOutdated,
    SnapshotUnavailable,
    norm_fingerprint,
    port_hash,
)


def test_every_port_func_resolves():
    for name in PORT_FUNCS:
        assert callable(snap_store._port_func(name)), name


def test_port_hash_matches_recorded():
    assert port_hash() == PORT_HASH, (
        'upstream code that SnapBackend repeats changed (snap_store.PORT_FUNCS). Read the diff of the '
        'changed functions, carry it into SnapBackend or show it does not change an answer, run the '
        'SnapBackend/JsonBackend parity tests, and only then update PORT_HASH. '
        f'Current hash: {port_hash()}')


@pytest.mark.parametrize('module, attr', [
    (serve, '_score_query'),
    (serve, '_ambiguity_message'),
    (serve, '_is_searchable'),
    (affected, 'resolve_seed'),
])
def test_a_changed_port_func_changes_the_hash(monkeypatch, module, attr):
    def replacement(*a, **k):
        return None

    replacement.__qualname__ = attr
    monkeypatch.setattr(module, attr, replacement)
    assert port_hash() != PORT_HASH


def test_a_changed_backend_method_changes_the_hash(monkeypatch):
    from graphify import query_backend as qb

    def query(self, *a, **k):
        return None

    query.__qualname__ = 'JsonBackend.query'
    monkeypatch.setattr(qb.JsonBackend, 'query', query)
    assert port_hash() != PORT_HASH


def test_backend_refuses_when_the_port_changed(monkeypatch, tmp_path):
    from graphify import snap_backend

    monkeypatch.setattr(snap_backend, 'PORT_HASH', '0' * 64)
    with pytest.raises(PortOutdated) as exc:
        snap_backend.SnapBackend(tmp_path / 'graph.snap')
    assert isinstance(exc.value, SnapshotUnavailable)
    assert 'PORT_FUNCS' in str(exc.value) and 'Update graphify' in str(exc.value)


def test_backend_with_a_current_port_opens_a_real_snapshot(tmp_path):
    pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')
    from graphify.snap_backend import SnapBackend
    from graphify.snap_writer import build_snapshot

    data = {'directed': False, 'multigraph': False, 'graph': {},
            'nodes': [{'id': 'a', 'label': 'alpha', 'source_file': 'a.py', 'community': 0}], 'links': []}
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    build_snapshot(p)
    assert SnapBackend(snap_store.snapshot_path_for(p), graph_json=p).graph_stats().nodes == 1


def test_normalisation_fingerprint_ignores_the_package_version(monkeypatch):
    '''D4: every upstream merge changes the version; no snapshot may turn incompatible because of it.'''
    before = norm_fingerprint()
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '99.0.0')
    assert norm_fingerprint() == before


def test_normalisation_fingerprint_follows_the_normalisation_code(monkeypatch):
    before = norm_fingerprint()

    def _search_tokens(text):
        return []

    monkeypatch.setattr(serve, '_search_tokens', _search_tokens)
    assert norm_fingerprint() != before
