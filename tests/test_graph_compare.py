'''rebuild-single-graph-parse (fork delta): fingerprint compare and one parse of graph.json per rebuild.

The equivalence tests are the main guard: graph_fingerprint / topology_fingerprint must give the same
"same graph" / "same topology" answer as upstream's canonical-string compare in watch._rebuild_code.
'''
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphify import build, graph_compare, watch
from graphify.graph_compare import (
    PORT_FUNCS,
    PORT_HASH,
    existing_graph_fingerprint,
    graph_fingerprint,
    load_existing_graph_json,
    port_hash,
    topology_fingerprint,
)


def _upstream_same(canon, a, b) -> bool:
    try:
        return (json.dumps(canon(a), sort_keys=True, ensure_ascii=False)
                == json.dumps(canon(b), sort_keys=True, ensure_ascii=False))
    except Exception:
        return False


def _ours_same(fp, a, b) -> bool:
    try:
        return fp(a) == fp(b)
    except Exception:
        return False


def _check(a, b):
    assert _ours_same(graph_fingerprint, a, b) == _upstream_same(watch._canonical_graph_for_compare, a, b)
    assert (_ours_same(topology_fingerprint, a, b)
            == _upstream_same(watch._canonical_topology_for_compare, a, b))


def _graph(**over):
    g = {
        'directed': False, 'multigraph': False, 'graph': {'schema_version': 2, 'graphify_version': '0.9.97'},
        'nodes': [
            {'id': 'a', 'label': 'A', 'community': 0, 'community_name': 'X', 'norm_label': 'a'},
            {'id': 'b', 'label': 'Б', 'community': 1, 'source_file': 'b.py'},
        ],
        'links': [
            {'source': 'a', 'target': 'b', 'relation': 'calls', 'confidence_score': 1.0},
            {'source': 'b', 'target': 'a', 'relation': 'imports'},
        ],
        'hyperedges': [{'nodes': ['a', 'b'], 'relation': 'group'}],
        'built_at_commit': 'abc',
    }
    g.update(over)
    return g


def _reordered(g):
    r = dict(g)
    for k in ('nodes', 'links', 'edges', 'hyperedges'):
        if isinstance(r.get(k), list):
            r[k] = list(reversed(r[k]))
    return r


# --- 3.1 equivalence, hand-picked pairs ---

PAIRS = {
    'identical': (_graph(), _graph()),
    'reordered and other commit': (_graph(), dict(_reordered(_graph()), built_at_commit='def')),
    'no directed key vs false': ({k: v for k, v in _graph().items() if k != 'directed'}, _graph()),
    'directed true vs false': (_graph(directed=True), _graph()),
    'edges instead of links': (
        {**{k: v for k, v in _graph().items() if k != 'links'}, 'edges': _graph()['links']}, _graph()),
    'non-list nodes': (_graph(nodes={'a': 1}), _graph(nodes={'a': 1})),
    'non-list nodes differ': (_graph(nodes={'a': 1}), _graph(nodes={'a': 2})),
    'list vs non-list': (_graph(nodes=[]), _graph(nodes={})),
    'NaN attribute': (_graph(graph={'x': math.nan}), _graph(graph={'x': math.nan})),
    'one link attribute': (
        _graph(), _graph(links=[{**_graph()['links'][0], 'relation': 'uses'}, _graph()['links'][1]])),
    'duplicate element vs single': (
        _graph(links=_graph()['links'] + [_graph()['links'][0]]), _graph()),
    'multiset counts swapped': (
        _graph(links=[_graph()['links'][0]] * 2 + [_graph()['links'][1]]),
        _graph(links=[_graph()['links'][0]] + [_graph()['links'][1]] * 2)),
    'only community differs': (
        _graph(), _graph(nodes=[{**n, 'community': 7, 'community_name': 'Y'} for n in _graph()['nodes']])),
    'only norm_label differs': (
        _graph(), _graph(nodes=[{**n, 'norm_label': 'zz'} for n in _graph()['nodes']])),
    '_src/_tgt on candidate': (
        _graph(links=[{'source': 'a', 'target': 'b', 'relation': 'calls'}]),
        _graph(links=[{'source': 'b', 'target': 'a', '_src': 'a', '_tgt': 'b', 'relation': 'calls'}])),
    'only _src': (
        _graph(links=[{'source': 'a', 'target': 'b', 'relation': 'calls'}]),
        _graph(links=[{'source': 'a', 'target': 'b', '_src': 'b', 'relation': 'calls'}])),
    'only confidence_score differs': (
        _graph(), _graph(links=[{**e, 'confidence_score': 0.1} for e in _graph()['links']])),
    'only schema_version differs': (_graph(), _graph(graph={'schema_version': 3})),
    'graph meta not a dict': (_graph(graph='x'), _graph(graph='x')),
    'non-dict node and link': (
        _graph(nodes=_graph()['nodes'] + [5], links=_graph()['links'] + ['x']), _graph()),
    'extra top-level key': (_graph(extra=1), _graph()),
    'nested key order': (_graph(graph={'a': 1, 'b': {'c': 1, 'd': 2}}),
                         _graph(graph={'b': {'d': 2, 'c': 1}, 'a': 1})),
    'int vs float': (_graph(graph={'x': 1}), _graph(graph={'x': 1.0})),
    'lone surrogate': (_graph(graph={'x': '\ud800'}), _graph(graph={'x': '\ud800'})),
    'surrogate pair vs astral': (_graph(graph={'x': '\ud83d\ude00'}), _graph(graph={'x': '\U0001f600'})),
    'unserializable value': (_graph(graph={'x': {1, 2}}), _graph(graph={'x': {1, 2}})),
    'mixed key types': ({**_graph(), 1: 'x'}, {**_graph(), 1: 'x'}),
}


@pytest.mark.parametrize('name', list(PAIRS))
def test_fingerprint_answers_as_upstream_strings(name):
    a, b = PAIRS[name]
    _check(a, b)
    _check(b, a)


def test_reorder_is_same_and_one_attribute_is_changed():
    '''Spec scenarios: other order + other built_at_commit → same; one link attribute → changed.'''
    a, b = PAIRS['reordered and other commit']
    assert graph_fingerprint(a) == graph_fingerprint(b)
    a, b = PAIRS['one link attribute']
    assert graph_fingerprint(a) != graph_fingerprint(b)
    a, b = PAIRS['only community differs']
    assert topology_fingerprint(a) == topology_fingerprint(b)
    assert graph_fingerprint(a) != graph_fingerprint(b)


# --- 3.1 equivalence, random graphs ---

_scalar = st.one_of(st.none(), st.booleans(), st.integers(-2, 2), st.sampled_from([0.5, 1.0]),
                    st.sampled_from(['a', 'b', 'Б', '']))
_value = st.recursive(_scalar, lambda inner: st.one_of(
    st.lists(inner, max_size=2), st.dictionaries(st.sampled_from(['k', 'm']), inner, max_size=2)), max_leaves=4)
_node = st.dictionaries(
    st.sampled_from(['id', 'label', 'community', 'community_name', 'norm_label', 'source_file']),
    _value, max_size=4)
_edge = st.dictionaries(
    st.sampled_from(['source', 'target', '_src', '_tgt', 'relation', 'confidence_score']), _value, max_size=4)
_graph_st = st.fixed_dictionaries({}, optional={
    'directed': st.one_of(st.booleans(), st.none()),
    'graph': st.one_of(st.dictionaries(st.sampled_from(['schema_version', 'graphify_version', 'x']),
                                       _scalar, max_size=2), _scalar),
    'nodes': st.one_of(st.lists(st.one_of(_node, _scalar), max_size=4), _scalar),
    'links': st.lists(st.one_of(_edge, _scalar), max_size=4),
    'edges': st.lists(_edge, max_size=3),
    'hyperedges': st.lists(_value, max_size=3),
    'built_at_commit': _scalar,
})


@settings(max_examples=400, deadline=None)
@given(_graph_st, _graph_st)
def test_random_pairs_answer_as_upstream(a, b):
    _check(a, b)


@settings(max_examples=200, deadline=None)
@given(_graph_st, st.randoms(use_true_random=False))
def test_random_permutations_answer_as_upstream(a, rnd):
    b = dict(a)
    for k in ('nodes', 'links', 'edges', 'hyperedges'):
        if isinstance(b.get(k), list):
            b[k] = list(b[k])
            rnd.shuffle(b[k])
    _check(a, b)
    if isinstance(b.get('nodes'), list) and b['nodes'] and isinstance(b['nodes'][0], dict):
        b['nodes'][0] = {**b['nodes'][0], 'community': 'changed'}
        _check(a, b)


# --- 3.3 refusals (#2251) as build._load_existing_graph ---

def _refusal(fn, path):
    with pytest.raises(Exception) as exc:
        fn(path)
    return type(exc.value), str(exc.value)


def test_missing_file_is_none(tmp_path):
    assert load_existing_graph_json(tmp_path / 'graph.json') is None


def test_unreadable_graph_refused_as_upstream(tmp_path):
    p = tmp_path / 'graph.json'
    p.write_text('{"nodes": [', encoding='utf-8')
    ours = _refusal(load_existing_graph_json, p)
    assert ours == _refusal(build._load_existing_graph, p)
    assert ours[0] is RuntimeError and 'Delete the file and run a full rebuild.' in ours[1]


def test_over_cap_graph_refused_as_upstream(tmp_path, monkeypatch):
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(_graph()), encoding='utf-8')
    monkeypatch.setenv('GRAPHIFY_MAX_GRAPH_BYTES', '10')
    ours = _refusal(load_existing_graph_json, p)
    assert ours == _refusal(build._load_existing_graph, p)
    assert ours[0] is ValueError


def test_non_object_document_fails_as_upstream(tmp_path):
    p = tmp_path / 'graph.json'
    p.write_text('[1, 2]', encoding='utf-8')
    assert _refusal(load_existing_graph_json, p)[0] is _refusal(build._load_existing_graph, p)[0]


# --- 3.4 the fingerprint follows the file on disk ---

def test_fingerprint_is_of_the_file_not_of_later_edits(tmp_path, monkeypatch):
    '''Reconcile stamps _origin on the parsed items; upstream re-read the raw file for the compare.'''
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(_graph()), encoding='utf-8')
    data = load_existing_graph_json(p)
    for n in data['nodes']:
        n['_origin'] = 'ast'
    loads = []
    monkeypatch.setattr(graph_compare.json, 'loads', lambda s, *a, **k: loads.append(1) or json.JSONDecoder().decode(s))
    assert existing_graph_fingerprint(p) == graph_fingerprint(_graph())
    assert loads == [], 'an unchanged file must not be parsed again'


def test_file_rewritten_after_the_parse_is_compared_by_its_new_content(tmp_path):
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(_graph()), encoding='utf-8')
    load_existing_graph_json(p)
    newer = _graph(nodes=[{'id': 'z'}])
    p.write_text(json.dumps(newer), encoding='utf-8')
    st_ = p.stat()
    os.utime(p, ns=(st_.st_atime_ns, st_.st_mtime_ns + 10_000_000))
    assert existing_graph_fingerprint(p) == graph_fingerprint(newer)


def test_unreadable_on_compare_raises(tmp_path):
    p = tmp_path / 'graph.json'
    p.write_text(json.dumps(_graph()), encoding='utf-8')
    load_existing_graph_json(p)
    p.write_text('{', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        existing_graph_fingerprint(p)


# --- 3.2 one parse of graph.json per rebuild, prior/ctx_graph freed ---

def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / 'corpus'
    c.mkdir()
    (c / 'a.py').write_text('def f():\n    return g()\n\ndef g():\n    return 1\n', encoding='utf-8')
    (c / 'b.py').write_text('from a import f\n\nclass C:\n    def m(self):\n        return f()\n', encoding='utf-8')
    (c / 'README.md').write_text('# Doc\n\nSee `a.py`.\n', encoding='utf-8')
    return c


@pytest.fixture
def graph_reads(monkeypatch):
    '''Counts full reads of <corpus>/graphify-out/graph.json and the rebuild locals seen at reconcile.'''
    reads: list[str] = []
    seen_locals: list[set] = []
    real_read_text = Path.read_text

    def read_text(self, *a, **k):
        if self.name == 'graph.json' and self.parent.name == 'graphify-out':
            reads.append(str(self))
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, 'read_text', read_text)
    real_reconcile = watch._reconcile_existing_graph

    def reconcile(*a, **k):
        seen_locals.append(set(sys._getframe(1).f_locals))
        return real_reconcile(*a, **k)

    monkeypatch.setattr(watch, '_reconcile_existing_graph', reconcile)
    return reads, seen_locals


@pytest.mark.parametrize('no_cluster', [True, False], ids=['no-cluster', 'clustered'])
def test_full_rebuild_parses_graph_json_once(tmp_path, graph_reads, no_cluster):
    corpus = _corpus(tmp_path)
    assert watch._rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
    reads, seen = graph_reads
    reads.clear(), seen.clear()
    (corpus / 'a.py').write_text('def f():\n    return 2\n', encoding='utf-8')
    assert watch._rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
    # streaming-rebuild-graph-reads: prior (#1915) and reconcile read element-wise, the compare
    # reuses the reconcile fingerprint — no whole-text read of graph.json at all.
    assert len(reads) == 0, reads
    assert seen and not ({'prior', 'prior_paths'} & seen[0]), 'prior must be freed after #1915'


@pytest.mark.parametrize('no_cluster', [True, False], ids=['no-cluster', 'clustered'])
def test_watch_rebuild_parses_graph_json_once(tmp_path, graph_reads, no_cluster):
    corpus = _corpus(tmp_path)
    assert watch._rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
    reads, seen = graph_reads
    reads.clear(), seen.clear()
    (corpus / 'a.py').write_text('def f():\n    return 3\n', encoding='utf-8')
    assert watch._rebuild_code(corpus, changed_paths=[Path('a.py')], no_cluster=no_cluster,
                               acquire_lock=False) is True
    # Only the resolution context (#2406) still parses the whole file; prior, reconcile and the
    # compare do not (streaming-rebuild-graph-reads).
    assert len(reads) == 1, reads
    assert seen and not ({'prior', 'prior_paths', 'ctx_graph', 'ctx_paths'} & seen[0])


def test_unchanged_rebuild_leaves_graph_json_untouched(tmp_path):
    corpus = _corpus(tmp_path)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    g = corpus / 'graphify-out' / 'graph.json'
    before = (g.read_bytes(), g.stat().st_mtime_ns)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    assert (g.read_bytes(), g.stat().st_mtime_ns) == before


# --- 3.5 port guard ---

def test_every_port_func_resolves():
    for name in PORT_FUNCS:
        assert callable(graph_compare._port_func(name)), name


def test_port_hash_matches_recorded():
    assert port_hash() == PORT_HASH, (
        'upstream code that graph_compare repeats changed (graph_compare.PORT_FUNCS). Read the diff, '
        'carry it into graph_compare, run tests/test_graph_compare.py, and only then update PORT_HASH. '
        f'Current hash: {port_hash()}')


@pytest.mark.parametrize('module, attr', [
    (watch, '_canonical_graph_for_compare'),
    (watch, '_canonical_topology_for_compare'),
    (build, '_load_existing_graph'),
])
def test_a_changed_port_func_changes_the_hash(monkeypatch, module, attr):
    def replacement(*a, **k):
        return None

    replacement.__qualname__ = attr
    monkeypatch.setattr(module, attr, replacement)
    assert port_hash() != PORT_HASH
