'''streaming-rebuild-graph-reads (fork delta): graph.json / graph.tmp.json read element by element.

Guards, in order of strength:
- the reader gives the dict json.loads gives, or upstream's exception (3.1, 3.2);
- the projections hold every field upstream reads and give upstream's answers (3.3, 3.4);
- a rebuild never reads the whole text and writes the same files as upstream's whole-file path (3.5);
- the upstream code the projections mirror is hashed (3.6).
'''
from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

from graphify import build, graph_compare as gc, watch
from graphify.graph_compare import (
    PORT_FUNCS,
    PORT_HASH,
    PRIOR_FIELDS,
    SHRINK_NODE_FIELDS,
    graph_fingerprint,
    load_existing_graph_json,
    load_graph_projection,
    port_hash,
    read_candidate,
)

pytest.importorskip('graphify_snap', reason='graphify-snap wheel not installed')


@pytest.fixture
def small_batches(monkeypatch):
    '''Batches of 2 elements, so every array crosses batch boundaries.'''
    monkeypatch.setattr(gc, '_BATCH', 2)


@pytest.fixture
def text_reads(monkeypatch):
    '''Whole-text reads (Path.read_text) of graph files.'''
    reads: list[str] = []
    real = Path.read_text

    def read_text(self, *a, **k):
        if self.name in ('graph.json', '.graph.tmp.json'):
            reads.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, 'read_text', read_text)
    return reads


@pytest.fixture
def no_wheel(monkeypatch):
    monkeypatch.setitem(sys.modules, 'graphify_snap', None)


def _graph():
    return {
        'directed': False, 'multigraph': False,
        'graph': {'schema_version': 2, 'hyperedges': [{'nodes': ['a', 'b'], 'relation': 'g'}]},
        'nodes': [
            {'id': 'a', 'label': 'A', 'source_file': 'a.py', 'file_type': 'code', '_origin': 'ast',
             'source_location': 'L1', 'community': 0, 'community_name': 'X', 'extra': [1, {'k': None}]},
            {'id': 'b', 'label': 'Б — юникод', 'source_file': 'docs/b.md', 'file_type': 'document',
             'community': 1, 'community_name': 'Y', 'w': -1.5e3},
            {'id': 'c', 'label': 'c\u00e9\U0001f600', 'community': 1, 'n': float('nan')},
            7,
        ],
        'links': [
            {'source': 'a', 'target': 'b', 'relation': 'calls', '_src': 'b', '_tgt': 'a', 'source_file': 'a.py'},
            {'source': 'b', 'target': 'c', 'relation': 'mentions', 'confidence_score': float('inf')},
            'odd',
        ],
        'hyperedges': [{'nodes': ['a', 'c'], 'relation': 'group', 'source_file': 'a.py'}],
        'built_at_commit': 'abc',
    }


LAYOUTS = {
    'indented': lambda d: json.dumps(d, indent=2),
    'compact': lambda d: json.dumps(d, separators=(',', ':')),
    'ascii': lambda d: json.dumps(d),
    'raw-unicode': lambda d: json.dumps(d, ensure_ascii=False, indent=1),
}


def _write(tmp_path, text: str, name='graph.json') -> Path:
    p = tmp_path / name
    p.write_bytes(text.encode('utf-8'))
    return p


def _same(a, b) -> bool:
    '''Equality that treats NaN as equal to itself (json round-trips keep it NaN).'''
    return json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)


# --- 3.1 reader ---

DOCS = {
    'graph': _graph(),
    'edges spelling': {**{k: v for k, v in _graph().items() if k != 'links'}, 'edges': _graph()['links']},
    'empty arrays': {'nodes': [], 'links': [], 'hyperedges': []},
    'scalars only': {'directed': True, 'graph': {}},
    'nodes not a list': {'nodes': {'a': 1}, 'links': []},
    'extra array': {'nodes': [{'id': 'a'}], 'extra': [1, 2, 3], 'links': []},
}


@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('doc', list(DOCS))
def test_reader_gives_json_loads(tmp_path, small_batches, text_reads, layout, doc):
    p = _write(tmp_path, LAYOUTS[layout](DOCS[doc]))
    expected = json.loads(p.read_text(encoding='utf-8'))
    text_reads.clear()
    got = load_existing_graph_json(p)
    assert text_reads == [], 'the element-wise read must not read the whole text'
    assert _same(got, expected) and list(got) == list(expected)
    saved = gc._FINGERPRINTS[str(p.resolve())][2]
    assert saved == gc._fingerprint_or_none(expected)


def test_repeated_top_level_keys_keep_first_position_last_value(tmp_path, small_batches, text_reads):
    text = ('{"nodes":[{"id":"x"}],"graph":{"v":1},"links":[],'
            '"nodes":[{"id":"y"},{"id":"z"}],"graph":{"v":2}}')
    p = _write(tmp_path, text)
    expected = json.loads(text)
    text_reads.clear()
    got = load_existing_graph_json(p)
    assert text_reads == []
    assert got == expected and list(got) == list(expected) == ['nodes', 'graph', 'links']


def test_damaged_shadowed_value_fails_as_the_whole_file_parse(tmp_path):
    '''A repeated key's first value is dropped by json.loads but still parsed: a bad one fails both.'''
    p = _write(tmp_path, '{"nodes":[{"id":01}],"nodes":[]}')
    with pytest.raises(RuntimeError) as ours:
        load_existing_graph_json(p)
    with pytest.raises(RuntimeError) as theirs:
        build._load_existing_graph(p)
    assert str(ours.value) == str(theirs.value)


def test_projection_and_candidate_match_the_whole_parse(tmp_path, small_batches):
    p = _write(tmp_path, json.dumps(_graph(), indent=2))
    full = json.loads(p.read_text(encoding='utf-8'))
    proj = load_graph_projection(p)
    for bucket, fields in PRIOR_FIELDS.items():
        for got, item in zip(proj.get(bucket, []), full.get(bucket, [])):
            assert got == ({f: item[f] for f in fields if f in item} if isinstance(item, dict) else item)
    assert proj['graph'] == full['graph'] and proj['directed'] == full['directed']
    c = read_candidate(p)
    assert c.fingerprint == graph_fingerprint(full)
    assert c.shrink_data == {'nodes': [
        {f: n[f] for f in SHRINK_NODE_FIELDS if f in n} if isinstance(n, dict) else n for n in full['nodes']]}


# --- 3.2 refusals ---

BAD = {
    'truncated': b'{"nodes": [{"id": "a"}, ',
    'trailing garbage': b'{"nodes": []} x',
    'bom': b'\xef\xbb\xbf{"nodes": []}',
    'invalid utf-8 in a value': b'{"nodes": [{"id": "a\xff"}]}',
    'invalid utf-8 in a key': b'{"no\xfedes": []}',
    'not an object': b'[1, 2]',
    'bad scalar': b'{"nodes": [{"id": tru}]}',
    'control char in string': b'{"nodes": [{"id": "a\rb"}]}',
    'empty': b'',
}


def _outcome(fn, p):
    try:
        return 'ok', fn(p)
    except Exception as exc:  # noqa: BLE001 — type and text are the contract
        return type(exc), str(exc)


@pytest.mark.parametrize('case', list(BAD))
def test_bad_file_refused_as_upstream(tmp_path, case):
    p = tmp_path / 'graph.json'
    p.write_bytes(BAD[case])
    ours = _outcome(load_existing_graph_json, p)
    theirs = _outcome(build._load_existing_graph, p)
    assert ours[0] == theirs[0] and ours[0] != 'ok'
    assert ours[1] == theirs[1]


@pytest.mark.parametrize('case', list(BAD))
def test_bad_candidate_raises_as_the_whole_file_parse(tmp_path, case):
    p = tmp_path / '.graph.tmp.json'
    p.write_bytes(BAD[case])
    theirs = _outcome(lambda q: json.loads(q.read_text(encoding='utf-8')), p)
    ours = _outcome(read_candidate, p)
    if theirs[0] == 'ok':  # a parseable non-object: every consumer gets the parsed value
        assert ours[0] == 'ok' and ours[1].shrink_data == theirs[1]
    else:
        assert ours == theirs


def test_over_cap_refused_as_upstream(tmp_path, monkeypatch):
    p = _write(tmp_path, json.dumps(_graph()))
    monkeypatch.setenv('GRAPHIFY_MAX_GRAPH_BYTES', '10')
    assert _outcome(load_existing_graph_json, p) == _outcome(build._load_existing_graph, p)


def test_without_the_wheel_the_result_is_the_same(tmp_path, no_wheel, text_reads):
    p = _write(tmp_path, json.dumps(_graph(), indent=2))
    expected = json.loads(Path.read_bytes(p).decode('utf-8'))
    assert _same(load_existing_graph_json(p), expected)
    assert _same(load_graph_projection(p)['graph'], expected['graph'])
    assert read_candidate(p).fingerprint == graph_fingerprint(expected)
    assert 'graph.json' in text_reads, 'the fallback is the whole-file read'


# --- 3.3 projections hold every field upstream reads ---

class _Rec(dict):
    '''A dict that records the keys read through it; whole-dict access records "*".'''

    log: set

    def __getitem__(self, k):
        self.log.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.log.add(k)
        return super().get(k, default)

    def __contains__(self, k):
        self.log.add(k)
        return super().__contains__(k)

    def _all(self):
        self.log.add('*')

    def __iter__(self):
        self._all()
        return super().__iter__()

    def keys(self):
        self._all()
        return super().keys()

    def items(self):
        self._all()
        return super().items()

    def values(self):
        self._all()
        return super().values()

    def copy(self):
        self._all()
        return super().copy()


def _recording(data: dict, log_by_bucket: dict) -> dict:
    out = dict(data)
    for bucket in ('nodes', 'links', 'edges', 'hyperedges'):
        if isinstance(out.get(bucket), list):
            log = log_by_bucket.setdefault(bucket, set())
            items = []
            for item in out[bucket]:
                if isinstance(item, dict):
                    r = _Rec(item)
                    r.log = log
                    items.append(r)
                else:
                    items.append(item)
            out[bucket] = items
    return out


def _corpus(root: Path) -> Path:
    c = root / 'corpus'
    c.mkdir()
    (c / 'a.py').write_text('def f():\n    return g()\n\ndef g():\n    return 1\n', encoding='utf-8')
    (c / 'b.py').write_text('from a import f\n\nclass C:\n    def m(self):\n        return f()\n', encoding='utf-8')
    (c / 'README.md').write_text('# Doc\n\nSee `a.py`.\n', encoding='utf-8')
    (c / 'notes.md').write_text('# Notes\n\nDesign.\n', encoding='utf-8')
    return c


def _add_semantic_layer(corpus: Path) -> None:
    '''Semantic nodes on notes.md (#1915 path) and a legacy unstamped AST-shaped node.'''
    g = corpus / 'graphify-out' / 'graph.json'
    data = json.loads(g.read_text(encoding='utf-8'))
    data['nodes'] += [
        {'id': 'notes_doc', 'label': 'Notes', 'file_type': 'document', 'source_file': 'notes.md'},
        {'id': 'notes_idea', 'label': 'Idea', 'file_type': 'concept', 'source_file': 'notes.md',
         'source_location': None},
        {'id': 'legacy', 'label': 'legacy', 'file_type': 'document', 'source_file': 'README.md',
         'source_location': 'L3'},
    ]
    data['links'].append({'source': 'notes_idea', 'target': 'notes_doc', 'relation': 'described_in',
                          'source_file': 'notes.md'})
    g.write_text(json.dumps(data), encoding='utf-8')


def test_prior_block_reads_only_projected_fields(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    _add_semantic_layer(corpus)
    logs: dict = {}

    def recording_projection(path, fields=PRIOR_FIELDS):
        # The whole graph, every element recording what the #1915 block reads from it.
        return _recording(json.loads(Path(path).read_bytes().decode('utf-8')), logs)

    monkeypatch.setattr(gc, 'load_graph_projection', recording_projection)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    assert logs.get('nodes'), 'the #1915 block must have run over the nodes'
    for bucket, log in logs.items():
        assert log <= set(PRIOR_FIELDS[bucket]), (bucket, log - set(PRIOR_FIELDS[bucket]))


@pytest.mark.parametrize('bucket', ['links', 'edges', 'hyperedges'])
@pytest.mark.parametrize('marker', ['absolute', 'relative'])
def test_source_paths_resolver_reads_projected_fields_and_answers_the_same(tmp_path, monkeypatch, bucket, marker):
    '''_StoredSourcePaths reaches the non-node buckets when no node carries a relative source_file.

    Absolute marker: the anchor sampling decides existing_source_root from those buckets.
    Relative marker: the legacy watch-relative check decides legacy_watch_relative from them.
    '''
    from graphify.build import _norm_source_file

    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'src' / 'a.py').write_text('x = 1\n', encoding='utf-8')
    out = project / 'graphify-out'
    out.mkdir()
    watch_root = project / 'src'
    if marker == 'absolute':
        (out / '.graphify_root').write_text(str(watch_root), encoding='utf-8')
    else:
        (out / '.graphify_root').write_text('src', encoding='utf-8')
        monkeypatch.chdir(project)
    item = {'source': 'x', 'target': 'x', 'relation': 'r', 'source_file': 'src/a.py', 'note': 'n'}
    if bucket == 'hyperedges':
        item = {'nodes': ['x'], 'relation': 'g', 'source_file': 'src/a.py', 'note': 'n'}
    graph = {'nodes': [{'id': 'x', 'label': 'x'}], bucket: [item]}
    p = _write(tmp_path, json.dumps(graph))

    def resolve(data):
        sp = watch._StoredSourcePaths(data, out=out, project_root=project, watch_root=watch_root,
                                      normalize_source=_norm_source_file)
        return sp.existing_source_root, sp.legacy_watch_relative

    logs: dict = {}
    whole = resolve(_recording(json.loads(p.read_text(encoding='utf-8')), logs))
    assert 'source_file' in logs.get(bucket, set()), 'the fixture must make the resolver read this bucket'
    for b, log in logs.items():
        assert log <= set(PRIOR_FIELDS[b]), (b, log - set(PRIOR_FIELDS[b]))
    assert resolve(load_graph_projection(p)) == whole


SHRINK_CASES = {
    # (existing nodes, new nodes, kwargs)
    'lost semantic node': (
        [{'id': 's', 'label': 'S', 'source_file': 'n.md', '_origin': 'semantic'}, {'id': 'a', '_origin': 'ast'}],
        [{'id': 'a', '_origin': 'ast', 'label': 'a', 'source_file': 'a.py', 'x': 1}],
        {}),
    'semantic folded into its AST twin': (
        [{'id': 's', 'label': 'f', 'source_file': 'a.py', '_origin': 'semantic'}],
        [{'id': 'a_f', '_origin': 'ast', 'label': 'f', 'source_file': 'a.py', 'x': 1},
         {'id': 'z', '_origin': 'ast', 'label': 'z', 'source_file': 'z.py'}],
        {}),
    'document folded into its file node': (
        [{'id': 'doc', 'label': 'a.py', 'source_file': 'docs/x.md', '_origin': 'semantic'}],
        [{'id': 'file_a', '_origin': 'ast', 'label': 'a.py', 'source_file': 'src/a.py'}],
        {}),
    'shrink accounted by rebuilt sources': (
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py'}, {'id': 'b', '_origin': 'ast', 'source_file': 'a.py'}],
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py', 'label': 'a'}],
        {'rebuilt_sources': {'a.py'}}),
    'shrink from a failed source': (
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py'}, {'id': 'b', '_origin': 'ast', 'source_file': 'b.py'}],
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py'}],
        {'rebuilt_sources': {'a.py', 'b.py'}, 'failed_sources': {'b.py'}}),
    'unaccounted shrink': (
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py'}, {'id': 'b', '_origin': 'ast', 'source_file': 'b.py'}],
        [{'id': 'a', '_origin': 'ast', 'source_file': 'a.py'}],
        {'rebuilt_sources': {'a.py'}}),
    'growth': (
        [{'id': 'a', '_origin': 'ast'}],
        [{'id': 'a', '_origin': 'ast'}, {'id': 'b', '_origin': 'ast', 'label': 'b', 'source_file': 'b.py'}],
        {}),
    'deleted source': (
        [{'id': 's', 'label': 'S', 'source_file': 'gone.md', '_origin': 'semantic'}, {'id': 'a', '_origin': 'ast'}],
        [{'id': 'a', '_origin': 'ast'}, {'id': 'b', '_origin': 'ast'}],
        {'deleted_sources': {'gone.md'}}),
}


@pytest.mark.parametrize('case', list(SHRINK_CASES))
def test_check_shrink_on_the_projection_answers_as_on_the_whole_graph(tmp_path, capsys, case):
    existing_nodes, new_nodes, kw = SHRINK_CASES[case]
    existing = {'nodes': existing_nodes, 'links': []}
    new = {'nodes': new_nodes, 'links': [{'source': 'a', 'target': 'a', 'relation': 'r'}], 'graph': {}}
    p = _write(tmp_path, json.dumps(new), '.graph.tmp.json')
    capsys.readouterr()
    whole = watch._check_shrink(False, existing, json.loads(p.read_text(encoding='utf-8')), **kw)
    whole_err = capsys.readouterr().err
    logs: dict = {}
    projected = read_candidate(p).shrink_data
    rec = _recording(projected, logs)
    ours = watch._check_shrink(False, existing, rec, **kw)
    assert (ours, capsys.readouterr().err) == (whole, whole_err)
    assert logs.get('nodes', set()) <= set(SHRINK_NODE_FIELDS)


# --- 3.4 graph.html from the projection ---

def _html_graph(n_nodes: int, *, directed=False, multigraph=False, dup_ids=0):
    nodes = [{'id': f'n{i}', 'label': f'Node {i}', 'community': i % 3, 'community_name': f'C{i % 3}',
              'source_file': f'f{i}.py', 'file_type': 'code', 'big': 'x' * 50} for i in range(n_nodes)]
    nodes += [dict(nodes[0]) for _ in range(dup_ids)]
    links = []
    for i in range(n_nodes - 1):
        e = {'source': f'n{i}', 'target': f'n{i + 1}', 'relation': 'calls', 'confidence': 'EXTRACTED',
             'weight': 1.0}
        if i % 2:
            e['_src'], e['_tgt'] = e['target'], e['source']
        if multigraph:
            e['key'] = i % 2
        links.append(e)
    if multigraph:
        # Parallel arcs across communities: a repeated key collapses into one edge, a new key adds one,
        # so the cross-community edge counts of the aggregated view depend on `key`.
        links += [{'source': 'n0', 'target': 'n1', 'relation': 'uses', 'key': 0},
                  {'source': 'n0', 'target': 'n1', 'relation': 'uses', 'key': 7}]
    return {'directed': directed, 'multigraph': multigraph,
            'graph': {'hyperedges': [{'id': 'h', 'nodes': ['n0', 'n4', 'n5'], 'relation': 'group_of'}]},
            'nodes': nodes, 'links': links,
            'hyperedges': [{'id': 'h2', 'nodes': ['n1', 'n2'], 'relation': 'pair'}]}


def _render(tmp_path, name, data):
    d = tmp_path / name
    d.mkdir()
    action = watch._reconcile_graph_html(d, data)
    p = d / 'graph.html'
    return action, (p.read_bytes() if p.exists() else None)


@pytest.mark.parametrize('variant', [
    {}, {'directed': True}, {'multigraph': True}, {'directed': True, 'multigraph': True}])
def test_aggregated_html_from_the_projection_is_byte_identical(tmp_path, monkeypatch, variant):
    monkeypatch.setenv('GRAPHIFY_VIZ_NODE_LIMIT', '5')
    data = _html_graph(12, **variant)
    p = _write(tmp_path, json.dumps(data), 'graph.json')
    c = read_candidate(p)
    projected = c.html_data(p)
    assert projected is not None and all(set(n) <= set(gc.HTML_NODE_FIELDS) for n in projected['nodes'])
    whole = _render(tmp_path, 'whole', json.loads(p.read_text(encoding='utf-8')))
    ours = _render(tmp_path, 'ours', projected)
    assert whole[0] == 'rendered' and whole == ours


def test_small_graph_html_uses_the_whole_graph(tmp_path, monkeypatch):
    monkeypatch.setenv('GRAPHIFY_VIZ_NODE_LIMIT', '50')
    p = _write(tmp_path, json.dumps(_html_graph(12)), 'graph.json')
    full = json.loads(p.read_text(encoding='utf-8'))
    assert read_candidate(p).html_data(p) == full
    assert _render(tmp_path, 'whole', full) == _render(tmp_path, 'ours', read_candidate(p).html_data(p))


def test_duplicate_ids_under_the_limit_fall_back_to_the_whole_graph(tmp_path, monkeypatch):
    '''More elements than the limit but not more distinct ids: to_html may render in full.'''
    monkeypatch.setenv('GRAPHIFY_VIZ_NODE_LIMIT', '12')
    data = _html_graph(10, dup_ids=5)
    p = _write(tmp_path, json.dumps(data), 'graph.json')
    full = json.loads(p.read_text(encoding='utf-8'))
    got = read_candidate(p).html_data(p)
    assert got == full
    assert _render(tmp_path, 'whole', full) == _render(tmp_path, 'ours', got)


def test_html_disabled_reads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv('GRAPHIFY_VIZ_NODE_LIMIT', '0')
    p = _write(tmp_path, json.dumps(_html_graph(12)), 'graph.json')
    assert read_candidate(p).html_data(p) == {}


# --- 3.5 rebuild: no whole-text read, same output as upstream's whole-file path ---

def _rebuild_twice(corpus: Path, **kw):
    assert watch._rebuild_code(corpus, acquire_lock=False, **kw) is True
    _add_semantic_layer(corpus)
    (corpus / 'a.py').write_text('def f():\n    return 2\n\ndef h():\n    return f()\n', encoding='utf-8')


@pytest.mark.parametrize('no_cluster', [True, False], ids=['no-cluster', 'clustered'])
def test_full_rebuild_reads_no_whole_text(tmp_path, text_reads, no_cluster):
    corpus = _corpus(tmp_path)
    _rebuild_twice(corpus, no_cluster=no_cluster)
    text_reads.clear()
    assert watch._rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
    assert 'graph.json' not in text_reads and '.graph.tmp.json' not in text_reads, text_reads


@pytest.mark.parametrize('no_cluster', [True, False], ids=['no-cluster', 'clustered'])
def test_rebuild_output_equals_upstream_whole_file_path(tmp_path, monkeypatch, no_cluster):
    '''One corpus at one path, one history: element-wise reads vs the whole-file fallback (no wheel).'''
    corpus = _corpus(tmp_path)
    _rebuild_twice(corpus, no_cluster=no_cluster)
    saved = tmp_path / 'saved'
    shutil.copytree(corpus, saved)
    runs = {}
    for name in ('stream', 'whole'):
        shutil.rmtree(corpus)
        shutil.copytree(saved, corpus)
        with monkeypatch.context() as m:
            if name == 'whole':
                m.setitem(sys.modules, 'graphify_snap', None)
            assert watch._rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
        out = corpus / 'graphify-out'
        runs[name] = {f: (out / f).read_bytes() for f in ('graph.json', 'graph.html', 'GRAPH_REPORT.md')
                      if (out / f).exists()}
    assert 'graph.json' in runs['stream'] and runs['stream'].keys() == runs['whole'].keys()
    for f in runs['stream']:
        assert runs['stream'][f] == runs['whole'][f], f


def test_semantic_layer_kept_on_rebuild(tmp_path):
    '''#1915 through the projection: the notes.md concept node survives a full rebuild.

    notes_doc folds into the AST heading node of notes.md (notes_notes), as on e07e9ac5 —
    checked against the whole-file path by test_rebuild_output_equals_upstream_whole_file_path.
    '''
    corpus = _corpus(tmp_path)
    _rebuild_twice(corpus)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    ids = {n['id'] for n in json.loads((corpus / 'graphify-out' / 'graph.json').read_text(encoding='utf-8'))['nodes']}
    assert {'notes_idea', 'notes_notes'} <= ids


def test_corrupt_existing_graph_still_refused(tmp_path, capsys):
    corpus = _corpus(tmp_path)
    assert watch._rebuild_code(corpus, acquire_lock=False) is True
    g = corpus / 'graphify-out' / 'graph.json'
    g.write_text(g.read_text(encoding='utf-8')[:40], encoding='utf-8')
    before = g.read_bytes()
    assert watch._rebuild_code(corpus, acquire_lock=False) is False
    assert g.read_bytes() == before
    assert 'Delete the file and run a full rebuild.' in capsys.readouterr().err


# --- 3.6 port guard ---

def test_prior_block_holds_the_fork_call_site():
    block = gc.prior_block_source()
    assert 'load_graph_projection(existing_graph)' in block
    assert block.rstrip().endswith('del prior, prior_paths')


def test_port_hash_matches_recorded():
    assert port_hash() == PORT_HASH, (
        'upstream code the projections or the compare repeat changed (graph_compare.PORT_FUNCS or the '
        '#1915 block). Read the diff, widen the projection or carry the change, run '
        f'tests/test_graph_streaming_reads.py and tests/test_graph_compare.py, then update PORT_HASH: {port_hash()}')


def _owner(name: str):
    '''The module or class that holds the last segment of a PORT_FUNCS name.'''
    import importlib
    parent = name.rsplit('.', 1)[0]
    try:
        return importlib.import_module('graphify.' + parent)
    except ImportError:
        return gc._port_func(parent)


@pytest.mark.parametrize('name', list(PORT_FUNCS))
def test_a_changed_port_func_changes_the_hash(monkeypatch, name):
    def replacement(*a, **k):
        return None

    replacement.__qualname__ = gc._port_func(name).__qualname__
    monkeypatch.setattr(_owner(name), name.rsplit('.', 1)[1], replacement)
    assert port_hash() != PORT_HASH


def test_a_changed_prior_block_changes_the_hash(monkeypatch):
    real = inspect.getsource

    def fake(obj):
        src = real(obj)
        if obj is watch._rebuild_code:
            assert '"document", "concept"' in src
            src = src.replace('"document", "concept"', '"document", "idea"', 1)
        return src

    monkeypatch.setattr(inspect, 'getsource', fake)
    assert port_hash() != PORT_HASH


def test_a_lost_anchor_fails_the_hash(monkeypatch):
    monkeypatch.setattr(gc, 'PRIOR_BLOCK_ANCHORS', ('no such anchor', 'del prior, prior_paths'))
    with pytest.raises(ValueError):
        port_hash()


def test_snapshot_writer_streams_the_same_repeated_key_rule(tmp_path):
    '''One rule for both element-wise readers: first position, last value.'''
    from graphify.snap_writer import _FileGraph
    text = '{"nodes":[{"id":"x"}],"links":[],"nodes":[{"id":"y"}]}'
    p = _write(tmp_path, text)
    fg = _FileGraph(p)
    assert list(fg._at) == list(load_existing_graph_json(p)) == ['nodes', 'links']
    assert [n for b in fg.node_batches(10) for n in b] == load_existing_graph_json(p)['nodes']
