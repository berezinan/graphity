"""Fork delta (rebuild-single-graph-parse): one parse of the existing graph.json per
rebuild, and the "same graph" / "same topology" checks decided by fingerprints
instead of graph-sized canonical strings.

``graph_fingerprint(a) == graph_fingerprint(b)`` decides exactly what
``json.dumps(_canonical_graph_for_compare(x), sort_keys=True, ensure_ascii=False)``
string equality decides upstream (same for topology), up to a sha256 collision:
two canonical strings are equal iff their non-list parts serialize equally and,
for every list, the multisets of element serializations are equal. Each element
is hashed from that same serialization; the sorted digests stand for the multiset.
An input upstream cannot serialize raises here too, so the caller's ``except``
gives the same "changed" answer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# path -> (size, mtime_ns, fingerprint or None) of the last graph.json parsed by
# load_existing_graph_json, taken before reconcile stamps _origin on its items.
_FINGERPRINTS: dict[str, tuple[int, int, bytes | None]] = {}

_GRAPH_LIST_KEYS = ("nodes", "links", "edges", "hyperedges")


def _dump(value) -> bytes:
    # surrogatepass: json.loads can yield lone surrogates; this keeps the
    # encoding injective where plain utf-8 would raise.
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8", "surrogatepass")


def _multiset(items, norm=None) -> bytes:
    digests = []
    for item in items:
        if norm is not None:
            item = norm(item)
            if item is None:
                continue
        digests.append(hashlib.sha256(_dump(item)).digest())
    return _finish_digests(digests)


def _finish_digests(digests: list) -> bytes:
    digests.sort()
    h = hashlib.sha256(len(digests).to_bytes(8, "little"))
    for d in digests:
        h.update(d)
    return h.digest()


def _combine(parts: dict) -> bytes:
    h = hashlib.sha256()
    for key in sorted(parts):
        tag, digest = parts[key]
        kb = _dump(key)
        h.update(len(kb).to_bytes(8, "little"))
        h.update(kb)
        h.update(tag)
        h.update(digest)
    return h.digest()


def _plain(value) -> tuple[bytes, bytes]:
    return b"V", hashlib.sha256(_dump(value)).digest()


def graph_fingerprint(graph_data: dict) -> bytes:
    """Fingerprint equal iff upstream's ``_canonical_graph_for_compare`` strings are equal."""
    canonical = dict(graph_data)
    canonical.pop("built_at_commit", None)
    canonical["directed"] = bool(canonical.get("directed", False))
    parts = {}
    for key, value in canonical.items():
        if key in _GRAPH_LIST_KEYS and isinstance(value, list):
            parts[key] = (b"L", _multiset(value))
        else:
            parts[key] = _plain(value)
    return _combine(parts)


def _topology_node(node):
    if not isinstance(node, dict):
        return None
    n = dict(node)
    n.pop("community", None)
    n.pop("community_name", None)
    n.pop("norm_label", None)
    return n


def _topology_edge(edge):
    if not isinstance(edge, dict):
        return None
    e = dict(edge)
    true_src = e.pop("_src", None)
    true_tgt = e.pop("_tgt", None)
    if true_src is not None and true_tgt is not None:
        e["source"] = true_src
        e["target"] = true_tgt
    e.pop("confidence_score", None)
    return e


def topology_fingerprint(graph_data: dict) -> bytes:
    """Fingerprint equal iff upstream's ``_canonical_topology_for_compare`` strings are equal."""
    canonical = dict(graph_data)
    canonical.pop("built_at_commit", None)
    graph_meta = canonical.get("graph")
    if isinstance(graph_meta, dict):
        graph_meta = dict(graph_meta)
        graph_meta.pop("schema_version", None)
        graph_meta.pop("graphify_version", None)
        canonical["graph"] = graph_meta
    parts = {}
    for key, value in canonical.items():
        if not isinstance(value, list):
            parts[key] = _plain(value)
        elif key == "nodes":
            parts[key] = (b"L", _multiset(value, _topology_node))
        elif key in ("links", "edges"):
            parts[key] = (b"L", _multiset(value, _topology_edge))
        elif key == "hyperedges":
            parts[key] = (b"L", _multiset(value))
        else:
            parts[key] = _plain(value)
    return _combine(parts)


def _stat(path: Path) -> tuple[str, int, int]:
    st = os.stat(path)
    return str(Path(path).resolve()), st.st_size, st.st_mtime_ns


def _fingerprint_or_none(data) -> bytes | None:
    try:
        return graph_fingerprint(data)
    except Exception:
        return None


# --- element-wise reads (streaming-rebuild-graph-reads) -------------------------
#
# A file is read member by member and array element by array element through
# graphify_snap.GraphJson; every byte still goes through json.loads, in batches,
# so the values are those json.loads of the whole file gives (a JSON element is
# self-contained). A repeated top-level key keeps its first position and its last
# value, as in json.loads; shadowed values are parsed too and dropped, so a damaged
# one fails the read as it would fail the whole-file parse. Any failure of the
# element-wise read falls back to upstream's whole-file read, which gives upstream's
# result or upstream's exception with the same type and text.

_BATCH = 20000


def _loads(raw: bytes):
    # Strict UTF-8 first, as read_text(encoding="utf-8") decodes: json.loads(bytes)
    # would decode with surrogatepass and accept bytes the whole-file read rejects.
    return json.loads(bytes(raw).decode("utf-8"))


def _stream(path: Path, sink) -> tuple[int, int]:
    import graphify_snap
    gj = graphify_snap.GraphJson(str(path))
    keys = [_loads(raw) for raw in gj.keys()]
    last = {k: i for i, k in enumerate(keys)}
    for i, k in enumerate(keys):
        if last[k] != i:
            n = gj.count(i)
            if n is None:
                _loads(gj.value(i))
            else:
                for s in range(0, n, _BATCH):
                    _loads(gj.batch(i, s, _BATCH))
    counts = {k: gj.count(last[k]) for k in last}
    sink.begin(counts)
    for k in dict.fromkeys(keys):
        i = last[k]
        n = counts[k]
        if n is None:
            sink.value(k, _loads(gj.value(i)))
            continue
        sink.array(k)
        for s in range(0, n, _BATCH):
            sink.items(k, _loads(gj.batch(i, s, _BATCH)))
    size, mtime_ns, _sha = gj.source
    return size, mtime_ns


def _feed(raw, sink) -> None:
    """Feed an already-parsed document to a sink, as _stream would have."""
    if not isinstance(raw, dict):
        sink.not_an_object(raw)
        return
    sink.begin({k: (len(v) if isinstance(v, list) else None) for k, v in raw.items()})
    for k, v in raw.items():
        if isinstance(v, list):
            sink.array(k)
            sink.items(k, v)
        else:
            sink.value(k, v)


def _try_stream(path: Path, make_sink):
    """(sink, (size, mtime_ns)) from the element-wise read, or None when it is not possible."""
    sink = make_sink()
    try:
        stamp = _stream(path, sink)
    except MemoryError:
        raise
    except Exception:
        return None
    return sink, stamp


def _read(path: Path, make_sink):
    """A sink fed with the document: element-wise, else from upstream's whole-file read."""
    done = _try_stream(path, make_sink)
    if done is not None:
        return done[0]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sink = make_sink()
    _feed(raw, sink)
    return sink


def _project(item, fields, intern):
    if not isinstance(item, dict):
        return item
    return {f: intern(item[f]) for f in fields if f in item}


def _interner():
    seen: dict = {}

    def intern(v):
        return seen.setdefault(v, v) if type(v) is str else v
    return intern


class _Fingerprint:
    """graph_fingerprint built from batches: element digests for the list keys, values otherwise."""

    def __init__(self):
        self.top: dict = {}
        self.digests: dict = {}
        self.ok = True
        self.raw = _UNSET

    def begin(self, counts):
        pass

    def not_an_object(self, raw):
        self.raw = raw

    def value(self, k, v):
        self.top[k] = v

    def array(self, k):
        if k in _GRAPH_LIST_KEYS:
            self.top[k] = None
            self.digests[k] = []
        else:
            self.top[k] = []

    def items(self, k, batch):
        if k not in self.digests:
            self.top[k].extend(batch)
            return
        if not self.ok:
            return
        try:
            self.digests[k].extend(hashlib.sha256(_dump(x)).digest() for x in batch)
        except Exception:
            self.ok = False

    def result(self) -> bytes | None:
        if self.raw is not _UNSET:
            return _fingerprint_or_none(self.raw)
        if not self.ok:
            return None
        try:
            parts = {}
            for k, v in self.top.items():
                if k in ("built_at_commit", "directed"):
                    continue
                parts[k] = (b"L", _finish_digests(self.digests[k])) if k in self.digests else _plain(v)
            parts["directed"] = _plain(bool(self.top.get("directed", False)))
            return _combine(parts)
        except Exception:
            return None


class _FullSink:
    """The dict json.loads gives, plus its graph_fingerprint."""

    def __init__(self):
        self.data: dict = {}
        self.fp = _Fingerprint()

    def begin(self, counts):
        pass

    def value(self, k, v):
        self.data[k] = v
        self.fp.value(k, v)

    def array(self, k):
        self.data[k] = lst = []
        self.fp.array(k)
        if k not in self.fp.digests:
            self.fp.top[k] = lst  # an array outside the list keys: its value, shared

    def items(self, k, batch):
        self.data[k].extend(batch)
        if k in self.fp.digests:
            self.fp.items(k, batch)

    def not_an_object(self, raw):
        self.data = raw


class _ProjectionSink:
    """The document with array elements cut down to the given fields (other arrays and values whole)."""

    def __init__(self, fields: dict):
        self.fields = fields
        self.data: dict = {}
        self.intern = _interner()

    def begin(self, counts):
        pass

    def value(self, k, v):
        self.data[k] = v

    def array(self, k):
        self.data[k] = []

    def items(self, k, batch):
        f = self.fields.get(k)
        if f is None:
            self.data[k].extend(batch)
        else:
            self.data[k].extend(_project(x, f, self.intern) for x in batch)

    def not_an_object(self, raw):
        self.data = raw


# Fields the #1915 block of watch._rebuild_code reads from the parsed graph:
# _StoredSourcePaths reads source_file of every bucket; the block reads, per node,
# _is_ast_tier (_origin, source_location), file_type and source_file.
PRIOR_FIELDS = {
    "nodes": ("source_file", "file_type", "_origin", "source_location"),
    "links": ("source_file",),
    "edges": ("source_file",),
    "hyperedges": ("source_file",),
}

# Fields _check_shrink (with build._lost_saved_semantic_nodes) reads from the new graph.
SHRINK_NODE_FIELDS = ("id", "_origin", "source_file", "label")

# Fields watch._reconcile_graph_html reads when the graph is over the viz node limit:
# communities and names per node, and the edges (direction markers, multigraph key)
# that load_node_link_graph and the aggregated to_html view use.
HTML_NODE_FIELDS = ("id", "community", "community_name")
HTML_LINK_FIELDS = ("source", "target", "_src", "_tgt", "key")


def load_graph_projection(path: Path, fields: dict = PRIOR_FIELDS):
    """The graph with array elements cut down to ``fields``; upstream's whole-file read on any failure."""
    return _read(path, lambda: _ProjectionSink(fields)).data


_UNSET = object()


class _CandidateSink:
    """One pass over the written graph: fingerprint, the nodes _check_shrink reads, the HTML input."""

    def __init__(self):
        self.fp = _Fingerprint()
        self.intern = _interner()
        self.shrink: dict = {}
        self.html: dict | None = None
        self.full: dict | None = None
        self.raw = _UNSET

    def begin(self, counts):
        from graphify.exporters.html import _viz_node_limit
        limit = _viz_node_limit()
        n_nodes = counts.get("nodes")
        if limit <= 0:
            pass  # no HTML: _reconcile_graph_html does not read the graph
        elif n_nodes is not None and n_nodes > limit:
            self.html = {}
        else:
            self.full = {}  # a full render needs every attribute; the graph is small

    def value(self, k, v):
        self.fp.value(k, v)
        if k == "nodes":
            self.shrink["nodes"] = v
        for d in (self.html, self.full):
            if d is not None:
                d[k] = v

    def array(self, k):
        self.fp.array(k)
        if k == "nodes":
            self.shrink["nodes"] = []
        for d in (self.html, self.full):
            if d is not None:
                d[k] = []

    def items(self, k, batch):
        self.fp.items(k, batch)
        if k == "nodes":
            self.shrink["nodes"].extend(_project(x, SHRINK_NODE_FIELDS, self.intern) for x in batch)
        if self.full is not None:
            self.full[k].extend(batch)
        if self.html is not None:
            f = HTML_NODE_FIELDS if k == "nodes" else HTML_LINK_FIELDS if k in ("links", "edges") else None
            self.html[k].extend(batch if f is None else (_project(x, f, self.intern) for x in batch))

    def not_an_object(self, raw):
        self.raw = raw


class Candidate:
    """What the clustered write path of _rebuild_code needs from the written graph.tmp.json.

    ``fingerprint`` — graph_fingerprint of the file (None: upstream's compare would have raised);
    ``shrink_data`` — the graph with nodes cut down to what _check_shrink reads;
    ``html_data()`` — the input for _reconcile_graph_html.
    """

    def __init__(self, sink: _CandidateSink):
        # Keep only the results: the element digests and the string interner go with the sink.
        self._raw = sink.raw
        self._html, self._full = sink.html, sink.full
        if sink.raw is not _UNSET:
            # Not an object: hand every consumer the parsed value, as upstream did.
            self.fingerprint = _fingerprint_or_none(sink.raw)
            self.shrink_data = sink.raw
            return
        self.fingerprint = sink.fp.result()
        self.shrink_data = sink.shrink

    def html_data(self, persisted: Path):
        """Input for _reconcile_graph_html; ``persisted`` is where the candidate now lives (graph.json)."""
        if self._raw is not _UNSET:
            return self._raw
        if self._full is not None:
            return self._full
        if self._html is None:
            return {}  # HTML disabled: _reconcile_graph_html does not read the graph
        # The projection is valid only when to_html surely takes the aggregated branch:
        # more distinct node ids than the limit means G has more nodes than the limit.
        from graphify.exporters.html import _viz_node_limit
        nodes = self._html.get("nodes")
        try:
            distinct = len({n.get("id") for n in nodes if isinstance(n, dict)}) if isinstance(nodes, list) else 0
        except TypeError:
            distinct = 0  # an unhashable id: let the full graph give upstream's error
        if distinct > _viz_node_limit():
            return self._html
        return _read(Path(persisted), _FullSink).data


def read_candidate(path: Path) -> Candidate:
    """Read the freshly written graph once; upstream's whole-file read on any failure."""
    return Candidate(_read(path, _CandidateSink))


def load_existing_graph_json(path: Path) -> dict | None:
    """Parse graph.json once for reconcile, failing closed as ``build._load_existing_graph`` (#2251).

    None when the file does not exist; ValueError over the size cap; RuntimeError
    with upstream's message when it cannot be read or parsed. Records the file's
    fingerprint for ``existing_graph_fingerprint``.
    """
    if not path.exists():
        return None
    from graphify.security import check_graph_file_size_cap
    check_graph_file_size_cap(path)
    streamed = _try_stream(path, _FullSink)
    if streamed is not None:
        sink, stamp = streamed
        data = sink.data
        # The shape accesses build._load_existing_graph makes, as below.
        links_key = "links" if "links" in data else "edges"
        list(data.get("nodes", []))
        list(data.get(links_key, []))
        list(data.get("hyperedges", []))
        _FINGERPRINTS[str(Path(path).resolve())] = (stamp[0], stamp[1], sink.fp.result())
        return data
    try:
        before = _stat(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        after = _stat(path)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Cannot read {path} for incremental merge: {exc}. "
            "Delete the file and run a full rebuild."
        ) from exc
    # The shape accesses build._load_existing_graph makes, so a malformed
    # document fails here exactly as it did there.
    links_key = "links" if "links" in data else "edges"
    list(data.get("nodes", []))
    list(data.get(links_key, []))
    list(data.get("hyperedges", []))
    if before == after:
        _FINGERPRINTS[before[0]] = (before[1], before[2], _fingerprint_or_none(data))
    return data


def existing_graph_fingerprint(path: Path) -> bytes | None:
    """Fingerprint of graph.json as it is on disk now.

    Taken from the parse in ``load_existing_graph_json`` when the file has not
    changed since (size and mtime_ns), else from a fresh parse. A read or parse
    failure raises, as upstream's re-read did; None when the parsed document
    cannot be fingerprinted (upstream's compare raised and answered "changed").
    """
    from graphify.security import check_graph_file_size_cap
    check_graph_file_size_cap(path)
    key, size, mtime_ns = _stat(path)
    saved = _FINGERPRINTS.get(key)
    if saved is not None and saved[:2] == (size, mtime_ns):
        return saved[2]
    return _read(path, _Fingerprint).result()


# Upstream functions this module mirrors, or whose field reads the projections
# (PRIOR_FIELDS, SHRINK_NODE_FIELDS, HTML_*_FIELDS) repeat. PORT_HASH is their source
# hash plus the text of the #1915 block of watch._rebuild_code, recorded after the
# equivalence and projection tests passed against them.
PORT_FUNCS = (
    "watch._canonical_graph_for_compare",
    "watch._canonical_topology_for_compare",
    "build._load_existing_graph",
    # #1915 block: what the source-path resolver and the tier test read.
    "watch._StoredSourcePaths.__init__",
    "watch._StoredSourcePaths._anchors_stored_sources",
    "build._is_ast_tier",
    # _check_shrink on the new graph.
    "watch._check_shrink",
    "build._lost_saved_semantic_nodes",
    "build._is_file_node_label",
    # graph.html from the written graph, aggregated view.
    "watch._reconcile_graph_html",
    "watch._node_community_map",
    "paths.load_node_link_graph",
    "paths.restore_arc_direction",
    "exporters.html.to_html",
)

# The #1915 block lives inside _rebuild_code, which every merge changes; its text
# between these anchors is hashed instead. A missing anchor fails the hash.
PRIOR_BLOCK_ANCHORS = ("semantic_doc_files: set[Path] = set()", "del prior, prior_paths")


def prior_block_source() -> str:
    import inspect
    from graphify import watch
    src = inspect.getsource(watch._rebuild_code)
    start = src.index(PRIOR_BLOCK_ANCHORS[0])
    end = src.index(PRIOR_BLOCK_ANCHORS[1], start) + len(PRIOR_BLOCK_ANCHORS[1])
    return src[start:end]


def _port_func(name: str):
    """'mod.sub.Class.attr' -> object; the longest importable graphify.<prefix> is the module."""
    import importlib
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module("graphify." + ".".join(parts[:cut]))
        except ImportError:
            continue
        for part in parts[cut:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(name)


def port_hash() -> str:
    from graphify.snap_store import _sources_hash
    h = _sources_hash([_port_func(n) for n in PORT_FUNCS])
    h.update(prior_block_source().encode())
    return h.hexdigest()


PORT_HASH = "ab4b6bbe617c472bcc00c2fdedb3fa672117cf7e16b3cefc13a6b0def802f5cd"
