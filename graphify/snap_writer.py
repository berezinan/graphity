"""Snapshot writer: graph.json -> published ``graph.snap`` generation.

Input is graph.json, in its own node and link order: the file read element by
element through ``graphify_snap.GraphJson`` (any layout), or the data as a dict
(``node_link_data(G)`` for a fixture). Normalisation runs through the upstream
``serve.py`` functions; the byte layout is done by the Rust encoder in the
``graphify-snap`` wheel.

The writer never touches graph.json. Publication is a rename over the live
name done in Rust (``std::fs::rename``): CPython ``os.replace`` fails with
error 5 over a mapped file on Windows. Publications of one snapshot are
serialised by a lock file (``msvcrt.locking`` / ``fcntl.flock``) that also
holds the writer's PID for the refusal message.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from graphify.snap_store import (
    LINK_FIELDS,
    NODE_FIELDS,
    Unsupported,
    lock_path_for,
    norm_fingerprint,
    snapshot_path_for,
)

GOD_NODES_KEPT = 100
_NODE_SET = frozenset(NODE_FIELDS) | {"id"}
_LINK_SET = frozenset(LINK_FIELDS) | {"source", "target"}
_PID_OFFSET = 1  # byte 0 is the locked byte; readers may read the rest


def _dumps(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def graphify_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("graphifyy")
    except Exception:
        return "unknown"


def links_of(data: dict) -> list:
    if "links" in data:
        return data["links"]
    if "edges" in data:
        return data["edges"]
    return []


def _has_nul(d: dict) -> bool:
    return any((isinstance(v, str) and "\0" in v) or (isinstance(k, str) and "\0" in k)
               for k, v in d.items())


_HINT = "; set GRAPHIFY_GRAPH_STORE=json for this project"
_ARRAYS = ("nodes", "links", "edges")
_ABSENT = object()


def file_info(path: Path) -> tuple[int, int, bytes]:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return st.st_size, st.st_mtime_ns, h.digest()


class _DictGraph:
    """graph.json data already in memory (fixtures, ``publish_snapshot(data)``)."""

    source = None

    def __init__(self, data: dict):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def rest(self) -> dict:
        return {k: v for k, v in self._data.items() if k not in _ARRAYS}

    def _nodes(self) -> list:
        nodes = self._data.get("nodes")
        if not isinstance(nodes, list):
            raise Unsupported("graph.json without a 'nodes' list" + _HINT)
        return nodes

    def node_count(self) -> int:
        return len(self._nodes())

    def node_batches(self, n: int):
        nodes = self._nodes()
        for s in range(0, len(nodes), n):
            yield nodes[s:s + n]

    def link_batches(self, n: int):
        links = links_of(self._data)
        if not isinstance(links, list):
            raise Unsupported("graph.json whose links are not a list" + _HINT)
        for s in range(0, len(links), n):
            yield links[s:s + n]


class _FileGraph:
    """graph.json read element by element through ``graphify_snap.GraphJson``.

    Rust finds the byte range of every top-level member and array element; each
    batch of elements is parsed here with ``json.loads``, so the values are the
    ones ``json.load`` of the whole file gives. A repeated top-level key resolves
    as in ``json.load``: first position, last value. ``source`` describes exactly
    the bytes that were scanned.
    """

    def __init__(self, path):
        import graphify_snap
        self._gj = graphify_snap.GraphJson(str(path))
        self._at: dict[str, int] = {}
        for i, raw in enumerate(self._gj.keys()):
            self._at[json.loads(raw)] = i
        size, mtime_ns, sha = self._gj.source
        self.source = (size, mtime_ns, bytes(sha))
        self._values: dict[int, object] = {}

    def _value(self, key):
        i = self._at[key]
        if i not in self._values:
            try:
                self._values[i] = json.loads(self._gj.value(i))
            except json.JSONDecodeError as exc:
                raise ValueError(f"graph.json is damaged in top-level member {key!r}: {exc.msg}") from None
        return self._values[i]

    def get(self, key, default=None):
        return self._value(key) if key in self._at else default

    def rest(self) -> dict:
        return {k: self._value(k) for k in self._at if k not in _ARRAYS}

    def _array(self, key: str | None, what: str) -> tuple[int | None, int]:
        if key is None:
            return None, 0
        i = self._at[key]
        n = self._gj.count(i)
        if n is None:
            raise Unsupported(f"graph.json whose {what} are not a list" + _HINT)
        return i, n

    def _links_key(self) -> str | None:
        return "links" if "links" in self._at else "edges" if "edges" in self._at else None

    def node_count(self) -> int:
        if "nodes" not in self._at:
            raise Unsupported("graph.json without a 'nodes' list" + _HINT)
        return self._array("nodes", "nodes")[1]

    def _batches(self, i: int | None, total: int, n: int):
        for s in range(0, total, n):
            raw = self._gj.batch(i, s, n)
            try:
                items = json.loads(raw)
            except json.JSONDecodeError as exc:
                off = self._gj.element_offset(i, s) + exc.pos - 1
                # exc.pos counts characters; equal to bytes for to_json's ASCII output.
                raise ValueError(f"graph.json is damaged near byte {off}: {exc.msg}") from None
            yield items

    def node_batches(self, n: int):
        yield from self._batches(*self._array("nodes", "nodes"), n)

    def link_batches(self, n: int):
        yield from self._batches(*self._array(self._links_key(), "links"), n)


class _LightNodes:
    """``G.nodes[nid]`` with the only attributes the ``analyze.god_nodes`` filters read."""

    def __init__(self, idx, labels, files):
        self._idx, self._labels, self._files = idx, labels, files

    def __getitem__(self, nid):
        i = self._idx[nid]
        d = {}
        if self._labels[i] is not _ABSENT:
            d["label"] = self._labels[i]
        if self._files[i] is not _ABSENT:
            d["source_file"] = self._files[i]
        return d


class _GodGraph:
    """The slice of nx.DiGraph that ``analyze.god_nodes`` reads."""

    def __init__(self, ids, idx, labels, files, deg):
        self._ids, self._idx, self._deg = ids, idx, deg
        self.nodes = _LightNodes(idx, labels, files)

    def degree(self, n=None):
        if n is None:
            return list(zip(self._ids, self._deg))
        return self._deg[self._idx[n]]


def _node_columns(d: dict) -> tuple:
    from graphify.serve import (
        _node_attributes_text, _node_rationale_text, _search_tokens, _strip_diacritics,
    )
    nid = d["id"]
    label = d.get("label")
    nl = d.get("norm_label") or _strip_diacritics(label or "").lower()
    if not isinstance(nl, str):
        raise Unsupported(f"norm_label that is not a string on {nid!r}")
    nli = (d.get("norm_label") or _strip_diacritics(d.get("label") or "")).lower()
    sf = d.get("source_file")
    if sf is not None and not isinstance(sf, str) and sf:
        raise Unsupported(f"source_file that is not a string on {nid!r}")
    atn, att = _node_attributes_text(d)
    search = (
        nl,
        "" if nli == nl else nli,
        " ".join(_search_tokens(label or "")),
        (sf or "").lower(),
        " ".join(_search_tokens(sf or "")),
        _node_rationale_text(d),
        atn,
        att,
    )
    low = nid.lower()
    fold = low if nid.isascii() else _strip_diacritics(nid).lower()
    vals = tuple(_dumps(d[f]) if f in d else None for f in NODE_FIELDS)
    rest = {k: v for k, v in d.items() if k not in _NODE_SET}
    return (nid, low if low != nid else None, fold if fold != low else None, search, vals,
            len(label or nid), _dumps(rest) if rest else None)


def encode(graph, out_path: Path, *, generation: int = 1, source: Path | None = None,
           batch: int = 50_000) -> dict:
    """Write one snapshot file at ``out_path`` (not published). Returns the report.

    ``graph`` is graph.json data (a dict; ``source`` then names the file it came
    from) or a ``_FileGraph`` read element by element. Both give the same bytes.
    The graph is never held whole: nodes and links are checked, normalised and
    encoded batch by batch, and only what is needed after the pass stays — the
    id index, degrees, communities, and label / source_file for ``god_nodes``.
    """
    import graphify_snap
    from graphify.analyze import god_nodes as _god_nodes

    g = graph if isinstance(graph, _FileGraph) else _DictGraph(graph)
    times: dict[str, float] = collections.defaultdict(float)
    t_all = time.perf_counter()
    if g.get("multigraph", True):
        raise Unsupported("multigraph graphs (or a file without 'multigraph': false)" + _HINT)
    n_nodes = g.node_count()
    directed = bool(g.get("directed", False))
    enc = graphify_snap.Encoder(NODE_FIELDS, LINK_FIELDS, directed)
    idx: dict[str, int] = {}
    ids: list[str] = []
    labels: list = []
    files: list = []
    communities: dict[int, list[int]] = {}
    part = hashlib.sha256()
    for chunk_nodes in g.node_batches(batch):
        t = time.perf_counter()
        for d in chunk_nodes:
            if not isinstance(d, dict):
                raise Unsupported("node that is not a JSON object" + _HINT)
            nid = d.get("id")
            if not isinstance(nid, str):
                raise Unsupported(f"node id that is not a string: {nid!r}" + _HINT)
            if nid in idx:
                raise Unsupported(f"duplicate node id {nid!r}" + _HINT)
            if _has_nul(d):
                raise Unsupported(f"strings with NUL on node {nid!r}, which the format cannot hold" + _HINT)
            i = len(ids)
            idx[nid] = i
            ids.append(nid)
            labels.append(d.get("label", _ABSENT))
            files.append(d.get("source_file", _ABSENT))
            part.update(_dumps([d.get("community"), d.get("community_name")]).encode())
            cid = d.get("community")
            if cid is not None:
                try:
                    key = int(cid)
                except (TypeError, ValueError):
                    raise Unsupported(f"community that int() rejects: {cid!r}") from None
                communities.setdefault(key, []).append(i)
        rows = [_node_columns(d) for d in chunk_nodes]
        cols = list(zip(*rows))
        search = [list(c) for c in zip(*cols[3])]
        vals = [list(c) for c in zip(*cols[4])]
        times["normalize"] += time.perf_counter() - t
        t = time.perf_counter()
        try:
            enc.add_nodes(list(cols[0]), list(cols[1]), list(cols[2]), search, vals,
                          list(cols[5]), list(cols[6]))
        except (ValueError, UnicodeEncodeError) as exc:
            raise Unsupported(f"node strings the format cannot hold: {exc}") from None
        times["encode"] += time.perf_counter() - t

    if len(ids) != n_nodes:
        raise ValueError(f"graph.json changed while it was read ({n_nodes} nodes expected, {len(ids)} read)")

    deg = [0] * n_nodes
    conf = collections.Counter()
    # Parallel links of one direction, as build.has_parallel_links(directed=True):
    # ids are unique and endpoints are checked, so a pair is its two node numbers.
    pairs: set[int] = set()
    n_links = 0
    for chunk in g.link_batches(batch):
        t = time.perf_counter()
        src: list[int] = []
        tgt: list[int] = []
        for l in chunk:
            if not isinstance(l, dict):
                raise Unsupported("link that is not a JSON object" + _HINT)
            s, e = idx.get(l.get("source")), idx.get(l.get("target"))
            if s is None or e is None:
                raise Unsupported("link endpoint that is not a node" + _HINT)
            if "_src" in l or "_tgt" in l:
                raise Unsupported("links with stored _src/_tgt markers" + _HINT)
            if _has_nul(l):
                raise Unsupported("link strings with NUL, which the format cannot hold" + _HINT)
            key = s * n_nodes + e
            if key in pairs:
                raise Unsupported("parallel links of one direction" + _HINT)
            pairs.add(key)
            src.append(s)
            tgt.append(e)
            deg[s] += 1
            deg[e] += 1
        n_links += len(chunk)
        conf.update(l.get("confidence", "EXTRACTED") for l in chunk)
        vals = [[_dumps(l[f]) if f in l else None for l in chunk] for f in LINK_FIELDS]
        cold = []
        for l in chunk:
            rest = {k: v for k, v in l.items() if k not in _LINK_SET}
            cold.append(_dumps(rest) if rest else None)
        times["normalize"] += time.perf_counter() - t
        t = time.perf_counter()
        try:
            enc.add_links(src, tgt, vals, cold)
        except (ValueError, UnicodeEncodeError) as exc:
            raise Unsupported(f"link strings the format cannot hold: {exc}") from None
        times["encode"] += time.perf_counter() - t

    del pairs
    t = time.perf_counter()
    gods = _god_nodes(_GodGraph(ids, idx, labels, files, deg), top_n=GOD_NODES_KEPT)
    cids = list(communities)
    comm_off = [0]
    comm_mem: list[int] = []
    for c in cids:
        comm_mem.extend(communities[c])
        comm_off.append(len(comm_mem))
    stats = {
        "nodes": n_nodes,
        "edges": n_links,
        "confidence": {k: v for k, v in conf.items() if isinstance(k, str)},
        "god_nodes": [{"label": gn["label"], "degree": gn["degree"]} for gn in gods],
        "god_nodes_kept": GOD_NODES_KEPT,
        "cids": cids,
        "built_at_commit": g.get("built_at_commit"),
    }
    meta = {"node_fields": NODE_FIELDS, "link_fields": LINK_FIELDS, "directed": directed,
            "graphify_version": graphify_version()}
    graph_rest = g.rest()
    if g.source is not None:
        size, mtime_ns, sha = g.source
    elif source is not None:
        size, mtime_ns, sha = file_info(source)
    else:
        size, mtime_ns, sha = 0, 0, bytes(32)
    times["stats"] += time.perf_counter() - t
    rust = enc.finish(str(out_path), generation, norm_fingerprint(), size, mtime_ns, sha,
                      part.digest(), _dumps(meta), _dumps(stats), _dumps(graph_rest),
                      comm_off, comm_mem)
    for k, v in rust:
        times[k] += v
    times["total"] = time.perf_counter() - t_all
    return {"nodes": n_nodes, "links": n_links, "bytes": out_path.stat().st_size,
            "generation": generation, "times": dict(times)}


def _try_lock(f) -> bool:
    f.seek(0)
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(f) -> None:
    try:
        f.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def writer_lock(snapshot: Path, timeout: float = 600.0):
    """Exclusive writer lock of one snapshot; the holder's PID is stored in the file."""
    path = lock_path_for(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    deadline = time.monotonic() + timeout
    try:
        while not _try_lock(f):
            if time.monotonic() > deadline:
                raise TimeoutError(f"snapshot lock {path} is held (pid {writer_pid(snapshot)})") from None
            time.sleep(0.05)
        try:
            f.seek(_PID_OFFSET)
            f.truncate(_PID_OFFSET)
            f.write(str(os.getpid()).encode())
            f.flush()
        except OSError:
            pass
        yield
    finally:
        _unlock(f)
        f.close()


def writer_pid(snapshot: Path) -> int | None:
    """PID of the process that currently holds the writer lock, else ``None``."""
    path = lock_path_for(snapshot)
    try:
        with open(path, "a+b") as f:
            if _try_lock(f):
                _unlock(f)
                return None
            f.seek(_PID_OFFSET)
            raw = f.read(32).strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def publish_snapshot(data: dict, out_dir: Path, *, name: str = "graph.snap",
                     source: Path | None = None, crash_before_publish: bool = False) -> dict:
    """Write the next generation and replace ``out_dir/name`` over."""
    import graphify_snap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = out_dir / name
    with writer_lock(live):
        removed = []
        for stale in out_dir.glob(f"{name}.*.tmp"):
            try:
                stale.unlink()
                removed.append(stale.name)
            except OSError:
                pass
        gen = (graphify_snap.peek_generation(str(live)) or 0) + 1
        tmp = out_dir / f"{name}.{os.getpid()}.tmp"
        try:
            report = encode(data, tmp, generation=gen, source=source)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if crash_before_publish:
            os._exit(3)
        t = time.perf_counter()
        graphify_snap.publish(str(tmp), str(live))
        report["times"]["publish"] = time.perf_counter() - t
        report["removed_tmp"] = removed
        report["path"] = str(live)
    return report


def build_snapshot(graph_json: str | Path) -> dict:
    """Build and publish ``<graph>.snap`` next to ``graph_json`` from the written file.

    ``graph_json`` is read element by element, whatever its layout (indented or
    one line, as ``to_json`` writes it), and never modified.
    """
    src = Path(graph_json)
    t = time.perf_counter()
    graph = _FileGraph(src)
    scan_s = time.perf_counter() - t
    live = snapshot_path_for(src)
    report = publish_snapshot(graph, live.parent, name=live.name)
    report["times"]["scan"] = scan_s
    return report


def snap_build_command(args):
    '''graphify snap build [<project-dir|graph.json>]; returns the exit code.'''
    from graphify.snap_store import check_api
    if not args or args[0] != 'build':
        print('Usage: graphify snap build [<project-dir|graph.json>]', file=sys.stderr)
        return 2
    if len(args) > 1:
        p = Path(args[1])
        if p.is_dir():
            cand = p / 'graphify-out' / 'graph.json'
            gj = cand if cand.exists() else p / 'graph.json'
        else:
            gj = p
    else:
        gj = Path('graphify-out') / 'graph.json'
    if not gj.exists():
        print('graph.json not found: ' + str(gj), file=sys.stderr)
        return 1
    check_api()
    try:
        rep = build_snapshot(gj)
    except ValueError as exc:  # Unsupported shape, or a damaged graph.json
        print('error: cannot build a snapshot of ' + str(gj) + ': ' + str(exc), file=sys.stderr)
        return 1
    secs = rep['times']['total'] + rep['times'].get('scan', 0.0)
    print('snapshot ' + rep['path'] + ': ' + str(rep['nodes']) + ' nodes, ' + str(rep['links']) + ' links, ' + str(rep['bytes']) + ' bytes, ' + format(secs, '.1f') + 's')
    return 0
