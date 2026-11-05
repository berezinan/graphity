"""Snapshot writer: graph.json data -> published ``graph.snap`` generation.

Input is graph.json as data (``graph_stream.load`` of the file, or
``node_link_data(G)`` for a fixture), in its own node and link order.
Normalisation runs through the upstream ``serve.py`` functions; the byte layout
is done by the Rust encoder in the ``graphify-snap`` wheel.

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


def check_supported(data: dict) -> None:
    """Refuse the graph shapes the snapshot does not reproduce."""
    from graphify.build import has_parallel_links
    hint = "; set GRAPHIFY_GRAPH_STORE=json for this project"
    if data.get("multigraph", True):
        raise Unsupported("multigraph graphs (or a file without 'multigraph': false)" + hint)
    links = links_of(data)
    seen = set()
    for d in data["nodes"]:
        nid = d.get("id")
        if not isinstance(nid, str):
            raise Unsupported(f"node id that is not a string: {nid!r}" + hint)
        if nid in seen:
            raise Unsupported(f"duplicate node id {nid!r}" + hint)
        if _has_nul(d):
            raise Unsupported(f"strings with NUL on node {nid!r}, which the format cannot hold" + hint)
        seen.add(nid)
    for l in links:
        if l.get("source") not in seen or l.get("target") not in seen:
            raise Unsupported("link endpoint that is not a node" + hint)
        if "_src" in l or "_tgt" in l:
            raise Unsupported("links with stored _src/_tgt markers" + hint)
        if _has_nul(l):
            raise Unsupported("link strings with NUL, which the format cannot hold" + hint)
    if has_parallel_links(links, directed=True):
        raise Unsupported("parallel links of one direction" + hint)


def partition_hash(nodes) -> bytes:
    """sha256 of (community, community_name) per node, in node order (information only)."""
    h = hashlib.sha256()
    for d in nodes:
        h.update(_dumps([d.get("community"), d.get("community_name")]).encode())
    return h.digest()


def file_info(path: Path) -> tuple[int, int, bytes]:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return st.st_size, st.st_mtime_ns, h.digest()


class _GodGraph:
    """The slice of nx.DiGraph that ``analyze.god_nodes`` reads."""

    def __init__(self, ids, nodes, deg):
        self._ids, self._deg = ids, deg
        self._pos = {nid: i for i, nid in enumerate(ids)}
        self.nodes = {nid: d for nid, d in zip(ids, nodes)}

    def degree(self, n=None):
        if n is None:
            return list(zip(self._ids, self._deg))
        return self._deg[self._pos[n]]


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


def encode(data: dict, out_path: Path, *, generation: int = 1, source: Path | None = None,
           batch: int = 50_000) -> dict:
    """Write one snapshot file at ``out_path`` (not published). Returns the report."""
    import graphify_snap
    from graphify.analyze import god_nodes as _god_nodes

    times: dict[str, float] = collections.defaultdict(float)
    t_all = time.perf_counter()
    check_supported(data)
    nodes = data["nodes"]
    links = links_of(data)
    directed = bool(data.get("directed", False))
    enc = graphify_snap.Encoder(NODE_FIELDS, LINK_FIELDS, directed)
    idx: dict[str, int] = {}
    communities: dict[int, list[int]] = {}
    for start in range(0, len(nodes), batch):
        t = time.perf_counter()
        chunk_nodes = nodes[start:start + batch]
        rows = [_node_columns(d) for d in chunk_nodes]
        for k, d in enumerate(chunk_nodes):
            i = start + k
            idx[d["id"]] = i
            cid = d.get("community")
            if cid is not None:
                try:
                    key = int(cid)
                except (TypeError, ValueError):
                    raise Unsupported(f"community that int() rejects: {cid!r}") from None
                communities.setdefault(key, []).append(i)
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

    deg = [0] * len(nodes)
    conf = collections.Counter()
    for start in range(0, len(links), batch):
        t = time.perf_counter()
        chunk = links[start:start + batch]
        src = [idx[l["source"]] for l in chunk]
        tgt = [idx[l["target"]] for l in chunk]
        for s, g in zip(src, tgt):
            deg[s] += 1
            deg[g] += 1
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

    t = time.perf_counter()
    ids = [d["id"] for d in nodes]
    gods = _god_nodes(_GodGraph(ids, nodes, deg), top_n=GOD_NODES_KEPT)
    cids = list(communities)
    comm_off = [0]
    comm_mem: list[int] = []
    for c in cids:
        comm_mem.extend(communities[c])
        comm_off.append(len(comm_mem))
    stats = {
        "nodes": len(nodes),
        "edges": len(links),
        "confidence": {k: v for k, v in conf.items() if isinstance(k, str)},
        "god_nodes": [{"label": g["label"], "degree": g["degree"]} for g in gods],
        "god_nodes_kept": GOD_NODES_KEPT,
        "cids": cids,
        "built_at_commit": data.get("built_at_commit"),
    }
    meta = {"node_fields": NODE_FIELDS, "link_fields": LINK_FIELDS, "directed": directed,
            "graphify_version": graphify_version()}
    graph_rest = {k: v for k, v in data.items() if k not in ("nodes", "links", "edges")}
    if source is not None:
        size, mtime_ns, sha = file_info(source)
    else:
        size, mtime_ns, sha = 0, 0, bytes(32)
    times["stats"] += time.perf_counter() - t
    rust = enc.finish(str(out_path), generation, norm_fingerprint(), size, mtime_ns, sha,
                      partition_hash(nodes), _dumps(meta), _dumps(stats), _dumps(graph_rest),
                      comm_off, comm_mem)
    for k, v in rust:
        times[k] += v
    times["total"] = time.perf_counter() - t_all
    return {"nodes": len(nodes), "links": len(links), "bytes": out_path.stat().st_size,
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

    ``graph_json`` is read with the streaming loader and never modified.
    """
    from graphify import graph_stream
    src = Path(graph_json)
    t = time.perf_counter()
    data = graph_stream.load(src)
    load_s = time.perf_counter() - t
    live = snapshot_path_for(src)
    report = publish_snapshot(data, live.parent, name=live.name, source=src)
    report["times"]["json_load"] = load_s
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
    except Unsupported as exc:
        print('error: cannot build a snapshot of ' + str(gj) + ': ' + str(exc), file=sys.stderr)
        return 1
    secs = rep['times']['total'] + rep['times'].get('json_load', 0.0)
    print('snapshot ' + rep['path'] + ': ' + str(rep['nodes']) + ' nodes, ' + str(rep['links']) + ' links, ' + str(rep['bytes']) + ' bytes, ' + format(secs, '.1f') + 's')
    return 0
