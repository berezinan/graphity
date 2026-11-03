"""Snapshot writer (task 3.1, D5, D7).

Input is graph.json as data (`json.load` of the file, or `node_link_data(G)`
for a fixture), in its own node and link order. Normalisation runs through the
upstream `serve.py` functions; the byte layout is done by the Rust encoder.

    python snap_writer.py <graph.json> <out_dir>      # prints the JSON report
"""
from __future__ import annotations

import collections
import hashlib
import json
import msvcrt
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import graphify_snap
from graphify.analyze import god_nodes as _god_nodes
from graphify.build import has_parallel_links
from graphify.serve import (
    _node_attributes_text,
    _node_rationale_text,
    _search_tokens,
    _strip_diacritics,
)

from snap_common import (
    LINK_FIELDS,
    NODE_FIELDS,
    Unsupported,
    graphify_version,
    norm_fingerprint,
)

GOD_NODES_KEPT = 100
_NODE_SET = frozenset(NODE_FIELDS) | {"id"}
_LINK_SET = frozenset(LINK_FIELDS) | {"source", "target"}


def _dumps(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def links_of(data: dict) -> list:
    if "links" in data:
        return data["links"]
    if "edges" in data:
        return data["edges"]
    return []


def check_supported(data: dict) -> None:
    """Refuse the graph shapes D13 lists as not reproduced."""
    if data.get("multigraph", True):
        raise Unsupported("multigraph graphs (or a file without 'multigraph': false)")
    links = links_of(data)
    seen = set()
    for d in data["nodes"]:
        nid = d.get("id")
        if not isinstance(nid, str):
            raise Unsupported(f"node id that is not a string: {nid!r}")
        if nid in seen:
            raise Unsupported(f"duplicate node id {nid!r}")
        seen.add(nid)
    for l in links:
        if l.get("source") not in seen or l.get("target") not in seen:
            raise Unsupported("link endpoint that is not a node")
        if "_src" in l or "_tgt" in l:
            raise Unsupported("links with stored _src/_tgt markers")
    if has_parallel_links(links, directed=True):
        raise Unsupported("parallel links of one direction")


def partition_hash(nodes) -> bytes:
    """sha256 of (community, community_name) per node, in node order (D9)."""
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
    """The slice of nx.DiGraph that `analyze.god_nodes` reads."""

    def __init__(self, ids, nodes, deg):
        self._ids, self._deg = ids, deg
        self._pos = {nid: i for i, nid in enumerate(ids)}
        self.nodes = {nid: d for nid, d in zip(ids, nodes)}

    def degree(self, n=None):
        if n is None:
            return list(zip(self._ids, self._deg))
        return self._deg[self._pos[n]]


def _node_columns(d: dict) -> tuple:
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
    """Write one snapshot file at `out_path` (not published). Returns the report."""
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
        rows = [_node_columns(d) for d in nodes[start:start + batch]]
        for k, d in enumerate(nodes[start:start + batch]):
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


@contextmanager
def _lock(path: Path, timeout: float = 600.0):
    """Exclusive writer lock (`msvcrt.locking`, as watch._rebuild_lock)."""
    f = open(path, "a+b")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"snapshot lock {path} is held") from None
                time.sleep(0.05)
        yield
    finally:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        f.close()


def publish_snapshot(data: dict, out_dir: Path, *, name: str = "graph.snap",
                     source: Path | None = None, crash_before_publish: bool = False) -> dict:
    """Write the next generation and replace `out_dir/name` over (D7)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = out_dir / name
    with _lock(out_dir / f"{name}.lock"):
        removed = []
        for stale in out_dir.glob(f"{name}.*.tmp"):
            try:
                stale.unlink()
                removed.append(stale.name)
            except OSError:
                pass
        gen = (graphify_snap.peek_generation(str(live)) or 0) + 1
        tmp = out_dir / f"{name}.{os.getpid()}.tmp"
        report = encode(data, tmp, generation=gen, source=source)
        if crash_before_publish:
            os._exit(3)
        t = time.perf_counter()
        graphify_snap.publish(str(tmp), str(live))
        report["times"]["publish"] = time.perf_counter() - t
        report["removed_tmp"] = removed
        report["path"] = str(live)
    return report


def _peak_memory() -> dict:
    try:
        import psutil
        m = psutil.Process().memory_info()
        return {"peak_wset_mb": round(m.peak_wset / 2**20, 1), "private_mb": round(m.private / 2**20, 1)}
    except Exception:
        return {}


class _MemWatch:
    """Samples this process's private bytes and the system's available memory."""

    def __init__(self, every: float = 1.0):
        import threading
        import psutil
        self._p, self._vm = psutil.Process(), psutil.virtual_memory
        self.peak_private = 0
        self.min_available = None
        self._stop = threading.Event()
        self._every = every
        self._t = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> int:
        priv = self._p.memory_info().private
        avail = self._vm().available
        self.peak_private = max(self.peak_private, priv)
        self.min_available = avail if self.min_available is None else min(self.min_available, avail)
        return priv

    def _run(self):
        while not self._stop.wait(self._every):
            self.sample()

    def __enter__(self):
        self.sample()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.sample()


def main(argv: list[str]) -> int:
    import graph_stream
    src, out_dir = Path(argv[1]), Path(argv[2])
    mb = lambda b: round(b / 2**20, 1)
    with _MemWatch() as mw:
        mem0 = _peak_memory()
        t = time.perf_counter()
        data = graph_stream.load(src)
        load_s = time.perf_counter() - t
        mem_loaded = _peak_memory()
        after_load_private = mw.sample()
        peak_during_load = mw.peak_private
        mw.peak_private = after_load_private
        report = publish_snapshot(data, out_dir, source=src)
    report["times"]["json_load"] = load_s
    report["memory"] = {
        "start": mem0, "after_json_load": mem_loaded, "end": _peak_memory(),
        "private_after_load_mb": mb(after_load_private),
        "peak_private_during_load_mb": mb(peak_during_load),
        "peak_private_during_write_mb": mb(mw.peak_private),
        "write_growth_over_data_pct": round((mw.peak_private / after_load_private - 1) * 100, 1),
        "min_system_available_mb": mb(mw.min_available),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
