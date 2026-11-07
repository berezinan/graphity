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
    return _fingerprint_or_none(json.loads(path.read_text(encoding="utf-8")))


# Upstream functions this module mirrors; PORT_HASH is their source hash,
# recorded after the equivalence tests passed against them.
PORT_FUNCS = (
    "watch._canonical_graph_for_compare",
    "watch._canonical_topology_for_compare",
    "build._load_existing_graph",
)


def port_hash() -> str:
    from graphify.snap_store import _port_func, _sources_hash
    return _sources_hash([_port_func(n) for n in PORT_FUNCS]).hexdigest()


PORT_HASH = "9885e49777aab422292041749cac614f151190b9a4372877167b31c484a8d723"
