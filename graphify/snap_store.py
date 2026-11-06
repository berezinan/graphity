"""Graph snapshot store: column layout, normalisation fingerprint, paths.

Fork-owned (not in upstream). The snapshot is an immutable memory-mapped file
built from a written graph.json (capability ``graph-snapshot-store``). The
writer is ``graphify.snap_writer``, the reader ``graphify.snap_backend``; this
module holds what both share.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

from graphify.arcade_server import GraphDBUnavailable

# Hot value columns. Every other attribute goes to the cold JSON, losslessly.
NODE_FIELDS = ["label", "norm_label", "source_file", "source_location", "community",
               "community_name", "file_type", "_origin"]
LINK_FIELDS = ["relation", "confidence", "confidence_score", "weight", "source_file",
               "source_location", "context", "_origin"]
# Order fixed by snap_core::encode::SEARCH.
SEARCH_FIELDS = ["nl", "nli", "lt", "src", "st", "rat", "atn", "att"]

# Version of the Python <-> graphify_snap contract. The wheel exports its own
# API_VERSION; a mismatch is a refusal that tells the user to reinstall it.
EXPECTED_API_VERSION = 2


class Unsupported(ValueError):
    """The graph has a shape the snapshot writer does not reproduce."""


class SnapshotUnavailable(GraphDBUnavailable):
    """The snapshot cannot answer: missing, stale, unfinished or incompatible.

    Carries the complete user-facing text (reason and action). A GraphDBUnavailable,
    so the CLI prints it and exits non-zero through the existing handler (D3).
    """


def snapshot_path_for(graph_json: str | Path) -> Path:
    """``graphify-out/graph.json`` -> ``graphify-out/graph.snap`` (D6)."""
    return Path(graph_json).with_suffix(".snap")


def lock_path_for(snapshot: str | Path) -> Path:
    return Path(str(snapshot) + ".lock")


def _norm_funcs() -> tuple:
    from graphify import serve
    return (serve._strip_diacritics, serve._search_tokens, serve._node_rationale_text,
            serve._node_attributes_text, serve._flatten_attr_text)


def _sources_hash(funcs) -> "hashlib._Hash":
    h = hashlib.sha256()
    for f in funcs:
        h.update(f.__qualname__.encode())
        h.update(inspect.getsource(f).encode())
    return h


def norm_fingerprint() -> bytes:
    """sha256 of the normalisation functions' source (D4).

    Deliberately without the package version: every upstream merge changes it,
    and every snapshot would turn incompatible after each production update.
    """
    return _sources_hash(_norm_funcs()).digest()


_BUILD_HINT = "Rebuild it with: graphify snap build <project-dir|graph.json>"


def _rebuilding_note(snapshot: Path) -> str:
    from graphify.snap_writer import writer_pid
    pid = writer_pid(Path(snapshot))
    if pid is None:
        return ""
    return f"\nThe snapshot is being rebuilt right now (writer pid {pid}); retry when it finishes."


def check_api() -> None:
    """Refuse unless graphify_snap is importable and speaks the expected API."""
    try:
        import graphify_snap
    except ImportError as exc:
        raise SnapshotUnavailable(
            "the graphify_snap module is not installed, so the graph snapshot cannot be read.\n"
            "Install the graphify-snap wheel (the one attached to the berezinan/graphity release)."
        ) from exc
    got = getattr(graphify_snap, "API_VERSION", None)
    if got != EXPECTED_API_VERSION:
        raise SnapshotUnavailable(
            f"the installed graphify_snap has API version {got!r}, this graphify needs "
            f"{EXPECTED_API_VERSION}.\nReinstall the graphify-snap wheel that matches this graphify.")


def check_fresh(snap, snapshot, graph_json) -> None:
    """Refuse a snapshot whose graph.json changed (size or mtime_ns) after it was built."""
    try:
        st = os.stat(graph_json)
    except OSError:
        return
    size, mtime_ns, _sha = snap.source
    if (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
        raise SnapshotUnavailable(
            f"graph snapshot {snapshot} is stale: {Path(graph_json).name} changed after it was "
            f"built (size or mtime differ).\n{_BUILD_HINT}" + _rebuilding_note(Path(snapshot)))


def open_snapshot(path, graph_json=None):
    """Open the snapshot at ``path`` or raise SnapshotUnavailable with the reason and the action."""
    check_api()
    import graphify_snap
    p = Path(path)
    if not p.exists():
        raise SnapshotUnavailable(
            f"graph snapshot {p} does not exist.\n{_BUILD_HINT}" + _rebuilding_note(p))
    try:
        snap = graphify_snap.Snap(str(p), norm_fingerprint())
    except (OSError, ValueError) as exc:
        raise SnapshotUnavailable(
            f"graph snapshot {p} cannot be used: {exc}\n"
            f"(it is unfinished, has another format version, or was built with different "
            f"normalisation code)\n{_BUILD_HINT}" + _rebuilding_note(p)) from exc
    if graph_json is not None:
        check_fresh(snap, p, graph_json)
    return snap


class PortOutdated(SnapshotUnavailable):
    """Upstream changed code that SnapBackend repeats; the port must be re-checked."""


# Upstream code whose behaviour SnapBackend repeats instead of calling, as
# "<module under graphify>.<qualname>". If any of it changes, the port may
# answer differently, so SnapBackend refuses until the port is re-checked (D10).
PORT_FUNCS = (
    "query_backend.JsonBackend.query", "query_backend.JsonBackend._query_from_seeds",
    "query_backend.JsonBackend._resolve", "query_backend.JsonBackend.get_node",
    "query_backend.JsonBackend.get_neighbors", "query_backend.JsonBackend.get_community",
    "query_backend.JsonBackend.god_nodes", "query_backend.JsonBackend.graph_stats",
    "query_backend.JsonBackend.shortest_path", "query_backend.JsonBackend.explain",
    "query_backend.JsonBackend.pr_impact",
    "serve._score_query", "serve._compute_idf", "serve._score_terms", "serve._bfs", "serve._dfs",
    "serve._traversal_view", "serve._filter_graph_by_context", "serve._resolve_path_endpoint",
    "serve._find_node_tiers", "serve._label_has_literal_exact_match",
    "serve._resolve_path_scoped_symbol", "serve._resolve_python_qualified",
    "serve._communities_from_graph", "serve._read_graph",
    "prs.compute_pr_impact", "analyze.god_nodes",
    "serve._resolve_single_node", "serve._query_name_refusal", "serve._tier_refusal",
    "serve._rivals_across_files", "serve._near_miss_message",
    "serve._same_file_ambiguity_message", "serve._complete_induced_edges",
    "serve._pick_seeds", "serve._pick_scored_endpoint", "serve._query_terms",
    "serve._is_searchable", "serve._ambiguity_message",
    # `query --seed` resolves through the upstream function on a candidate mini-graph (D8).
    "affected.resolve_seed",
)


def _port_func(name: str):
    import importlib
    mod, *path = name.split(".")
    obj = importlib.import_module("graphify." + mod)
    for part in path:
        obj = getattr(obj, part)
    return obj


def port_hash() -> str:
    return _sources_hash([_port_func(n) for n in PORT_FUNCS]).hexdigest()


# Recorded after the port was checked against PORT_FUNCS (parity run green).
PORT_HASH = "daa994c5ad85066e72db3b168e600d7aec6138d3a0a3e5b4bfb4c8a3e03619ba"


def publish_snapshot_for(graph_json) -> None:
    '''Build and publish the snapshot of a written graph.json in a child process (D2).

    A child, because the writer peaks at gigabytes on big graphs and a long-lived
    parent (watch) would never hand that heap back to the OS. A non-zero exit is an
    error: the command that wrote graph.json did not succeed.
    '''
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, '-m', 'graphify', 'snap', 'build', str(graph_json)],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip()[-2000:]
        raise SnapshotUnavailable(
            f'the snapshot writer failed for {graph_json} (exit {proc.returncode}).\n{tail}')
    if proc.stdout.strip():
        print(proc.stdout.strip())
