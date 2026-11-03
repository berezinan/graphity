"""Shared constants of the snapshot spike: column layout, normalisation
fingerprint (D5) and the port guard (D13)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect

from graphify import analyze, prs, serve
from graphify import query_backend as qb

# Hot value columns. Every other attribute goes to COLD as JSON, losslessly.
NODE_FIELDS = ["label", "norm_label", "source_file", "source_location", "community",
               "community_name", "file_type", "_origin"]
LINK_FIELDS = ["relation", "confidence", "confidence_score", "weight", "source_file",
               "source_location", "context", "_origin"]
# Order fixed by snap_core::encode::SEARCH.
SEARCH_FIELDS = ["nl", "nli", "lt", "src", "st", "rat", "atn", "att"]

NORM_FUNCS = (serve._strip_diacritics, serve._search_tokens, serve._node_rationale_text,
              serve._node_attributes_text, serve._flatten_attr_text)


def graphify_version() -> str:
    try:
        return importlib.metadata.version("graphifyy")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _sources_hash(funcs) -> hashlib._Hash:
    h = hashlib.sha256()
    for f in funcs:
        h.update(f.__qualname__.encode())
        h.update(inspect.getsource(f).encode())
    return h


def norm_fingerprint() -> bytes:
    """sha256 of the normalisation functions' source plus the graphify version."""
    h = _sources_hash(NORM_FUNCS)
    h.update(graphify_version().encode())
    return h.digest()


# Upstream code whose behaviour SnapBackend repeats instead of calling. If any
# of it changes, the port may answer differently, so SnapBackend refuses.
PORT_FUNCS = (
    qb.JsonBackend.query, qb.JsonBackend._query_from_seeds, qb.JsonBackend._resolve,
    qb.JsonBackend.get_node, qb.JsonBackend.get_neighbors, qb.JsonBackend.get_community,
    qb.JsonBackend.god_nodes, qb.JsonBackend.graph_stats, qb.JsonBackend.shortest_path,
    qb.JsonBackend.explain, qb.JsonBackend.pr_impact,
    serve._score_query, serve._compute_idf, serve._score_terms, serve._bfs, serve._dfs,
    serve._traversal_view, serve._filter_graph_by_context, serve._resolve_path_endpoint,
    serve._find_node_tiers, serve._label_has_literal_exact_match,
    serve._resolve_path_scoped_symbol, serve._resolve_python_qualified,
    serve._communities_from_graph, serve._read_graph, prs.compute_pr_impact,
    analyze.god_nodes,
)


def port_hash() -> str:
    return _sources_hash(PORT_FUNCS).hexdigest()


# Recorded when the port was written (graphify 0.9.97, staging 8d9526c3).
PORT_HASH = "20c5bfa2f18370dfb173230f90e0c87f5ce24632a8f4939d2eb4b5aa78e2dea1"


class Unsupported(ValueError):
    """The graph uses a feature the spike does not reproduce (D13)."""


class PortOutdated(RuntimeError):
    """Upstream changed code that SnapBackend repeats."""
