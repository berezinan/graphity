"""The networkx graphs the CLI builds from graph.json, per command (D13).

Copies of the loading code in `graphify/cli.py` (`query`, `path`, `explain`
branches) and `serve._read_graph`, minus the size cap and messages, so the
parity tests compare SnapBackend with JsonBackend over exactly those graphs.
"""
from __future__ import annotations

from networkx.readwrite import json_graph

from graphify.build import has_parallel_links


def _links(raw: dict) -> dict:
    if "links" not in raw and "edges" in raw:
        raw = dict(raw, links=raw["edges"])
    return raw


def _markers(raw: dict) -> dict:
    return dict(raw, links=[{**l, "_src": l.get("_src", l.get("source")),
                             "_tgt": l.get("_tgt", l.get("target"))} for l in raw.get("links", [])])


def query_graph(raw: dict):
    return json_graph.node_link_graph(_markers(_links(raw)), edges="links")


def path_graph(raw: dict, undirected: bool = False):
    raw = _markers(_links(raw))
    return json_graph.node_link_graph({**raw, "directed": not undirected, "multigraph": True}, edges="links")


def explain_graph(raw: dict):
    raw = _links(raw)
    multi = has_parallel_links(raw.get("links"), directed=True)
    return json_graph.node_link_graph({**raw, "directed": True, **({"multigraph": True} if multi else {})},
                                      edges="links")


def mcp_graph(raw: dict):
    """`serve._read_graph` without the file I/O."""
    raw = _links(raw)
    data = {**raw, "directed": True}
    if has_parallel_links(data.get("links"), directed=True):
        data = {**data, "multigraph": True}
    G = json_graph.node_link_graph(data, edges="links")
    commit = raw.get("built_at_commit")
    if isinstance(commit, str) and commit.strip():
        G.graph["_built_at_commit"] = commit.strip()
    return G
