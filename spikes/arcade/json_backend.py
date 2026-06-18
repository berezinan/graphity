"""Baseline backend: the current JSON + in-memory NetworkX model.

Reuses graphify.serve so the comparison is against the REAL query code, not a
re-implementation. The whole graph is parsed and held in RAM - this is exactly
the model the spike is testing the limits of.
"""
from __future__ import annotations

import json
import time
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from graphify.serve import _query_graph_text, _query_terms, _score_nodes


class JsonBackend:
    def __init__(self, graph_path: str) -> None:
        self.graph_path = graph_path
        self.G: nx.Graph | None = None

    def load(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw = json.loads(open(self.graph_path, encoding="utf-8").read())
        if "links" not in raw and "edges" in raw:
            raw = dict(raw, links=raw["edges"])
        try:
            self.G = json_graph.node_link_graph(raw, edges="links")
        except TypeError:
            self.G = json_graph.node_link_graph(raw)
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "load_seconds": round(time.perf_counter() - t0, 2),
        }

    def query(self, question: str, *, depth: int = 2) -> dict[str, Any]:
        text = _query_graph_text(self.G, question, mode="bfs", depth=depth, token_budget=2000)
        return {"node_count": text.count("\nNODE "), "preview": text[:120]}

    def shortest_path(self, source: str, target: str, *, max_hops: int = 8) -> dict[str, Any]:
        src = _score_nodes(self.G, _query_terms(source))
        tgt = _score_nodes(self.G, _query_terms(target))
        if not src or not tgt:
            return {"found": False, "reason": "endpoint not resolved"}
        s_id, t_id = src[0][1], tgt[0][1]
        if s_id == t_id:
            return {"found": False, "reason": "same node"}
        try:
            path = nx.shortest_path(self.G.to_undirected(as_view=True), s_id, t_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"found": False, "reason": "no path"}
        return {"found": True, "hops": len(path) - 1}
