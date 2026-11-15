"""Pluggable read query backend for the knowledge graph.

The graph is built and exported as ``graph.json`` exactly as before; this module
is only about *serving queries*. ``GraphBackend`` is the interface the MCP tools
(serve.py) and the CLI (``graphify query|path|explain``) read through, so the
same graph can be backed by the in-process NetworkX model (``JsonBackend``,
explicit) or the memory-mapped graph snapshot (``SnapBackend``, default)
without the callers caring which.

Design contract that makes a DB backend possible:
  - backend methods return **materialized records** (the dataclasses below), with
    no live ``nx.Graph`` reference — a backend may hold the whole graph in RAM
    (JsonBackend) or fetch only what each query touches (a DB backend);
  - all *formatting* lives in the ``render_*`` functions here, so text output and
    the F-010 ``sanitize_label`` pass happen in exactly one place regardless of
    backend, and golden tests can compare backends at the record level.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx as nx

from graphify.build import edge_data, edge_datas
from graphify.security import sanitize_label
from graphify.serve import (
    _bfs,
    _bfs_core,
    _complete_induced_edges,
    _dfs,
    _dfs_core,
    _filter_graph_by_context,
    _find_node_tiers,
    _hub_threshold,
    _idf_value,
    _is_ambiguity_refusal,
    _node_attributes_text,
    _node_rationale_text,
    _no_scoped_match_message,
    _path_search_graph,
    _pick_scored_endpoint,
    _pick_seeds,
    _PY_QUALIFIED_RE,
    _query_name_refusal,
    _query_terms,
    _RELATIONAL_INTENT_TERMS,
    _resolve_path_endpoint,
    _resolve_single_node,
    _score_query,
    _tier_refusal,
    _resolve_context_filters,
    _score_nodes,
    _score_record,
    _score_terms,
    _search_tokens,
    _strip_diacritics,
    _traversal_view,
)


# --------------------------------------------------------------------------- #
# Materialized result records (no nx.Graph reference crosses this boundary)
# --------------------------------------------------------------------------- #
@dataclass
class Refusal:
    """A lookup that has no single answer: the name is ambiguous or matches nothing.

    ``message`` is the text `serve.py` produces for the same lookup, so both
    backends refuse in the upstream wording. ``ambiguous`` picks the exit code:
    3 when candidates are listed, 4 when nothing matched.
    """
    message: str
    ambiguous: bool


def _refusal(message: str) -> Refusal:
    return Refusal(message, _is_ambiguity_refusal(message))


@dataclass
class NodeRecord:
    label: str
    source_file: str = ""
    source_location: str = ""
    community: str = ""


@dataclass
class EdgeRecord:
    source_label: str
    target_label: str
    relation: str = ""
    confidence: str = ""
    context: str = ""
    source_file: str = ""      # relation site (#BUG1), not a def line
    source_location: str = ""  # shown as `at=<file>:<loc>` in EDGE lines


@dataclass
class QueryResult:
    mode: str
    depth: int
    seeds: list[str]                       # seed labels, in seed order
    total_nodes: int                       # size of the traversal frontier
    nodes: list[NodeRecord]                # already ordered (seeds first, then degree desc)
    edges: list[EdgeRecord]
    filters: list[str] = field(default_factory=list)
    filter_source: str | None = None
    graph_nodes: int | None = None         # size of the WHOLE graph, for the header
    # The question is itself the exact name of several nodes (#4517): refused
    # like `explain`/`path`, never answered from one of them.
    refusal: Refusal | None = None
    given: bool = False                    # True when seeds came from --seed
    unresolved: list[str] = field(default_factory=list)  # --seed values that could not be resolved


@dataclass
class NodeDetail:
    label: str
    id: str
    source_file: str = ""
    source_location: str = ""
    file_type: str = ""
    community: str = ""
    degree: int = 0
    # C/C++/ObjC: символ, объявленный в заголовке и определённый в соседнем
    # файле, — ОДИН узел, привязанный к заголовку, поэтому Source указывает на
    # объявление. Называем и место определения, когда оно известно.
    definition_file: str = ""
    definition_location: str = ""


@dataclass
class NeighborRecord:
    outgoing: bool                         # True: --> , False: <--
    label: str
    relation: str = ""
    confidence: str = ""
    source_file: str = ""      # relation site (#BUG1)
    source_location: str = ""


@dataclass
class Neighbors:
    label: str
    neighbors: list[NeighborRecord]


@dataclass
class CommunityRecord:
    cid: int
    members: list[NodeRecord]
    name: str | None = None


@dataclass
class GodNode:
    label: str
    degree: int


@dataclass
class GraphStats:
    nodes: int
    edges: int
    communities: int
    extracted_pct: int
    inferred_pct: int
    ambiguous_pct: int
    built_at_commit: str = ""   # commit the graph was built on, empty when unknown


@dataclass
class PathSegment:
    outgoing: bool                         # direction of this hop
    relation: str
    confidence: str
    label: str                             # label of the node reached by this hop


@dataclass
class PathResult:
    found: bool
    start_label: str = ""
    end_label: str = ""                    # resolved target label (for the no-path message)
    hops: int = 0
    segments: list[PathSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)   # ambiguity warnings only
    reason: str = ""                       # populated when found is False
    refusal: Refusal | None = None         # an endpoint is ambiguous or a file::symbol miss


@dataclass
class Connection:
    outgoing: bool                         # True: --> , False: <--
    label: str
    relation: str = ""
    confidence: str = ""
    degree: int = 0                        # neighbour degree, used to rank connections
    # Место САМОЙ связи (строка вызова/импорта в файле источника), а не строка
    # определения соседа: на вопрос «кто это вызывает» нужна кликабельная точка
    # вызова, а не def вызывающего (#BUG1).
    source_file: str = ""
    source_location: str = ""


@dataclass
class ExplainResult:
    detail: NodeDetail
    connections: list[Connection] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shortest route, shared by both backends
# --------------------------------------------------------------------------- #
def _lexmin_shortest_path(expand, s: str, t: str) -> list[str] | None:
    """The shortest s→t route, and among equal-length routes the smallest by node ids.

    ``expand(frontier, forward)`` returns, for each node of ``frontier``, its
    neighbours in the search graph: successors when ``forward``, predecessors
    otherwise. Both backends search through it, so they pick the same route
    (#2074) — JSON over the in-memory search graph, the snapshot over its CSR
    arrays, one BFS level at a time, never enumerating paths.

    Bidirectional: the smaller frontier expands by a full level until the two
    sides meet. Every parent link of a newly reached node is kept, so the nodes
    lying on SOME shortest route are known exactly when they meet, and a greedy
    walk from ``s`` taking the smallest valid id at each step is the
    lexicographically smallest shortest route. None when ``t`` is unreachable.
    """
    if s == t:
        return [s]
    ds, dt = {s: 0}, {t: 0}
    fs, ft = {s}, {t}
    par_s: dict[str, set[str]] = {}       # node -> its parents one level nearer s
    kid_t: dict[str, set[str]] = {}       # node -> its children one level nearer t
    while fs and ft:
        if len(fs) <= len(ft):
            frontier, dist, links, forward = fs, ds, par_s, True
        else:
            frontier, dist, links, forward = ft, dt, kid_t, False
        level = dist[next(iter(frontier))] + 1
        adj = expand(frontier, forward)
        nxt: set[str] = set()
        for u in frontier:
            for x in adj.get(u, ()):
                if x not in dist:
                    dist[x] = level
                    nxt.add(x)
                if dist[x] == level:
                    links.setdefault(x, set()).add(u)
        if forward:
            fs = nxt
        else:
            ft = nxt
        meet = [x for x in nxt if x in (dt if forward else ds)]
        if meet:
            break
    else:
        return None
    length = min(ds[x] + dt[x] for x in meet)
    pos = {x: ds[x] for x in meet if ds[x] + dt[x] == length}
    stack = list(pos)
    while stack:                                  # back toward s
        x = stack.pop()
        for p in par_s.get(x, ()):
            if p not in pos:
                pos[p] = ds[p]
                stack.append(p)
    stack = [x for x in pos if x in dt and ds.get(x, -1) + dt[x] == length]
    while stack:                                  # on toward t
        x = stack.pop()
        for c in kid_t.get(x, ()):
            if c not in pos:
                pos[c] = length - dt[c]
                stack.append(c)
    kids_s: dict[str, set[str]] = {}
    for x, parents in par_s.items():
        for p in parents:
            kids_s.setdefault(p, set()).add(x)
    route, cur = [s], s
    for i in range(1, length + 1):
        cur = min((x for x in kids_s.get(cur, set()) | kid_t.get(cur, set()) if pos.get(x) == i),
                  key=str)
        route.append(cur)
    return route


def _path_segment(u: str, v: str, edges, label: str) -> PathSegment:
    """One hop of a route, naming every stored relation of the pair (#2074).

    ``edges`` are ``(true_source, relation, confidence)`` for each edge between
    ``u`` and ``v``. Edges running ``u``→``v`` win; only without them is the hop
    the reverse of a stored edge (a `contains` walked back, #3878, or any edge
    under --undirected). As in `serve._shortest_path_text`.
    """
    fwd = [e for e in edges if e[0] == u]
    datas = fwd or [e for e in edges if e[0] != u]
    return PathSegment(outgoing=bool(fwd),
                       relation="/".join(sorted({str(r) for _, r, _ in datas if r})),
                       confidence="/".join(sorted({str(c) for _, _, c in datas if c})),
                       label=label)


def _nx_expand(H: nx.Graph):
    if H.is_directed():
        return lambda frontier, forward: {
            u: list(H.successors(u) if forward else H.predecessors(u)) for u in frontier}
    return lambda frontier, forward: {u: list(H.neighbors(u)) for u in frontier}


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class GraphBackend(ABC):
    """Read-only view over a knowledge graph, addressed by labels/ids.

    A point lookup (`get_node`, `get_neighbors`, `explain`, the `shortest_path`
    endpoints) resolves its label by `serve._resolve_single_node` rules and
    returns a `Refusal` instead of guessing when that label names several nodes
    or none.
    """

    @abstractmethod
    def query(self, question: str, *, mode: str = "bfs", depth: int = 3,
              context_filters: list[str] | None = None,
              seeds: list[str] | None = None,
              seed_root: "Path | None" = None) -> QueryResult: ...

    @abstractmethod
    def get_node(self, label: str) -> NodeDetail | Refusal: ...

    @abstractmethod
    def get_neighbors(self, label: str, relation_filter: str = "") -> Neighbors | Refusal: ...

    @abstractmethod
    def get_community(self, community_id: int) -> CommunityRecord | None: ...

    @abstractmethod
    def god_nodes(self, top_n: int = 10) -> list[GodNode]: ...

    @abstractmethod
    def graph_stats(self) -> GraphStats: ...

    @abstractmethod
    def shortest_path(self, source: str, target: str, *, max_hops: int = 8,
                      undirected: bool = False) -> PathResult: ...

    @abstractmethod
    def explain(self, label: str) -> ExplainResult | Refusal: ...

    @abstractmethod
    def pr_impact(self, files: list[str]) -> tuple[list[int], int]: ...


# --------------------------------------------------------------------------- #
# JsonBackend - the current in-process NetworkX model, behind the interface
# --------------------------------------------------------------------------- #
class JsonBackend(GraphBackend):
    """Holds the whole graph in RAM and answers from it, reusing the existing
    serve.py helpers so behaviour is identical to the pre-abstraction tools."""

    def __init__(self, G: nx.Graph, communities: dict[int, list[str]] | None = None) -> None:
        from graphify.serve import _communities_from_graph
        self.G = G
        self.communities = communities if communities is not None else _communities_from_graph(G)

    def query(self, question, *, mode="bfs", depth=3, context_filters=None,
              seeds=None, seed_root=None) -> QueryResult:
        G = self.G
        if seeds:
            from graphify.affected import resolve_seed as _rs
            start_nodes: list[str] = []
            unresolved: list[str] = []
            ambiguous: list[str] = []
            for val in seeds:
                nid = _rs(G, val, seed_root)
                if nid is None:
                    ref = _query_name_refusal(G, val)
                    if ref:
                        if ref not in ambiguous:
                            ambiguous.append(ref)
                    elif val not in unresolved:
                        unresolved.append(val)
                elif nid not in start_nodes:
                    start_nodes.append(nid)
            if not start_nodes:
                msg = "\n".join(ambiguous) if ambiguous else \
                      "No unique node match for --seed: " + ", ".join(seeds)
                return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[],
                                   refusal=_refusal(msg))
            return self._query_from_seeds(G, question, start_nodes, mode=mode, depth=depth,
                                          context_filters=context_filters, unresolved=unresolved)
        refused = _query_name_refusal(G, question)
        if refused:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[],
                               refusal=_refusal(refused))
        terms = _query_terms(question)
        qs = _score_query(G, terms, collect_per_term_seeds=True)
        best_seed_by_term = qs.best_seed_by_term
        intent = {t for t in best_seed_by_term if t in _RELATIONAL_INTENT_TERMS}
        if intent and any(t not in _RELATIONAL_INTENT_TERMS for t in terms):
            best_seed_by_term = {t: nid for t, nid in best_seed_by_term.items() if t not in intent}
        seeds = _pick_seeds(qs.ranked, G=G, best_seed_by_term=best_seed_by_term, term_order=qs.term_order)
        if not seeds:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[])
        return self._query_from_seeds(G, question, seeds, mode=mode, depth=depth,
                                      context_filters=context_filters)

    def _query_from_seeds(self, G, question, seeds, *, mode, depth, context_filters=None,
                          unresolved=None) -> QueryResult:
        filters, fsource = _resolve_context_filters(question, context_filters)
        # Through _traversal_view, exactly as serve._query_graph_text does: on a
        # DiGraph `neighbors()` yields successors only, so a leaf that is merely
        # called/imported expands to nothing and the answer silently loses its
        # callers. Direction survives per-edge via _src/_tgt, so rendering is
        # unaffected; an undirected graph is returned as-is.
        TG = _filter_graph_by_context(_traversal_view(G), filters)
        visited, edges = (_dfs if mode == "dfs" else _bfs)(TG, seeds, depth)

        # Match _subgraph_to_text exactly. It no longer orders by degree alone:
        # the query path now passes its seeds (#BUG2), so the queried symbol
        # renders first and the rest rank by hop distance from it — otherwise an
        # incidental high-degree hub pushes the node that answers the question
        # past the budget cut. Distance is recomputed over BOTH edge directions
        # because _bfs/_dfs return a set and discovery order is gone.
        seed_set = set(seeds)
        seed_hits = [n for n in seeds if n in visited]

        def _adj(n):
            if TG.is_directed():
                yield from TG.successors(n)
                yield from TG.predecessors(n)
            else:
                yield from TG.neighbors(n)

        dist: dict[str, int] = {n: 0 for n in seed_hits}
        frontier, hop = seed_hits, 0
        while frontier:
            hop += 1
            nxt = []
            for n in frontier:
                for nb in _adj(n):
                    if nb in visited and nb not in dist:
                        dist[nb] = hop
                        nxt.append(nb)
            frontier = nxt
        ordered = seed_hits + sorted(
            visited - seed_set,
            key=lambda n: (dist.get(n, 1 << 30), -TG.degree(n), str(n)),
        )
        node_recs = [
            NodeRecord(
                label=TG.nodes[n].get("label", n),
                source_file=str(TG.nodes[n].get("source_file", "")),
                source_location=str(TG.nodes[n].get("source_location", "")),
                community=str(TG.nodes[n].get("community", "")),
            )
            for n in ordered
        ]
        edge_recs = []
        for u, v in edges:
            if u in visited and v in visited:
                for d in edge_datas(TG, u, v):
                    # (u, v) is visit order, not direction: traversing undirected
                    # walks callers and callees alike, so a caller->callee edge
                    # renders backwards whenever the callee is reached first.
                    # _src/_tgt (stashed by _traversal_view and the query loader)
                    # carry the real direction; trust them only when they name THIS
                    # edge's endpoints, else a stray marker would KeyError below.
                    src, tgt = d.get("_src", u), d.get("_tgt", v)
                    if {src, tgt} != {u, v}:
                        src, tgt = u, v
                    edge_recs.append(EdgeRecord(
                        source_label=TG.nodes[src].get("label", src),
                        target_label=TG.nodes[tgt].get("label", tgt),
                        relation=str(d.get("relation", "")),
                        confidence=str(d.get("confidence", "")),
                        context=str(d.get("context", "") or ""),
                        source_file=str(d.get("source_file", "") or ""),
                        source_location=str(d.get("source_location", "") or ""),
                    ))
        result = QueryResult(
            mode=mode, depth=depth,
            seeds=[G.nodes[n].get("label", n) for n in seeds],
            total_nodes=len(visited), nodes=node_recs, edges=edge_recs,
            filters=filters, filter_source=fsource,
            graph_nodes=G.number_of_nodes(),
            given=bool(unresolved is not None),
            unresolved=list(unresolved or []),
        )
        return result

    def _resolve(self, label: str) -> tuple[str | None, Refusal | None]:
        nid, err = _resolve_single_node(self.G, label)
        return nid, (_refusal(err) if err else None)

    def get_node(self, label) -> NodeDetail | Refusal:
        # Lowercased like upstream's get_node tool, so a refusal quotes the same label.
        nid, refused = self._resolve(label.lower())
        if refused:
            return refused
        d = self.G.nodes[nid]
        return NodeDetail(
            label=d.get("label", nid), id=nid,
            source_file=str(d.get("source_file", "")),
            source_location=str(d.get("source_location", "")),
            file_type=str(d.get("file_type", "")),
            community=str(d.get("community", "")),
            degree=self.G.degree(nid),
        )

    def get_neighbors(self, label, relation_filter="") -> Neighbors | Refusal:
        nid, refused = self._resolve(label.lower())
        if refused:
            return refused
        rf = relation_filter.lower()
        recs: list[NeighborRecord] = []
        for nb in self.G.successors(nid):
            for d in edge_datas(self.G, nid, nb):
                rel = d.get("relation", "")
                if rf and rf not in rel.lower():
                    continue
                recs.append(NeighborRecord(True, self.G.nodes[nb].get("label", nb), str(rel),
                                           str(d.get("confidence", "")),
                                           str(d.get("source_file") or ""),
                                           str(d.get("source_location") or "")))
        for nb in self.G.predecessors(nid):
            for d in edge_datas(self.G, nb, nid):
                rel = d.get("relation", "")
                if rf and rf not in rel.lower():
                    continue
                recs.append(NeighborRecord(False, self.G.nodes[nb].get("label", nb), str(rel),
                                           str(d.get("confidence", "")),
                                           str(d.get("source_file") or ""),
                                           str(d.get("source_location") or "")))
        return Neighbors(label=self.G.nodes[nid].get("label", nid), neighbors=recs)

    def get_community(self, community_id) -> CommunityRecord | None:
        members = self.communities.get(int(community_id))
        if not members:
            return None
        recs = [
            NodeRecord(label=self.G.nodes[n].get("label", n),
                       source_file=str(self.G.nodes[n].get("source_file", "")))
            for n in members
        ]
        name = self.G.nodes[members[0]].get("community_name")
        return CommunityRecord(cid=int(community_id), members=recs, name=name)

    def god_nodes(self, top_n=10) -> list[GodNode]:
        from graphify.analyze import god_nodes as _god
        return [GodNode(label=n["label"], degree=n["degree"]) for n in _god(self.G, top_n=top_n)]

    def graph_stats(self) -> GraphStats:
        confs = [d.get("confidence", "EXTRACTED") for _, _, d in self.G.edges(data=True)]
        total = len(confs) or 1
        return GraphStats(
            nodes=self.G.number_of_nodes(), edges=self.G.number_of_edges(),
            communities=len(self.communities),
            extracted_pct=round(confs.count("EXTRACTED") / total * 100),
            inferred_pct=round(confs.count("INFERRED") / total * 100),
            ambiguous_pct=round(confs.count("AMBIGUOUS") / total * 100),
            built_at_commit=str(self.G.graph.get("_built_at_commit") or ""),
        )

    def shortest_path(self, source, target, *, max_hops=8, undirected=False) -> PathResult:
        G = self.G
        # An exact hit (path::symbol, node id, exact label or path) names the
        # endpoint outright, a tie or a file::symbol miss is refused, and only
        # then does scoring pick one (#3913, #4264, #1785).
        s_id, src_scored, s_err = _resolve_path_endpoint(G, source)
        if s_err:
            return PathResult(found=False, reason="refused", refusal=_refusal(s_err))
        if s_id is None:
            return PathResult(found=False, reason="no-source")
        t_id, tgt_scored, t_err = _resolve_path_endpoint(G, target)
        if t_err:
            return PathResult(found=False, reason="refused", refusal=_refusal(t_err))
        if t_id is None:
            return PathResult(found=False, reason="no-target")
        if s_id == t_id:
            return PathResult(found=False, reason="same-node", start_label=s_id)
        warnings: list[str] = []
        for name, scored, nid in (("source", src_scored, s_id), ("target", tgt_scored, t_id)):
            # Only meaningful when the raw score head is what got picked — a
            # full-token override was chosen on token coverage, not score.
            if len(scored) >= 2 and nid == scored[0][1]:
                top, runner = scored[0][0], scored[1][0]
                if top > 0 and (top - runner) / top < 0.10:
                    warnings.append(f"warning: {name} match was ambiguous (top score {top:g}, runner-up {runner:g})")
        # Honour the graph's own directedness instead of always flattening it
        # (#2487): the CLI loads directed by default and undirected only on
        # --undirected, so forcing to_undirected() here made the flag a no-op and
        # reported a backwards walk as a real path.
        # Направление — намерение вызывающего, а не свойство загрузки: иначе
        # тот же флаг приходится кодировать дважды, и бэкенды расходятся.
        directed = G.is_directed() and not undirected
        # Upstream's search graph: directed by the `_src`/`_tgt` markers (#2309)
        # plus the implied reverse of every `contains` (#3878), cached on G.
        # Among equal-length routes the smallest by node ids, so the answer does
        # not depend on hash seed (#2074) and matches ArcadeDB.
        path_nodes = _lexmin_shortest_path(
            _nx_expand(_path_search_graph(G, not directed)), s_id, t_id)
        if path_nodes is None:
            # "No path at all" and "no path THAT WAY" are different answers, and
            # only the second one is fixed by --undirected. Suggesting the flag
            # for genuinely disconnected nodes sends the caller down a retry that
            # cannot work, so ask the undirected question before choosing.
            blocked_by_direction = directed and _lexmin_shortest_path(
                _nx_expand(_path_search_graph(G, True)), s_id, t_id) is not None
            return PathResult(found=False,
                              reason=("no-path-directed" if blocked_by_direction
                                      else "no-path-directed-none" if directed else "no-path"),
                              warnings=warnings,
                              start_label=G.nodes[s_id].get("label", s_id),
                              end_label=G.nodes[t_id].get("label", t_id))
        hops = len(path_nodes) - 1
        if hops > max_hops:
            return PathResult(found=False, reason=f"too-long:{hops}", warnings=warnings)
        segments: list[PathSegment] = []
        for u, v in zip(path_nodes, path_nodes[1:]):
            # Direction from the stored markers, not from which way the walk
            # crossed the edge: on an undirected load has_edge() is true both
            # ways (#2309). A marker is trusted only when it names this pair.
            edges = []
            for a, b in ((u, v), (v, u)):
                if G.has_edge(a, b):
                    for d in edge_datas(G, a, b):
                        src = d.get("_src", a)
                        edges.append((src if src in (u, v) else a,
                                      d.get("relation"), d.get("confidence")))
            segments.append(_path_segment(u, v, edges, G.nodes[v].get("label", v)))
        return PathResult(found=True, start_label=G.nodes[s_id].get("label", s_id),
                          hops=hops, segments=segments, warnings=warnings)

    def explain(self, label) -> ExplainResult | Refusal:
        nid, refused = self._resolve(label)
        if refused:
            return refused
        d = self.G.nodes[nid]
        detail = NodeDetail(
            label=d.get("label", nid), id=nid,
            source_file=str(d.get("source_file", "")), source_location=str(d.get("source_location", "")),
            file_type=str(d.get("file_type", "")), community=str(d.get("community", "")),
            degree=self.G.degree(nid),
            definition_file=str(d.get("definition_file", "") or ""),
            definition_location=str(d.get("definition_location", "") or ""),
        )
        conns: list[Connection] = []

        def _outgoing(e: dict, other: str, default: bool) -> bool:
            """Whether this edge leaves `nid`, by the stored markers.

            Arc order alone is not the truth: a link persisted in flipped
            endpoint order carries its direction in `_src` (#2309), so
            classifying by which adjacency list it came from showed a caller as
            a callee. Markers are trusted only when they name this edge.
            """
            src, tgt = e.get("_src"), e.get("_tgt")
            if src is None or tgt is None or {src, tgt} != {nid, other}:
                return default          # which adjacency list it came from
            return src == nid

        for nb in self.G.successors(nid):
            for e in edge_datas(self.G, nid, nb):
                conns.append(Connection(_outgoing(e, nb, True), self.G.nodes[nb].get("label", nb),
                                        str(e.get("relation", "")), str(e.get("confidence", "")), self.G.degree(nb),
                                        source_file=str(e.get("source_file", "") or ""),
                                        source_location=str(e.get("source_location", "") or "")))
        for nb in self.G.predecessors(nid):
            for e in edge_datas(self.G, nb, nid):
                conns.append(Connection(_outgoing(e, nb, False), self.G.nodes[nb].get("label", nb),
                                        str(e.get("relation", "")), str(e.get("confidence", "")), self.G.degree(nb),
                                        source_file=str(e.get("source_file", "") or ""),
                                        source_location=str(e.get("source_location", "") or "")))
        return ExplainResult(detail=detail, connections=conns)

    def pr_impact(self, files) -> tuple[list[int], int]:
        from graphify.prs import compute_pr_impact
        return compute_pr_impact(files, self.G)


# --------------------------------------------------------------------------- #
# Renderers - the single place text is formatted + sanitized (F-010)
# --------------------------------------------------------------------------- #
def render_query(r: QueryResult, token_budget: int = 2000, *,
                 context_filters=None, graph_path: str | None = None) -> str:
    """Render a query answer.

    ``graph_path`` prefixes the header with the graph the answer came from
    (#2789), matching `serve._query_graph_text`: `graphify-out/` resolves
    against the CWD, so an answer taken from a parent project's graph is
    otherwise indistinguishable from the one you meant. ``context_filters`` is
    accepted so the CLI can show what it resolved even when the backend
    reported none.
    """
    if r.refusal:
        return r.refusal.message
    if not r.seeds:
        return "No matching nodes found."
    seed_label = f"Start (given): {r.seeds}" if r.given else f"Start: {r.seeds}"
    parts = [f"Traversal: {r.mode.upper()} depth={r.depth}", seed_label]
    filters = r.filters or (list(context_filters) if context_filters else [])
    if filters:
        parts.append(f"Context: {', '.join(filters)} ({r.filter_source})")
    if r.unresolved:
        parts.append(f"Unresolved seeds (skipped): {r.unresolved}")
    parts.append(f"{r.total_nodes} nodes found")
    if graph_path:
        from graphify.serve import _display_graph_path
        # Размер ГРАФА, не обхода: «355 nodes» против «3178 nodes» — обычно
        # первое, что выдаёт ответ из чужого корпуса (#2789).
        _n = r.graph_nodes if r.graph_nodes is not None else r.total_nodes
        parts.insert(0, f"Graph: {_display_graph_path(graph_path)} ({_n} nodes)")
    header = " | ".join(parts) + "\n\n"

    lines = [
        f"NODE {sanitize_label(n.label)} "
        f"[src={sanitize_label(n.source_file)} loc={sanitize_label(n.source_location)} "
        f"community={sanitize_label(n.community)}]"
        for n in r.nodes
    ]
    for e in r.edges:
        ctx = f" context={sanitize_label(e.context)}" if e.context else ""
        loc = str(e.source_location or "")
        at = (f" at={sanitize_label(str(e.source_file or ''))}:{sanitize_label(loc)}"
              if loc else "")
        lines.append(
            f"EDGE {sanitize_label(e.source_label)} --{sanitize_label(e.relation)} "
            f"[{sanitize_label(e.confidence)}{ctx}]--> {sanitize_label(e.target_label)}{at}"
        )
    output = "\n".join(lines)
    char_budget = token_budget * 3
    if len(output) > char_budget:
        cut_at = output[:char_budget].rfind("\n")
        cut_at = cut_at if cut_at > 0 else char_budget
        # Never cut the seed nodes: they render first, so if the budget lands
        # inside the seed block, extend the cut to cover it. The symbol the
        # question named must always be in the answer (#BUG2).
        n_seed_lines = min(len(r.seeds), len(lines))
        if n_seed_lines:
            seed_block_end = sum(len(lines[i]) + 1 for i in range(n_seed_lines)) - 1
            cut_at = max(cut_at, min(seed_block_end, len(output)))
        total_nodes = sum(1 for l in lines if l.startswith("NODE "))
        shown = output[:cut_at].count("\nNODE ") + (1 if output.startswith("NODE ") else 0)
        cut_count = total_nodes - shown
        if cut_count == 0:
            # Every node fits but nodes+edges do not. Edges are never dropped
            # once every node fits, so this is the complete answer — say it is
            # over budget rather than warning about nodes that were never cut
            # (#2601, #2784).
            total_edges = sum(1 for l in lines if l.startswith("EDGE "))
            est_tokens = len(output) // 3
            return header + (
                f"[i] Complete answer over budget: all {total_nodes} nodes and "
                f"{total_edges} edges shown (~{est_tokens} tokens vs the "
                f"requested ~{token_budget}-token budget). Edges are never "
                f"dropped once every node fits, so this is already the full "
                f"answer — raising --budget further will not shrink it. Narrow "
                f"with context_filter=['call'] or use get_node for a specific "
                f"symbol to reduce size instead.\n\n"
            ) + output
        # Prominent notice at the TOP as well as the end marker, mirroring
        # `serve._subgraph_to_text`: silence reads as absence, so a truncated
        # answer must never be mistakable for a complete one. Both wrapper lines
        # sit OUTSIDE char_budget by design.
        # Smallest budget whose cut keeps every NODE line; past that, edges are never dropped.
        node_chars = sum(len(l) + 1 for l in lines if l.startswith("NODE "))
        full_budget = (min(node_chars, len(output)) + 2) // 3
        output = (
            f"[!] TRUNCATED: showing {shown} of {total_nodes} nodes "
            f"(~{token_budget}-token budget). The answer may be among the "
            f"{cut_count} cut nodes — raise the token budget to {full_budget} to get "
            f"all {total_nodes} (CLI: --budget {full_budget}, MCP: "
            f"token_budget={full_budget}) or narrow the query (e.g. "
            f"context_filter=['call'], or get_node for a specific symbol).\n\n"
            + output[:cut_at]
            + f"\n... (truncated — {cut_count} more nodes cut by ~{token_budget}-token budget."
            f" Narrow with context_filter=['call'] or use get_node for a specific symbol)"
        )
    return header + output


def render_node(d: NodeDetail | Refusal, label: str) -> str:
    if isinstance(d, Refusal):
        return d.message
    return "\n".join([
        f"Node: {sanitize_label(d.label)}",
        f"  ID: {sanitize_label(d.id)}",
        f"  Source: {sanitize_label(d.source_file)} {sanitize_label(d.source_location)}",
        f"  Type: {sanitize_label(d.file_type)}",
        f"  Community: {sanitize_label(d.community)}",
        f"  Degree: {d.degree}",
    ])


def render_neighbors(n: Neighbors | Refusal, label: str,
                     token_budget: int | None = None) -> str:
    """Соседи узла; при заданном бюджете вывод режется по границе строки.

    Бюджет живёт здесь, а не в бэкенде: узел-хаб на большом корпусе иначе
    заливает контекст вызывающего целиком, и оба бэкенда обязаны обрезать
    одинаково (#2069).
    """
    if isinstance(n, Refusal):
        return n.message
    lines = [f"Neighbors of {sanitize_label(n.label)}:"]
    for nb in n.neighbors:
        arrow = "  -->" if nb.outgoing else "  <--"
        loc = str(nb.source_location or "")
        at = (f" at={sanitize_label(str(nb.source_file or ''))}:{sanitize_label(loc)}"
              if loc else "")
        lines.append(f"{arrow} {sanitize_label(nb.label)} [{sanitize_label(nb.relation)}] [{sanitize_label(nb.confidence)}]{at}")
    if token_budget:
        from graphify.serve import _cut_lines_to_budget
        return _cut_lines_to_budget(
            lines, token_budget,
            "Narrow with relation_filter or use get_node for a specific symbol.")
    return "\n".join(lines)


def render_community(c: CommunityRecord | None, cid: int,
                     token_budget: int | None = None) -> str:
    if c is None:
        return f"Community {cid} not found."
    # Header "Community N — Name" (#1448); skip the name when it is just the
    # "Community N" placeholder written for unnamed communities, and fall back
    # to the bare id when there is no name. Name is sanitised (F-010).
    base = f"Community {c.cid}"
    header = base
    if c.name:
        clean = sanitize_label(str(c.name))
        if clean and clean != base:
            header = f"{base} — {clean}"
    lines = [f"{header} ({len(c.members)} nodes):"]
    for m in c.members:
        lines.append(f"  {sanitize_label(m.label)} [{sanitize_label(m.source_file)}]")
    if token_budget:
        from graphify.serve import _cut_lines_to_budget
        return _cut_lines_to_budget(
            lines, token_budget,
            "Use get_node for a specific symbol or query for a focused subgraph.")
    return "\n".join(lines)


def render_god_nodes(nodes: list[GodNode]) -> str:
    lines = ["God nodes (most connected):"]
    lines += [f"  {i}. {n.label} - {n.degree} edges" for i, n in enumerate(nodes, 1)]
    return "\n".join(lines)


def render_stats(s: GraphStats, *, head: str | None = None) -> str:
    """Graph statistics; ``head`` is the repo's current commit, None when unknown.

    The build-commit line is upstream's `_build_commit_line` (748de12): an MCP
    client cannot stat graph.json, so this is its only staleness signal (#3354).
    """
    out = (
        f"Nodes: {s.nodes}\n"
        f"Edges: {s.edges}\n"
        f"Communities: {s.communities}\n"
        f"EXTRACTED: {s.extracted_pct}%\n"
        f"INFERRED: {s.inferred_pct}%\n"
        f"AMBIGUOUS: {s.ambiguous_pct}%\n"
    )
    commit = s.built_at_commit
    if not commit:
        return out
    if not head:
        return out + f"Built at commit: {commit}\n"
    if head == commit:
        return out + f"Built at commit: {commit} (matches HEAD)\n"
    return out + (f"Built at commit: {commit}\n"
                  f"HEAD is {head[:7]}, graph built at {commit[:7]}: graph may be stale\n")


def render_path(p: PathResult, source: str, target: str, max_hops: int = 8) -> str:
    if not p.found:
        if p.refusal:
            return p.refusal.message
        if p.reason == "no-source":
            return f"No node matching source '{source}' found."
        if p.reason == "no-target":
            return f"No node matching target '{target}' found."
        if p.reason == "same-node":
            return (f"'{source}' and '{target}' both resolved to the same node "
                    f"'{p.start_label}'. Use a more specific label or the exact node ID.")
        if p.reason == "no-path" or p.reason.startswith("no-path-directed"):
            # Name the direction when the search respected it (#2487): otherwise
            # "no path" reads as "these are unrelated" when the real answer is
            # "not in that direction — try --undirected".
            if p.reason.startswith("no-path-directed"):
                # The mode that was searched is named either way (#2487), but the
                # --undirected hint is added only when an undirected path really
                # exists: advising a retry that cannot succeed is worse than
                # silence for two genuinely disconnected nodes.
                hint = (" Retry with --undirected to walk edges in either direction."
                        if p.reason == "no-path-directed" else "")
                return (f"No directed path found between '{p.start_label}' and "
                        f"'{p.end_label or target}'.{hint}")
            return f"No path found between '{p.start_label}' and '{p.end_label or target}'."
        if p.reason.startswith("too-long:"):
            return f"Path exceeds max_hops={max_hops} ({p.reason.split(':', 1)[1]} hops found)."
        return f"No path found between '{source}' and '{target}'."
    segments = [p.start_label]
    for seg in p.segments:
        conf = f" [{seg.confidence}]" if seg.confidence else ""
        # An edge with no stored relation prints an honest "related" rather than
        # an empty "---->" arrow, which read as a rendering bug (#2074).
        rel = seg.relation or "related"
        if seg.outgoing:
            segments.append(f"--{rel}{conf}--> {seg.label}")
        else:
            segments.append(f"<--{rel}{conf}-- {seg.label}")
    prefix = ("\n".join(p.warnings) + "\n") if p.warnings else ""
    return prefix + f"Shortest path ({p.hops} hops):\n  " + " ".join(segments)


def render_explain(r: ExplainResult | Refusal, label: str, *, max_connections: int = 20) -> str:
    """Reproduce the `graphify explain` node card + degree-ranked connections."""
    if isinstance(r, Refusal):
        return r.message
    d = r.detail
    lines = [
        f"Node: {d.label}",
        f"  ID:        {d.id}",
        f"  Source:    {d.source_file} {d.source_location}".rstrip(),
    ]
    if d.definition_file:
        lines.append(f"  Defined in: {d.definition_file} {d.definition_location}".rstrip())
    lines += [
        f"  Type:      {d.file_type}",
        f"  Community: {d.community}",
        f"  Degree:    {d.degree}",
    ]
    conns = r.connections
    if conns:
        ranked = sorted(conns, key=lambda c: c.degree, reverse=True)
        lines.append(f"\nConnections ({len(conns)}):")
        for c in ranked[:max_connections]:
            arrow = "-->" if c.outgoing else "<--"
            # Место РЕБРА — точка вызова в файле источника, а не def соседа.
            at = f" {c.source_file}:{c.source_location}" if c.source_location else ""
            lines.append(f"  {arrow} {c.label} [{c.relation}] [{c.confidence}]{at}")
        cut = ranked[max_connections:]
        if cut:
            lines.append(f"  ... and {len(cut)} more (--all lists all {len(conns)})")
            # Голое «и ещё N» прячет ответ на вопрос «кто это вызывает» на узле
            # высокой степени и отправляет обратно к grep. Сводим срезанное по
            # файлам — счётчики обязаны сходиться к len(cut).
            groups: dict[tuple[bool, str], int] = {}
            for c in cut:
                key = (c.outgoing, c.source_file or "<unknown>")
                groups[key] = groups.get(key, 0) + 1
            lines.append("  Grouped by file:")
            for (outgoing, sfile), n in sorted(
                    groups.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0])):
                arrow = "-->" if outgoing else "<--"
                lines.append(f"    {arrow} {sfile}: {n} connection{'s' if n != 1 else ''}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Store selection: env-driven, the snapshot by default
# --------------------------------------------------------------------------- #
from graphify.store_errors import GraphDBUnavailable  # noqa: E402,F401  (re-exported)

# Values of GRAPHIFY_GRAPH_STORE. `snap` is the default; anything else is an
# error, because a typo such as `snapshot` would otherwise silently select JSON,
# i.e. loading a whole graph.json into memory.
GRAPH_STORE_VALUES = frozenset({"snap", "json"})

# ArcadeDB was a graph store until 2026-11-15 (retire-arcadedb). An explicit
# choice of it is refused with its own message rather than "unknown value".
RETIRED_STORE_VALUES = frozenset({"arcadedb", "arcade"})

# The values the fork used to put in GRAPHIFY_BACKEND before upstream took that
# variable for the LLM provider (#4361): llm._graphify_backend_setting() filters
# them out so a stale setting does not select an LLM backend. Kept for good, after
# ArcadeDB's retirement too: the settings outlive the code that read them.
STALE_BACKEND_VALUES = frozenset({"arcadedb", "arcade", "json"})


def resolve_backend_config(graph_path: str | None = None) -> dict:
    """Resolve the graph store from ``GRAPHIFY_GRAPH_STORE``.

    The snapshot (``snap``) is the default: a memory-mapped file next to each
    graph.json (capability ``graph-snapshot-store``). ``json`` is the explicit
    in-process model, kept for tests, CI and upstream compatibility, not as a
    fallback the system chooses on its own. Any other value is an error.
    ``GRAPHIFY_BACKEND`` is not read: upstream uses it to pick the LLM provider.
    """
    import os
    kind = os.environ.get("GRAPHIFY_GRAPH_STORE", "").strip().lower() or "snap"
    if kind in RETIRED_STORE_VALUES:
        raise GraphDBUnavailable(
            f"GRAPHIFY_GRAPH_STORE={kind!r}: ArcadeDB was retired; the graph store is "
            f"the snapshot (snap, the default).\n"
            f"Remove GRAPHIFY_GRAPH_STORE, or set it to snap or json.")
    if kind not in GRAPH_STORE_VALUES:
        raise GraphDBUnavailable(
            f"GRAPHIFY_GRAPH_STORE={kind!r} is not a graph store.\n"
            f"Valid values: snap (default), json.")
    if kind == "snap":
        from graphify.snap_store import snapshot_path_for
        return {"kind": "snap", "graph_path": graph_path,
                "snapshot_path": str(snapshot_path_for(graph_path)) if graph_path else None}
    return {"kind": "json", "graph_path": graph_path}


def open_backend(*, config: dict | None = None, graph=None, graph_path: str | None = None) -> GraphBackend:
    """Construct the configured backend.

    Snapshot: opens the memory-mapped file next to the graph (refuses when it is
    missing, stale or incompatible). JSON: reuses a preloaded ``graph`` when
    given (so callers that already enforced the size cap don't reload), else
    loads from ``graph_path``.
    """
    cfg = config or resolve_backend_config(graph_path)
    if cfg["kind"] == "snap":
        if not cfg.get("snapshot_path"):
            raise GraphDBUnavailable("the graph snapshot needs a graph.json path to sit next to.")
        from graphify.snap_backend import SnapBackend
        return SnapBackend(cfg["snapshot_path"], graph_json=cfg.get("graph_path") or graph_path)
    if graph is None:
        from graphify.serve import _load_graph
        graph = _load_graph(cfg.get("graph_path") or graph_path)
    return JsonBackend(graph)


def require_backend(*, config: dict | None = None, graph=None, graph_path: str | None = None,
                    **_kw) -> GraphBackend:
    """`open_backend`, checked: an unavailable store is fatal, never a fallback.

    The snapshot backend refuses on its own when it is missing, stale or
    incompatible (``SnapshotUnavailable``). Lives in this fork-owned file so the
    churn-file call sites stay one line: upstream merges can sever a call, and
    the guard tests make that visible, but they cannot carry off the logic.
    """
    return open_backend(config=config, graph=graph, graph_path=graph_path)


def require_service(graph_path: str | None = None, *, config: dict | None = None, **_kw) -> None:
    """Fail at the START of a pipeline if the graph store cannot be written.

    The store is written at the very end, after AST extraction, LLM calls and
    real API spend; learning there that the snapshot module is missing is the
    expensive way to learn it. A no-op for the JSON store.
    """
    cfg = config or resolve_backend_config(graph_path)
    if cfg["kind"] == "snap":
        from graphify.snap_store import check_api
        check_api()
