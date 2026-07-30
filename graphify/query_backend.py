"""Pluggable read query backend for the knowledge graph.

The graph is built and exported as ``graph.json`` exactly as before; this module
is only about *serving queries*. ``GraphBackend`` is the interface the MCP tools
(serve.py) and the CLI (``graphify query|path|explain``) read through, so the
same graph can be backed by the in-process NetworkX model (``JsonBackend``,
default) or a server graph database (``ArcadeDBBackend``, added in a later phase)
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

from graphify.build import edge_data
from graphify.security import sanitize_label
from graphify.serve import (
    _bfs,
    _bfs_core,
    _dfs,
    _dfs_core,
    _filter_graph_by_context,
    _find_node,
    _hub_threshold,
    _idf_value,
    _pick_scored_endpoint,
    _pick_seeds,
    _query_terms,
    _resolve_context_filters,
    _score_nodes,
    _score_record,
    _score_terms,
    _search_tokens,
    _strip_diacritics,
)


# --------------------------------------------------------------------------- #
# Materialized result records (no nx.Graph reference crosses this boundary)
# --------------------------------------------------------------------------- #
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


@dataclass
class NodeDetail:
    label: str
    id: str
    source_file: str = ""
    source_location: str = ""
    file_type: str = ""
    community: str = ""
    degree: int = 0


@dataclass
class NeighborRecord:
    outgoing: bool                         # True: --> , False: <--
    label: str
    relation: str = ""
    confidence: str = ""


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


@dataclass
class Connection:
    outgoing: bool                         # True: --> , False: <--
    label: str
    relation: str = ""
    confidence: str = ""
    degree: int = 0                        # neighbour degree, used to rank connections


@dataclass
class ExplainResult:
    detail: NodeDetail
    connections: list[Connection] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class GraphBackend(ABC):
    """Read-only view over a knowledge graph, addressed by labels/ids."""

    @abstractmethod
    def query(self, question: str, *, mode: str = "bfs", depth: int = 3,
              context_filters: list[str] | None = None) -> QueryResult: ...

    @abstractmethod
    def get_node(self, label: str) -> NodeDetail | None: ...

    @abstractmethod
    def get_neighbors(self, label: str, relation_filter: str = "") -> Neighbors | None: ...

    @abstractmethod
    def get_community(self, community_id: int) -> CommunityRecord | None: ...

    @abstractmethod
    def god_nodes(self, top_n: int = 10) -> list[GodNode]: ...

    @abstractmethod
    def graph_stats(self) -> GraphStats: ...

    @abstractmethod
    def shortest_path(self, source: str, target: str, *, max_hops: int = 8) -> PathResult: ...

    @abstractmethod
    def explain(self, label: str) -> ExplainResult | None: ...

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

    def query(self, question, *, mode="bfs", depth=3, context_filters=None) -> QueryResult:
        G = self.G
        terms = _query_terms(question)
        scored = _score_nodes(G, terms)
        seeds = _pick_seeds(scored)
        if not seeds:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[])
        filters, fsource = _resolve_context_filters(question, context_filters)
        TG = _filter_graph_by_context(G, filters)
        visited, edges = (_dfs if mode == "dfs" else _bfs)(TG, seeds, depth)

        # Match _subgraph_to_text exactly: the query path calls it WITHOUT seeds,
        # so nodes are ordered purely by degree desc (no seeds-first promotion).
        ordered = sorted(visited, key=lambda n: TG.degree(n), reverse=True)
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
                d = edge_data(TG, u, v)
                edge_recs.append(EdgeRecord(
                    source_label=TG.nodes[u].get("label", u),
                    target_label=TG.nodes[v].get("label", v),
                    relation=str(d.get("relation", "")),
                    confidence=str(d.get("confidence", "")),
                    context=str(d.get("context", "") or ""),
                ))
        return QueryResult(
            mode=mode, depth=depth,
            seeds=[G.nodes[n].get("label", n) for n in seeds],
            total_nodes=len(visited), nodes=node_recs, edges=edge_recs,
            filters=filters, filter_source=fsource,
        )

    def get_node(self, label) -> NodeDetail | None:
        needle = label.lower()
        for nid, d in self.G.nodes(data=True):
            if needle in (d.get("label") or "").lower() or needle == nid.lower():
                return NodeDetail(
                    label=d.get("label", nid), id=nid,
                    source_file=str(d.get("source_file", "")),
                    source_location=str(d.get("source_location", "")),
                    file_type=str(d.get("file_type", "")),
                    community=str(d.get("community", "")),
                    degree=self.G.degree(nid),
                )
        return None

    def get_neighbors(self, label, relation_filter="") -> Neighbors | None:
        matches = _find_node(self.G, label.lower())
        if not matches:
            return None
        nid = matches[0]
        rf = relation_filter.lower()
        recs: list[NeighborRecord] = []
        for nb in self.G.successors(nid):
            d = edge_data(self.G, nid, nb)
            rel = d.get("relation", "")
            if rf and rf not in rel.lower():
                continue
            recs.append(NeighborRecord(True, self.G.nodes[nb].get("label", nb), str(rel), str(d.get("confidence", ""))))
        for nb in self.G.predecessors(nid):
            d = edge_data(self.G, nb, nid)
            rel = d.get("relation", "")
            if rf and rf not in rel.lower():
                continue
            recs.append(NeighborRecord(False, self.G.nodes[nb].get("label", nb), str(rel), str(d.get("confidence", ""))))
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
        )

    def shortest_path(self, source, target, *, max_hops=8) -> PathResult:
        G = self.G
        src_scored = _score_nodes(G, [t.lower() for t in source.split()])
        tgt_scored = _score_nodes(G, [t.lower() for t in target.split()])
        if not src_scored:
            return PathResult(found=False, reason=f"no-source")
        if not tgt_scored:
            return PathResult(found=False, reason=f"no-target")
        # Full-token label matches win over the raw score head (#1785).
        s_id = _pick_scored_endpoint(G, src_scored, source)
        t_id = _pick_scored_endpoint(G, tgt_scored, target)
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
        try:
            path_nodes = nx.shortest_path(G.to_undirected(as_view=True), s_id, t_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return PathResult(found=False, reason="no-path", warnings=warnings,
                              start_label=G.nodes[s_id].get("label", s_id),
                              end_label=G.nodes[t_id].get("label", t_id))
        hops = len(path_nodes) - 1
        if hops > max_hops:
            return PathResult(found=False, reason=f"too-long:{hops}", warnings=warnings)
        segments: list[PathSegment] = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if G.has_edge(u, v):
                d, outgoing = edge_data(G, u, v), True
            else:
                d, outgoing = edge_data(G, v, u), False
            segments.append(PathSegment(outgoing=outgoing, relation=d.get("relation", ""),
                                        confidence=d.get("confidence", ""),
                                        label=G.nodes[v].get("label", v)))
        return PathResult(found=True, start_label=G.nodes[s_id].get("label", s_id),
                          hops=hops, segments=segments, warnings=warnings)

    def explain(self, label) -> ExplainResult | None:
        matches = _find_node(self.G, label)
        if not matches:
            return None
        nid = matches[0]
        d = self.G.nodes[nid]
        detail = NodeDetail(
            label=d.get("label", nid), id=nid,
            source_file=str(d.get("source_file", "")), source_location=str(d.get("source_location", "")),
            file_type=str(d.get("file_type", "")), community=str(d.get("community", "")),
            degree=self.G.degree(nid),
        )
        conns: list[Connection] = []
        for nb in self.G.successors(nid):
            e = edge_data(self.G, nid, nb)
            conns.append(Connection(True, self.G.nodes[nb].get("label", nb),
                                    str(e.get("relation", "")), str(e.get("confidence", "")), self.G.degree(nb)))
        for nb in self.G.predecessors(nid):
            e = edge_data(self.G, nb, nid)
            conns.append(Connection(False, self.G.nodes[nb].get("label", nb),
                                    str(e.get("relation", "")), str(e.get("confidence", "")), self.G.degree(nb)))
        return ExplainResult(detail=detail, connections=conns)

    def pr_impact(self, files) -> tuple[list[int], int]:
        from graphify.prs import compute_pr_impact
        return compute_pr_impact(files, self.G)


# --------------------------------------------------------------------------- #
# Renderers - the single place text is formatted + sanitized (F-010)
# --------------------------------------------------------------------------- #
def render_query(r: QueryResult, token_budget: int = 2000) -> str:
    if not r.seeds:
        return "No matching nodes found."
    parts = [f"Traversal: {r.mode.upper()} depth={r.depth}", f"Start: {r.seeds}"]
    if r.filters:
        parts.append(f"Context: {', '.join(r.filters)} ({r.filter_source})")
    parts.append(f"{r.total_nodes} nodes found")
    header = " | ".join(parts) + "\n\n"

    lines = [
        f"NODE {sanitize_label(n.label)} "
        f"[src={sanitize_label(n.source_file)} loc={sanitize_label(n.source_location)} "
        f"community={sanitize_label(n.community)}]"
        for n in r.nodes
    ]
    for e in r.edges:
        ctx = f" context={sanitize_label(e.context)}" if e.context else ""
        lines.append(
            f"EDGE {sanitize_label(e.source_label)} --{sanitize_label(e.relation)} "
            f"[{sanitize_label(e.confidence)}{ctx}]--> {sanitize_label(e.target_label)}"
        )
    output = "\n".join(lines)
    char_budget = token_budget * 3
    if len(output) > char_budget:
        cut_at = output[:char_budget].rfind("\n")
        cut_at = cut_at if cut_at > 0 else char_budget
        total_nodes = sum(1 for l in lines if l.startswith("NODE "))
        shown = output[:cut_at].count("\nNODE ") + (1 if output.startswith("NODE ") else 0)
        output = (
            output[:cut_at]
            + f"\n... (truncated — {total_nodes - shown} more nodes cut by ~{token_budget}-token budget."
            f" Narrow with context_filter=['call'] or use get_node for a specific symbol)"
        )
    return header + output


def render_node(d: NodeDetail | None, label: str) -> str:
    if d is None:
        return f"No node matching '{label}' found."
    return "\n".join([
        f"Node: {sanitize_label(d.label)}",
        f"  ID: {sanitize_label(d.id)}",
        f"  Source: {sanitize_label(d.source_file)} {sanitize_label(d.source_location)}",
        f"  Type: {sanitize_label(d.file_type)}",
        f"  Community: {sanitize_label(d.community)}",
        f"  Degree: {d.degree}",
    ])


def render_neighbors(n: Neighbors | None, label: str) -> str:
    if n is None:
        return f"No node matching '{label}' found."
    lines = [f"Neighbors of {sanitize_label(n.label)}:"]
    for nb in n.neighbors:
        arrow = "  -->" if nb.outgoing else "  <--"
        lines.append(f"{arrow} {sanitize_label(nb.label)} [{sanitize_label(nb.relation)}] [{sanitize_label(nb.confidence)}]")
    return "\n".join(lines)


def render_community(c: CommunityRecord | None, cid: int) -> str:
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
    return "\n".join(lines)


def render_god_nodes(nodes: list[GodNode]) -> str:
    lines = ["God nodes (most connected):"]
    lines += [f"  {i}. {n.label} - {n.degree} edges" for i, n in enumerate(nodes, 1)]
    return "\n".join(lines)


def render_stats(s: GraphStats) -> str:
    return (
        f"Nodes: {s.nodes}\n"
        f"Edges: {s.edges}\n"
        f"Communities: {s.communities}\n"
        f"EXTRACTED: {s.extracted_pct}%\n"
        f"INFERRED: {s.inferred_pct}%\n"
        f"AMBIGUOUS: {s.ambiguous_pct}%\n"
    )


def render_path(p: PathResult, source: str, target: str, max_hops: int = 8) -> str:
    if not p.found:
        if p.reason == "no-source":
            return f"No node matching source '{source}' found."
        if p.reason == "no-target":
            return f"No node matching target '{target}' found."
        if p.reason == "same-node":
            return (f"'{source}' and '{target}' both resolved to the same node "
                    f"'{p.start_label}'. Use a more specific label or the exact node ID.")
        if p.reason == "no-path":
            return f"No path found between '{p.start_label}' and '{p.end_label or target}'."
        if p.reason.startswith("too-long:"):
            return f"Path exceeds max_hops={max_hops} ({p.reason.split(':', 1)[1]} hops found)."
        return f"No path found between '{source}' and '{target}'."
    segments = [p.start_label]
    for seg in p.segments:
        conf = f" [{seg.confidence}]" if seg.confidence else ""
        if seg.outgoing:
            segments.append(f"--{seg.relation}{conf}--> {seg.label}")
        else:
            segments.append(f"<--{seg.relation}{conf}-- {seg.label}")
    prefix = ("\n".join(p.warnings) + "\n") if p.warnings else ""
    return prefix + f"Shortest path ({p.hops} hops):\n  " + " ".join(segments)


def render_explain(r: ExplainResult | None, label: str, *, max_connections: int = 20) -> str:
    """Reproduce the `graphify explain` node card + degree-ranked connections."""
    if r is None:
        return f"No node matching '{label}' found."
    d = r.detail
    lines = [
        f"Node: {d.label}",
        f"  ID:        {d.id}",
        f"  Source:    {d.source_file} {d.source_location}".rstrip(),
        f"  Type:      {d.file_type}",
        f"  Community: {d.community}",
        f"  Degree:    {d.degree}",
    ]
    conns = r.connections
    if conns:
        lines.append(f"\nConnections ({len(conns)}):")
        for c in sorted(conns, key=lambda c: c.degree, reverse=True)[:max_connections]:
            arrow = "-->" if c.outgoing else "<--"
            lines.append(f"  {arrow} {c.label} [{c.relation}] [{c.confidence}]")
        if len(conns) > max_connections:
            lines.append(f"  ... and {len(conns) - max_connections} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ArcadeDBBackend - same interface, served from a graph database (connect-only)
# --------------------------------------------------------------------------- #
def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def shortfall_warning(stats: dict) -> str | None:
    """One line naming a load that did not fully arrive, or None when it did.

    ``load_from_graph_json``/``sync_graph`` used to report the counts they
    *intended* to write, so a load that silently dropped 30% of the edges still
    printed success. Callers pass their stats through here instead of restating
    them.
    """
    if stats.get("reconciled") is False:
        return "ArcadeDB load could not be verified: the count query returned nothing"
    missing_n, missing_e = stats.get("missing_nodes", 0), stats.get("missing_edges", 0)
    if not (missing_n or missing_e):
        return None
    parts = []
    if missing_n:
        parts.append(f"{missing_n} of {stats.get('expected_nodes', '?')} nodes")
    if missing_e:
        parts.append(f"{missing_e} of {stats.get('expected_edges', '?')} edges")
    msg = "ArcadeDB load incomplete: " + " and ".join(parts) + " did not arrive"
    unresolved = stats.get("unresolved_endpoint_edges")
    if missing_e and unresolved:
        msg += (f"; {unresolved} of the missing edges had an endpoint that does not "
                f"resolve through the index")
    return msg


def _by_id(row: dict) -> str:
    """Sort key standing in for a dropped ``ORDER BY id``.

    Dropping the unique index on ``id`` changes what ``ORDER BY id`` costs. With
    the index the planner did not sort at all: it walked ``FETCH FROM INDEX
    VALUES ASC Node[id]`` and filtered afterwards, reading every entry in the
    index. Measured on 1.18M nodes, that plan ran 10.9s against 1.5s for the
    same query without the clause - so the clause was the expensive part and
    losing the index makes these queries faster, not slower.

    Without the index the sort lands in the heap, and that is a hard ceiling
    rather than a slowdown: on this server it refuses at 500 000 elements
    ("Limit of allowed elements for in-heap ORDER BY in a single query
    exceeded"). Broad terms on a real corpus are well past it - ``module``
    matches 868 649 rows, ``_`` matches 1 180 620. Today the HTTP default limit
    of 20 000 rows bounds the sort and hides this, so the failure is latent and
    rides on a server default the client never states.

    Sorting here removes that dependency for nothing: every one of these call
    sites already pulls the whole result set into the client. Python compares
    str by code point and UTF-8 preserves that order, so the sequence matches
    what a correct ORDER BY would produce.
    """
    return str(row.get("id") or "")


def _id_key(node_id: str) -> str:
    """ASCII index key for a node id.

    The unique index is built over this, never over ``id`` itself. ArcadeDB's
    LSM index writes a compacted series with an unsigned byte comparison
    (``LSMTreeIndexAbstract.compareKeys``) but seeks it with a signed one
    (``compareKey`` -> ``BinaryComparator.compareBytes``), so on a keyspace that
    mixes ASCII with bytes >= 0x80 the two orders disagree and a point lookup
    silently misses - measured at 68% of probes on a mixed keyspace after
    compaction, and 0% both on pure ASCII and before compaction. Node ids of any
    non-English project are mixed by construction, so the digest keeps the key
    inside the range where both comparisons agree. Upstream fixed the engine in
    02d5800e (#5321), which no release carries yet; this does not depend on it.

    Not reversible on purpose: nothing needs it. ``id`` stays on the record as a
    plain property, so a scan, a LIKE or a manual look in Studio still work.
    """
    from hashlib import blake2b

    return blake2b(str(node_id).encode("utf-8"), digest_size=16).hexdigest()


class ArcadeDBBackend(GraphBackend):
    """Reads from a per-project ArcadeDB database over HTTP.

    Faithfulness to JsonBackend is structural, not coincidental: seed resolution
    reuses the exact scoring kernel (``_score_record`` + ``_idf_value`` with df/N
    fetched from the DB), and traversal reuses ``_bfs_core``/``_dfs_core`` over a
    DB-fetched neighborhood carrying the global degree the hub guard needs.
    ``god_nodes`` is precomputed at load (a global analytic that only changes on
    rebuild) and stored, so it matches ``analyze.god_nodes`` exactly.
    """

    def __init__(self, database, host="127.0.0.1", port=2480, user="root", password="playwithdata"):
        try:
            import requests  # lazy: keep JsonBackend users free of the dependency
        except ImportError as e:
            raise ImportError('ArcadeDB backend needs requests. Run: pip install "graphifyy[arcadedb]"') from e
        self.base = f"http://{host}:{port}"
        self.db = database
        self._s = requests.Session()
        self._s.auth = (user, password)

    # ---- HTTP ---------------------------------------------------------------
    def _server(self, command: str) -> dict:
        r = self._s.post(f"{self.base}/api/v1/server", json={"command": command}, timeout=60)
        r.raise_for_status()
        return r.json() if r.text else {}

    def _run(self, command: str, *, kind="command", language="sql", params=None) -> list[dict]:
        body = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = self._s.post(f"{self.base}/api/v1/{kind}/{self.db}", json=body, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"ArcadeDB {r.status_code}: {r.text[:300]}\n  {command[:160]}")
        return r.json().get("result", [])

    def ready(self) -> bool:
        try:
            return self._s.get(f"{self.base}/api/v1/ready", timeout=5).status_code in (200, 204)
        except Exception:
            return False

    def is_populated(self) -> bool:
        """True if the database already has the schema + at least one node
        (i.e. an incremental sync is possible rather than a first full load)."""
        try:
            return bool(self._run("SELECT count(*) AS c FROM Node")[0]["c"])
        except Exception:
            return False

    def schema_has_id_key(self) -> bool:
        """Whether ``Node`` carries the ASCII index key this backend writes by.

        Read from the schema, not from the data: a database can be empty and
        still be on the current schema, and a full one can predate it. An
        unreadable schema counts as absent, so the caller is told to reload
        rather than allowed to write into something unknown.
        """
        try:
            rows = self._run("SELECT properties FROM schema:types WHERE name = 'Node'",
                             kind="query")
        except Exception:
            return False
        if not rows:
            return False
        return any(p.get("name") == "id_key" for p in (rows[0].get("properties") or []))

    # ---- load (graph.json -> ArcadeDB; one-off materialized view) -----------
    def ensure_database(self, *, drop=False) -> None:
        names = self._server("list databases").get("result", []) or []
        if self.db in names and drop:
            self._server(f"drop database {self.db}")
            names = []
        if self.db not in names:
            self._server(f"create database {self.db}")

    def _reconcile(self, expected_nodes: int, expected_edges: int, *,
                   node_ids, edges, batch=1000) -> dict:
        """What the DB actually holds, against the graph that was just written.

        ``count()`` and not ``count(*)``: the latter answers from a cached
        counter (``cachedRecordCount``, persisted in statistics.json) plus the
        transaction delta rather than from a scan, so it is able to confirm
        itself and is the wrong tool for checking correctness.

        Attribution runs only when something IS missing. Resolving every id
        costs one query per ``batch`` ids - worth paying to name the cause,
        not worth paying to restate a load that already adds up.
        """
        rows_n = self._run("SELECT count() AS c FROM Node")
        rows_e = self._run("SELECT count() AS c FROM Rel")
        if not (rows_n and rows_e):
            # No count came back, so nothing can be compared. Say that instead of
            # crashing the load or, worse, inventing either a shortfall or a
            # clean bill of health.
            return {"nodes": expected_nodes, "edges": expected_edges,
                    "expected_nodes": expected_nodes, "expected_edges": expected_edges,
                    "reconciled": False}
        got_n = int(rows_n[0]["c"] or 0)
        got_e = int(rows_e[0]["c"] or 0)
        out = {"nodes": got_n, "edges": got_e,
               "expected_nodes": expected_nodes, "expected_edges": expected_edges,
               "missing_nodes": max(0, expected_nodes - got_n),
               "missing_edges": max(0, expected_edges - got_e)}
        if not (out["missing_nodes"] or out["missing_edges"]):
            return out

        # Endpoints, not just the node list: an edge may name an id that was
        # never among the nodes at all, and that end is exactly the one that
        # made CREATE EDGE do nothing. Accumulated in a loop rather than with
        # zip(*edges), which would unpack one argument per edge - 2.4M of them
        # on a graph this size.
        seen = set(node_ids)
        for src, tgt in edges:
            seen.add(src)
            seen.add(tgt)
        ids = list(seen)
        unresolved = set()
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            found = {r["id"] for r in self._run(
                "SELECT id FROM Node WHERE id_key IN :k",
                params={"k": [_id_key(x) for x in chunk]})}
            unresolved.update(x for x in chunk if x not in found)
        out["unresolved_nodes"] = len(unresolved)
        # An edge is created from a subquery per endpoint; an empty one makes
        # CREATE EDGE a silent no-op (CreateEdgesStep), so these are the edges
        # the loader could not have written whatever else went wrong.
        out["unresolved_endpoint_edges"] = sum(
            1 for s, t in edges if s in unresolved or t in unresolved)
        return out

    def load_from_graph_json(self, path: str, *, batch=2000) -> dict:
        import json
        from networkx.readwrite import json_graph
        from graphify.analyze import god_nodes as _god

        raw = json.loads(open(path, encoding="utf-8").read())
        if "links" not in raw and "edges" in raw:
            raw = dict(raw, links=raw["edges"])
        nodes = raw["nodes"]
        edges = raw["links"]
        # Build the full graph once: needed for global degree + the god_nodes
        # analytic (computed by the same code JsonBackend would use).
        try:
            G = json_graph.node_link_graph({**raw, "directed": True}, edges="links")
        except TypeError:
            G = json_graph.node_link_graph({**raw, "directed": True})
        degree = dict(G.degree())
        gods = _god(G, top_n=64)

        def _ddl(cmd):
            try:
                self._run(cmd)
            except RuntimeError as exc:
                if "already exist" not in str(exc).lower():
                    raise

        for cmd in ("CREATE VERTEX TYPE Node", "CREATE EDGE TYPE Rel", "CREATE VERTEX TYPE Meta",
                    "CREATE PROPERTY Node.id STRING", "CREATE PROPERTY Node.norm_label STRING",
                    "CREATE PROPERTY Node.degree INTEGER", "CREATE PROPERTY Node.id_key STRING",
                    # Unique over the ASCII digest, never over `id` itself - see _id_key.
                    "CREATE INDEX ON Node (id_key) UNIQUE", "CREATE INDEX ON Node (norm_label) FULL_TEXT",
                    # Indexed so the p99 hub threshold (ORDER BY degree) avoids the
                    # in-heap sort cap on large graphs.
                    "CREATE INDEX ON Node (degree) NOTUNIQUE"):
            _ddl(cmd)

        keep_n = ("id", "label", "source_file", "source_location", "file_type", "community")
        for i in range(0, len(nodes), batch):
            stmts = []
            for n in nodes[i:i + batch]:
                doc = {k: n[k] for k in keep_n if k in n and isinstance(n[k], (str, int, float, bool))}
                doc["norm_label"] = (n.get("norm_label") or _strip_diacritics(str(n.get("label", "")))).lower()
                doc["degree"] = int(degree.get(n["id"], 0))
                doc["id_key"] = _id_key(n["id"])
                stmts.append("INSERT INTO Node CONTENT " + json.dumps(doc, ensure_ascii=False))
            self._run(";".join(stmts), language="sqlscript")

        keep_e = ("relation", "confidence", "weight", "context")
        for i in range(0, len(edges), batch):
            stmts = []
            for e in edges[i:i + batch]:
                doc = {k: e[k] for k in keep_e if k in e and isinstance(e[k], (str, int, float, bool))}
                stmts.append(
                    f"CREATE EDGE Rel FROM (SELECT FROM Node WHERE id_key={_sql_str(_id_key(e['source']))}) "
                    f"TO (SELECT FROM Node WHERE id_key={_sql_str(_id_key(e['target']))}) CONTENT {json.dumps(doc, ensure_ascii=False)}"
                )
            self._run(";".join(stmts), language="sqlscript")

        self._run("INSERT INTO Meta CONTENT " + json.dumps({"key": "god_nodes", "value": gods}, ensure_ascii=False))
        return self._reconcile(len(nodes), len(edges),
                               node_ids=[n["id"] for n in nodes],
                               edges=[(e["source"], e["target"]) for e in edges])

    def sync_graph(self, G, *, changed_sources, pruned_sources, batch=2000) -> dict:
        """Incrementally sync the DB to the updated graph ``G``.

        Touches only dirty files: deletes every node from a changed-or-pruned
        source (cascading its edges), re-inserts the changed sources' nodes and
        the edges incident to them, refreshes the stored degree on affected
        neighbours, and recomputes the ``god_nodes`` analytic. Works for both a
        directed graph and the undirected build graph (edge direction recovered
        from the `_src`/`_tgt` attrs export uses, falling back to (u, v))."""
        import json
        from graphify.analyze import god_nodes as _god

        # A database written before the ASCII index key has no id_key at all, so
        # an incremental pass would delete nodes under one key and re-insert them
        # under another, leaving the DB half-converted. That is the failure mode
        # that does not self-heal: is_populated() stays true, so the next run
        # patches only the newly-changed files and the damage compounds.
        if not self.schema_has_id_key():
            raise RuntimeError(
                f"ArcadeDB database '{self.db}' predates the ASCII index key: Node.id_key is "
                f"absent from its schema, so an incremental sync would leave it half-converted. "
                f"Rebuild it with: graphify arcade reload <project-dir|graph.json>")

        changed = {s for s in changed_sources if s}
        pruned = {s for s in pruned_sources if s}
        dirty = changed | pruned
        if not dirty:
            return {"deleted_sources": 0, "upserted_nodes": 0, "upserted_edges": 0}

        self._run("DELETE VERTEX FROM Node WHERE source_file IN :d", params={"d": list(dirty)})

        degree = dict(G.degree())
        changed_ids = [n for n, d in G.nodes(data=True) if d.get("source_file") in changed]
        changed_set = set(changed_ids)
        # A node id is global to the project, but the dirty-source DELETE above only
        # removes nodes whose stored source_file is dirty. A shared node whose
        # representative source_file drifted to a changed file across runs still
        # lives in the DB under its old (unchanged) source_file, so re-inserting it
        # by id would hit the UNIQUE Node(id_key) index (DuplicatedKeyException).
        # Delete the changed ids directly first; their incident edges are rebuilt below.
        for i in range(0, len(changed_ids), batch):
            self._run("DELETE VERTEX FROM Node WHERE id_key IN :ids",
                      params={"ids": [_id_key(n) for n in changed_ids[i:i + batch]]})
        keep_n = ("id", "label", "source_file", "source_location", "file_type", "community")
        for i in range(0, len(changed_ids), batch):
            stmts = []
            for nid in changed_ids[i:i + batch]:
                data = G.nodes[nid]
                doc = {k: data[k] for k in keep_n if k in data and isinstance(data[k], (str, int, float, bool))}
                doc["id"] = nid
                doc["norm_label"] = (data.get("norm_label") or _strip_diacritics(str(data.get("label", "")))).lower()
                doc["degree"] = int(degree.get(nid, 0))
                doc["id_key"] = _id_key(nid)
                stmts.append("INSERT INTO Node CONTENT " + json.dumps(doc, ensure_ascii=False))
            if stmts:
                self._run(";".join(stmts), language="sqlscript")

        keep_e = ("relation", "confidence", "weight", "context")
        new_edges, neighbors = [], set()
        for u, v, d in G.edges(data=True):
            if u in changed_set or v in changed_set:
                src, tgt = d.get("_src") or u, d.get("_tgt") or v
                new_edges.append((src, tgt, d))
                neighbors.add(u if v in changed_set else v)
        for i in range(0, len(new_edges), batch):
            stmts = []
            for src, tgt, d in new_edges[i:i + batch]:
                doc = {k: d[k] for k in keep_e if k in d and isinstance(d[k], (str, int, float, bool))}
                stmts.append(
                    f"CREATE EDGE Rel FROM (SELECT FROM Node WHERE id_key={_sql_str(_id_key(src))}) "
                    f"TO (SELECT FROM Node WHERE id_key={_sql_str(_id_key(tgt))}) CONTENT {json.dumps(doc, ensure_ascii=False)}"
                )
            if stmts:
                self._run(";".join(stmts), language="sqlscript")

        # Refresh stored degree on unchanged neighbours (their edge count moved).
        stale = [n for n in (neighbors - changed_set) if n in degree]
        for i in range(0, len(stale), batch):
            stmts = [f"UPDATE Node SET degree={int(degree.get(n, 0))} WHERE id_key={_sql_str(_id_key(n))}"
                     for n in stale[i:i + batch]]
            if stmts:
                self._run(";".join(stmts), language="sqlscript")

        gods = _god(G, top_n=64)
        self._run("DELETE FROM Meta WHERE key = 'god_nodes'")
        self._run("INSERT INTO Meta CONTENT " + json.dumps({"key": "god_nodes", "value": gods}, ensure_ascii=False))
        stats = {"deleted_sources": len(dirty), "upserted_nodes": len(changed_ids),
                 "upserted_edges": len(new_edges)}
        # sync_graph's contract is that the DB ends up equal to G, so G is the
        # yardstick here exactly as graph.json is for a full load.
        stats.update(self._reconcile(
            G.number_of_nodes(), G.number_of_edges(),
            node_ids=list(G.nodes()),
            edges=[(d.get("_src") or u, d.get("_tgt") or v) for u, v, d in G.edges(data=True)]))
        return stats

    # ---- scoring / seed resolution (exact parity via shared kernel) ---------
    def _idf(self, norm_terms: list[str]) -> dict[str, float]:
        N = (self._run("SELECT count(*) AS c FROM Node")[0]["c"]) or 1
        idf = {}
        for t in set(norm_terms):
            df = self._run("SELECT count(*) AS c FROM Node WHERE norm_label LIKE :p",
                           params={"p": f"%{t}%"})[0]["c"]
            idf[t] = _idf_value(N, df)
        return idf

    def _score(self, terms: list[str]) -> list[tuple[float, str]]:
        norm_terms = [tok for t in terms for tok in _search_tokens(t)]
        if not norm_terms:
            return []
        idf = self._idf(norm_terms)
        norm_terms, joined, joined_w = _score_terms(terms, idf)
        # Candidate superset: anything that could score > 0 (label/source/id match).
        conds, params = [], {}
        for i, t in enumerate(set(norm_terms)):
            params[f"t{i}"] = f"%{t}%"
            conds.append(f"norm_label LIKE :t{i}")
            conds.append(f"source_file LIKE :t{i}")
        if joined:
            params["j"] = f"%{joined}%"
            conds.append("id LIKE :j")
        rows = self._run(
            f"SELECT id, label, norm_label, source_file FROM Node WHERE {' OR '.join(conds)}",
            params=params,
        )
        scored = []
        for r in rows:
            nl = r.get("norm_label") or _strip_diacritics(str(r.get("label", ""))).lower()
            s = _score_record(r["id"], nl, r.get("label") or "", r.get("source_file") or "",
                              norm_terms, idf, joined, joined_w)
            if s > 0:
                scored.append((s, r["id"], r.get("label") or r["id"]))
        scored.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
        return [(s, nid) for s, nid, _ in scored]

    # ---- traversal (shared core over a DB-fetched neighborhood) -------------
    def _neighborhood(self, seeds: list[str], depth: int):
        rows = self._run(
            "SELECT id, label, source_file, source_location, community, degree FROM "
            f"(TRAVERSE out('Rel') FROM (SELECT FROM Node WHERE id_key IN :s) MAXDEPTH {depth}) WHERE @type = 'Node'",
            params={"s": [_id_key(s) for s in seeds]},
        )
        vids = {r["id"] for r in rows}
        node_attrs = {r["id"]: r for r in rows}
        # Edge-endpoint access only works via Cypher on ArcadeDB (SQL `out.id`
        # projects null); MATCH binds both endpoints and the edge props.
        erows = self._run(
            "MATCH (a:Node)-[r:Rel]->(b:Node) WHERE a.id_key IN $v AND b.id_key IN $v "
            "RETURN a.id AS s, b.id AS t, r.relation AS relation, r.confidence AS confidence, r.context AS context",
            kind="query", language="cypher", params={"v": [_id_key(v) for v in vids]},
        )
        return node_attrs, erows

    def _node_count(self) -> int:
        return int(self._run("SELECT count(*) AS c FROM Node")[0]["c"] or 0)

    def _degree_model(self, node_attrs: dict, fset: set):
        """Resolve (hub_threshold, degree_fn) for the traversal, matching
        JsonBackend: the threshold is the p99 of the WHOLE graph's degree
        distribution (filtered by context when a filter is active), and degree_fn
        returns that same per-node degree. Falls back to the fetched
        neighborhood's degrees if the global computation isn't available (e.g. a
        graph loaded without a degree index) — never worse than before."""
        try:
            if fset:
                fdeg = self._filtered_degrees(fset)
                n_total = self._node_count()
                degrees = list(fdeg.values()) + [0] * max(0, n_total - len(fdeg))
                return _hub_threshold(degrees), (lambda n: fdeg.get(n, 0))
            return self._global_hub_threshold(), (lambda n: int(node_attrs.get(n, {}).get("degree") or 0))
        except RuntimeError:
            neigh = [int(a.get("degree") or 0) for a in node_attrs.values()]
            return _hub_threshold(neigh), (lambda n: int(node_attrs.get(n, {}).get("degree") or 0))

    def _global_hub_threshold(self) -> int:
        """p99 of the stored (total) degree distribution, floored at 50 — exactly
        what _hub_threshold computes, but via ORDER BY + SKIP so only one row
        crosses the wire instead of every node's degree (needs the degree index)."""
        n = self._node_count()
        if not n:
            return 50
        rows = self._run("SELECT degree FROM Node ORDER BY degree SKIP :s LIMIT 1",
                        params={"s": int(n * 0.99)})
        return max(50, int((rows[0].get("degree") if rows else 0) or 0))

    def _filtered_degrees(self, fset) -> dict:
        """In+out degree per node counting only edges whose context is in fset
        (the degree distribution of _filter_graph_by_context's subgraph)."""
        from collections import Counter
        fdeg: Counter = Counter()
        for proj in ("a.id", "b.id"):
            rows = self._run(
                f"MATCH (a:Node)-[r:Rel]->(b:Node) WHERE r.context IN $f RETURN {proj} AS id, count(*) AS c",
                kind="query", language="cypher", params={"f": list(fset)})
            for r in rows:
                if r.get("id") is not None:
                    fdeg[r["id"]] += int(r.get("c") or 0)
        return dict(fdeg)

    def query(self, question, *, mode="bfs", depth=3, context_filters=None) -> QueryResult:
        scored = self._score(_query_terms(question))
        seeds = _pick_seeds(scored)
        if not seeds:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[])
        filters, fsource = _resolve_context_filters(question, context_filters)
        node_attrs, erows = self._neighborhood(seeds, depth)

        # Rebuild the (context-filtered) adjacency locally and run the SAME kernel.
        H = nx.DiGraph()
        for nid, a in node_attrs.items():
            H.add_node(nid, **a)
        fset = set(filters)
        for e in erows:
            if not fset or (e.get("context") in fset):
                H.add_edge(e["s"], e["t"], relation=e.get("relation", ""),
                           confidence=e.get("confidence", ""), context=e.get("context") or "")
        # Hub threshold + degree match JsonBackend: over the WHOLE graph's degree
        # distribution (filtered when a context filter is active), not just the
        # fetched neighborhood.
        threshold, deg = self._degree_model(node_attrs, fset)
        core = _dfs_core if mode == "dfs" else _bfs_core
        visited, edges = core(lambda n: list(H.successors(n)) if n in H else [], deg, seeds, depth, threshold)

        ordered = sorted(visited, key=deg, reverse=True)
        nodes_rec = [
            NodeRecord(label=node_attrs[n].get("label", n), source_file=str(node_attrs[n].get("source_file", "") or ""),
                       source_location=str(node_attrs[n].get("source_location", "") or ""),
                       community=str(node_attrs[n].get("community", "") if node_attrs[n].get("community") is not None else ""))
            for n in ordered if n in node_attrs
        ]
        edge_rec = []
        for u, v in edges:
            if u in visited and v in visited and H.has_edge(u, v):
                d = H[u][v]
                edge_rec.append(EdgeRecord(
                    source_label=node_attrs[u].get("label", u), target_label=node_attrs[v].get("label", v),
                    relation=str(d.get("relation", "")), confidence=str(d.get("confidence", "")),
                    context=str(d.get("context", "") or ""),
                ))
        seed_labels = [node_attrs.get(s, {}).get("label", s) for s in seeds]
        return QueryResult(mode=mode, depth=depth, seeds=seed_labels, total_nodes=len(visited),
                           nodes=nodes_rec, edges=edge_rec, filters=filters, filter_source=fsource)

    # ---- point lookups ------------------------------------------------------
    def get_node(self, label) -> NodeDetail | None:
        needle = label.lower()
        rows = self._run(
            "SELECT id, label, source_file, source_location, file_type, community, degree FROM Node "
            "WHERE norm_label LIKE :p OR id_key = :idk",
            params={"p": f"%{needle}%", "idk": _id_key(label)},
        )
        rows.sort(key=_by_id)
        for r in rows:
            if needle in (r.get("label") or "").lower() or needle == r["id"].lower():
                return NodeDetail(label=r.get("label", r["id"]), id=r["id"],
                                  source_file=str(r.get("source_file", "") or ""),
                                  source_location=str(r.get("source_location", "") or ""),
                                  file_type=str(r.get("file_type", "") or ""),
                                  community=str(r.get("community") if r.get("community") is not None else ""),
                                  degree=int(r.get("degree") or 0))
        return None

    def _resolve_one(self, label: str) -> dict | None:
        """Mirror _find_node's first match (exact > prefix > substring)."""
        term = " ".join(_search_tokens(label))
        if not term:
            return None
        rows = self._run("SELECT id, label, norm_label FROM Node WHERE norm_label LIKE :p OR id LIKE :p",
                         params={"p": f"%{term}%"})
        rows.sort(key=_by_id)
        tiers: list[list[dict]] = [[], [], []]
        for r in rows:
            nl = (r.get("norm_label") or _strip_diacritics(str(r.get("label", "")))).lower()
            bare, nidl = nl.rstrip("()"), r["id"].lower()
            if term in (nl, bare, nidl):
                tiers[0].append(r)
            elif nl.startswith(term) or bare.startswith(term) or nidl.startswith(term):
                tiers[1].append(r)
            elif term in nl:
                tiers[2].append(r)
        for t in tiers:
            if t:
                return t[0]
        return None

    def get_neighbors(self, label, relation_filter="") -> Neighbors | None:
        node = self._resolve_one(label.lower())
        if node is None:
            return None
        nid = node["id"]
        rf = relation_filter.lower()
        out = self._run(
            "MATCH (a:Node {id_key:$id})-[r:Rel]->(b:Node) "
            "RETURN b.id AS nb, b.label AS lbl, r.relation AS relation, r.confidence AS confidence",
            kind="query", language="cypher", params={"id": _id_key(nid)})
        inc = self._run(
            "MATCH (a:Node)-[r:Rel]->(b:Node {id_key:$id}) "
            "RETURN a.id AS nb, a.label AS lbl, r.relation AS relation, r.confidence AS confidence",
            kind="query", language="cypher", params={"id": _id_key(nid)})
        recs = []
        for r in out:
            if rf and rf not in (r.get("relation") or "").lower():
                continue
            recs.append(NeighborRecord(True, r.get("lbl", r["nb"]), str(r.get("relation", "")), str(r.get("confidence", ""))))
        for r in inc:
            if rf and rf not in (r.get("relation") or "").lower():
                continue
            recs.append(NeighborRecord(False, r.get("lbl", r["nb"]), str(r.get("relation", "")), str(r.get("confidence", ""))))
        return Neighbors(label=node.get("label", nid), neighbors=recs)

    def get_community(self, community_id) -> CommunityRecord | None:
        rows = self._run("SELECT id, label, source_file, community_name FROM Node WHERE community = :c",
                        params={"c": int(community_id)})
        if not rows:
            return None
        rows.sort(key=_by_id)
        return CommunityRecord(cid=int(community_id),
                               members=[NodeRecord(label=r.get("label", r["id"]),
                                                   source_file=str(r.get("source_file", "") or "")) for r in rows],
                               name=rows[0].get("community_name"))

    def god_nodes(self, top_n=10) -> list[GodNode]:
        rows = self._run("SELECT value FROM Meta WHERE key = 'god_nodes' LIMIT 1")
        gods = rows[0]["value"] if rows else []
        return [GodNode(label=g["label"], degree=g["degree"]) for g in gods[:top_n]]

    def graph_stats(self) -> GraphStats:
        n = self._run("SELECT count(*) AS c FROM Node")[0]["c"]
        e = self._run("SELECT count(*) AS c FROM Rel")[0]["c"]
        comm = len(self._run("SELECT community FROM Node WHERE community IS NOT NULL GROUP BY community"))
        rows = self._run("SELECT confidence, count(*) AS c FROM Rel GROUP BY confidence")
        counts = {r.get("confidence") or "EXTRACTED": r["c"] for r in rows}
        total = (e or 1)
        return GraphStats(nodes=n, edges=e, communities=comm,
                          extracted_pct=round(counts.get("EXTRACTED", 0) / total * 100),
                          inferred_pct=round(counts.get("INFERRED", 0) / total * 100),
                          ambiguous_pct=round(counts.get("AMBIGUOUS", 0) / total * 100))

    def shortest_path(self, source, target, *, max_hops=8) -> PathResult:
        src = self._score([t.lower() for t in source.split()])
        tgt = self._score([t.lower() for t in target.split()])
        if not src:
            return PathResult(found=False, reason="no-source")
        if not tgt:
            return PathResult(found=False, reason="no-target")
        s_id, t_id = src[0][1], tgt[0][1]
        if s_id == t_id:
            return PathResult(found=False, reason="same-node", start_label=s_id)
        warnings = []
        for name, sc in (("source", src), ("target", tgt)):
            if len(sc) >= 2 and sc[0][0] > 0 and (sc[0][0] - sc[1][0]) / sc[0][0] < 0.10:
                warnings.append(f"warning: {name} match was ambiguous (top score {sc[0][0]:g}, runner-up {sc[1][0]:g})")
        s_label = self._label_of(s_id)
        # Clamp the variable-length upper bound so an "unlimited" caller can't ask
        # ArcadeDB for a pathological `*..1000000000` expansion.
        bound = min(max(max_hops, 1), 64)
        rows = self._run(
            f"MATCH (a:Node {{id_key:$s}}),(b:Node {{id_key:$t}}), p=shortestPath((a)-[:Rel*..{bound}]-(b)) "
            "RETURN [n IN nodes(p) | n.id] AS ids",
            kind="query", language="cypher", params={"s": _id_key(s_id), "t": _id_key(t_id)})
        ids = rows[0]["ids"] if rows and rows[0].get("ids") else None
        if not ids:
            return PathResult(found=False, reason="no-path", start_label=s_label,
                              end_label=self._label_of(t_id), warnings=warnings)
        hops = len(ids) - 1
        if hops > max_hops:
            return PathResult(found=False, reason=f"too-long:{hops}", warnings=warnings)
        segments = []
        labels = {nid: self._label_of(nid) for nid in ids}
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i + 1]
            fwd = self._run(
                "MATCH (a:Node {id_key:$u})-[r:Rel]->(b:Node {id_key:$v}) RETURN r.relation AS relation, r.confidence AS confidence",
                kind="query", language="cypher", params={"u": _id_key(u), "v": _id_key(v)})
            if fwd:
                d, outgoing = fwd[0], True
            else:
                back = self._run(
                    "MATCH (a:Node {id_key:$v})-[r:Rel]->(b:Node {id_key:$u}) RETURN r.relation AS relation, r.confidence AS confidence",
                    kind="query", language="cypher", params={"u": _id_key(u), "v": _id_key(v)})
                d, outgoing = (back[0] if back else {}), False
            segments.append(PathSegment(outgoing=outgoing, relation=d.get("relation", ""),
                                        confidence=d.get("confidence", ""), label=labels[v]))
        return PathResult(found=True, start_label=s_label, hops=hops, segments=segments, warnings=warnings)

    def _label_of(self, nid: str) -> str:
        rows = self._run("SELECT label FROM Node WHERE id_key = :id LIMIT 1", params={"id": _id_key(nid)})
        return rows[0]["label"] if rows and rows[0].get("label") else nid

    def explain(self, label) -> ExplainResult | None:
        node = self._resolve_one(label)
        if node is None:
            return None
        nid = node["id"]
        rows = self._run("SELECT id, label, source_file, source_location, file_type, community, degree "
                        "FROM Node WHERE id_key = :id LIMIT 1", params={"id": _id_key(nid)})
        if not rows:
            return None
        r = rows[0]
        detail = NodeDetail(
            label=r.get("label", nid), id=nid,
            source_file=str(r.get("source_file", "") or ""), source_location=str(r.get("source_location", "") or ""),
            file_type=str(r.get("file_type", "") or ""),
            community=str(r.get("community") if r.get("community") is not None else ""),
            degree=int(r.get("degree") or 0),
        )
        out = self._run(
            "MATCH (a:Node {id_key:$id})-[r:Rel]->(b:Node) "
            "RETURN b.label AS lbl, b.degree AS deg, r.relation AS relation, r.confidence AS confidence",
            kind="query", language="cypher", params={"id": _id_key(nid)})
        inc = self._run(
            "MATCH (a:Node)-[r:Rel]->(b:Node {id_key:$id}) "
            "RETURN a.label AS lbl, a.degree AS deg, r.relation AS relation, r.confidence AS confidence",
            kind="query", language="cypher", params={"id": _id_key(nid)})
        conns = [Connection(True, x.get("lbl", ""), str(x.get("relation", "")), str(x.get("confidence", "")),
                            int(x.get("deg") or 0)) for x in out]
        conns += [Connection(False, x.get("lbl", ""), str(x.get("relation", "")), str(x.get("confidence", "")),
                             int(x.get("deg") or 0)) for x in inc]
        return ExplainResult(detail=detail, connections=conns)

    def pr_impact(self, files) -> tuple[list[int], int]:
        from graphify.prs import _path_match
        rows = self._run("SELECT source_file, community, count(*) AS c FROM Node "
                        "WHERE source_file IS NOT NULL GROUP BY source_file, community")
        file_comms: dict[str, set[int]] = {}
        file_count: dict[str, int] = {}
        for r in rows:
            src = r.get("source_file") or ""
            if not src:
                continue
            file_comms.setdefault(src, set())
            file_count[src] = file_count.get(src, 0) + r["c"]
            if r.get("community") is not None:
                file_comms[src].add(int(r["community"]))
        comms: set[int] = set()
        nodes = 0
        matched: set[str] = set()
        for f in files:
            for src, src_comms in file_comms.items():
                if src not in matched and _path_match(src, f):
                    comms |= src_comms
                    nodes += file_count[src]
                    matched.add(src)
        return sorted(comms), nodes


# --------------------------------------------------------------------------- #
# Backend selection (connect-only): env-driven, JSON by default
# --------------------------------------------------------------------------- #
def derive_db_name(graph_path: str | None) -> str:
    """Per-project database name from the graph path: ``proj_<project-root>``.

    Lets each repo map to its own ArcadeDB database automatically (a project
    being the parent of its ``graphify-out/`` directory), so many projects can
    share one server without manual `GRAPHIFY_ARCADE_DB` per repo.
    """
    import re
    from pathlib import Path
    if not graph_path:
        return "graphify"
    p = Path(graph_path).resolve()
    root = p.parent.parent.name if p.parent.name == "graphify-out" else p.parent.name
    safe = re.sub(r"[^A-Za-z0-9_]", "_", root).strip("_")
    return f"proj_{safe}" if safe else "graphify"


def resolve_backend_config(graph_path: str | None = None) -> dict:
    """Resolve which backend to use from the environment.

    ``GRAPHIFY_BACKEND=arcadedb`` switches to the DB; otherwise the in-process
    JSON model is used (so nothing changes for existing users). ArcadeDB is
    connect-only here — the server is assumed already running. The database
    name defaults to a per-project name derived from ``graph_path`` unless
    ``GRAPHIFY_ARCADE_DB`` is set.
    """
    import os
    kind = os.environ.get("GRAPHIFY_BACKEND", "json").strip().lower()
    if kind in ("arcadedb", "arcade"):
        url = os.environ.get("GRAPHIFY_ARCADE_URL", "http://127.0.0.1:2480")
        rest = url.split("://", 1)[-1]
        host, _, port = rest.partition(":")
        return {
            "kind": "arcadedb",
            "host": host or "127.0.0.1",
            "port": int(port or 2480),
            "database": os.environ.get("GRAPHIFY_ARCADE_DB") or derive_db_name(graph_path),
            "user": os.environ.get("GRAPHIFY_ARCADE_USER", "root"),
            "password": os.environ.get("GRAPHIFY_ARCADE_PASSWORD", ""),
        }
    return {"kind": "json", "graph_path": graph_path}


def open_backend(*, config: dict | None = None, graph=None, graph_path: str | None = None) -> GraphBackend:
    """Construct the configured backend.

    JSON: reuses a preloaded ``graph`` when given (so callers that already
    enforced the size cap don't reload), else loads from ``graph_path``.
    ArcadeDB: connects to the running server named in the config.
    """
    cfg = config or resolve_backend_config(graph_path)
    if cfg["kind"] == "arcadedb":
        return ArcadeDBBackend(cfg["database"], host=cfg["host"], port=cfg["port"],
                               user=cfg["user"], password=cfg["password"])
    if graph is None:
        from graphify.serve import _load_graph
        graph = _load_graph(cfg.get("graph_path") or graph_path)
    return JsonBackend(graph)


def open_backend_for_sync(config: dict) -> GraphBackend:
    """`open_backend` for the extract/update SYNC path only.

    Honors the opt-in ArcadeDB auto-start (GRAPHIFY_ARCADE_AUTOSTART, see
    `arcade_server.maybe_autostart`) before connecting; queries stay
    connect-only. Lives in this fork-owned file so the churn-file call sites
    (cli.py extract sync, watch.py rebuild sync) stay one-line thin.
    """
    if config.get("kind") == "arcadedb":
        from graphify import arcade_server
        arcade_server.maybe_autostart(config)
    return open_backend(config=config)
