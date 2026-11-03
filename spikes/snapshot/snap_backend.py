"""SnapBackend: GraphBackend over a memory-mapped snapshot (tasks 4.1-4.3).

Each method repeats the body of the matching `JsonBackend` method, with the
graph reads replaced by snapshot primitives. Which networkx graph each method
emulates (D13):

- `query`: the CLI `query` loader (`directed` from the file, `_src`/`_tgt`
  markers on every link), traversed through `serve._traversal_view`.
- `explain`, `get_node`, `get_neighbors`: a DiGraph without markers (CLI
  `explain`, MCP `_read_graph`).
- `shortest_path`: the CLI `path` loader (MultiDiGraph / MultiGraph, markers).
- `graph_stats`, `god_nodes`, `get_community`: `_read_graph` (MCP).

Name resolution runs the upstream functions on a mini-graph of candidates in
graph.json order (as `ArcadeDBBackend` does). Traversal runs the upstream
kernels `_bfs_core` / `_dfs_core` over neighbours from Rust (decision
2026-11-03: the BFS frontier is a hash-ordered set, so only the Python kernel
reproduces its edge order).

Not reproduced natively, refused or marked xfail(strict): `query --seed`
(`affected.resolve_seed`), `god_nodes(top_n)` beyond the stored list, and the
graph shapes `snap_writer.check_supported` rejects.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import networkx as nx

import graphify_snap
from graphify.query_backend import (
    CommunityRecord,
    Connection,
    EdgeRecord,
    ExplainResult,
    GodNode,
    GraphBackend,
    GraphStats,
    NeighborRecord,
    Neighbors,
    NodeDetail,
    NodeRecord,
    PathResult,
    PathSegment,
    QueryResult,
    Refusal,
    _lexmin_shortest_path,
    _path_segment,
    _refusal,
)
from graphify.serve import (
    _PY_QUALIFIED_RE,
    _QueryScores,
    _RELATIONAL_INTENT_TERMS,
    _bfs_core,
    _complete_induced_edges,
    _dfs_core,
    _find_node_tiers,
    _idf_value,
    _no_scoped_match_message,
    _pick_scored_endpoint,
    _pick_seeds,
    _query_name_refusal,
    _query_terms,
    _resolve_context_filters,
    _resolve_single_node,
    _score_terms,
    _search_tokens,
    _strip_diacritics,
    _tier_refusal,
)

from snap_common import PORT_HASH, PortOutdated, norm_fingerprint, port_hash

UNSUPPORTED = (
    "query(seeds=...): --seed resolves through affected.resolve_seed, not ported",
    "god_nodes(top_n) beyond the stored list",
)
_CACHE_LIMIT = 200_000


class _Ranked(Sequence):
    """`(score, node_id)` ranking from Rust, decoded on access."""

    def __init__(self, be: "SnapBackend", idx: bytes, score: bytes):
        self._be = be
        self._i = memoryview(idx).cast("I")
        self._s = memoryview(score).cast("d")

    def __len__(self):
        return len(self._i)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[j] for j in range(*k.indices(len(self)))]
        return (self._s[k], self._be._id(self._i[k]))


class _LazyNodes:
    def __init__(self, be):
        self._be = be

    def __getitem__(self, nid):
        return self._be._node(self._be._idx[nid])


class _Duck:
    """What `_pick_seeds` / `_pick_scored_endpoint` read from G: `G.nodes[nid]`."""

    def __init__(self, be):
        self.nodes = _LazyNodes(be)


class _TG:
    """The (context-filtered) traversal graph of `JsonBackend._query_from_seeds`."""

    def __init__(self, be: "SnapBackend", fcodes):
        self._be, self._f = be, fcodes
        self._nbrs: dict[str, list[str]] = {}
        self.nodes = _LazyNodes(be)

    def is_directed(self):
        return False

    def neighbors(self, n):
        out = self._nbrs.get(n)
        if out is None:
            be = self._be
            out = [be._id(j) for j in be.snap.tneighbors(be._idx[n], self._f)]
            self._nbrs[n] = out
        return out

    def degree(self, n):
        return self._be.snap.tdegree(self._be._idx[n], self._f)

    def hub_threshold(self):
        return self._be.snap.hub_threshold(self._f)

    def edges(self, nbunch):
        # nx.Graph.edges(nbunch): each pair once, at its first endpoint in nbunch order.
        seen = set()
        for n in nbunch:
            for nbr in self.neighbors(n):
                if nbr not in seen:
                    yield (n, nbr)
            seen.add(n)

    def pair_data(self, u, v) -> dict:
        be = self._be
        d: dict = {}
        for j in be.snap.pair_links(be._idx[u], be._idx[v]):
            d.update(be._link(j, markers=True))
        return d


class SnapBackend(GraphBackend):
    def __init__(self, path, *, graph_json=None, partition=None, check_port: bool = True):
        """`graph_json`: the source file, compared by size and mtime_ns.
        `partition`: callable returning the current partition hash (hex) when
        communities can change without graph.json being rewritten (D9); the
        package integration decides where that hash lives."""
        if check_port and port_hash() != PORT_HASH:
            raise PortOutdated(
                "upstream changed code SnapBackend repeats (snap_common.PORT_FUNCS); "
                "re-check the port and update PORT_HASH")
        self._path = Path(path)
        self._graph_json = Path(graph_json) if graph_json else None
        self._partition = partition
        self.stale_warning: str | None = None
        self._open(graphify_snap.Snap(str(self._path), norm_fingerprint()))

    # ---- generations and staleness (D7, D9) --------------------------------
    def _open(self, snap):
        self.snap = snap
        meta = json.loads(snap.block_text("meta.json"))
        self._stats = json.loads(snap.block_text("stats.json"))
        self._nf, self._lf = meta["node_fields"], meta["link_fields"]
        self.directed = meta["directed"]
        self._ids: dict[int, str] = {}
        self._idx: dict[str, int] = {}
        self._ncache: dict[int, dict] = {}
        self._vcache: dict[tuple, object] = {}
        self._ctx = {}
        for c in range(1, snap.pool_size("l", "context") + 1):
            v = json.loads(snap.value("l", "context", c))
            if isinstance(v, str):
                self._ctx[v] = c
        self._contains = 0
        for c in range(1, snap.pool_size("l", "relation") + 1):
            if json.loads(snap.value("l", "relation", c)) == "contains":
                self._contains = c
        self._cids = {c: k for k, c in enumerate(self._stats["cids"])}
        self._duck = _Duck(self)

    def _begin(self):
        """Per request: follow a newer generation, re-check staleness."""
        gen = graphify_snap.peek_generation(str(self._path))
        if gen is not None and gen != self.snap.generation:
            for _ in range(3):
                try:
                    self._open(graphify_snap.Snap(str(self._path), norm_fingerprint()))
                    break
                except OSError:
                    continue
        if len(self._ncache) > _CACHE_LIMIT:
            self._ncache.clear()
        self.stale_warning = None
        if self._graph_json is not None:
            try:
                st = os.stat(self._graph_json)
            except OSError:
                st = None
            size, mtime_ns, _sha = self.snap.source
            if st is not None and (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
                self.stale_warning = (
                    f"warning: graph snapshot {self._path.name} is stale: "
                    f"{self._graph_json.name} changed after it was built; rebuild the snapshot")
                return
        if self._partition is not None and self._partition() != self.snap.partition_hash:
            self.stale_warning = (
                f"warning: graph snapshot {self._path.name} is stale: "
                "communities changed after it was built; rebuild the snapshot")

    # ---- records -------------------------------------------------------------
    def _id(self, i: int) -> str:
        nid = self._ids.get(i)
        if nid is None:
            nid = self.snap.node_id(i)
            self._ids[i] = nid
            self._idx[nid] = i
        return nid

    def _val(self, kind, field, code):
        key = (kind, field, code)
        try:
            return self._vcache[key]
        except KeyError:
            v = self._vcache[key] = json.loads(self.snap.value(kind, field, code))
            return v

    def _node(self, i: int) -> dict:
        d = self._ncache.get(i)
        if d is None:
            codes, cold = self.snap.record("n", self._nf, i)
            d = {f: self._val("n", f, c) for f, c in zip(self._nf, codes) if c}
            if cold:
                d.update(json.loads(cold))
            self._ncache[i] = d
        return d

    def _link(self, j: int, *, markers: bool) -> dict:
        codes, cold = self.snap.record("l", self._lf, j)
        d = {f: self._val("l", f, c) for f, c in zip(self._lf, codes) if c}
        if cold:
            d.update(json.loads(cold))
        if markers:
            s, t = self.snap.link_endpoints(j)
            d["_src"], d["_tgt"] = self._id(s), self._id(t)
        return d

    def _label(self, i: int) -> str:
        return self._node(i).get("label", self._id(i))

    # ---- name resolution (mini-graph of candidates) ----------------------------
    def _tier_graph(self, label: str, *, substring: bool = False) -> nx.DiGraph:
        term = " ".join(_search_tokens(label))
        nq = _strip_diacritics(str(label)).lower().strip()
        needles = {term, nq}
        symbols = []
        if "::" in label:
            symbols.append(label.partition("::")[2].strip())
        m = _PY_QUALIFIED_RE.match(label.strip())
        if m:
            symbols.append(m.group(2))
        for s in symbols:
            needles |= {" ".join(_search_tokens(s)), _strip_diacritics(s).lower().strip()}
        needles = sorted(n for n in needles if n)
        cand: set[int] = set()
        snap = self.snap
        if needles:
            for f in ("nl", "lt"):
                cand.update(snap.match_nodes(f, "prefix", needles))
                if substring:
                    cand.update(snap.match_nodes(f, "substring", needles))
            cand.update(snap.match_ids("lower", "prefix", needles))
            cand.update(snap.match_ids("norm", "equal", needles))
        if term:
            cand.update(snap.match_nodes("st", "equal", [term]))
        H = nx.DiGraph()
        H.add_nodes_from((self._id(i), self._node(i)) for i in sorted(cand))
        return H

    def _resolve(self, label: str) -> tuple[str | None, Refusal | None]:
        H = self._tier_graph(label)
        if not any(_find_node_tiers(H, label)[:3]):
            H = self._tier_graph(label, substring=True)
        nid, err = _resolve_single_node(H, label)
        return nid, (_refusal(err) if err else None)

    # ---- scoring (mirror of serve._score_query) ----------------------------------
    def _score_query(self, terms, *, collect_per_term_seeds: bool) -> _QueryScores:
        norm_terms = list(dict.fromkeys(tok for t in terms for tok in _search_tokens(t)))
        N = self.snap.node_count or 1
        dfs = self.snap.df(norm_terms) if norm_terms else []
        idf = {t: _idf_value(N, d) for t, d in zip(norm_terms, dfs)}
        norm_terms, joined, joined_w = _score_terms(terms, idf)
        n_terms = len(norm_terms)
        cov = [(m / n_terms) ** 2 for m in range(n_terms + 1)] if n_terms else [0.0]
        idx, sc, best = self.snap.score(norm_terms, [idf.get(t, 1.0) for t in norm_terms],
                                        joined, joined_w, cov, collect_per_term_seeds)
        ranked = _Ranked(self, idx, sc)
        best_seed_by_term: dict[str, str] = {}
        term_order: tuple[str, ...] = ()
        if collect_per_term_seeds:
            by_term = {}
            for t, b in zip(norm_terms, best):
                if b is not None:
                    s, i = b
                    nid = self._id(i)
                    by_term[t] = ((-s, -self.snap.deg_q(i), self.snap.label_len(i), nid), nid)
            if by_term:
                best_seed_by_term = {t: nid for t, (_key, nid) in by_term.items()}
                term_order = tuple(t for t, _ in sorted(by_term.items(), key=lambda kv: (kv[1][0], kv[0])))
        return _QueryScores(ranked=ranked, best_seed_by_term=best_seed_by_term, term_order=term_order)

    def _score_nodes(self, terms):
        return self._score_query(terms, collect_per_term_seeds=False).ranked

    # ---- GraphBackend -------------------------------------------------------------
    def query(self, question, *, mode="bfs", depth=3, context_filters=None,
              seeds=None, seed_root=None) -> QueryResult:
        self._begin()
        if seeds:
            raise NotImplementedError(UNSUPPORTED[0])
        refused = _query_name_refusal(self._tier_graph(question), question)
        if refused:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[],
                               refusal=_refusal(refused))
        terms = _query_terms(question)
        qs = self._score_query(terms, collect_per_term_seeds=True)
        best_seed_by_term = qs.best_seed_by_term
        intent = {t for t in best_seed_by_term if t in _RELATIONAL_INTENT_TERMS}
        if intent and any(t not in _RELATIONAL_INTENT_TERMS for t in terms):
            best_seed_by_term = {t: nid for t, nid in best_seed_by_term.items() if t not in intent}
        seeds = _pick_seeds(qs.ranked, G=self._duck, best_seed_by_term=best_seed_by_term,
                            term_order=qs.term_order)
        if not seeds:
            return QueryResult(mode=mode, depth=depth, seeds=[], total_nodes=0, nodes=[], edges=[])
        return self._query_from_seeds(question, seeds, mode=mode, depth=depth,
                                      context_filters=context_filters)

    def _query_from_seeds(self, question, seeds, *, mode, depth, context_filters=None,
                          unresolved=None) -> QueryResult:
        filters, fsource = _resolve_context_filters(question, context_filters)
        # `_filter_graph_by_context` keeps every pair whose merged `context` is
        # one of the filters; an empty filter list means the unfiltered graph.
        fcodes = [self._ctx[f] for f in filters if f in self._ctx] if filters else None
        TG = _TG(self, fcodes)
        core = _dfs_core if mode == "dfs" else _bfs_core
        visited, edges = core(TG.neighbors, TG.degree, seeds, depth, TG.hub_threshold())
        _complete_induced_edges(TG, visited, edges)

        seed_set = set(seeds)
        seed_hits = [n for n in seeds if n in visited]
        dist: dict[str, int] = {n: 0 for n in seed_hits}
        frontier, hop = seed_hits, 0
        while frontier:
            hop += 1
            nxt = []
            for n in frontier:
                for nb in TG.neighbors(n):
                    if nb in visited and nb not in dist:
                        dist[nb] = hop
                        nxt.append(nb)
            frontier = nxt
        ordered = seed_hits + sorted(
            visited - seed_set,
            key=lambda n: (dist.get(n, 1 << 30), -TG.degree(n), str(n)),
        )
        node_recs = []
        for n in ordered:
            d = TG.nodes[n]
            node_recs.append(NodeRecord(
                label=d.get("label", n),
                source_file=str(d.get("source_file", "")),
                source_location=str(d.get("source_location", "")),
                community=str(d.get("community", "")),
            ))
        edge_recs = []
        for u, v in edges:
            if u in visited and v in visited:
                for d in [TG.pair_data(u, v)]:
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
        return QueryResult(
            mode=mode, depth=depth,
            seeds=[self._duck.nodes[n].get("label", n) for n in seeds],
            total_nodes=len(visited), nodes=node_recs, edges=edge_recs,
            filters=filters, filter_source=fsource,
            graph_nodes=self.snap.node_count,
            given=bool(unresolved is not None),
            unresolved=list(unresolved or []),
        )

    def get_node(self, label) -> NodeDetail | Refusal:
        self._begin()
        nid, refused = self._resolve(label.lower())
        if refused:
            return refused
        i = self._idx[nid]
        d = self._node(i)
        return NodeDetail(
            label=d.get("label", nid), id=nid,
            source_file=str(d.get("source_file", "")),
            source_location=str(d.get("source_location", "")),
            file_type=str(d.get("file_type", "")),
            community=str(d.get("community", "")),
            degree=self.snap.deg_dir(i),
        )

    def _arcs(self, i: int):
        """(outgoing, neighbour index, link) in DiGraph successor / predecessor order."""
        for j in self.snap.out_links(i):
            yield True, self.snap.link_endpoints(j)[1], j
        for j in self.snap.in_links(i):
            yield False, self.snap.link_endpoints(j)[0], j

    def get_neighbors(self, label, relation_filter="") -> Neighbors | Refusal:
        self._begin()
        nid, refused = self._resolve(label.lower())
        if refused:
            return refused
        rf = relation_filter.lower()
        recs: list[NeighborRecord] = []
        for out, nb, j in self._arcs(self._idx[nid]):
            d = self._link(j, markers=False)
            rel = d.get("relation", "")
            if rf and rf not in rel.lower():
                continue
            recs.append(NeighborRecord(out, self._label(nb), str(rel),
                                       str(d.get("confidence", "")),
                                       str(d.get("source_file") or ""),
                                       str(d.get("source_location") or "")))
        return Neighbors(label=self._label(self._idx[nid]), neighbors=recs)

    def get_community(self, community_id) -> CommunityRecord | None:
        self._begin()
        k = self._cids.get(int(community_id))
        if k is None:
            return None
        off = memoryview(self.snap.block_bytes("c.off")).cast("Q")
        mem = memoryview(self.snap.block_bytes("c.mem")).cast("I")
        members = list(mem[off[k]:off[k + 1]])
        if not members:
            return None
        recs = [NodeRecord(label=self._label(i), source_file=str(self._node(i).get("source_file", "")))
                for i in members]
        name = self._node(members[0]).get("community_name")
        return CommunityRecord(cid=int(community_id), members=recs, name=name)

    def god_nodes(self, top_n=10) -> list[GodNode]:
        self._begin()
        gods = self._stats["god_nodes"]
        if top_n > self._stats["god_nodes_kept"]:
            raise NotImplementedError(UNSUPPORTED[1])
        return [GodNode(label=g["label"], degree=g["degree"]) for g in gods[:top_n]]

    def graph_stats(self) -> GraphStats:
        self._begin()
        s = self._stats
        total = s["edges"] or 1
        conf = s["confidence"]
        commit = s.get("built_at_commit")
        commit = commit.strip() if isinstance(commit, str) and commit.strip() else ""
        return GraphStats(
            nodes=s["nodes"], edges=s["edges"], communities=len(s["cids"]),
            extracted_pct=round(conf.get("EXTRACTED", 0) / total * 100),
            inferred_pct=round(conf.get("INFERRED", 0) / total * 100),
            ambiguous_pct=round(conf.get("AMBIGUOUS", 0) / total * 100),
            built_at_commit=commit,
        )

    def _path_endpoint(self, query: str):
        """`serve._resolve_path_endpoint`, tiers on the candidate mini-graph,
        scoring over the whole snapshot."""
        H = self._tier_graph(query)
        source_exact, exact, _, _ = _find_node_tiers(H, query)
        hits = source_exact or exact
        if hits:
            refusal = _tier_refusal(H, query, hits, file_query=bool(source_exact) and "::" not in query)
            if refusal:
                return None, [], refusal
            return hits[0], [], None
        path_part, sep, symbol_part = query.partition("::")
        if sep and path_part.strip() and symbol_part.strip():
            return None, [], _no_scoped_match_message(H, path_part.strip(), symbol_part.strip())
        scored = self._score_nodes([t.lower() for t in query.split()])
        if not scored:
            return None, [], None
        return _pick_scored_endpoint(self._duck, scored, query), scored, None

    def _expand(self, undirected: bool):
        def expand(frontier, forward):
            fr = list(frontier)
            res = self.snap.path_expand([self._idx[u] for u in fr], forward, undirected, self._contains)
            return {u: [self._id(x) for x in lst] for u, lst in zip(fr, res)}
        return expand

    def shortest_path(self, source, target, *, max_hops=8, undirected=False) -> PathResult:
        self._begin()
        s_id, src_scored, s_err = self._path_endpoint(source)
        if s_err:
            return PathResult(found=False, reason="refused", refusal=_refusal(s_err))
        if s_id is None:
            return PathResult(found=False, reason="no-source")
        t_id, tgt_scored, t_err = self._path_endpoint(target)
        if t_err:
            return PathResult(found=False, reason="refused", refusal=_refusal(t_err))
        if t_id is None:
            return PathResult(found=False, reason="no-target")
        if s_id == t_id:
            return PathResult(found=False, reason="same-node", start_label=s_id)
        warnings: list[str] = []
        for name, scored, nid in (("source", src_scored, s_id), ("target", tgt_scored, t_id)):
            if len(scored) >= 2 and nid == scored[0][1]:
                top, runner = scored[0][0], scored[1][0]
                if top > 0 and (top - runner) / top < 0.10:
                    warnings.append(f"warning: {name} match was ambiguous (top score {top:g}, runner-up {runner:g})")
        directed = not undirected
        path_nodes = _lexmin_shortest_path(self._expand(not directed), s_id, t_id)
        if path_nodes is None:
            blocked_by_direction = directed and _lexmin_shortest_path(
                self._expand(True), s_id, t_id) is not None
            return PathResult(found=False,
                              reason=("no-path-directed" if blocked_by_direction
                                      else "no-path-directed-none" if directed else "no-path"),
                              warnings=warnings,
                              start_label=self._label(self._idx[s_id]),
                              end_label=self._label(self._idx[t_id]))
        hops = len(path_nodes) - 1
        if hops > max_hops:
            return PathResult(found=False, reason=f"too-long:{hops}", warnings=warnings)
        segments: list[PathSegment] = []
        for u, v in zip(path_nodes, path_nodes[1:]):
            iu, iv = self._idx[u], self._idx[v]
            between = self.snap.links_between(iu, iv)
            edges = []
            for a, b in ((u, v), (v, u)):
                ia = self._idx[a]
                # MultiDiGraph: arcs a->b; MultiGraph: every link of the pair, both times.
                for j in between:
                    if directed and self.snap.link_endpoints(j)[0] != ia:
                        continue
                    d = self._link(j, markers=True)
                    src = d.get("_src", a)
                    edges.append((src if src in (u, v) else a, d.get("relation"), d.get("confidence")))
            segments.append(_path_segment(u, v, edges, self._label(iv)))
        return PathResult(found=True, start_label=self._label(self._idx[s_id]),
                          hops=hops, segments=segments, warnings=warnings)

    def explain(self, label) -> ExplainResult | Refusal:
        self._begin()
        nid, refused = self._resolve(label)
        if refused:
            return refused
        i = self._idx[nid]
        d = self._node(i)
        detail = NodeDetail(
            label=d.get("label", nid), id=nid,
            source_file=str(d.get("source_file", "")), source_location=str(d.get("source_location", "")),
            file_type=str(d.get("file_type", "")), community=str(d.get("community", "")),
            degree=self.snap.deg_dir(i),
            definition_file=str(d.get("definition_file", "") or ""),
            definition_location=str(d.get("definition_location", "") or ""),
        )
        conns: list[Connection] = []

        def _outgoing(e: dict, other: str, default: bool) -> bool:
            src, tgt = e.get("_src"), e.get("_tgt")
            if src is None or tgt is None or {src, tgt} != {nid, other}:
                return default
            return src == nid

        for out, nb, j in self._arcs(i):
            e = self._link(j, markers=False)
            other = self._id(nb)
            conns.append(Connection(_outgoing(e, other, out), self._label(nb),
                                    str(e.get("relation", "")), str(e.get("confidence", "")),
                                    self.snap.deg_dir(nb),
                                    source_file=str(e.get("source_file", "") or ""),
                                    source_location=str(e.get("source_location", "") or "")))
        return ExplainResult(detail=detail, connections=conns)

    def pr_impact(self, files) -> tuple[list[int], int]:
        self._begin()
        from graphify.prs import _path_match
        sf = memoryview(self.snap.block_bytes("n.v.source_file")).cast("I")
        cm = memoryview(self.snap.block_bytes("n.v.community")).cast("I")
        file_comms: dict[str, set[int]] = {}
        file_count: dict[str, int] = {}
        for a, b in zip(sf, cm):
            src = (self._val("n", "source_file", a) if a else None) or ""
            if not src:
                continue
            if src not in file_comms:
                file_comms[src] = set()
                file_count[src] = 0
            c = self._val("n", "community", b) if b else None
            if c is not None:
                file_comms[src].add(int(c))
            file_count[src] += 1
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
