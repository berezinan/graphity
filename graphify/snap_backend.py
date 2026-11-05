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

Name resolution: for an ordinary label the tiers of `_find_node_tiers` come
from Rust (`Snap.tiers`, same predicates), and the upstream refusal and
message functions run on a mini-graph of the winning tier only. A label with
`::` or a dotted Python symbol runs the upstream functions on a mini-graph of
candidates (as `ArcadeDBBackend` does). Traversal runs the upstream kernels
`_bfs_core` / `_dfs_core` over neighbours from Rust (the BFS frontier is a
hash-ordered set, so only the Python kernel reproduces its edge order);
`_complete_induced_edges` and the answer's edge attributes are computed in
Rust with the upstream order (decision 2026-11-04: speed-ups must not change
the output).

Hot columns are read through zero-copy views of the mapping
(`Snap.view`), so the process holds decoded values only in bounded caches.

`query --seed` resolves through `affected.resolve_seed` on a mini-graph of
candidates (D8; the `ß`/`ss` gap is pinned by an xfail(strict) test).
`god_nodes(top_n)` beyond the stored list walks the nodes by degree (D9).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import networkx as nx

try:
    import graphify_snap
except ImportError:  # snap_store.check_api() refuses with the install hint when a backend is opened
    graphify_snap = None
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
    _dfs_core,
    _find_node_tiers,
    _idf_value,
    _is_ambiguity_refusal,
    _near_miss_message,
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

from graphify.snap_store import (
    PORT_HASH,
    PortOutdated,
    check_fresh,
    norm_fingerprint,
    open_snapshot,
    port_hash,
)

# Bounded caches: cleared at the start of a request once over the limit.
_ID_CACHE = 50_000
_NODE_CACHE = 10_000
_VAL_CACHE = 50_000
_NONE = 0xFFFFFFFF
_EDGE_FIELDS = ("relation", "confidence", "context", "source_file", "source_location")
_LIGHT_FIELDS = ("label", "norm_label", "source_file", "source_location")


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


class _LazyIds(Sequence):
    """Node ids of an index list, decoded on access (`_near_miss_message`
    reads only the first few and the length)."""

    def __init__(self, be: "SnapBackend", idx):
        self._be, self._idx = be, idx

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return self._be._ids_of(self._idx[k])
        return self._be._id(self._idx[k])


class _LightNodes:
    """`G.nodes[nid]` for the upstream helpers that read only hot fields
    (`_pick_seeds`, `_pick_scored_endpoint`; see snap_store.PORT_FUNCS)."""

    def __init__(self, be):
        self._be = be

    def __getitem__(self, nid):
        return self._be._light(self._be._idx[nid])


class _Duck:
    def __init__(self, be):
        self.nodes = _LightNodes(be)


class _OneNode:
    """The slice of nx.DiGraph the `analyze` node filters read, for one node (key 0)."""

    def __init__(self, attrs: dict, degree: int):
        self.nodes = {0: attrs}
        self._degree = degree

    def degree(self, n):
        return self._degree


class _TG:
    """The (context-filtered) traversal graph of `JsonBackend._query_from_seeds`."""

    def __init__(self, be: "SnapBackend", fcodes):
        self._be, self._f = be, fcodes
        self._nbrs: dict[str, list[str]] = {}

    def is_directed(self):
        return False

    def neighbors(self, n):
        out = self._nbrs.get(n)
        if out is None:
            be = self._be
            out = be._ids_of(be.snap.tneighbors(be._idx[n], self._f))
            self._nbrs[n] = out
        return out

    def degree(self, n):
        be = self._be
        if self._f is None:
            return be._degt[be._idx[n]]
        return be.snap.tdegree(be._idx[n], self._f)

    def hub_threshold(self):
        return self._be.snap.hub_threshold(self._f)

    def pair_data(self, u, v) -> dict:
        """Merged attributes of the pair (tests; the query path uses pair_codes)."""
        be = self._be
        d: dict = {}
        for j in be.snap.pair_links(be._idx[u], be._idx[v]):
            d.update(be._link(j, markers=True))
        return d


class SnapBackend(GraphBackend):
    def __init__(self, path, *, graph_json=None, check_port: bool = True):
        """`graph_json`: the source file, compared by size and mtime_ns. A snapshot
        that is missing, stale, unfinished or incompatible is refused (D3)."""
        if check_port and port_hash() != PORT_HASH:
            raise PortOutdated(
                "the upstream code that SnapBackend repeats changed (snap_store.PORT_FUNCS), "
                "so snapshot answers are not trusted.\nUpdate graphify (the port must be re-checked).")
        self._path = Path(path)
        self._graph_json = Path(graph_json) if graph_json else None
        self._open(open_snapshot(self._path, self._graph_json))

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
        view = lambda name, fmt: memoryview(snap.view(name)).cast(fmt)
        self._ncol = {f: view(f"n.v.{f}", "I") for f in ("label", "norm_label", "source_file",
                                                         "source_location", "community")}
        self._lcol = {f: view(f"l.v.{f}", "I") for f in _EDGE_FIELDS}
        self._lsrc, self._ltgt = view("l.src", "I"), view("l.tgt", "I")
        self._ooff, self._olink = view("o.off", "Q"), view("o.link", "I")
        self._ioff, self._ilink = view("i.off", "Q"), view("i.link", "I")
        self._degt = view("n.degt", "I")
        self._coff, self._cmem = view("c.off", "Q"), view("c.mem", "I")
        self._cold_repo: bool | None = None

    def _begin(self):
        """Per request: follow a newer generation, re-check staleness, bound caches."""
        gen = graphify_snap.peek_generation(str(self._path))
        if gen is not None and gen != self.snap.generation:
            for _ in range(3):
                try:
                    self._open(graphify_snap.Snap(str(self._path), norm_fingerprint()))
                    break
                except OSError:
                    continue
        if len(self._ids) > _ID_CACHE:
            self._ids.clear()
            self._idx.clear()
        if len(self._ncache) > _NODE_CACHE:
            self._ncache.clear()
        if len(self._vcache) > _VAL_CACHE:
            self._vcache.clear()
        if self._graph_json is not None:
            check_fresh(self.snap, self._path, self._graph_json)

    # ---- records -------------------------------------------------------------
    def _id(self, i: int) -> str:
        nid = self._ids.get(i)
        if nid is None:
            nid = self.snap.node_id(i)
            self._ids[i] = nid
            self._idx[nid] = i
        return nid

    def _ids_of(self, idx) -> list[str]:
        """Ids of many nodes, decoding the missing ones in one call."""
        ids = self._ids
        missing = [i for i in idx if i not in ids]
        if missing:
            for i, nid in zip(missing, self.snap.node_ids(missing)):
                ids[i] = nid
                self._idx[nid] = i
        return [ids[i] for i in idx]

    def _val(self, kind, field, code):
        key = (kind, field, code)
        try:
            return self._vcache[key]
        except KeyError:
            v = self._vcache[key] = json.loads(self.snap.value(kind, field, code))
            return v

    def _nget(self, i: int, field: str, default=None):
        """`G.nodes[nid].get(field, default)` for a hot node field."""
        c = self._ncol[field][i]
        return self._val("n", field, c) if c else default

    def _lget(self, j: int, field: str, default=None):
        c = self._lcol[field][j]
        return self._val("l", field, c) if c else default

    def _label(self, i: int) -> str:
        """`G.nodes[nid].get("label", nid)`."""
        c = self._ncol["label"][i]
        return self._val("n", "label", c) if c else self._id(i)

    def _light(self, i: int) -> dict:
        """The hot fields the upstream message and seed helpers read."""
        return {f: self._val("n", f, c) for f in _LIGHT_FIELDS if (c := self._ncol[f][i])}

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
            d["_src"], d["_tgt"] = self._id(self._lsrc[j]), self._id(self._ltgt[j])
        return d

    def _deg_dir(self, i: int) -> int:
        return (self._ooff[i + 1] - self._ooff[i]) + (self._ioff[i + 1] - self._ioff[i])

    # ---- name resolution ---------------------------------------------------------
    @staticmethod
    def _special(label: str) -> bool:
        """Labels `_find_node_tiers` handles with whole-graph helpers first."""
        return "::" in label or bool(_PY_QUALIFIED_RE.match(label.strip()))

    def _has_cold_repo(self) -> bool:
        """Whether any node may carry `repo` (read by `_rivals_across_files`).
        Cold JSON is written by json.dumps, so a `repo` key appears as `"repo": `."""
        if self._cold_repo is None:
            self._cold_repo = self.snap.cold_contains("n", '"repo": ')
        return self._cold_repo

    def _light_graph(self, idx) -> tuple[nx.DiGraph, list[str]]:
        """Mini-graph of `idx` (graph order) for the upstream refusal helpers."""
        ids = self._ids_of(idx)
        H = nx.DiGraph()
        if self._has_cold_repo():
            H.add_nodes_from(zip(ids, (self._node(i) for i in idx)))
        else:
            H.add_nodes_from(zip(ids, (self._light(i) for i in idx)))
        return H, ids

    def _tiers(self, label: str, level: int):
        """`_find_node_tiers(G, label)` as index lists, tiers up to `level` complete."""
        term = " ".join(_search_tokens(label))
        nq = _strip_diacritics(str(label)).lower().strip()
        se, ex, pr, sub = self.snap.tiers(term, nq, level)
        if se:
            query_basename = _strip_diacritics(Path(label).name).lower()
            preferred = []
            for i in se:
                if str(self._nget(i, "source_location", "")) != "L1":
                    continue
                lbl = _strip_diacritics(str(self._nget(i, "label") or "")).lower()
                if lbl == query_basename or lbl.endswith("/" + query_basename):
                    preferred.append(i)
            if len(preferred) == 1:
                se = preferred + [i for i in se if i != preferred[0]]
        return se, ex, pr, sub

    def _tier_graph(self, label: str, level: str = "prefix") -> nx.DiGraph:
        """For `::` and dotted labels: mini-graph holding every node
        `_find_node_tiers` can place in the tiers up to `level`, plus the
        symbol half's candidates, in graph.json order."""
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
            cand.update(snap.match_nodes("nl", "bare", needles))
            cand.update(snap.match_nodes("lt", "equal", needles))
            cand.update(snap.match_ids("lower", "equal", needles))
            cand.update(snap.match_ids("norm", "equal", needles))
            if level in ("prefix", "substring"):
                for f in ("nl", "lt"):
                    cand.update(snap.match_nodes(f, "prefix", needles))
                cand.update(snap.match_ids("lower", "prefix", needles))
            if level == "substring":
                for f in ("nl", "lt"):
                    cand.update(snap.match_nodes(f, "substring", needles))
        if term:
            cand.update(snap.match_nodes("st", "equal", [term]))
        order = sorted(cand)
        H = nx.DiGraph()
        H.add_nodes_from(zip(self._ids_of(order), (self._node(i) for i in order)))
        return H

    def _resolve(self, label: str) -> tuple[str | None, Refusal | None]:
        """`serve._resolve_single_node`."""
        if self._special(label):
            H = self._tier_graph(label, "exact")
            tiers = _find_node_tiers(H, label)
            if not (tiers[0] or tiers[1]):
                H = self._tier_graph(label, "prefix")
                if not any(_find_node_tiers(H, label)[:3]):
                    H = self._tier_graph(label, "substring")
            nid, err = _resolve_single_node(H, label)
            return nid, (_refusal(err) if err else None)
        se, ex, pr, sub = self._tiers(label, 0)
        if not (se or ex):
            se, ex, pr, sub = self._tiers(label, 1)
            if not pr:
                se, ex, pr, sub = self._tiers(label, 2)
        if not (se or ex or pr):
            if not sub:
                return None, _refusal(f"No node matching '{label}' found.")
            H, _ = self._light_graph(sub[:5])
            return None, _refusal(_near_miss_message(H, label, _LazyIds(self, sub)))
        tier = se or ex or pr
        H, ids = self._light_graph(tier)
        err = _tier_refusal(H, label, ids, file_query=bool(se) and "::" not in label)
        if err:
            return None, _refusal(err)
        return ids[0], None

    def _query_name_refusal(self, label: str) -> str | None:
        """`serve._query_name_refusal`."""
        if self._special(label):
            return _query_name_refusal(self._tier_graph(label, "exact"), label)
        se, ex, _, _ = self._tiers(label, 0)
        tier = se or ex
        if not tier:
            return None
        H, ids = self._light_graph(tier)
        refusal = _tier_refusal(H, label, ids, file_query=bool(se) and "::" not in label)
        return refusal if _is_ambiguity_refusal(refusal) else None

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
            return self._query_with_seeds(question, seeds, mode=mode, depth=depth,
                                          context_filters=context_filters, seed_root=seed_root)
        refused = self._query_name_refusal(question)
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

    # ---- --seed (D8) ---------------------------------------------------------------
    def _seed_candidates(self, value: str, seed_root) -> nx.DiGraph:
        """Mini-graph on which `affected.resolve_seed` answers as on the whole graph.

        As `ArcadeDBBackend._candidate_graph_for_seed`: every node an equality step of
        `resolve_seed` can accept (id, normalised label, bare name, source path) in
        full, plus at most three label-substring matches (more than one already
        means no answer). Known gap, pinned by an xfail(strict) test: `resolve_seed`
        compares by `casefold`, the snapshot by `norm_label` (diacritics stripped,
        `lower`), so labels that differ only by `ß`/`ss` can resolve differently.
        """
        from graphify.affected import _as_repo_relative
        q = value.rstrip("/\\") or value
        norm_q = _strip_diacritics(q).lower()
        bare = norm_q[:-2] if norm_q.endswith("()") else norm_q
        paths = sorted({v.lower() for p in (q, _as_repo_relative(q, seed_root)) if p
                        for v in (p, p.replace("\\", "/"))})
        snap = self.snap
        cand: set[int] = set()
        cand.update(snap.match_ids("lower", "equal", [q.lower()]))
        cand.update(snap.match_nodes("nl", "equal", sorted({norm_q, bare, bare + "()"})))
        cand.update(snap.match_nodes("nl", "bare", [bare]))
        cand.update(snap.match_nodes("src", "equal", paths))
        cand.update(sorted(snap.match_nodes("nl", "substring", [norm_q]))[:3])
        order = sorted(cand)
        H = nx.DiGraph()
        H.add_nodes_from(zip(self._ids_of(order), (self._light(i) for i in order)))
        return H

    def _query_with_seeds(self, question, seeds, *, mode, depth, context_filters,
                          seed_root) -> QueryResult:
        """`JsonBackend.query` with `seeds`: each value goes through `affected.resolve_seed`."""
        from graphify.affected import resolve_seed as _rs
        start_nodes: list[str] = []
        unresolved: list[str] = []
        ambiguous: list[str] = []
        for val in seeds:
            nid = _rs(self._seed_candidates(val, seed_root), val, seed_root)
            if nid is None:
                ref = self._query_name_refusal(val)
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
        return self._query_from_seeds(question, start_nodes, mode=mode, depth=depth,
                                      context_filters=context_filters, unresolved=unresolved)

    def _query_from_seeds(self, question, seeds, *, mode, depth, context_filters=None,
                          unresolved=None) -> QueryResult:
        filters, fsource = _resolve_context_filters(question, context_filters)
        # `_filter_graph_by_context` keeps every pair whose merged `context` is
        # one of the filters; an empty filter list means the unfiltered graph.
        fcodes = [self._ctx[f] for f in filters if f in self._ctx] if filters else None
        TG = _TG(self, fcodes)
        core = _dfs_core if mode == "dfs" else _bfs_core
        visited, edges = core(TG.neighbors, TG.degree, seeds, depth, TG.hub_threshold())
        # serve._complete_induced_edges(TG, visited, edges), in Rust.
        idx = self._idx
        eu, ev = self.snap.complete_induced([idx[n] for n in visited], [idx[u] for u, _ in edges],
                                            [idx[v] for _, v in edges], fcodes)
        edges.extend(zip(self._ids_of(eu), self._ids_of(ev)))

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
            i = idx[n]
            node_recs.append(NodeRecord(
                label=self._nget(i, "label", n),
                source_file=str(self._nget(i, "source_file", "")),
                source_location=str(self._nget(i, "source_location", "")),
                community=str(self._nget(i, "community", "")),
            ))
        pairs = [(u, v) for u, v in edges if u in visited and v in visited]
        codes = memoryview(self.snap.pair_codes([idx[u] for u, _ in pairs], [idx[v] for _, v in pairs],
                                                list(_EDGE_FIELDS))).cast("I")
        w = len(_EDGE_FIELDS) + 2
        lv = lambda f, c, default: self._val("l", f, c) if c else default
        edge_recs = []
        for k, (u, v) in enumerate(pairs):
            b = k * w
            if codes[b + 5] == _NONE:
                raise KeyError((u, v))      # not adjacent: edge_datas(TG, u, v) raises too
            src, tgt = self._id(codes[b + 5]), self._id(codes[b + 6])
            if {src, tgt} != {u, v}:
                src, tgt = u, v
            edge_recs.append(EdgeRecord(
                source_label=self._label(idx[src]),
                target_label=self._label(idx[tgt]),
                relation=str(lv("relation", codes[b], "")),
                confidence=str(lv("confidence", codes[b + 1], "")),
                context=str(lv("context", codes[b + 2], "") or ""),
                source_file=str(lv("source_file", codes[b + 3], "") or ""),
                source_location=str(lv("source_location", codes[b + 4], "") or ""),
            ))
        return QueryResult(
            mode=mode, depth=depth,
            seeds=[self._label(idx[n]) for n in seeds],
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
            degree=self._deg_dir(i),
        )

    def _arcs(self, i: int):
        """(outgoing, neighbour index, link) in DiGraph successor / predecessor order."""
        for j in self._olink[self._ooff[i]:self._ooff[i + 1]]:
            yield True, self._ltgt[j], j
        for j in self._ilink[self._ioff[i]:self._ioff[i + 1]]:
            yield False, self._lsrc[j], j

    def get_neighbors(self, label, relation_filter="") -> Neighbors | Refusal:
        self._begin()
        nid, refused = self._resolve(label.lower())
        if refused:
            return refused
        rf = relation_filter.lower()
        recs: list[NeighborRecord] = []
        for out, nb, j in self._arcs(self._idx[nid]):
            rel = self._lget(j, "relation", "")
            if rf and rf not in rel.lower():
                continue
            recs.append(NeighborRecord(out, self._label(nb), str(rel),
                                       str(self._lget(j, "confidence", "")),
                                       str(self._lget(j, "source_file") or ""),
                                       str(self._lget(j, "source_location") or "")))
        return Neighbors(label=self._label(self._idx[nid]), neighbors=recs)

    def get_community(self, community_id) -> CommunityRecord | None:
        self._begin()
        k = self._cids.get(int(community_id))
        if k is None:
            return None
        members = list(self._cmem[self._coff[k]:self._coff[k + 1]])
        if not members:
            return None
        recs = [NodeRecord(label=self._label(i), source_file=str(self._nget(i, "source_file", "")))
                for i in members]
        name = self._node(members[0]).get("community_name")
        return CommunityRecord(cid=int(community_id), members=recs, name=name)

    def god_nodes(self, top_n=10) -> list[GodNode]:
        self._begin()
        if top_n <= self._stats["god_nodes_kept"]:
            return [GodNode(label=g["label"], degree=g["degree"])
                    for g in self._stats["god_nodes"][:top_n]]
        # Beyond the stored list (D9): `analyze.god_nodes` over the nodes by degree,
        # descending, ties in graph order (sorted(..., reverse=True) is stable).
        from graphify.analyze import (
            _BUILTIN_NOISE_LABELS, _is_concept_node, _is_file_node, _is_json_key_node)
        n = self.snap.node_count
        deg = [self._deg_dir(i) for i in range(n)]
        out: list[GodNode] = []
        for i in sorted(range(n), key=deg.__getitem__, reverse=True):
            G = _OneNode(self._light(i), deg[i])
            if _is_file_node(G, 0) or _is_concept_node(G, 0) or _is_json_key_node(G, 0):
                continue
            if G.nodes[0].get("label", "") in _BUILTIN_NOISE_LABELS:
                continue
            out.append(GodNode(label=self._label(i), degree=deg[i]))
            if len(out) >= top_n:
                break
        return out

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
        """`serve._resolve_path_endpoint`, exact tiers from Rust (or the
        candidate mini-graph for `::` / dotted labels), scoring over the
        whole snapshot."""
        if self._special(query):
            H = self._tier_graph(query, "exact")
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
        else:
            se, ex, _, _ = self._tiers(query, 0)
            hits = se or ex
            if hits:
                H, ids = self._light_graph(hits)
                refusal = _tier_refusal(H, query, ids, file_query=bool(se) and "::" not in query)
                if refusal:
                    return None, [], refusal
                return ids[0], [], None
        scored = self._score_nodes([t.lower() for t in query.split()])
        if not scored:
            return None, [], None
        return _pick_scored_endpoint(self._duck, scored, query), scored, None

    def _expand(self, undirected: bool):
        def expand(frontier, forward):
            fr = list(frontier)
            res = self.snap.path_expand([self._idx[u] for u in fr], forward, undirected, self._contains)
            return {u: self._ids_of(lst) for u, lst in zip(fr, res)}
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
                    if directed and self._lsrc[j] != ia:
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
            degree=self._deg_dir(i),
            definition_file=str(d.get("definition_file", "") or ""),
            definition_location=str(d.get("definition_location", "") or ""),
        )
        # JsonBackend's `_outgoing` trusts `_src`/`_tgt` markers; the explain
        # graph has none (the writer refuses links that carry them), so the
        # direction is the adjacency list each arc came from.
        conns = [
            Connection(out, self._label(nb),
                       str(self._lget(j, "relation", "")), str(self._lget(j, "confidence", "")),
                       self._deg_dir(nb),
                       source_file=str(self._lget(j, "source_file", "") or ""),
                       source_location=str(self._lget(j, "source_location", "") or ""))
            for out, nb, j in self._arcs(i)
        ]
        return ExplainResult(detail=detail, connections=conns)

    def pr_impact(self, files) -> tuple[list[int], int]:
        self._begin()
        from graphify.prs import _path_match
        sf, cm = self._ncol["source_file"], self._ncol["community"]
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
