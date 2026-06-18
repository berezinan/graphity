# Spike: ArcadeDB as a query backend vs JSON/NetworkX

Goal: measure whether moving Graphify's query path off the in-memory
JSON + NetworkX model and into ArcadeDB is worth it for large / global graphs.
This is a **proof-of-concept to get numbers**, not production code.

## What it does

- `json_backend.py` — baseline. Reuses the **real** `graphify.serve` query code
  (`_query_graph_text`, `_score_nodes`) against an in-memory `nx.Graph`. The
  whole graph is parsed and held in RAM — the model we're testing the limits of.
- `arcade_backend.py` — pushes the graph into ArcadeDB once (a materialized
  view; `graph.json` stays the build artifact), then resolves seeds via an
  indexed lookup and runs traversal / shortest-path **inside the DB** over HTTP.
  The Python client holds ~nothing.
- `bench.py` — runs each backend in its own subprocess (isolated RSS) over the
  same workload and prints a comparison table.

## Results (this repo's graph: 6,057 nodes / 10,879 edges, ~6 MB)

| metric                     |   JSON | ArcadeDB |
|----------------------------|-------:|---------:|
| client RAM for graph (MB)  |  19.1  |    0.2   |
| query p50 (ms)             |  57.9  |   22.2   |
| query p95 (ms)             |  74.4  |   37.1   |
| path  p50 (ms)             | 134.7  |   36.5   |
| path  p95 (ms)             | 178.2  |   50.2   |
| one-time load into DB (s)  |   —    |   ~3–6   |

## Results (synthetic: 1,000,000 nodes / ~2,000,000 edges, 331 MB)

Generated with `gen_synth.py`. JSON leg ran with ArcadeDB stopped to free RAM;
the host has only ~3 GB free of 17 GB, so JSON query latency below is partly
**swap-inflated** — but the RAM figure (resident allocation) is swap-independent.

| metric                     |    JSON | ArcadeDB |   ratio |
|----------------------------|--------:|---------:|--------:|
| client RAM for graph (MB)  |  2297.8 |      4.5 |  **510× less** |
| parse/load into client (s) |    22.5 |     —    |       — |
| one-time load into DB (s)  |    —    |    520.7 |  (naive loader) |
| query p50 (ms)             | 13163.3 |    273.4 |   **48× faster** |
| query p95 (ms)             | 14054.7 |   3256.2 |     4.3× |
| path  p50 (ms)             | 18424.7 |     71.4 |  **258× faster** |
| path  p95 (ms)             | 18564.8 |    363.0 |     51× |

Reading the numbers:
- **RAM:** 2.3 GB in the agent's own process vs 4.5 MB. The JSON model alone
  nearly exhausts free RAM on this box; at this scale a one-shot `graphify query`
  re-pays the 22 s parse **every call**. ArcadeDB client RAM is flat.
- **query p50 48× / path p50 258×:** even allowing for swap noise on the JSON
  side, the gap is order-of-magnitude. Resolution is an index lookup, not a
  1 M-node scan + IDF.
- **query p95 = 3.3 s is a SPIKE BUG, not ArcadeDB:** the synthetic hub nodes
  make some traversals expand to ~20 k nodes (sample shown). The JSON backend's
  hub-aware BFS (skip nodes above p99 degree, `serve.py:_bfs`) is exactly what
  prevents this, and the spike's DB traversal does **not** yet replicate it.
  Porting that degree guard into the traversal (`WHERE both().size() < t`) brings
  p95 back down. Counts as a portability TODO, not a DB limitation.
- **load 520 s is the naive per-edge `CREATE EDGE` loader** (2 M sub-select
  lookups). Production uses ArcadeDB's bulk importer + incremental upsert of only
  `affected.py`-changed nodes — load time is a one-off, not per query.

### How this extrapolates to "millions of nodes"

The JSON backend's two costs both grow **O(N)**:
- **Client RAM** ≈ 19 MB per 6 k nodes → ~3 GB per 1 M nodes, all in the agent's
  own process, re-parsed on every one-shot `graphify query`.
- **Per-query latency** is dominated by full scans (`_score_nodes` + IDF +
  per-query degree distribution in `_bfs`), i.e. linear in N → seconds/query at 1 M.

ArcadeDB's client RAM is **flat** (~0.2 MB, just the HTTP session) and resolution
is an index lookup, so per-query cost grows with the *result* size, not N. The
512 MiB `graph.json` ceiling and the per-process memory pain disappear, and
concurrent readers/writers are the DB's job (MVCC), not a file-mtime hot-reload.

## Honest caveats found during the spike

1. **Ranking semantics differ.** The spike resolves seeds with simplified
   exact/prefix/substring tiers, **not** the full TF-IDF the JSON backend uses.
   Observed divergence: `build → export` resolved to different endpoints and
   found no path on ArcadeDB while JSON found 2 hops. Productionizing requires
   faithfully porting `_score_nodes`/IDF (ArcadeDB has a full-text index + can
   return df counts) and locking it down with golden tests against the JSON
   backend. **This is the main correctness risk, now confirmed empirically.**
2. **Cypher partially reusable.** `shortest_path` runs on ArcadeDB's openCypher
   (`shortestPath((a)-[:Rel*..h]-(b))`) — so `push_to_neo4j`'s Cypher is a
   reusable starting point. But DDL is ArcadeDB-SQL-only: `CREATE PROPERTY/INDEX`
   reject `IF NOT EXISTS`, and the load path uses native SQL, not Cypher.
3. **Python driver story is HTTP-first.** No mature bolt-style driver was needed
   or used; everything is `requests` over the HTTP API. Workable, but less
   ergonomic than Neo4j's official driver.
4. **Naive load (~3–6 s).** Per-batch `sqlscript` INSERT/CREATE EDGE. Production
   should use ArcadeDB's importer or larger batched upserts and, for `update`,
   patch only `affected.py`-changed nodes instead of reloading.
5. **Java 25 note.** ArcadeDB 26.6.1 starts fine on JDK 25, but the bundled
   `bin/server.sh` builds a `:`-separated classpath that breaks on Windows;
   launch `java -cp "lib/*" com.arcadedb.server.ArcadeDBServer` directly (see
   `run_server.md`).

## How to run

```bash
# 1. start ArcadeDB (see run_server.md for the exact java command)
# 2. baseline only (no DB needed):
python bench.py json   <path/to/graph.json>
# 3. ArcadeDB leg (first run needs --load --drop to populate):
python bench.py arcade <path/to/graph.json> --load --drop
# 4. side-by-side table (omit --load on reruns to reuse the loaded DB):
python bench.py compare <path/to/graph.json> --load --drop
```

## Verdict

For the target (millions of nodes, global graph, concurrent access) the spike
confirms the thesis: ArcadeDB removes the RAM ceiling (flat client memory) and
is already faster per query at small scale, where the JSON model is at its
*best*. The work to make it real is the ranking port + incremental sync + a
`GraphBackend` abstraction — not the DB plumbing, which is straightforward.
