# GraphBackend + ArcadeDB backend — change summary

Pluggable read-query backend for the knowledge graph. The build/export pipeline
and `graph.json` are unchanged — this only affects how queries are *served*, so
the same graph can be answered from the in-process NetworkX model (default) or a
server graph database (ArcadeDB) for large / many-project / concurrent use.

Status: Phases 1–8 landed, test-guarded. Focused suite: **249 passed, 1 skipped**.

---

## Why

`graphify query|path|explain` and the MCP server loaded the whole `graph.json`
into NetworkX and scanned all nodes per call. That caps scale: a 512 MiB file
cap, O(N) scans, whole-file rewrite on `update`, a monolithic global graph, and
no concurrent access. A spike (`spikes/arcade/`) measured ArcadeDB at **510×
less client RAM and ~48× faster queries on 1M nodes**. This change makes the
backend swappable so the DB can serve queries without the callers caring.

---

## New files

| File | What |
|------|------|
| `graphify/query_backend.py` | `GraphBackend` (ABC), `JsonBackend`, `ArcadeDBBackend`, materialized result dataclasses, `render_*` formatters, `resolve_backend_config()` + `open_backend()` factory. |
| `tests/test_query_backend.py` | JsonBackend parity (incl. 40-case byte-for-byte `render_query` vs legacy `_query_graph_text`) + factory selection tests. |
| `tests/test_arcade_backend.py` | Golden parity of ArcadeDBBackend vs JsonBackend on a fixture loaded into a live ArcadeDB (query, get_node/neighbors/community, stats, shortest_path incl. no-path, explain, god_nodes, pr_impact). Skipped when no server is reachable. |
| `tests/test_path_explain_cli.py` | Characterization + parity for the `path`/`explain` CLI (pins exact stdout/stderr/exit, kept identical after routing through the backend). |
| `graphify/arcade_server.py` | ArcadeDB lifecycle helper: download (pinned, cached), start (detached), stop, status; version-aware Java pick. |
| `tests/test_arcade_server.py` | Unit tests for the pure helpers (paths, download URL, java argv, `pick_java` version logic). |

## Modified files

**`graphify/serve.py`**
- Extracted pure kernels so both backends compute identically: `_idf_value`,
  `_score_record`, `_score_terms` (scoring) and `_hub_threshold`, `_bfs_core`,
  `_dfs_core` (traversal). Public `_score_nodes` / `_bfs` / `_dfs` are now thin
  wrappers over the kernels (signatures unchanged — existing imports/tests intact).
- `_build_server`: selects the backend via `resolve_backend_config()`. JSON
  (default) loads the graph and keeps `G` for the G-dependent resources +
  hot-reload, exactly as before. ArcadeDB connects and leaves `G=None`; the
  `surprises` / `audit` / `questions` resources and hot-reload degrade gracefully.
- The 7 tool handlers + PR-impact now go through `backend` + `render_*` (single
  place for formatting and the F-010 `sanitize_label` pass). Removed the
  now-unused `edge_data` import.

**`graphify/__main__.py`**
- CLI `query`, `path`, and `explain` all select the backend via the factory:
  the JSON branch keeps the existing checks (not-found, size cap) and is
  behaviour-identical (stdout/stderr/exit pinned by characterization tests);
  the ArcadeDB branch connects. `path` is routed through `backend.shortest_path`
  and `explain` through the new `backend.explain` + `render_explain`. Removed the
  now-unused `_query_graph_text` / `sanitize_label` / `_score_nodes` / `_find_node`
  imports in those blocks.

**`pyproject.toml`** — added the `arcadedb = ["requests"]` optional extra (also in
`all`). `ArcadeDBBackend.__init__` raises a clear `pip install "graphifyy[arcadedb]"`
hint when `requests` is missing.

**Per-project databases + sync on update (Phase 5).**
- `derive_db_name()` maps each repo to `proj_<project-root>` automatically; the
  factory uses it unless `GRAPHIFY_ARCADE_DB` is set, so many projects share one
  server without per-repo config.
- `ArcadeDBBackend.sync_graph(G, changed_sources, pruned_sources)` incrementally
  reconciles the DB to an updated graph: deletes nodes from changed/pruned
  sources (cascading edges via `DELETE VERTEX FROM Node`), re-inserts the changed
  sources' nodes + incident edges (direction from `_src`/`_tgt`), refreshes stored
  degree on affected neighbours, and recomputes `god_nodes`. `is_populated()`
  picks full-load vs incremental.
- A guarded hook in `watch._rebuild_code` (the `graphify update` fast path) and in
  the full-build path (`__main__` after `to_json`) syncs the DB when
  `GRAPHIFY_BACKEND=arcadedb`: first/full builds load fresh, watch-hook
  incremental rebuilds (`changed_paths` set) call `sync_graph`. No-op for the
  JSON default; wrapped in try/except so a DB hiccup never fails a rebuild.

**Global database sync (Phase 6).** `global_graph._sync_global_db()` reloads a
separate `global` database (`GRAPHIFY_ARCADE_GLOBAL_DB`, default
`graphify_global`) from `global-graph.json` after every `global_add` /
`global_remove`. The global graph is a single dedup'd merge that those functions
rewrite whole, so a full reload matches their semantics and sidesteps the
external-node dedup that makes a per-repo incremental sync unsafe. Guarded +
never-raises, like the per-project hooks. This completes the two-tier model:
per-project `proj_<repo>` databases for local queries, one `global` database for
cross-project questions.

---

## Design notes

- **Materialized records cross the backend boundary** (dataclasses, no live
  `nx.Graph`), so a backend may hold the whole graph (JsonBackend) or fetch only
  what a query touches (ArcadeDBBackend). All text rendering lives in `render_*`.
- **Parity is structural, not coincidental.** ArcadeDBBackend resolves seeds via
  the same `_score_record` kernel (df/N fetched from the DB) and traverses via
  the same `_bfs_core`/`_dfs_core` over a DB-fetched neighborhood carrying the
  global degree the hub guard needs. `god_nodes` is precomputed at load (a global
  analytic that only changes on rebuild) and stored in a `Meta` vertex.
- **Edge-endpoint reads use Cypher `MATCH`** — ArcadeDB SQL `out.id` projects null.

---

## Using the ArcadeDB backend (connect-only)

The server is assumed already running (lifecycle management is a later phase).

```bash
export GRAPHIFY_BACKEND=arcadedb
export GRAPHIFY_ARCADE_URL=http://127.0.0.1:2480   # default
export GRAPHIFY_ARCADE_DB=proj_myrepo              # database name (default: graphify)
export GRAPHIFY_ARCADE_USER=root                   # default
export GRAPHIFY_ARCADE_PASSWORD=...                # required

graphify query "build graph"        # routed to ArcadeDB
```

Unset `GRAPHIFY_BACKEND` (or leave it `json`) for the in-process default — nothing
changes for existing users. Loading a `graph.json` into a database is done with
`ArcadeDBBackend.load_from_graph_json()` (stores `norm_label`, `degree`, and the
precomputed `god_nodes`). Verified live end-to-end against a 1M-node database.

The server itself is turnkey via the lifecycle helper (downloads a pinned
ArcadeDB once, launches it with the right JVM flags, picks a Java >= 21):

```bash
graphify arcade start     # download if needed + launch + wait for ready
graphify arcade status
graphify arcade stop
```

---

**Exact hub threshold (Phase 7).** ArcadeDBBackend.query now computes the hub
threshold + per-node degree over the WHOLE graph, matching JsonBackend: no-filter
uses `ORDER BY degree SKIP p99 LIMIT 1` (one row over the wire; a `NOTUNIQUE`
index on `degree` keeps it off ArcadeDB's in-heap sort cap), context-filtered
uses a grouped Cypher aggregation of the filtered degree distribution. Both fall
back to the fetched neighborhood's degrees on any failure (never worse than the
prior approximation). Locked by `test_query_context_filter_parity`.

## Limitations / remaining work

- `get_neighbors` / `get_community` / `explain` connection ordering is DB-defined
  (tests compare sets / degree-ranked prefix); text order may differ from
  JsonBackend on real graphs with degree ties.
- Explicit `graphify update` re-extracts the whole corpus (`changed_paths=None`),
  so it does a full DB reload; the incremental `sync_graph` path fires for
  file-watch hook rebuilds (which pass `changed_paths`). The `global` database is
  full-reloaded on `global_add` (per-repo incremental global sync is a follow-up
  blocked on the external-node dedup model).
- Per-repo incremental global sync (vs full reload) is blocked on the
  external-node dedup model.
- The lifecycle helper pins one ArcadeDB version and tracks a single local server
  via a PID file (it honours the `GRAPHIFY_ARCADE_URL` port, but is not a
  multi-instance manager).

---

## Tests

```bash
pytest tests/test_query_backend.py tests/test_serve.py tests/test_serve_http.py \
       tests/test_query_cli.py tests/test_build.py tests/test_analyze.py -q
# ArcadeDB parity (needs a running server; otherwise skipped):
pytest tests/test_arcade_backend.py -q
```
