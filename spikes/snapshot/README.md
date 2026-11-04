# Spike: immutable memory-mapped graph snapshot

OpenSpec change: `openspec/changes/graph-snapshot-spike/` (root repo).
Proof-of-concept to get numbers. Nothing in `graphify/graphify/`, `tests/`
or `pyproject.toml` is touched.

## Decision (task 7.1, 2026-11-04): go, with conditions

Measured against the D12 thresholds as revised by the user on 2026-11-04
(design.md D12; the 2026-11-02 values are kept there next to them).

| Threshold (D12) | psn_diadoc | Buh30 | ERP2 | Result |
|---|---|---|---|---|
| Golden parity, except the listed branches | 59 cases + 12 CLI runs byte-equal; no xfail needed | — | — | pass |
| Real-query parity vs JsonBackend | 112 / 112 | 105 / 105 | JsonBackend does not fit in memory | pass (psn, Buh30) |
| Snapshot size ≤ 20 % of graph.json | 17.8 % | 11.8 % | 10.5 % | pass |
| ERP2 initialisation: open ≤ 200 ms; `snap-query` process p50 ≤ 500 ms | | | 0.42 ms; 288 ms | pass |
| ERP2 warm `query` p50 ≤ 100 ms, p95 ≤ 5 s | | | 37 ms, 4.4 s | pass, 12 % margin on p95 |
| ERP2 private bytes ≤ 100 MB after open, ≤ 200 MB after a series | | | 42 MB, 135 MB | pass |
| ERP2 write from `G` ≤ 10 min, peak growth over `G` ≤ 50 % | | | 564 s, +25 % (from graph.json data, not `G`) | pass, 36 s margin |
| Generations: 20 rebuilds under 4 readers | 0 errors, 0 mixed answers | | | pass |

Against the 2026-11-02 thresholds three ERP2 read lines fail:
`snap-query` ≤ 200 ms (288 ms), `query` p95 ≤ 1 s (4.4 s), private bytes
≤ 100 MB (135 MB).

**Conditions of the go**, each a separate change:

1. Port the two `spike_patches.py` fixes to `serve.py` for every backend.
   One of them (no short Cyrillic query terms) changes `query` output; the
   ERP2 numbers above depend on it.
2. Before graphs past ERP2 size: a streaming writer in Rust (D5
   alternative). Extrapolated (not measured) for a 6 M-node graph: ~28 min
   and ~53 GB private bytes with the Python writer, above this machine's 48 GB.
3. The `query` p95 on hubs is upstream's own cost (`render_query` builds the
   whole answer before cutting it to budget). It falls below 1 s only if the
   rendering moves to Rust or the output changes; neither is part of the go.

**Compared with what is in use today, on ERP2:** first `query` in a new
process 1.3 s (ArcadeDB 386 s, timeout; JsonBackend does not load: ~25 GB
needed, 19.5 GB free); `path` / `explain` p50 13 ms (ArcadeDB 30 s).

**Limits of the measurements:** one machine, OS file cache warm (no elevation
to flush the standby list); ERP2 and Buh30 one run each (psn two); the
synthetic 30 GB graph does not exercise hub answers (§6.6); ArcadeDB answers
differ from JsonBackend on most items and were not investigated; the share of
snapshot writing in a real `graphify update` was not measured (§6.7).

**Open questions answered by the spike:**

- *How many generations to keep* (D7): one; replaced generations live while
  readers hold them (closed 2026-11-03).
- *What "30 GB" means*: graph.json size (closed 2026-11-03).
- *Is dictionary appending worth it* (D8): not as a first step; incremental
  normalisation gives the gain (§6.7).
- *Context filter by pass or per-context degree arrays* (D10): the pass is
  enough (ERP2 `query` with filter p50 43 ms, p95 0.66 s).

Still open: the staleness policy (D9: refuse, warn or fall back), how often
`watch` runs on large graphs, and how often agents use `context_filter`.

## Layout

| Path | What |
|---|---|
| `Cargo.toml` | Cargo workspace |
| `snap-core/` | format, encoder, memory-mapped reader |
| `graphify_snap/` | PyO3 module over `snap-core` (built with `maturin`) |
| `snap-query/` | minimal binary: start, open, one `query` to stdout |

## Environment

The spike has its own venv, `spikes/snapshot/.venv` (git-ignored). The staging
`graphify/.venv` and the production tool-env are not modified.

```powershell
cd graphify\spikes\snapshot
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe "maturin>=1.5,<2" pytest psutil -e ..\..
$env:VIRTUAL_ENV = "$PWD\.venv"
.venv\Scripts\maturin.exe develop --release -m graphify_snap\Cargo.toml
cargo build --release -p snap-query
```

Requirements: Rust stable `x86_64-pc-windows-msvc` (1.98 used) and the MSVC
C++ build tools with a Windows SDK.

Deviation from task 1.2: no tool-env on this machine has `graspologic-native`
(neither staging `.venv` nor prod `graphifyy`). The read path does not
cluster, so the spike venv is the editable staging `graphify` without it.

## Code

| File | What |
|---|---|
| `snap-core/src/header.rs`, `writer.rs`, `reader.rs` | header page, section table, commit footer, publish |
| `snap-core/src/blocks.rs`, `pool.rs` | named blocks in a section; string pools, front coding, `memmem` search |
| `snap-core/src/encode.rs` | encoder: networkx-ordered topology, postings, id permutations |
| `snap-core/src/graph.rs` | reader primitives: candidates, df, score, traversal, directed view, path step |
| `snap_writer.py` | Python writer over graph.json data; lock, `.tmp`, publish |
| `snap_backend.py` | `SnapBackend(GraphBackend)` |
| `snap_common.py` | columns, normalisation fingerprint, port guard (`PORT_HASH`) |
| `ref_loaders.py`, `query_set.py` | reference networkx graphs per CLI command; seeded query set |
| `tests/` | parity and format tests (`.venv\Scripts\python -m pytest spikes\snapshot\tests`) |

### Which networkx graph the snapshot reproduces (D13)

Each CLI command loads graph.json differently, so the snapshot keeps links in
file order and derives each view: the undirected traversal graph of `query`
(pairs in first-appearance order, attributes merged by `dict.update`, filtered
neighbours in `G.edges()` rank order), the directed view of `explain` /
`get_neighbors` / MCP, and the search step of `path`. Name resolution runs the
upstream `_find_node_tiers` & co. on a mini-graph of candidates; traversal runs
the upstream `_bfs_core` / `_dfs_core` over neighbours from Rust, because the
BFS frontier is a hash-ordered `set` and only the Python kernel reproduces the
order of EDGE lines.

Not supported (refused by the writer with a reason, or by `SnapBackend`):
`multigraph: true` files, parallel links of one direction, links with stored
`_src`/`_tgt`, NUL inside strings, `query --seed`, `god_nodes(top_n > 100)`.

`SnapBackend` refuses to start when the upstream functions it repeats have
changed (`snap_common.PORT_FUNCS`, sha256 in `PORT_HASH`).

## Results

### Parity (2026-11-03)

| Suite | Cases | Result |
|---|---:|---|
| `test_golden.py`: cases of `tests/test_query_backend.py` + path/explain/pr_impact | 59 | byte-equal |
| `test_cli_parity.py`: real CLI, stdout + stderr + exit code | 12 | byte-equal |
| `test_psn.py`: psn_diadoc round trip, networkx order, scores, stats | 11 | equal |
| `test_psn.py`: psn_diadoc seeded queries (`query` BFS/DFS, 5 context filters, 30 `path` x2, 40 `explain`) | 4 | byte-equal |
| `test_shapes.py`, `test_generations.py` | 18 | pass |

### Generations under live readers (tasks 5.1 / 5.2, psn_diadoc)

`python stand_generations.py <graph.json> <work> 20 4`: 4 reader processes
query continuously; the writer publishes 20 generations alternating graph A
(psn_diadoc) and graph B (same, every label suffixed); then a writer is killed
after writing its `.tmp`.

| | |
|---|---:|
| answers | 50 912 (A 42 523, B 8 389) |
| errors / answers matching neither A nor B | 0 / 0 |
| generations seen by each reader | 21 of 21 |
| killed writer | published generation untouched; next run removed its `.tmp` |
| directory after | `graph.snap`, `graph.snap.lock` |
| free disk space vs. start | +4.4 MB (snapshot 26.4 MB) |

All processes run with one `PYTHONHASHSEED`. With different seeds the same
stand reported 1 223 "mismatches": BFS answers order their EDGE lines by a
hash-ordered set (upstream `serve._bfs_core`), so two processes can print the
same answer in different orders. JsonBackend has the same property.

### 6.1 / 6.4 psn_diadoc: snapshot vs JsonBackend vs ArcadeDB (2026-11-03)

`python bench.py run <graph.json> <graph.snap> <out> [--passes N]`. One
subprocess per backend and command, `PYTHONHASHSEED=0` in every child.
JsonBackend loads graph.json the way that CLI command does (`ref_loaders.py`).
"Cold process" is spawn to first answer, OS file cache warm (the shell is not
elevated, so the standby list could not be flushed: no cold-cache numbers).
Warm figures are over 4 more passes of the same set (`--passes 5`); ArcadeDB
one pass.

| | snapshot | JsonBackend | ArcadeDB |
|---|---:|---:|---:|
| cold process, first `query` | 1.2–1.5 s | 14.5–16.8 s | 24.4–26.7 s |
| of which open (imports + backend init) | 0.6 s | 3.1–4.1 s | 0.6–0.7 s |
| `query` warm p50 / p95 | 15 / 520 ms | 30 / 2 100 ms | 600 / 23 000 ms |
| `query` with `context_filter` p50 / p95 | 16 / 80 ms | 600–670 / 1 900 ms | 2 080 / 9 500 ms |
| `path` p50 / p95 | 3 / 55 ms | 1.1 / 38 ms | 560 / 3 800 ms |
| `explain` p50 / p95 | 2.5 / 9 ms | 0.6 / 14 ms | 580 / 960 ms |
| private bytes at end | 35–62 MB | 320–470 MB | 39–78 MB |
| answers equal to JsonBackend | 112 / 112 | — | 49 / 112 |

- Warm `path` and `explain` are faster on JsonBackend once its graph is in
  memory: each snapshot record crosses PyO3 and its values are JSON-decoded.
- ArcadeDB answers differ from JsonBackend on 28 of 47 `query`, 19 of 30
  `path` and 16 of 30 `explain`; not investigated in this spike.
- Reproducibility (two runs, `--passes 5`): identical answers; p50 spread
  3 % (snapshot) and 7 % (JsonBackend) for `query`, 1 % / 11 % with a context
  filter, but 22 % for snapshot `path` (3.7 vs 2.9 ms) and 19 % for
  JsonBackend `explain` (0.65 vs 0.52 ms). Below ~4 ms the 10 % criterion of
  task 6.1 is not met.

### 6.4 Buh30: snapshot vs JsonBackend vs ArcadeDB (2026-11-03)

885 006 nodes, 1 716 200 links, graph.json 4 094 MB. One run, `--budget 300`
s per backend and command, one pass. JsonBackend answered only the items
that fit the budget (4 of 40 `query`, 6 of 30 `path`), so its percentiles
rest on few samples.

| | snapshot | JsonBackend | ArcadeDB |
|---|---:|---:|---:|
| snapshot size | 505 MB (11.8 % of graph.json) | — | — |
| write: json.load / normalise / encode + Rust / total | 161 s / 191 s / 43 s / 247 s (+161) | — | — |
| writer peak working set | 12.3 GB (all of it `json.load`) | — | — |
| open | 0.6–0.7 s | 72–119 s | 0.6–0.8 s |
| cold process, first answer: `query` / `path` / `explain` | 2.8 / 2.0 / 7.1 s | 402 / 337 / 336 s | timeout / 27 / 18 s |
| `query` p50 / p95 / max | 44 ms / 4.1 s / 13.1 s | 5.1 s / 9.7 s (n=3) | first query > 300 s, HTTP read timeout |
| `query` with `context_filter` p50 / p95 | 137 / 660 ms | 205 / 570 ms | 10.3 / 13.0 s |
| `path` p50 / p95 | 106 / 900 ms | 425 ms / 55 s (n=5) | 13.2 / 27.4 s |
| `explain` p50 / p95 | 18 ms / 2.2 s | 138 ms / 2.5 s | 13.3 / 17.0 s |
| private bytes at end | 45–382 MB | 5.6–9.0 GB | 43–68 MB |
| answers equal to JsonBackend (items both answered) | 45 / 45 | — | 7 / 34 |

- JsonBackend's first answer costs ~4.5 minutes beyond the load: the
  trigram index (`serve._get_trigram_index`) is built on first use.
- Snapshot `query` private bytes reach 382 MB: the Python-side caches of
  `SnapBackend` (decoded nodes, ids). The Rust side holds no copy.
- Snapshot `query` p95 4.1 s and max 13.1 s: broad labels on Buh30 score
  hundreds of thousands of candidates and traverse large neighbourhoods in
  the Python kernel. The ERP2 threshold (p95 ≤ 1 s, D12) is at risk.

### 6.5 ERP2 (2026-11-03)

2 019 923 nodes, 4 903 605 links, graph.json 11 047 MB. Machine at the start:
19.5 GB of 48 GB available (ArcadeDB service, other sessions running).

**Write.** `python snap_writer.py <graph.json> <out>`, graph.json loaded by
`graph_stream.load` (streams the `nodes` / `links` elements; `json.load`
peaked at 3x the file on Buh30, ~33 GB here).

| | | D12 |
|---|---:|---|
| snapshot | 1 165 MB (10.5 % of graph.json) | ≤ 20 % — pass |
| load graph.json (streaming) | 263 s | not in the threshold |
| write from loaded data: normalise (Python) / stats / encode + Rust | 391 s / 107 s / 66 s = **564 s** | ≤ 10 min — pass, 36 s to spare |
| private bytes: after load / peak while writing | 14.1 GB / 17.6 GB = +25 % | ≤ +50 % — pass |
| least available system memory | 4.9 GB | — |

Deviation from task 6.5: the writer reads graph.json data in file order
(D13), not the upstream `G`; `G` on ERP2 (~32 GB) was not built.

**Read.** `bench.py --backends snap,arcade --passes 3`, after the scoring
fix below. JsonBackend not run: on Buh30 it took up to 9 GB for a 4 GB file,
~25 GB here against 19.5 GB available (refused for memory, as task 6.5
allows).

| | snapshot | ArcadeDB | D12 (snapshot) |
|---|---:|---:|---|
| cold process, first `query` (`SnapBackend`, Python) | 5.1 s | 386 s (HTTP read timeout at 300 s) | — |
| cold process, `snap-query` (Rust) p50 / p95 | 387 ms / 4.2 s | — | ≤ 200 ms — **fail** |
| `query` warm p50 / p95 / max | 135 ms / 20.6 s / 23.9 s | — | p50 ≤ 100 ms, p95 ≤ 1 s — **fail** |
| `query` with `context_filter` p50 / p95 | 422 ms / 1.7 s | 39 s / 157 s | — |
| `path` p50 / p95 | 39 ms / 5.3 s | 31 s / 41 s | — |
| `explain` p50 / p95 | 32 ms / 6.3 s | 30 s / 47 s | — |
| private bytes, `query` process: after open / end | 42 MB / 612 MB | 39–56 MB | ≤ 100 MB — **fail** at end |
| open | 0.6–0.7 s (Python imports), 0.5 ms in Rust | 0.7–1.0 s | — |

Where the slow `query` answers spend their time (profile on the query set):

| query kind | example | time | where |
|---|---|---:|---|
| exact name of tens of thousands of nodes | `Массив` (20 266), `ПриСозданииНаСервере()` (21 528) | 3–27 s | upstream `_find_node_tiers` + `_query_name_refusal` on the candidate mini-graph: 4–9 s + 3–14 s, to print "Ambiguous" with every rival |
| god-node seed | `ПустаяСтрока`: 80 045 nodes, 184 186 edges | 14 s | building the answer's records in Python; `_complete_induced_edges` 3.9 s |
| short Cyrillic words match every node | "…Товары в пути и …" (2 017 507 candidates) | 4.4 s | Rust scoring: 1.8 µs per candidate |

Scoring fix during the run: `Graph::score` resolved block names and decoded
front-coded ids inside the per-node loop. Hoisting the lookups and testing
`nid.lower() == term` with bitsets from the sorted id permutation took the
worst `snap-query` from 19.1 s to 3.7 s and its median from 169 to 118 ms;
parity tests unchanged (109 pass). The remaining cost is scoring every
candidate; only a different algorithm (e.g. pruning by an upper bound on the
score) removes it.

Upstream reproduces these costs on the full graph: `_is_searchable` keeps
one- and two-letter Cyrillic words (it drops only short Latin ones), the
ambiguity message lists every rival, and a hub seed expands in full. The
snapshot removes the load (402 s on Buh30 for JsonBackend), not the per-query
work of the upstream functions it calls.

### 6.5b Re-run after the 2026-11-04 fixes

Fixes, none of which changes the output (parity re-checked, 121 tests):

- `_find_node_tiers` tiers computed in Rust with the upstream predicates;
  the upstream refusal / message helpers run on the winning tier only.
- `_complete_induced_edges` and the merged edge attributes of the answer in
  Rust, in upstream order.
- Hot columns read through zero-copy buffers over the mapping; bounded caches.
- Spike-only, applied to JsonBackend and SnapBackend alike
  (`spike_patches.py`, decided 2026-11-04): one- and two-letter Cyrillic
  words are not query terms; `serve._ambiguity_message`'s quadratic
  `sources.count` replaced by a `Counter` (byte-identical message, tested).

| | ERP2 before | ERP2 after | Buh30 after | D12 (ERP2) |
|---|---:|---:|---:|---|
| cold process, first `query` (`SnapBackend`) | 5.1 s | 1.3 s | 1.8 s | — |
| `snap-query` cold process p50 / p95 | 387 ms / 4.2 s | 288 / 516 ms | — | ≤ 200 ms — fail |
| `query` warm p50 / p95 / max | 135 ms / 20.6 s / 23.9 s | 37 ms / 4.4 s / 8.6 s | 15 ms / 1.4 s | p50 pass, p95 fail |
| `query` with `context_filter` p50 / p95 | 422 ms / 1.7 s | 43 / 659 ms | 17 / 66 ms | — |
| `path` p50 / p95 | 39 ms / 5.3 s | 13 / 138 ms | 15 / 201 ms | — |
| `explain` p50 / p95 | 32 ms / 6.3 s | 13 / 146 ms | 10 / 81 ms | — |
| private bytes, `query` process after the set | 612 MB | 135 MB | 87 MB | ≤ 100 MB — fail |

Buh30, same items on JsonBackend (`--budget 600`): 105 / 105 answers equal.
JsonBackend there: open 64–109 s, first answer 4–6 min, `query` p50 218 ms,
private bytes 5.6–10.0 GB.

`snap-query` time is the query, not the start: open 0.42 ms, the rest is
`df` + scoring of short-term candidate sets (the ≤ 200 ms threshold was
meant as initialisation cost).

What still exceeds 1 s on ERP2, all upstream behaviour reproduced exactly:

| query | time | where |
|---|---:|---|
| `ПустаяСтрока` (hub: 80 045 nodes, 184 186 edges in the answer) | 6–8 s | records for 80 k nodes + `render_query` (1.5 M `sanitize_label` calls) before cutting to the 2 000-token budget |
| `ОткрытьФорму`, `ПоказатьВопрос` (hubs, 37–46 k nodes) | 3–4 s | same |
| `Массив`, `ПриСозданииНаСервере()` (20–21 k nodes share the name) | ~1 s | Ambiguous message of 4.3 MB / 40 k lines, built in Python |

### 6.6 Synthetic graph at the scale of a 30 GB graph.json (2026-11-04)

`python gen_synth_snap.py <erp2.snap> <out.snap> 3`: three copies of ERP2
(different id prefix and source root per copy, same labels), 6 059 769
nodes and 14 710 815 links: ERP2's density, about 30 GB of graph.json.
Columns are fed to the Rust encoder straight from the ERP2 snapshot, with no
`G` and no Python normalisation.

| | synthetic (3 x ERP2) | ERP2 |
|---|---:|---:|
| snapshot | 3 240 MB | 1 165 MB |
| encoder: feeding columns / Rust topology + index + write + flush | 456 s / 34 s | — / 10 s |
| open | 0.45 ms | 0.42 ms |
| cold process, first `query` (`SnapBackend`) | 1.6 s | 1.3 s |
| `snap-query` cold process p50 / p95 | 429 / 913 ms | 288 / 516 ms |
| `query` warm p50 / p95 / max | 22 ms / 0.77 s / 2.2 s | 37 ms / 4.4 s / 8.6 s |
| `path` / `explain` p95 | 416 / 277 ms | 138 / 146 ms |
| private bytes after open / after the set | 46 / 99 MB | 42 / 135 MB |

**The `query` tail is not comparable.** With three copies every label occurs
three times, so a hub label (`ПустаяСтрока`, `ОткрытьФорму`, `ПоказатьВопрос`)
becomes an "Ambiguous" refusal instead of an 80 k-node answer: 40 of 47
`query` items are refusals on the synthetic graph, 37 on ERP2, and no
synthetic answer exceeds 10 k nodes (ERP2: 3). The synthetic graph checks how
the reader scales in open time, scoring and memory; it does not check the
p95 of hub answers.

**Writer, extrapolated (not measured):** on ERP2 the writer spent 391 s on
Python normalisation, 107 s on stats and 66 s in encode + Rust for 2.02 M
nodes. Linear in nodes and links, a real 6 M-node graph would take about
3 x 564 s = 28 min from loaded data, plus streaming the 30 GB graph.json
(~13 min at ERP2's rate). Private bytes would be about 3 x 17.6 GB = 53 GB,
above the 48 GB of this machine: the streaming Rust writer (D5 alternative)
is required at that scale.

### 6.7 Would appending to the dictionary pay off? (D8)

`python dict_append_exp.py <graph.snap> ...`: for 1 % and 5 % of the
distinct `source_file` values (seeded), the nodes and links they carry, the
cost to normalise just those nodes with the upstream functions, and the cost
to build their dictionary segment (pools + postings) in Rust. `graphify
update` itself was not run: the spike does not rewrite projects' graph.json.

| | Buh30 1 % | Buh30 5 % | ERP2 1 % | ERP2 5 % |
|---|---:|---:|---:|---:|
| files changed | 454 | 2 274 | 1 026 | 5 130 |
| nodes / links touched | 8 631 / 16 877 | 44 762 / 88 266 | 25 191 / 77 770 | 114 286 / 320 993 |
| normalise the changed nodes | 0.9 s | 4.8 s | 2.7 s | 12.9 s |
| build their dictionary segment (Rust) | 0.1 s | 0.7 s | 0.4 s | 1.8 s |
| segment size | 2.6 MB | 12.6 MB | 7.6 MB | 32.2 MB |
| full rewrite from loaded data | 247 s | 247 s | 564 s | 564 s |

The full rewrite is 95–98 % Python normalisation and stats over every node,
and the changed files touch 1–6 % of the nodes. An incremental writer that
normalises only the changed nodes and appends their segment would take
seconds instead of minutes. But the gain comes from *incremental
normalisation*, not from appending as such: rebuilding the pools from cached
normalised columns would also take seconds (Rust phases for all of ERP2 are
10 s). Appending only removes the remaining rewrite of a 1.2 GB file and
adds a compaction policy plus the watermark logic of D8.

Conclusion: dictionary appending does not pay off as a first step. The first
step is a writer that keeps the normalised columns of the previous
generation and re-normalises only changed files; appending is reconsidered
if the rewrite of the file itself shows up in `update` times.

Not measured: the share of snapshot writing in a real `graphify update` on
Buh30 / ERP2 (it needs `update` on those projects).

### 6.2 `snap-query` (Rust, no interpreter), psn_diadoc

`python snap_query_check.py <graph.snap> <qset.json> 5`.

| | |
|---|---:|
| tokenisation vs `_query_terms` + `_search_tokens` | 0 mismatches on 137 queries |
| cold process (spawn, open, score, exit), 50 runs | p50 152 ms, p95 269 ms |
| of which open | 0.46 ms |
| in-process open + df + score | p50 116 ms, p95 227 ms |

`snap-query` answers with the combined ranking only (no seed picking, no
traversal, no rendering); it bounds the start-up cost of a native reader.

### 6.3 Are the ArcadeDB databases comparable? (2026-11-03)

`python arcade_check.py <graph.json> ...`: graph.json counted by streaming
its lines, the database by `count(*)` over HTTP. Read-only.

| project | graph.json | nodes / links in graph.json | database | nodes / edges in database | comparable |
|---|---:|---|---|---|---|
| psn_diadoc | 155 MB | 40 988 / 96 315 | `proj_psn_diadoc_4_67_3_pravka_f7b6d35e` | 40 988 / 96 315 | yes |
| Buh30 | 4 094 MB | 885 006 / 1 716 200 | `proj_buh30_c2bb55ef` | 885 006 / 1 716 200 | yes |
| ERP2 | 11 047 MB | 2 019 923 / 4 903 605 | `proj_erp2_e368a057` | 2 019 923 / 4 903 605 | yes |

Equal counts do not prove equal content. The ERP2 database was not reloaded
from the 6.18 GB file the design table mentions; its counts match the
current 11 GB graph.json.

### Size and write time, psn_diadoc (40 988 nodes, 96 315 links, graph.json 155.5 MB)

| | |
|---|---:|
| snapshot | 27.7 MB (17.8 % of graph.json) |
| before front-coding ids | 32.6 MB (21 %) |
| write: json.load / normalise (Python) / encode / stats / Rust | 2.1 s / 6.1 s / 0.5 s / 0.6 s / 0.14 s |


### 1.3 Windows: what a memmap2-mapped file allows (2026-11-03)

Probe: `cargo run --release -p snap-core --example winfs_probe -- table <scratch-dir>`.
The reader opens the file with `std::fs::File::open` (share mode R|W|D, the
Rust default) and maps it with `memmap2` 0.9. Another party then tries one
operation. "Reader view" is whether the live mapping still shows the original
bytes. Both variants were run: with the `File` kept open next to the `Mmap`,
and with the `File` dropped right after mapping. The results are identical.

| operation | result | reader view |
|---|---|---|
| rename to a new name | ok | intact |
| delete | ok | intact |
| replace over with `std::fs::rename` | **ok** | intact |
| replace over with `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` | **error 5** (access denied) | intact |
| `os.replace` from CPython over a file mapped in another process (`probe_os_replace.py`) | **error 5** | intact |
| overwrite in place (write at offset 0) | ok | **changed** |
| truncate below mapped size | error 1224 (`ERROR_USER_MAPPED_FILE`) | intact |
| append at end | ok | intact |
| delete, then create a file under the same name | ok, ok | intact |

Conclusions:

- `memmap2` does not need a hand-rolled `CreateFileW`: the share mode of the
  opening handle carries over, and `std` already opens with R|W|D.
- Replacing over a mapped file works only through `std::fs::rename`. With
  `MoveFileExW`, which is what CPython's `os.replace` calls, it always fails.
  The difference matches `std` using a rename with POSIX semantics on NTFS;
  the probe shows the behaviour, not the mechanism.
- The 2026-11-02 fact in `design.md` ("cannot replace over while a reader
  holds a handle") was true for the call it tested (`os.replace`), not for
  replacement as such.
- Writing inside the mapped range is visible to the reader at once. Nothing
  protects a published file from an in-place writer.

### D7 follow-up: replace over one name under live readers (2026-11-03)

`winfs_probe race <dir> <mode> <iters> <readers> <size_kib>`: a writer thread
writes `graph.snap.tmp`, flushes it (`sync_all`), and replaces `graph.snap`
with it. Reader threads loop: open `graph.snap`, map, check that the bytes
are one complete generation, and check that the generation number never goes
down for that reader.

| mode | replaces | readers | file size | writer errors | reads ok | open / map errors | torn reads | went back |
|---|---:|---:|---:|---|---:|---|---:|---:|
| `std::fs::rename` | 1 000 | 4 | 1 MiB | none | 8 071 | none | 0 | 0 |
| `std::fs::rename` | 100 | 4 | 64 MiB | none | 635 | none | 0 | 0 |
| `MoveFileExW` | 200 | 4 | 1 MiB | error 5 on every attempt (50 retries each), 0 published | 35 571 | none | 0 | 0 |

`winfs_probe diskfree <dir> <mode> 512`: map a 512 MiB generation, replace
it with a 4 KiB one, measure free disk space before and after unmapping.

| mode | replace | old mapping | space freed while mapped | space freed after unmap |
|---|---|---|---:|---:|
| `std::fs::rename` | ok | intact | 0 MiB | **+512 MiB** |
| `MoveFileExW` | error 5 | intact | 0 MiB | 0 MiB |

Conclusions:

- A single published name, replaced over with `std::fs::rename`, gives
  readers a consistent generation on every open, with no open errors in the
  replacement window. A numbered-generation directory is not needed.
- Publication must be done by Rust. The Python writer calls the encoder's
  publish function, not `os.replace`.
- A replaced generation keeps its disk space until its last mapping closes.
  A long-lived reader must notice the new generation and drop the old
  mapping, or it pins the old file on disk.
- Not checked: file systems without POSIX rename semantics (FAT, network
  shares).

### 1.4 Append-only handle (`FILE_APPEND_DATA` without `FILE_WRITE_DATA`)

`OpenOptions::new().append(true)` in Rust `std` requests exactly this access.

| attempt | result | effect on file |
|---|---|---|
| `WriteFile` with explicit offset 0 | ok | 4 B appended at end, prefix intact |
| `WriteFile` with explicit offset `LEN/2` | ok | 4 B appended at end, prefix intact |
| `SetEndOfFile` to 16 B | error 5 (access denied) | unchanged |

An append-only handle cannot change bytes below the end of file. Explicit
offsets are ignored and the write goes to EOF. Truncation is refused. This
is the guarantee D8 needs for an appended dictionary.
