# Spike: immutable memory-mapped graph snapshot

OpenSpec change: `openspec/changes/graph-snapshot-spike/` (root repo).
Proof-of-concept to get numbers. Nothing in `graphify/graphify/`, `tests/`
or `pyproject.toml` is touched.

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
