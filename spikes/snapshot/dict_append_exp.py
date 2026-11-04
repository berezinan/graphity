"""Task 6.7: would appending to the string dictionary pay off? (D8)

    python dict_append_exp.py <graph.snap> [<graph.snap> ...]

A typical `update` re-extracts the files that changed. For 1 % and 5 % of
the distinct `source_file` values (seeded choice) this measures:

- how many nodes and links those files carry;
- the delta: their distinct search and value strings, upper bound "every
  string of a changed file is new" and lower bound "none is" (a re-extracted
  file whose labels did not change);
- the cost to normalise those nodes with the upstream functions (what an
  incremental writer pays instead of normalising every node);
- the cost to build the delta's pools and postings in Rust (an Encoder over
  the changed nodes only), i.e. the appended dictionary segment.

It does not implement appending and does not run `graphify update`: the
spike never rewrites a project's graph.json (proposal, Impact).
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path

import graphify_snap
import spike_patches
from snap_backend import SnapBackend
from snap_common import LINK_FIELDS, NODE_FIELDS
from snap_writer import _node_columns

SEED = 20261104


def measure(path: Path, frac: float) -> dict:
    sb = SnapBackend(path, check_port=False)
    n, e = sb.snap.node_count, sb.snap.link_count
    sf = sb._ncol["source_file"]
    files = sorted({c for c in sf if c})
    rnd = random.Random(SEED)
    picked = set(rnd.sample(files, max(1, int(len(files) * frac))))
    nodes = [i for i in range(n) if sf[i] in picked]
    nodeset = set(nodes)
    links = sum(1 for j in range(e) if sb._lsrc[j] in nodeset or sb._ltgt[j] in nodeset)

    t = time.perf_counter()
    dicts = []
    for i in nodes:
        d = {"id": sb._id(i)}
        d.update(sb._node(i))
        dicts.append(d)
    read_s = time.perf_counter() - t
    sb._ncache.clear()

    t = time.perf_counter()
    rows = [_node_columns(d) for d in dicts]
    normalise_s = time.perf_counter() - t
    distinct = set()
    for r in rows:
        distinct.update(s for s in r[3] if s)
        distinct.update(v for v in r[4] if v is not None)

    t = time.perf_counter()
    enc = graphify_snap.Encoder(NODE_FIELDS, LINK_FIELDS, sb.directed)
    cols = list(zip(*rows)) if rows else [[]] * 7
    enc.add_nodes(list(cols[0]), list(cols[1]), list(cols[2]),
                  [list(c) for c in zip(*cols[3])] if rows else [[] for _ in range(8)],
                  [list(c) for c in zip(*cols[4])] if rows else [[] for _ in NODE_FIELDS],
                  list(cols[5]), list(cols[6]))
    encode_s = time.perf_counter() - t
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "delta.snap"
        rust = dict(enc.finish(str(out), 1, bytes(32), 0, 0, bytes(32), bytes(32), "{}", "{}", "{}", [0], []))
        delta_bytes = out.stat().st_size
        dict_bytes = dict(graphify_snap.Snap(str(out), bytes(32)).section_sizes()).get("Dict", 0)
    full = dict(sb.snap.section_sizes())
    return {
        "frac": frac, "files": len(picked), "files_total": len(files),
        "nodes": len(nodes), "nodes_pct": round(len(nodes) / n * 100, 2),
        "links_touching": links, "links_pct": round(links / e * 100, 2),
        "distinct_strings_upper": len(distinct),
        "read_nodes_s": round(read_s, 2), "normalise_s": round(normalise_s, 2),
        "delta_encode_s": round(encode_s + sum(rust.values()), 2),
        "delta_dict_mb": round(dict_bytes / 2**20, 1), "delta_file_mb": round(delta_bytes / 2**20, 1),
        "full_dict_mb": round(full.get("Dict", 0) / 2**20, 1),
        "full_snapshot_mb": round(sb.snap.file_len / 2**20, 1),
    }


def main(argv) -> int:
    spike_patches.apply()
    for p in argv[1:]:
        for frac in (0.01, 0.05):
            r = measure(Path(p), frac)
            r["snapshot"] = p
            print(json.dumps(r, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
