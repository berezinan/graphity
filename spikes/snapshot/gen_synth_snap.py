"""Task 6.6: a synthetic snapshot at the scale of a 30 GB graph.json.

    python gen_synth_snap.py <erp2.snap> <out.snap> [copies=3]

Three copies of ERP2 form one "global graph of three configurations": the
same labels (1C object names repeat across configurations), a different
source root and id prefix per copy. ERP2 has 2.02 M nodes and 4.90 M links,
so three copies give 6.06 M nodes / 14.7 M links: ERP2's density, about
30 GB of graph.json in the current format (D11, "30 GB" = graph.json size).

Columns are read from the ERP2 snapshot and fed to the Rust encoder
directly: no `G`, no Python normalisation (the search strings of a copy are
ERP2's, with the source root prepended). The writer's own time at this scale
is therefore not measured here; README extrapolates it and marks it so.

Not representative: label distribution is ERP2's three times over, so each
ambiguous name has three times the rivals and hubs keep their size per copy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import graphify_snap
from graphify.serve import _strip_diacritics

from snap_common import SEARCH_FIELDS, norm_fingerprint

BATCH = 50_000


def _fold(nid: str) -> tuple[str | None, str | None]:
    low = nid.lower()
    fold = low if nid.isascii() else _strip_diacritics(nid).lower()
    return (low if low != nid else None), (fold if fold != low else None)


def main(argv) -> int:
    src, out = Path(argv[1]), Path(argv[2])
    copies = int(argv[3]) if len(argv) > 3 else 3
    snap = graphify_snap.Snap(str(src), norm_fingerprint())
    meta = json.loads(snap.block_text("meta.json"))
    stats = json.loads(snap.block_text("stats.json"))
    nf, lf = meta["node_fields"], meta["link_fields"]
    N, E = snap.node_count, snap.link_count
    view = lambda name, fmt: memoryview(snap.view(name)).cast(fmt)
    ncodes = {f: view(f"n.v.{f}", "I") for f in nf}
    lcodes = {f: view(f"l.v.{f}", "I") for f in lf}
    scodes = {f: view(f"n.s.{f}", "I") for f in SEARCH_FIELDS}
    llen = view("n.llen", "I")
    cn_off, cl_off = view("cn.off", "Q"), view("cl.off", "Q")
    lsrc, ltgt = view("l.src", "I"), view("l.tgt", "I")
    vcache: dict[tuple, str] = {}
    scache: dict[tuple, str] = {}

    def val(kind, f, c):
        k = (kind, f, c)
        v = vcache.get(k)
        if v is None:
            v = vcache[k] = snap.value(kind, f, c)
        return v

    def sstr(f, i):
        c = scodes[f][i]
        if not c:
            return ""
        k = (f, c)
        s = scache.get(k)
        if s is None:
            s = scache[k] = snap.search_string(f, i)
        return s

    def cold(kind, i):
        off = cn_off if kind == "n" else cl_off
        if off[i + 1] - off[i] <= 1:
            return None
        return snap.record(kind, [], i)[1] or None

    sf_cache: dict[tuple, str] = {}

    def rooted_sf(kind, c, root):
        k = (kind, c, root)
        v = sf_cache.get(k)
        if v is None:
            s = json.loads(val(kind, "source_file", c))
            v = sf_cache[k] = json.dumps(f"{root}/{s}" if isinstance(s, str) and s else s, ensure_ascii=False)
        return v

    enc = graphify_snap.Encoder(nf, lf, meta["directed"])
    times = {}
    t_all = time.perf_counter()
    for k in range(copies):
        t = time.perf_counter()
        root, pre = f"cfg{k}", f"k{k}_"
        for start in range(0, N, BATCH):
            rng = range(start, min(N, start + BATCH))
            old_ids = snap.node_ids(list(rng))
            ids, idl, idn = [], [], []
            for oid in old_ids:
                nid = pre + oid
                lo, fo = _fold(nid)
                ids.append(nid)
                idl.append(lo)
                idn.append(fo)
            search = []
            for f in SEARCH_FIELDS:
                col = [sstr(f, i) for i in rng]
                if f == "src":
                    col = [f"{root}/{s}" if s else s for s in col]
                elif f == "st":
                    col = [f"{root} {s}" if s else s for s in col]
                search.append(col)
            vals = []
            for f in nf:
                codes = ncodes[f]
                if f == "source_file":
                    vals.append([rooted_sf("n", codes[i], root) if codes[i] else None for i in rng])
                else:
                    vals.append([val("n", f, codes[i]) if codes[i] else None for i in rng])
            lab = ncodes["label"]
            lens = [llen[i] if lab[i] else llen[i] + len(pre) for i in rng]
            enc.add_nodes(ids, idl, idn, search, vals, lens, [cold("n", i) for i in rng])
        t_nodes = time.perf_counter() - t
        t = time.perf_counter()
        base = k * N
        for start in range(0, E, BATCH):
            rng = range(start, min(E, start + BATCH))
            vals = []
            for f in lf:
                codes = lcodes[f]
                if f == "source_file":
                    vals.append([rooted_sf("l", codes[j], root) if codes[j] else None for j in rng])
                else:
                    vals.append([val("l", f, codes[j]) if codes[j] else None for j in rng])
            enc.add_links([base + lsrc[j] for j in rng], [base + ltgt[j] for j in rng], vals,
                          [cold("l", j) for j in rng])
        times[f"copy{k}"] = {"nodes_s": round(t_nodes, 1), "links_s": round(time.perf_counter() - t, 1)}
        print(json.dumps({f"copy{k}": times[f"copy{k}"]}), flush=True)

    # communities: same ids in every copy -> CSR over the concatenated nodes
    cids = stats["cids"]
    pos = {c: n for n, c in enumerate(cids)}
    comm = ncodes["community"]
    cval: dict[int, int] = {}
    members: list[list[int]] = [[] for _ in cids]
    for k in range(copies):
        base = k * N
        for i in range(N):
            c = comm[i]
            if not c:
                continue
            cid = cval.get(c)
            if cid is None:
                cid = cval[c] = int(json.loads(val("n", "community", c)))
            members[pos[cid]].append(base + i)
    comm_off, comm_mem = [0], []
    for m in members:
        comm_mem.extend(m)
        comm_off.append(len(comm_mem))
    del members
    stats2 = dict(stats, nodes=N * copies, edges=E * copies,
                  confidence={c: v * copies for c, v in stats["confidence"].items()},
                  god_nodes=[{"label": g["label"], "degree": g["degree"]}
                             for g in stats["god_nodes"] for _ in range(copies)][:stats["god_nodes_kept"]])
    meta2 = dict(meta, synthetic=f"{copies} copies of {src.name}")
    t = time.perf_counter()
    rust = enc.finish(str(out), 1, norm_fingerprint(), 0, 0, bytes(32), bytes(32),
                      json.dumps(meta2, ensure_ascii=False), json.dumps(stats2, ensure_ascii=False),
                      snap.block_text("graph.json") or "{}", comm_off, comm_mem)
    times["finish_s"] = round(time.perf_counter() - t, 1)
    times["rust_phases"] = {a: round(b, 1) for a, b in rust}
    times["total_s"] = round(time.perf_counter() - t_all, 1)
    print(json.dumps({"nodes": N * copies, "links": E * copies, "bytes": out.stat().st_size,
                      "times": times}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
