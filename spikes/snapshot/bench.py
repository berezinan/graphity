"""Task 6.1: snapshot vs JsonBackend vs ArcadeDB, one subprocess per backend
and command (D11).

    python bench.py run <graph.json> <graph.snap> <out_dir> [--backends snap,json,arcade]
                        [--kinds query,context,path,explain] [--budget 300] [--tag name]

Every child runs with PYTHONHASHSEED=0, so BFS answers are comparable across
processes. A child opens its backend the way the CLI does for that command
(JsonBackend: json.load + the command's loader, see ref_loaders.py), prints
OPENED, answers the first item, prints FIRST, then answers the rest of the
set until the time budget runs out. The parent measures wall time from spawn
to OPENED and to FIRST (the cold process); the child measures each answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = ("query", "context", "path", "explain")


# --------------------------------------------------------------------------- #
# child
# --------------------------------------------------------------------------- #
def _mem() -> dict:
    import psutil
    p = psutil.Process()
    m = p.memory_info()
    full = p.memory_full_info()
    return {"rss_mb": m.rss / 2**20, "private_mb": m.private / 2**20, "uss_mb": full.uss / 2**20,
            "peak_wset_mb": m.peak_wset / 2**20}


def _open(backend: str, kind: str, graph_json: str, snap: str):
    if backend == "snap":
        from snap_backend import SnapBackend
        return SnapBackend(snap, graph_json=graph_json), {}
    if backend == "json":
        import ref_loaders
        from graphify.query_backend import JsonBackend
        t = time.perf_counter()
        with open(graph_json, encoding="utf-8") as f:
            raw = json.load(f)
        t_json = time.perf_counter() - t
        loader = {"query": ref_loaders.query_graph, "context": ref_loaders.query_graph,
                  "path": ref_loaders.path_graph, "explain": ref_loaders.explain_graph}[kind]
        G = loader(raw)
        del raw
        return JsonBackend(G), {"json_load_s": t_json, "graph_build_s": time.perf_counter() - t - t_json}
    if backend == "arcade":
        os.environ["GRAPHIFY_GRAPH_STORE"] = "arcadedb"
        from graphify.query_backend import open_backend, resolve_backend_config
        return open_backend(config=resolve_backend_config(graph_json)), {}
    raise ValueError(backend)


def _answer(b, kind: str, item) -> str:
    from graphify.query_backend import render_explain, render_path, render_query
    if kind == "query":
        return render_query(b.query(item, depth=2), token_budget=2000)
    if kind == "context":
        q, f = item
        return render_query(b.query(q, depth=2, context_filters=f), token_budget=2000)
    if kind == "path":
        a, c = item
        return render_path(b.shortest_path(a, c, max_hops=10**9), a, c)
    if kind == "explain":
        return render_explain(b.explain(item), item)
    raise ValueError(kind)


def child(a) -> int:
    sys.path.insert(0, str(HERE))
    items = json.loads(Path(a.qset).read_text(encoding="utf-8"))[a.kind]
    mem0 = _mem()
    t = time.perf_counter()
    b, open_detail = _open(a.backend, a.kind, a.graph_json, a.snap)
    open_s = time.perf_counter() - t
    mem_open = _mem()
    print("OPENED", flush=True)
    times, hashes, errors = [], [], []
    t_budget = time.perf_counter()
    done = False
    for pas in range(a.passes):
        for k, item in enumerate(items):
            t = time.perf_counter()
            try:
                out = _answer(b, a.kind, item)
            except Exception as exc:  # noqa: BLE001 -- every error is reported
                out = f"<error {type(exc).__name__}: {exc}>"
                errors.append(out[:300])
            times.append(time.perf_counter() - t)
            h = hashlib.sha1(out.encode("utf-8")).hexdigest()
            if pas == 0:
                hashes.append(h)
            elif h != hashes[k]:
                errors.append(f"pass {pas} item {k}: answer changed")
            if pas == 0 and k == 0:
                print("FIRST", flush=True)
            if time.perf_counter() - t_budget > a.budget:
                done = True
                break
        if done:
            break
    warm = times[1:]
    res = {
        "backend": a.backend, "kind": a.kind, "n": len(times), "n_items": len(items), "passes": a.passes,
        "open_s": open_s, **open_detail,
        "first_ms": times[0] * 1000 if times else None,
        "p50_ms": statistics.median(warm) * 1000 if warm else None,
        "p95_ms": (sorted(warm)[max(0, int(round(0.95 * len(warm))) - 1)] * 1000) if warm else None,
        "max_ms": max(warm) * 1000 if warm else None,
        "mem_start": mem0, "mem_open": mem_open, "mem_end": _mem(),
        "hashes": hashes, "errors": errors,
    }
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def _qset(graph_json: Path, snap: Path, out_dir: Path) -> Path:
    path = out_dir / "qset.json"
    if path.exists():
        return path
    sys.path.insert(0, str(HERE))
    import query_set
    from snap_backend import SnapBackend
    sb = SnapBackend(snap)
    gods = [g.label for g in sb.god_nodes(top_n=10)]
    qs = query_set.build(query_set.labels_of_snapshot(sb), graph_json, gods)
    path.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _spawn(backend, kind, graph_json, snap, qset, out, budget, passes=1) -> dict:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    env.pop("GRAPHIFY_GRAPH_STORE", None)
    # No -I: it implies -E, which ignores PYTHONHASHSEED and PYTHONIOENCODING.
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", backend, kind,
           str(graph_json), str(snap), str(qset), str(out), "--budget", str(budget),
           "--passes", str(passes)]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True,
                         encoding="utf-8")
    marks = {}
    for line in p.stdout:
        line = line.strip()
        if line in ("OPENED", "FIRST"):
            marks[line] = (time.perf_counter() - t0) * 1000
    err = p.stderr.read()
    p.wait()
    if p.returncode != 0:
        return {"backend": backend, "kind": kind, "failed": p.returncode, "stderr": err[-2000:]}
    res = json.loads(Path(out).read_text(encoding="utf-8"))
    res["spawn_to_opened_ms"] = marks.get("OPENED")
    res["cold_process_ms"] = marks.get("FIRST")
    return res


def run(a) -> int:
    graph_json, snap, out_dir = Path(a.graph_json), Path(a.snap), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qset = _qset(graph_json, snap, out_dir)
    results = []
    for backend in a.backends.split(","):
        for kind in a.kinds.split(","):
            out = out_dir / f"{a.tag}_{backend}_{kind}.json"
            r = _spawn(backend, kind, graph_json, snap, qset, out, a.budget,
                       1 if backend == "arcade" else a.passes)
            results.append(r)
            brief = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("backend", "kind", "n", "open_s", "cold_process_ms", "first_ms", "p50_ms",
                              "p95_ms", "failed")}
            brief["private_mb"] = round(r.get("mem_end", {}).get("private_mb", 0))
            brief["errors"] = len(r.get("errors", []))
            print(json.dumps(brief, ensure_ascii=False), flush=True)
    # parity of every backend against json, item by item over the common prefix
    by = {(r["backend"], r["kind"]): r for r in results if "hashes" in r}
    parity = {}
    for (backend, kind), r in by.items():
        ref = by.get(("json", kind))
        if backend == "json" or ref is None:
            continue
        n = min(len(r["hashes"]), len(ref["hashes"]))
        same = sum(x == y for x, y in zip(r["hashes"][:n], ref["hashes"][:n]))
        parity[f"{backend}/{kind}"] = {"compared": n, "equal": same}
    summary = {"graph_json": str(graph_json), "snap": str(snap), "snap_bytes": snap.stat().st_size,
               "graph_json_bytes": graph_json.stat().st_size, "results": results, "parity": parity}
    (out_dir / f"{a.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    print(json.dumps({"parity": parity}, ensure_ascii=False))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("graph_json")
    r.add_argument("snap")
    r.add_argument("out_dir")
    r.add_argument("--backends", default="snap,json,arcade")
    r.add_argument("--kinds", default=",".join(KINDS))
    r.add_argument("--budget", type=float, default=300.0)
    r.add_argument("--tag", default="run")
    r.add_argument("--passes", type=int, default=1,
                   help="passes over the item set (snap and json; arcade always 1)")
    c = sub.add_parser("child")
    c.add_argument("backend")
    c.add_argument("kind")
    c.add_argument("graph_json")
    c.add_argument("snap")
    c.add_argument("qset")
    c.add_argument("out")
    c.add_argument("--budget", type=float, default=300.0)
    c.add_argument("--passes", type=int, default=1)
    a = ap.parse_args(argv[1:])
    return run(a) if a.cmd == "run" else child(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
