"""Benchmark harness: JSON/NetworkX baseline vs ArcadeDB backend.

Each backend runs in its OWN subprocess so resident-memory (RSS) numbers are
isolated and honest - the headline question is "how much RAM does the query
client hold?", and the JSON model holds the whole graph while the DB client
holds ~nothing.

Usage:
  python bench.py json   <graph.json>                  # baseline, prints metrics JSON
  python bench.py arcade <graph.json> [--load] [--drop]  # ArcadeDB, prints metrics JSON
  python bench.py compare <graph.json> [--load] [--drop] # spawns both, prints table
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import psutil

# Representative workload. Terms are deliberately generic identifiers likely to
# exist in any Graphify graph; the path pair stresses traversal/resolution.
QUERIES = ["build graph", "export", "query", "shortest path", "cache", "extract", "security validate"]
PATHS = [("build", "export"), ("query", "cache"), ("extract", "security")]
REPEAT = int(os.environ.get("GRAPH_BENCH_REPEAT", "5"))


def _rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1_048_576, 1)


def _timeit(fn, *args, **kw) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return (time.perf_counter() - t0) * 1000, out


def _run_workload(backend, query_call, path_call) -> dict:
    q_ms: list[float] = []
    last = {}
    for _ in range(REPEAT):
        for q in QUERIES:
            ms, out = _timeit(query_call, q)
            q_ms.append(ms)
            last[q] = out
    p_ms: list[float] = []
    paths = {}
    for _ in range(REPEAT):
        for a, b in PATHS:
            ms, out = _timeit(path_call, a, b)
            p_ms.append(ms)
            paths[f"{a}->{b}"] = out
    return {
        "query_ms_p50": round(statistics.median(q_ms), 2),
        "query_ms_p95": round(sorted(q_ms)[int(len(q_ms) * 0.95)], 2),
        "path_ms_p50": round(statistics.median(p_ms), 2),
        "path_ms_p95": round(sorted(p_ms)[int(len(p_ms) * 0.95)], 2),
        "sample_query": last.get(QUERIES[0]),
        "sample_paths": paths,
    }


def run_json(graph_path: str) -> dict:
    from json_backend import JsonBackend
    rss0 = _rss_mb()
    be = JsonBackend(graph_path)
    load = be.load()
    rss_loaded = _rss_mb()
    work = _run_workload(be, lambda q: be.query(q), lambda a, b: be.shortest_path(a, b))
    return {
        "backend": "json+networkx",
        "load": load,
        "rss_baseline_mb": rss0,
        "rss_after_load_mb": rss_loaded,
        "rss_graph_cost_mb": round(rss_loaded - rss0, 1),
        **work,
    }


def run_arcade(graph_path: str, do_load: bool, drop: bool) -> dict:
    from arcade_backend import ArcadeDBBackend
    rss0 = _rss_mb()
    be = ArcadeDBBackend()
    if not be.ready():
        return {"backend": "arcadedb", "error": "ArcadeDB not reachable on http://127.0.0.1:2480 (start the server first)"}
    be.ensure_database(drop=drop)
    load = {}
    if do_load:
        load = be.load_from_graph_json(graph_path)
    rss_after = _rss_mb()
    from graphify.serve import _query_terms
    work = _run_workload(
        be,
        lambda q: be.query(_query_terms(q)),
        lambda a, b: be.shortest_path(_query_terms(a), _query_terms(b)),
    )
    return {
        "backend": "arcadedb",
        "load": load,
        "rss_baseline_mb": rss0,
        "rss_after_connect_mb": rss_after,
        "rss_graph_cost_mb": round(rss_after - rss0, 1),
        **work,
    }


def _spawn(mode: str, graph_path: str, extra: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), mode, graph_path, *extra],
        capture_output=True, text=True,
    )
    out = proc.stdout.strip()
    try:
        return json.loads(out.splitlines()[-1])
    except Exception:
        return {"backend": mode, "error": f"subprocess failed\nSTDOUT:\n{out[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"}


def _fmt(v) -> str:
    return "-" if v is None else (f"{v}" if not isinstance(v, float) else f"{v:.2f}")


def compare(graph_path: str, do_load: bool, drop: bool) -> None:
    arcade_extra = (["--load"] if do_load else []) + (["--drop"] if drop else [])
    j = _spawn("json", graph_path, [])
    a = _spawn("arcade", graph_path, arcade_extra)
    print("\n================ SPIKE RESULT: JSON/NetworkX vs ArcadeDB ================\n")
    for label, r in (("JSON+NetworkX", j), ("ArcadeDB", a)):
        print(f"--- {label} ---")
        print(json.dumps(r, indent=2, ensure_ascii=False))
        print()
    if "error" in a:
        print("ArcadeDB leg did not run (see error above). JSON baseline numbers are still valid.")
        return
    rows = [
        ("graph load (s)", j["load"].get("load_seconds"), a["load"].get("load_seconds")),
        ("client RAM for graph (MB)", j.get("rss_graph_cost_mb"), a.get("rss_graph_cost_mb")),
        ("query p50 (ms)", j.get("query_ms_p50"), a.get("query_ms_p50")),
        ("query p95 (ms)", j.get("query_ms_p95"), a.get("query_ms_p95")),
        ("path p50 (ms)", j.get("path_ms_p50"), a.get("path_ms_p50")),
        ("path p95 (ms)", j.get("path_ms_p95"), a.get("path_ms_p95")),
    ]
    w = 30
    print(f"{'metric':<{w}}{'JSON':>14}{'ArcadeDB':>14}")
    print("-" * (w + 28))
    for name, jv, av in rows:
        print(f"{name:<{w}}{_fmt(jv):>14}{_fmt(av):>14}")
    print("\nNote: 'client RAM for graph' is the key scaling metric - JSON holds the whole")
    print("graph in the agent's process; ArcadeDB keeps it server-side.\n")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode, graph_path = sys.argv[1], sys.argv[2]
    do_load = "--load" in sys.argv
    drop = "--drop" in sys.argv
    if mode == "json":
        print(json.dumps(run_json(graph_path), ensure_ascii=False))
    elif mode == "arcade":
        print(json.dumps(run_arcade(graph_path, do_load, drop), ensure_ascii=False))
    elif mode == "compare":
        compare(graph_path, do_load, drop)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
