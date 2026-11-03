"""Tasks 5.1 / 5.2: rebuilds under live readers, and a writer that dies.

    python stand_generations.py <graph.json> <work_dir> [rebuilds=20] [readers=4]

Graph A is the given graph.json, graph B the same with every label suffixed,
so every answer tells which graph it came from. Readers query continuously;
each answer must equal the reference answer of A or of B.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERIES_N = 12


def _variant(data: dict) -> dict:
    b = json.loads(json.dumps(data))
    for n in b["nodes"]:
        if isinstance(n.get("label"), str):
            n["label"] += " v2"
        n.pop("norm_label", None)
    return b


def _answers(snap_path: Path, queries: list[str]) -> list[str]:
    sys.path.insert(0, str(HERE))
    from graphify.query_backend import render_query
    from snap_backend import SnapBackend
    sb = SnapBackend(snap_path)
    return [render_query(sb.query(q, depth=2)) for q in queries]


def _reader(k: int, live: str, queries, ref_a, ref_b, stop: str, out: str):
    sys.path.insert(0, str(HERE))
    from graphify.query_backend import render_query
    from snap_backend import SnapBackend
    stats = {"ok_a": 0, "ok_b": 0, "mismatch": 0, "errors": {}, "generations": set()}
    sb = None
    i = k
    while not Path(stop).exists():
        q = i % len(queries)
        i += 1
        try:
            if sb is None:
                sb = SnapBackend(live)
            got = render_query(sb.query(queries[q], depth=2))
            stats["generations"].add(sb.snap.generation)
            if got == ref_a[q]:
                stats["ok_a"] += 1
            elif got == ref_b[q]:
                stats["ok_b"] += 1
            else:
                stats["mismatch"] += 1
        except Exception as exc:  # every error counts; the stand reports them
            key = f"{type(exc).__name__}: {exc}"[:200]
            stats["errors"][key] = stats["errors"].get(key, 0) + 1
    stats["generations"] = sorted(stats["generations"])
    Path(out).write_text(json.dumps(stats), encoding="utf-8")


def main(argv):
    sys.path.insert(0, str(HERE))
    import snap_writer
    src, work = Path(argv[1]), Path(argv[2])
    rebuilds = int(argv[3]) if len(argv) > 3 else 20
    nreaders = int(argv[4]) if len(argv) > 4 else 4
    if work.exists():
        shutil.rmtree(work)
    live_dir = work / "live"
    with open(src, encoding="utf-8") as f:
        a = json.load(f)
    b = _variant(a)
    labels = [n["label"] for n in a["nodes"] if isinstance(n.get("label"), str)]
    import random
    rnd = random.Random(5)
    queries = rnd.sample(labels, QUERIES_N)
    for name, data in (("a", a), ("b", b)):
        snap_writer.publish_snapshot(data, work / f"ref_{name}")
    # BFS answers depend on the process hash seed (the frontier is a set), so
    # every process -- references included -- runs with one fixed seed.
    os.environ["PYTHONHASHSEED"] = "0"
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        ref_a = pool.apply(_answers, (work / "ref_a" / "graph.snap", queries))
        ref_b = pool.apply(_answers, (work / "ref_b" / "graph.snap", queries))
    assert all(x != y for x, y in zip(ref_a, ref_b)), "graphs A and B must answer differently"
    snap_writer.publish_snapshot(a, live_dir)
    free0 = shutil.disk_usage(work).free
    snap_size = (live_dir / "graph.snap").stat().st_size

    stop = work / "STOP"
    procs =[ctx.Process(target=_reader, args=(k, str(live_dir / "graph.snap"), queries, ref_a, ref_b,
                                               str(stop), str(work / f"reader{k}.json")))
             for k in range(nreaders)]
    for p in procs:
        p.start()
    time.sleep(5)
    t0 = time.perf_counter()
    gens = []
    for r in range(rebuilds):
        rep = snap_writer.publish_snapshot(b if r % 2 == 0 else a, live_dir)
        gens.append(rep["generation"])
    rebuild_s = time.perf_counter() - t0

    # 5.2: a writer that dies after writing its .tmp, before publishing.
    crash = subprocess.run(
        [sys.executable, "-c",
         "import sys, json; sys.path.insert(0, sys.argv[1]); import snap_writer; "
         "d = json.load(open(sys.argv[2], encoding='utf-8')); "
         "snap_writer.publish_snapshot(d, sys.argv[3], crash_before_publish=True)",
         str(HERE), str(src), str(live_dir)])
    tmp_after_crash = sorted(p.name for p in live_dir.glob("*.tmp"))
    gen_after_crash = __import__("graphify_snap").peek_generation(str(live_dir / "graph.snap"))
    time.sleep(3)          # readers keep answering from the last published generation
    stop.write_text("stop")
    for p in procs:
        p.join(120)
    readers = [json.loads((work / f"reader{k}.json").read_text(encoding="utf-8")) for k in range(nreaders)]
    rep = snap_writer.publish_snapshot(a, live_dir)
    time.sleep(1)
    free1 = shutil.disk_usage(work).free
    result = {
        "rebuilds": rebuilds, "readers": nreaders, "rebuild_seconds": round(rebuild_s, 1),
        "generations_published": gens,
        "answers_a": sum(r["ok_a"] for r in readers), "answers_b": sum(r["ok_b"] for r in readers),
        "mismatches": sum(r["mismatch"] for r in readers),
        "errors": [r["errors"] for r in readers],
        "generations_seen_per_reader": [len(r["generations"]) for r in readers],
        "crash_exit_code": crash.returncode, "tmp_after_crash": tmp_after_crash,
        "generation_after_crash": gen_after_crash, "next_run_removed": rep["removed_tmp"],
        "dir_after": sorted(p.name for p in live_dir.iterdir()),
        "free_space_delta_mb": round((free1 - free0) / 2**20, 1), "snapshot_mb": round(snap_size / 2**20, 1),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
