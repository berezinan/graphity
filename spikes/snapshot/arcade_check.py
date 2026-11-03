"""Task 6.3: do the ArcadeDB databases hold the graph.json they are compared with?

    python arcade_check.py <graph.json> [<graph.json> ...]

Counts graph.json nodes and links by streaming the indented file line by line
(no full load), and the database's Node / Rel counts over HTTP. Read-only: no
database is created, reloaded or written.
"""
from __future__ import annotations

import json
import os
import sys
import time


def stream_counts(path: str) -> tuple[int, int]:
    """Items of the top-level `nodes` and `links` arrays of an indent=2 file."""
    section = None
    counts = {"nodes": 0, "links": 0}
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b'  "') and not line.startswith(b'   '):
                key = line[3:line.index(b'"', 3)].decode()
                section = key if key in counts else None
            elif section and line.rstrip(b"\r\n") == b"    {":
                counts[section] += 1
    return counts["nodes"], counts["links"]


def db_counts(graph_json: str) -> dict:
    os.environ["GRAPHIFY_GRAPH_STORE"] = "arcadedb"
    from graphify.query_backend import open_backend, resolve_backend_config
    cfg = resolve_backend_config(graph_json)
    out = {"database": cfg.get("database")}
    try:
        be = open_backend(config=cfg)
        out["nodes"] = int(be._run("SELECT count(*) AS c FROM Node")[0]["c"])
        out["edges"] = int(be._run("SELECT count(*) AS c FROM Rel")[0]["c"])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def main(paths: list[str]) -> int:
    rows = []
    for p in paths:
        t = time.perf_counter()
        n, e = stream_counts(p)
        row = {"graph_json": p, "size_mb": round(os.path.getsize(p) / 2**20),
               "json_nodes": n, "json_links": e, "count_seconds": round(time.perf_counter() - t, 1)}
        db = db_counts(p)
        row.update({f"db_{k}": v for k, v in db.items()})
        row["comparable"] = "error" not in db and (db.get("nodes"), db.get("edges")) == (n, e)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
