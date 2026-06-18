"""Generate a synthetic graph.json in Graphify's node-link format.

Streams JSON to disk so generating 1M nodes doesn't itself blow up RAM. Shapes
the graph to look like a real Graphify graph:
  - searchable labels built from a vocabulary that INCLUDES the bench query terms
    (so queries actually resolve), each suffixed with the index for variety;
  - communities (label clusters);
  - a backbone chain + per-node random edges + a handful of hub nodes (god nodes)
    so shortest paths stay short and the graph is one giant component.

norm_label is intentionally omitted: each backend derives it (serve.py already
does; arcade_backend lowercases at load), which keeps the file lean.

Usage: python gen_synth.py <out.json> <n_nodes> [avg_degree=2]
"""
from __future__ import annotations

import json
import random
import sys

VOCAB = [
    "build", "graph", "export", "query", "cache", "extract", "security", "validate",
    "service", "handler", "module", "parser", "loader", "writer", "reader", "client",
    "server", "config", "schema", "index", "node", "edge", "token", "vector",
    "session", "request", "response", "manager", "worker", "queue", "router", "store",
]
RELATIONS = ["calls", "imports", "uses", "contains", "references"]
CONFS = ["EXTRACTED", "INFERRED", "AMBIGUOUS"]


def gen(out_path: str, n: int, avg_degree: int = 2) -> None:
    rnd = random.Random(42)
    n_comm = max(1, n // 500)
    n_hubs = max(10, n // 10_000)
    hub_ids = [f"n{rnd.randrange(n)}" for _ in range(n_hubs)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"directed": true, "multigraph": false, "graph": {}, "nodes": [')
        for i in range(n):
            tok = VOCAB[i % len(VOCAB)]
            node = {
                "id": f"n{i}",
                "label": f"{tok}_{i}",
                "source_file": f"src/mod_{i % 5000}.py",
                "file_type": "code",
                "community": i % n_comm,
            }
            f.write(("" if i == 0 else ",") + json.dumps(node, ensure_ascii=False))
        f.write('], "links": [')

        first = True
        # Backbone: i -> i+1 keeps everything reachable.
        # Per-node: one random edge + one hub edge -> short diameter via hubs.
        for i in range(n):
            targets = []
            if i + 1 < n:
                targets.append(i + 1)
            for _ in range(max(0, avg_degree - 1)):
                targets.append(rnd.randrange(n))
            targets.append(None)  # sentinel -> hub edge
            for t in targets:
                tgt = rnd.choice(hub_ids) if t is None else f"n{t}"
                if tgt == f"n{i}":
                    continue
                edge = {
                    "source": f"n{i}",
                    "target": tgt,
                    "relation": RELATIONS[i % len(RELATIONS)],
                    "confidence": CONFS[i % len(CONFS)],
                    "weight": 1.0,
                }
                f.write(("" if first else ",") + json.dumps(edge, ensure_ascii=False))
                first = False
        f.write("]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    n = int(sys.argv[2])
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    gen(out, n, deg)
    import os
    print(f"wrote {out}: {n} nodes, ~{n * (deg + 1)} edges, {os.path.getsize(out) / 1e6:.1f} MB")
