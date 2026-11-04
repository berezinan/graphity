"""Fixed, seeded query set for a graph (D11)."""
from __future__ import annotations

import random
import re
from pathlib import Path

SEED = 20261103


def report_questions(graph_json: Path, limit: int = 10) -> list[str]:
    report = Path(graph_json).with_name("GRAPH_REPORT.md")
    if not report.exists():
        return []
    qs = []
    for line in report.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- \*\*(.+?)\*\*\s*$", line.strip())
        if m:
            q = m.group(1).replace("`", "")
            qs.append(q[:300])
    return qs[:limit]


def labels_of(raw: dict) -> list:
    return [n.get("label") for n in raw["nodes"]]


def labels_of_snapshot(sb) -> list:
    """Node labels in graph.json order, read from the snapshot's label column."""
    import json
    codes = sb._ncol["label"]
    cache: dict[int, object] = {}
    out = []
    for c in codes:
        if not c:
            out.append(None)
            continue
        v = cache.get(c)
        if v is None:
            v = cache[c] = json.loads(sb.snap.value("n", "label", c))
        out.append(v)
    return out


def build(labels: list, graph_json: Path | None, god_labels: list[str]) -> dict:
    """`labels`: every node's label in graph.json order (None when absent)."""
    rnd = random.Random(SEED)
    labels = [l for l in labels if isinstance(l, str) and l.strip()]
    pick = lambda k: rnd.sample(labels, min(k, len(labels)))
    queries = (report_questions(graph_json) if graph_json else []) + god_labels + pick(30)
    return {
        "query": queries,
        "path": [tuple(pick(2)) for _ in range(30)],
        "explain": pick(30),
        "context": [(q, f) for q, f in zip(pick(5), (["call"], ["new"], ["call", "new"], ["form-handler"], ["metadata"]))],
    }
