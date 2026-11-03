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


def build(raw: dict, graph_json: Path | None, god_labels: list[str]) -> dict:
    rnd = random.Random(SEED)
    labels = [n["label"] for n in raw["nodes"] if isinstance(n.get("label"), str) and n["label"].strip()]
    pick = lambda k: rnd.sample(labels, min(k, len(labels)))
    queries = (report_questions(graph_json) if graph_json else []) + god_labels + pick(30)
    return {
        "query": queries,
        "path": [tuple(pick(2)) for _ in range(30)],
        "explain": pick(30),
        "context": [(q, f) for q, f in zip(pick(5), (["call"], ["new"], ["call", "new"], ["form-handler"], ["metadata"]))],
    }
