"""Task 6.2: cold-process time of `snap-query` and its tokenisation vs
`serve._query_terms` / `_score_terms` on a query set.

    python snap_query_check.py <graph.snap> <qset.json> [repeats=5]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE / "target" / "release" / "snap-query.exe"


def main(argv) -> int:
    sys.path.insert(0, str(HERE))
    from graphify.serve import _query_terms, _search_tokens
    from snap_common import norm_fingerprint
    snap, qset = argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    repeats = int(argv[3]) if len(argv) > 3 else 5
    questions = qset["query"] + qset["explain"] + [a for p in qset["path"] for a in p]
    mismatches = []
    for q in questions:
        out = subprocess.run([str(EXE), "--terms", q], capture_output=True, encoding="utf-8").stdout
        terms, norm = (out.rstrip("\n").split("\t") + [""])[:2]
        want_terms = _query_terms(q)
        want_norm = list(dict.fromkeys(tok for t in want_terms for tok in _search_tokens(t)))
        if terms.split() != want_terms or norm.split() != want_norm:
            mismatches.append({"q": q, "rust": [terms, norm], "python": [" ".join(want_terms), " ".join(want_norm)]})
    fp = norm_fingerprint().hex()
    walls, opens, totals = [], [], []
    for q in qset["query"][:10]:
        for _ in range(repeats):
            t = time.perf_counter()
            r = subprocess.run([str(EXE), snap, fp, q], capture_output=True, encoding="utf-8")
            walls.append((time.perf_counter() - t) * 1000)
            if r.returncode != 0:
                raise SystemExit(r.stderr)
            kv = dict(x.split("=") for x in r.stderr.split())
            opens.append(float(kv["open_ms"]))
            totals.append(float(kv["total_ms"]))
    q95 = lambda v: sorted(v)[max(0, int(round(0.95 * len(v))) - 1)]
    print(json.dumps({
        "questions_checked": len(questions), "tokenisation_mismatches": mismatches,
        "runs": len(walls),
        "wall_ms_p50": round(statistics.median(walls), 1), "wall_ms_p95": round(q95(walls), 1),
        "open_ms_p50": round(statistics.median(opens), 2),
        "in_process_ms_p50": round(statistics.median(totals), 1), "in_process_ms_p95": round(q95(totals), 1),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
