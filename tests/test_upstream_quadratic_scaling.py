"""Two quadratic spots upstream added after the fork linearized its own (#321758d).

`label_still_applies` (#2298) scanned the whole previous assignment once per
community, and the shrink-guard twin search (`_lost_saved_semantic_nodes`)
tested every lost node against every AST file node. Both must keep the exact
upstream answer and grow linearly with the graph.
"""
from __future__ import annotations

import random
import time

from graphify.build import _is_file_node_label, _lost_saved_semantic_nodes
from graphify.cluster import label_still_applies, previous_members_by_community


def _label_case(rng):
    nodes = [f"n{i}" for i in range(rng.randint(1, 80))]
    prev = {n: rng.randint(0, 6) for n in nodes if rng.random() < 0.8}
    current: dict[int, list[str]] = {}
    for n in nodes:
        current.setdefault(rng.randint(0, 6), []).append(n)
    return prev, current


def test_label_still_applies_with_index_matches_scan():
    rng = random.Random(4321)
    for _ in range(300):
        prev, current = _label_case(rng)
        index = previous_members_by_community(prev)
        for cid in range(8):
            members = current.get(cid, [])
            for sigs in ((None, None), ("a", "a"), ("a", "b")):
                assert label_still_applies(cid, members, prev, *sigs, index) == \
                    label_still_applies(cid, members, prev, *sigs)


def _time_labels(n_communities: int, per: int = 20) -> float:
    prev = {f"n{c}_{i}": c for c in range(n_communities) for i in range(per)}
    current = {c: [f"n{c}_{i}" for i in range(per)] for c in range(n_communities)}
    t0 = time.perf_counter()
    index = previous_members_by_community(prev)
    for cid, members in current.items():
        assert label_still_applies(cid, members, prev, None, None, index)
    return time.perf_counter() - t0


def test_label_still_applies_scales_linearly():
    small, large = _time_labels(1000), _time_labels(4000)
    # Quadratic would be ~16x; allow generous noise for a linear ~4x.
    assert large < small * 9 + 0.05, (small, large)


def _twin_graph(n: int):
    new_nodes = [
        {"id": f"f{i}", "label": f"mod{i}.py", "source_file": f"src/pkg{i % 7}/mod{i}.py", "_origin": "ast"}
        for i in range(n)
    ]
    saved = [
        {"id": f"doc{i}", "label": f"mod{i}.py", "source_file": f"docs/d{i}.md", "file_type": "document"}
        for i in range(n)
    ]
    return saved, new_nodes


def test_file_twin_fold_matches_unbucketed_search():
    saved, new_nodes = _twin_graph(50)
    # Two AST files sharing a basename make that label ambiguous (not folded).
    new_nodes.append({"id": "dup", "label": "mod3.py", "source_file": "other/mod3.py", "_origin": "ast"})
    folded: dict[str, str] = {}
    lost = _lost_saved_semantic_nodes(saved, new_nodes, folded=folded)
    ast_files = [(n["id"], n["source_file"]) for n in new_nodes]
    for node in saved:
        twins = [nid for nid, sf in ast_files if _is_file_node_label(node["label"], sf)]
        if len(twins) == 1:
            assert folded[node["id"]] == twins[0]
        else:
            assert node in lost
    assert {n["id"] for n in lost} == {"doc3"}


def _time_twins(n: int) -> float:
    saved, new_nodes = _twin_graph(n)
    t0 = time.perf_counter()
    assert _lost_saved_semantic_nodes(saved, new_nodes) == []
    return time.perf_counter() - t0


def test_file_twin_fold_scales_linearly():
    small, large = _time_twins(2000), _time_twins(8000)
    assert large < small * 9 + 0.05, (small, large)
