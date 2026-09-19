"""File-label disambiguation must scale linearly in the basename-group size
and keep the labels the pairwise implementation (#2032) assigned.

A 1C:EDT export has tens of thousands of files per basename (``Module.bsl``,
``ManagerModule.bsl``), which made the pairwise ``_shortest_unique_suffix`` loop
take hours inside ``build_from_json``. The reference below is a verbatim copy of
that pairwise implementation; the production code must match it on any input.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

from graphify.build import _file_label_reassignments, _is_file_node_label


def _ref_shortest_unique_suffix(sf: str, all_sfs: "set[str]") -> str:
    parts = [p for p in sf.replace("\\", "/").split("/") if p]
    others = [
        [p for p in o.replace("\\", "/").split("/") if p]
        for o in all_sfs if o != sf
    ]
    for k in range(1, len(parts) + 1):
        suffix = parts[-k:]
        if all(o[-k:] != suffix for o in others):
            return "/".join(suffix)
    return "/".join(parts)


def _ref_file_label_reassignments(items: "list[tuple]") -> dict:
    groups: dict[str, list[tuple]] = defaultdict(list)
    for key, label, sf in items:
        if sf and label and _is_file_node_label(str(label), str(sf)):
            basename = str(sf).replace("\\", "/").rsplit("/", 1)[-1]
            groups[basename].append((key, str(sf)))
    out: dict = {}
    for members in groups.values():
        distinct = {sf for _, sf in members}
        if len(distinct) < 2:
            continue
        for key, sf in members:
            out[key] = _ref_shortest_unique_suffix(sf, distinct)
    return out


def _items(paths: "list[str]") -> "list[tuple]":
    return [(i, p.replace("\\", "/").rsplit("/", 1)[-1], p) for i, p in enumerate(paths)]


def test_matches_pairwise_reference_on_handwritten_edges():
    paths = [
        "a/b/index.ts", "c/b/index.ts",          # needs two parent dirs
        "x/index.ts",                             # one parent dir is enough
        "b/index.ts",                             # a full path that is a suffix of others
        "index.ts",                               # bare basename colliding with everything
        "a\\b\\Module.bsl", "a/b/Module.bsl",     # differ only by separator
        "q/Module.bsl",
        "solo/unique.py",                         # no collision: stays bare
        "/abs/deep/er/path/index.ts",
    ]
    items = _items(paths)
    assert _file_label_reassignments(items) == _ref_file_label_reassignments(items)


def test_matches_pairwise_reference_on_generated_groups():
    rng = random.Random(2032)
    dirs = ["Catalogs", "Documents", "Forms", "X", "Y", "Z", "src", "Ext"]
    names = ["Module.bsl", "ObjectModule.bsl", "Form.form", "index.ts"]
    for _ in range(200):
        paths = set()
        for _ in range(rng.randint(1, 40)):
            depth = rng.randint(0, 5)
            sep = rng.choice(["/", "/", "/", "\\"])
            paths.add(sep.join([rng.choice(dirs) for _ in range(depth)] + [rng.choice(names)]))
        items = _items(sorted(paths))
        rng.shuffle(items)
        assert _file_label_reassignments(items) == _ref_file_label_reassignments(items)


def test_same_file_twice_is_not_a_collision():
    items = [(0, "Module.bsl", "a/Module.bsl"), (1, "Module.bsl", "a/Module.bsl")]
    assert _file_label_reassignments(items) == _ref_file_label_reassignments(items) == {}


def test_edt_scale_group_is_not_quadratic():
    # 40 000 files sharing one basename: the pairwise loop needs ~1.6e9 suffix
    # comparisons (hours); the linear pass is well under the bound below.
    paths = [f"src/Catalogs/Obj{i}/Forms/Form{i % 7}/Module.bsl" for i in range(40_000)]
    items = _items(paths)
    t = time.perf_counter()
    out = _file_label_reassignments(items)
    elapsed = time.perf_counter() - t
    assert len(out) == 40_000
    assert out[0] == "Obj0/Forms/Form0/Module.bsl"
    assert elapsed < 20, f"disambiguation took {elapsed:.1f}s - quadratic again?"


def test_file_node_label_match_implies_same_last_segment():
    # build_from_json's ghost pass (#3344) buckets AST file nodes by the last
    # path segment instead of testing every non-AST node against every one of
    # them. That is only sound if a match always shares that segment.
    rng = random.Random(3344)
    segs = ["a", "b", "apps", "customer-app", "index.ts", "App.tsx", "Module.bsl", ""]
    hits = 0
    for _ in range(20_000):
        sf = rng.choice(["/", "\\"]).join(rng.choice(segs) for _ in range(rng.randint(1, 4)))
        label = "/".join(rng.choice(segs) for _ in range(rng.randint(1, 3)))
        if _is_file_node_label(label, sf):
            hits += 1
            assert label.rsplit("/", 1)[-1] == sf.replace("\\", "/").rsplit("/", 1)[-1]
    assert hits > 100
