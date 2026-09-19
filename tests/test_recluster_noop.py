"""A re-clustering that reproduces the stored partition has nothing to write:
community_changes() is empty exactly when to_json would reproduce the file."""
from __future__ import annotations

import json

from graphify.build import build_from_json
from graphify.export import to_json
from graphify.recluster import community_changes

COMMUNITIES = {0: ["a_mod", "a_fn"], 1: ["b_mod", "справочник_товары"]}
LABELS = {0: "Alpha", 1: "Бета"}


def _extraction() -> dict:
    return {
        "nodes": [
            {"id": "a_mod", "label": "a.py", "source_file": "src/a.py", "file_type": "code",
             "source_location": "L1", "extra_attr": {"kept": True}},
            {"id": "a_fn", "label": "run()", "source_file": "src/a.py", "file_type": "code",
             "source_location": "L5"},
            {"id": "b_mod", "label": "b.py", "source_file": "src/b.py", "file_type": "code",
             "source_location": "L1"},
            {"id": "справочник_товары", "label": "Товары", "source_file": "src/b.py",
             "file_type": "code", "source_location": "L9"},
        ],
        "edges": [
            {"source": "a_mod", "target": "a_fn", "relation": "contains", "confidence": "EXTRACTED"},
            {"source": "a_fn", "target": "справочник_товары", "relation": "calls",
             "confidence": "INFERRED"},
            {"source": "b_mod", "target": "справочник_товары", "relation": "contains",
             "confidence": "EXTRACTED"},
        ],
    }


def _stored_graph(tmp_path):
    """graph.json as a previous cluster-only run left it."""
    path = tmp_path / "graph.json"
    G = build_from_json(_extraction())
    assert to_json(G, COMMUNITIES, str(path), community_labels=LABELS, built_at_commit="c0ffee")
    return path


def _reload(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, build_from_json(raw, directed=bool(raw.get("directed", False)))


def test_unchanged_partition_is_empty_and_skipping_equals_writing(tmp_path):
    path = _stored_graph(tmp_path)
    before = path.read_bytes()
    raw, G = _reload(path)
    assert community_changes(raw, G, COMMUNITIES, LABELS) == {}
    # The write the CLI skips would have reproduced the file byte for byte.
    assert to_json(G, COMMUNITIES, str(path), community_labels=LABELS,
                   built_at_commit=raw["built_at_commit"])
    assert path.read_bytes() == before


def test_moved_node_and_renamed_community_are_reported(tmp_path):
    raw, G = _reload(_stored_graph(tmp_path))
    moved = {0: ["a_mod"], 1: ["a_fn", "b_mod", "справочник_товары"]}
    assert community_changes(raw, G, moved, LABELS) == {"a_fn": (1, "Бета")}
    renamed = community_changes(raw, G, COMMUNITIES, {0: "Alpha", 1: "Гамма"})
    assert renamed == {"b_mod": (1, "Гамма"), "справочник_товары": (1, "Гамма")}


def test_no_labels_keeps_stored_names(tmp_path):
    raw, G = _reload(_stored_graph(tmp_path))
    assert community_changes(raw, G, COMMUNITIES, None) == {}


def test_shape_mismatch_is_not_vouched_for(tmp_path):
    raw, G = _reload(_stored_graph(tmp_path))
    G.remove_node("a_fn")
    assert community_changes(raw, G, {0: ["a_mod"], 1: COMMUNITIES[1]}, LABELS) is None


def test_missing_commit_stamp_forces_a_write_but_not_a_database_reload(tmp_path):
    # A stamp to_json would add is a reason to write the file, not a reason to
    # distrust the community diff: the database still has nothing to update.
    from graphify.recluster import needs_write
    raw, G = _reload(_stored_graph(tmp_path))
    changed = community_changes(raw, G, COMMUNITIES, LABELS)
    assert changed == {}
    assert not needs_write(raw, changed, "c0ffee")
    assert not needs_write(raw, changed, None)
    assert needs_write(raw, changed, "deadbeef")
    assert needs_write(raw, {"a_fn": (1, "Бета")}, "c0ffee")
    assert needs_write(raw, None, "c0ffee")
