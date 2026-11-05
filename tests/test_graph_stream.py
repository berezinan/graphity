"""The streaming reader returns exactly what json.load returns."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify import graph_stream

PSN = Path(r"C:\_dev\ANB\AI\psn_diadoc_4_67_3_Pravka\graphify-out\graph.json")


def _data() -> dict:
    nodes = [
        {"id": i, "label": lbl, "source_file": f"{lbl}.py", "community": c}
        for i, lbl, c in (("a", "alpha", 0), ("b", "beta", 0), ("c", "gamma", 1), ("d", "delta", 1))
    ]
    links = [
        {"source": "a", "target": "a", "relation": "calls", "confidence": "EXTRACTED", "context": "call"},
        {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED", "weight": 1.0},
        {"source": "c", "target": "b", "relation": "uses", "confidence": "INFERRED"},
        {"source": "b", "target": "a", "relation": "imports", "confidence": "AMBIGUOUS",
         "extra": [1, {"k": None}]},
    ]
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


@pytest.mark.parametrize("indent", [2, None])
def test_fixture(tmp_path, indent):
    data = _data()
    data["graph"] = {"hyperedges": [{"id": "h", "nodes": ["a", "b"]}], "x": {"y": []}}
    data["built_at_commit"] = "abc"
    data["nodes"][0]["label"] = 'tricky "    },\\n" label'
    data["empty"] = []
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    got = graph_stream.load(p)
    want = json.loads(p.read_text(encoding="utf-8"))
    assert got == want
    assert list(got) == list(want)


def test_empty_arrays(tmp_path):
    data = {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": []}
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    assert graph_stream.load(p) == data


def test_edges_key_is_streamed_like_links(tmp_path):
    data = _data()
    data["edges"] = data.pop("links")
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    assert graph_stream.load(p) == data


def test_unexpected_layout_falls_back_to_json_load(tmp_path):
    data = _data()
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data), encoding="utf-8")  # one line: no indent=2 layout
    assert graph_stream.load(p) == data


@pytest.mark.skipif(not PSN.exists(), reason="psn_diadoc graph.json not present")
def test_psn():
    with open(PSN, encoding="utf-8") as f:
        want = json.load(f)
    got = graph_stream.load(PSN)
    assert got == want
    assert list(got) == list(want)
