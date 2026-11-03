"""The streaming reader returns exactly what json.load returns."""
import json

import pytest
from conftest import PSN

import graph_stream
from test_shapes import _data


@pytest.mark.parametrize("indent", [2, None])
def test_fixture(tmp_path, indent):
    data = _data(True)
    data["graph"] = {"hyperedges": [{"id": "h", "nodes": ["a", "b"]}], "x": {"y": []}}
    data["built_at_commit"] = "abc"
    data["nodes"][0]["label"] = 'tricky "    },\\n" label'
    data["empty"] = []
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    assert graph_stream.load(p) == json.loads(p.read_text(encoding="utf-8"))


def test_empty_arrays(tmp_path):
    data = {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": []}
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    assert graph_stream.load(p) == data


@pytest.mark.skipif(not PSN.exists(), reason="psn_diadoc graph.json not present")
def test_psn(psn_data):
    got = graph_stream.load(PSN)
    assert got == psn_data
    assert list(got) == list(psn_data)
