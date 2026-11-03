"""Task 2.4 synthetic shapes, task 3.1 source immutability, D13 refusals."""
import json
import shutil

import pytest
from conftest import PSN, snap_of

import ref_loaders
import snap_writer
from graphify.query_backend import JsonBackend, render_explain, render_path, render_query
from snap_common import Unsupported


def _data(directed: bool) -> dict:
    nodes = [{"id": i, "label": lbl, "source_file": f"{lbl}.py", "community": c}
             for i, lbl, c in (("a", "alpha", 0), ("b", "beta", 0), ("c", "gamma", 1), ("d", "delta", 1))]
    links = [
        {"source": "a", "target": "a", "relation": "calls", "confidence": "EXTRACTED", "context": "call"},
        {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED", "context": "call", "weight": 1.0},
        {"source": "c", "target": "b", "relation": "uses", "confidence": "INFERRED"},
        # the same pair the other way, with other attributes and no context
        {"source": "b", "target": "a", "relation": "imports", "confidence": "AMBIGUOUS", "extra": [1, {"k": None}]},
        {"source": "d", "target": "c", "relation": "contains", "confidence": "EXTRACTED", "context": "new"},
        {"source": "c", "target": "a", "relation": "calls", "confidence": "EXTRACTED", "context": "call"},
    ]
    return {"directed": directed, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


@pytest.mark.parametrize("directed", [False, True])
def test_loop_and_repeated_pair(tmp_path, directed):
    data = _data(directed)
    sb = snap_of(data, tmp_path)
    G = ref_loaders.query_graph(data)
    from graphify.serve import _traversal_view
    TG = _traversal_view(G)
    for i, nid in enumerate(TG.nodes()):
        assert [sb._id(j) for j in sb.snap.tneighbors(i)] == list(TG.neighbors(nid)), nid
        assert sb.snap.tdegree(i) == TG.degree(nid), nid
        assert sb.snap.deg_q(i) == G.degree(nid), nid
    # one pair a-b, at the position of its first appearance, attributes merged in order
    merged = {}
    for l in (data["links"][1], data["links"][3]) if not directed else (data["links"][1], data["links"][3]):
        merged.update({k: v for k, v in l.items() if k not in ("source", "target")})
    from snap_backend import _TG
    got = _TG(sb, None).pair_data("a", "b")
    assert {k: v for k, v in got.items() if not k.startswith("_")} == {k: v for k, v in TG["a"]["b"].items() if not k.startswith("_")}
    jq = JsonBackend(G)
    for q in ("alpha", "beta", "who calls gamma", "delta"):
        for mode in ("bfs", "dfs"):
            assert render_query(sb.query(q, mode=mode, depth=3)) == render_query(jq.query(q, mode=mode, depth=3)), (q, mode)
    je = JsonBackend(ref_loaders.explain_graph(data))
    for label in ("alpha", "beta", "gamma"):
        assert render_explain(sb.explain(label), label) == render_explain(je.explain(label), label)
    for undirected in (False, True):
        jp = JsonBackend(ref_loaders.path_graph(data, undirected))
        for a, b in (("delta", "alpha"), ("alpha", "delta"), ("beta", "delta")):
            assert render_path(sb.shortest_path(a, b, undirected=undirected), a, b) == \
                render_path(jp.shortest_path(a, b, undirected=undirected), a, b), (a, b, undirected)


@pytest.mark.parametrize("mutate, why", [
    (lambda d: d.update(multigraph=True), "multigraph"),
    (lambda d: d.pop("multigraph"), "multigraph"),
    (lambda d: d["nodes"].append(dict(d["nodes"][0])), "duplicate node id"),
    (lambda d: d["links"].append({"source": "a", "target": "zz"}), "endpoint"),
    (lambda d: d["links"].append(dict(d["links"][1])), "parallel"),
    (lambda d: d["links"][0].update(_src="a"), "_src"),
    (lambda d: d["nodes"][0].update(label="a\0b"), "cannot hold"),
])
def test_refuses_unsupported_shapes(tmp_path, mutate, why):
    data = _data(False)
    mutate(data)
    with pytest.raises(Unsupported, match=why):
        snap_writer.publish_snapshot(data, tmp_path)
    assert not (tmp_path / "graph.snap").exists()


def test_writer_leaves_graph_json_untouched(tmp_path):
    src = tmp_path / "graph.json"
    src.write_text(json.dumps(_data(False)), encoding="utf-8")
    before = snap_writer.file_info(src)
    snap_writer.main(["x", str(src), str(tmp_path / "out")])
    assert snap_writer.file_info(src) == before
    assert (tmp_path / "out" / "graph.snap").exists()


@pytest.mark.skipif(not PSN.exists(), reason="psn_diadoc graph.json not present")
def test_writer_leaves_real_graph_json_untouched(tmp_path):
    before = snap_writer.file_info(PSN)
    snap_writer.main(["x", str(PSN), str(tmp_path)])
    assert snap_writer.file_info(PSN) == before
