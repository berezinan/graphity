"""The writer reads graph.json element by element (change streaming-snapshot-writer).

The snapshot built from the file, in either layout, must be byte-identical to the
one built from ``json.load`` of the same file; refusals must be the same; the
recorded source must describe exactly the bytes that were read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

graphify_snap = pytest.importorskip("graphify_snap", reason="graphify-snap wheel not installed")
if getattr(graphify_snap, "API_VERSION", None) != 2 or not hasattr(graphify_snap, "GraphJson"):
    pytest.skip("graphify-snap wheel older than API 2", allow_module_level=True)

from graphify.snap_backend import SnapBackend  # noqa: E402
from graphify.snap_store import SnapshotUnavailable, Unsupported, snapshot_path_for  # noqa: E402
from graphify.snap_writer import (  # noqa: E402
    _FileGraph,
    build_snapshot,
    encode,
    publish_snapshot,
    snap_build_command,
)

PSN = Path(r"C:\_dev\ANB\AI\psn_diadoc_4_67_3_Pravka\graphify-out\graph.json")
NO_SOURCE = (0, 0, bytes(32))


def _members(edges_key: str = "links") -> list[tuple[str, object]]:
    """Top-level members in an order to_json never writes, with a repeated key."""
    nodes = [
        {"id": "a", "label": 'say "hi" \\ there', "source_file": "src/a.py", "community": 0,
         "attributes": {"tags": ["x", {"k": None}]}},
        {"id": "b", "label": "Номенклатура 😀", "source_file": "src/b.bsl", "community": 0,
         "source_location": "L1", "norm_label": "nomenklatura"},
        {"id": "c", "label": "gamma()", "source_file": "src/c.py", "community": 1, "rationale": "why"},
        {"id": "d", "label": "delta", "community": 1, "community_name": "D"},
    ]
    links = [
        {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED", "weight": 1.5},
        {"source": "b", "target": "c", "relation": "uses", "confidence": "INFERRED", "weight": float("nan")},
        {"source": "c", "target": "a", "relation": "calls", "context": "call", "extra": [1, [2, {}]]},
        {"source": "d", "target": "c", "relation": "contains"},
    ]
    return [
        ("directed", True),
        (edges_key, links),
        ("multigraph", False),
        ("nodes", nodes),
        ("directed", False),
        ("built_at_commit", "abc123"),
        ("graph", {"hyperedges": [{"id": "h", "nodes": ["a", "b"]}]}),
    ]


def _indented(members) -> str:
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, indent=2, ensure_ascii=False)}" for k, v in members)
    return "{\n" + body + "\n}\n"


def _one_line(members) -> str:
    return "{" + ",".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}" for k, v in members) + "}"


def _snap_from_file(path: Path, out: Path) -> bytes:
    fg = _FileGraph(path)
    fg.source = NO_SOURCE  # same source for every variant: only the encoding is compared
    encode(fg, out, generation=1)
    return out.read_bytes()


def _snap_from_dict(path: Path, out: Path) -> bytes:
    with open(path, encoding="utf-8") as f:
        encode(json.load(f), out, generation=1)
    return out.read_bytes()


@pytest.mark.parametrize("edges_key", ["links", "edges"])
def test_layouts_and_json_load_give_identical_snapshots(tmp_path, edges_key):
    members = _members(edges_key)
    ind, one = tmp_path / "indented.json", tmp_path / "one.json"
    ind.write_text(_indented(members), encoding="utf-8")
    one.write_text(_one_line(members), encoding="utf-8")
    want = _snap_from_dict(ind, tmp_path / "dict.snap")
    assert _snap_from_file(ind, tmp_path / "ind.snap") == want
    assert _snap_from_file(one, tmp_path / "one.snap") == want
    assert _snap_from_dict(one, tmp_path / "dict_one.snap") == want


def test_repeated_key_resolves_like_json_load(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(_one_line(_members()), encoding="utf-8")
    fg = _FileGraph(p)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    assert fg.get("directed") is False and data["directed"] is False
    assert list(fg.rest()) == [k for k in data if k not in ("nodes", "links", "edges")]


@pytest.mark.skipif(not PSN.exists(), reason="psn_diadoc graph.json not present")
def test_real_one_line_graph(tmp_path):
    assert PSN.read_bytes()[:2] == b'{"', "psn_diadoc graph.json is expected to be one line (to_json)"
    assert _snap_from_file(PSN, tmp_path / "file.snap") == _snap_from_dict(PSN, tmp_path / "dict.snap")


def _graph() -> dict:
    nodes = [{"id": i, "label": lbl, "source_file": f"{lbl}.py", "community": 0}
             for i, lbl in (("a", "alpha"), ("b", "beta"), ("c", "gamma"))]
    links = [{"source": "a", "target": "b", "relation": "calls"},
             {"source": "b", "target": "c", "relation": "uses"}]
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


@pytest.mark.parametrize("mutate, why", [
    (lambda d: d.update(multigraph=True), "multigraph"),
    (lambda d: d.pop("multigraph"), "multigraph"),
    (lambda d: d["nodes"].append(dict(d["nodes"][0])), "duplicate node id"),
    (lambda d: d["nodes"].append({"id": 7}), "not a string"),
    (lambda d: d["links"].append({"source": "a", "target": "zz"}), "endpoint"),
    (lambda d: d["links"].append(dict(d["links"][0], relation="other")), "parallel"),
    (lambda d: d["links"][0].update(_src="a"), "_src"),
    (lambda d: d["nodes"][0].update(label="a\0b"), "NUL"),
    (lambda d: d["links"][0].update(context="x\0"), "NUL"),
])
def test_refusals_from_a_one_line_file_match_the_dict_path(tmp_path, mutate, why):
    data = _graph()
    mutate(data)
    with pytest.raises(Unsupported) as from_dict:
        encode(data, tmp_path / "dict.snap")
    p = tmp_path / "graphify-out" / "graph.json"
    p.parent.mkdir()
    p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(Unsupported, match=why) as from_file:
        build_snapshot(p)
    assert str(from_file.value) == str(from_dict.value)
    assert "GRAPHIFY_GRAPH_STORE=json" in str(from_file.value)
    assert not snapshot_path_for(p).exists()
    assert not list(p.parent.glob("*.tmp"))


@pytest.mark.parametrize("damage, where", [
    (lambda s: s[: s.index('"links"') + 30], "truncated"),              # cut inside `links`
    (lambda s: s.replace('},{"id":"b"', '} x {"id":"b"', 1), "between array elements"),
    (lambda s: s.replace('"relation":"uses"}', '"relation":"uses",}', 1), "damaged near byte"),
    (lambda s: s + "garbage", "extra data"),
])
def test_damaged_file_is_refused_with_an_offset(tmp_path, capsys, damage, where):
    p = tmp_path / "graphify-out" / "graph.json"
    p.parent.mkdir()
    p.write_text(damage(json.dumps(_graph(), separators=(",", ":"))), encoding="utf-8")
    with pytest.raises(ValueError, match=where):
        build_snapshot(p)
    assert snap_build_command(["build", str(p)]) == 1
    err = capsys.readouterr().err
    assert "graph.json is damaged" in err and "byte" in err
    assert not snapshot_path_for(p).exists()
    assert not list(p.parent.glob("*.tmp"))


def test_source_describes_the_bytes_read(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(_graph(), separators=(",", ":")), encoding="utf-8")
    size, mtime_ns, sha = graphify_snap.GraphJson(str(p)).source
    st = os.stat(p)
    assert (size, mtime_ns) == (st.st_size, st.st_mtime_ns)
    assert bytes(sha) == hashlib.sha256(p.read_bytes()).digest()


@pytest.mark.skipif(not PSN.exists(), reason="psn_diadoc graph.json not present")
def test_source_of_a_real_graph(tmp_path):
    size, mtime_ns, _sha = graphify_snap.GraphJson(str(PSN)).source
    st = os.stat(PSN)
    assert (size, mtime_ns) == (st.st_size, st.st_mtime_ns)


def test_rewrite_while_reading_succeeds_and_the_snapshot_is_stale(tmp_path):
    """graphify rewrites graph.json while a writer holds it open.

    The rewrite goes through ``paths.write_json_atomic`` (``to_json``, the
    ``--no-cluster`` dump, the global graph): on Windows a plain ``os.replace``
    over a file another handle holds open fails with error 5, and
    ``os_replace_with_fallback`` then renames the old file aside, which the
    writer's share-delete handle allows. The writer keeps reading the old bytes.
    """
    from graphify.paths import write_json_atomic

    out = tmp_path / "graphify-out"
    out.mkdir()
    p = out / "graph.json"
    old = _graph()
    p.write_text(json.dumps(old, separators=(",", ":")), encoding="utf-8")
    fg = _FileGraph(p)  # pass 1 done, handle open
    new = _graph()
    new["nodes"].append({"id": "z", "label": "zeta", "source_file": "z.py"})
    write_json_atomic(p, new, compact=True)
    assert json.loads(p.read_text(encoding="utf-8")) == new
    publish_snapshot(fg, out, name="graph.snap")
    snap = snapshot_path_for(p)
    assert SnapBackend(snap).graph_stats().nodes == len(old["nodes"])  # built from the old bytes
    with pytest.raises(SnapshotUnavailable, match="stale"):
        SnapBackend(snap, graph_json=p)
