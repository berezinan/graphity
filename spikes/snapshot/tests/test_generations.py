"""Task 4.2: generation per request, Arc hold, staleness (D7, D9)."""
import json
import os
import time

import pytest
from conftest import snap_of
from test_shapes import _data

import graphify_snap
import snap_writer
from graphify.query_backend import render_query
from snap_backend import SnapBackend
from snap_common import norm_fingerprint


def _relabel(data, suffix):
    data = json.loads(json.dumps(data))
    for n in data["nodes"]:
        n["label"] += suffix
    return data


def test_reader_follows_new_generation_and_old_mapping_survives(tmp_path):
    a = _data(False)
    sb = snap_of(a, tmp_path)
    assert sb.snap.generation == 1
    old = sb.snap                                   # a request in flight holds this
    first = render_query(sb.query("alpha"))
    snap_writer.publish_snapshot(_relabel(a, "x"), tmp_path)
    second = render_query(sb.query("alphax"))
    assert sb.snap.generation == 2
    assert "alphax" in second and first != second
    # the replaced generation still answers from its own mapping
    assert json.loads(old.value("n", "label", old.record("n", ["label"], 0)[0][0])) == "alpha"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.snap", "graph.snap.lock"]


def test_crashed_writer_leaves_published_generation_and_next_run_cleans(tmp_path):
    a = _data(False)
    sb = snap_of(a, tmp_path)
    stale = tmp_path / "graph.snap.99999.tmp"
    stale.write_bytes(b"half a snapshot")
    assert render_query(sb.query("alpha")).startswith("Traversal")
    assert graphify_snap.peek_generation(str(stale)) is None
    rep = snap_writer.publish_snapshot(a, tmp_path)
    assert rep["removed_tmp"] == ["graph.snap.99999.tmp"] and not stale.exists()


def test_stale_when_graph_json_rewritten(tmp_path):
    src = tmp_path / "graph.json"
    src.write_text(json.dumps(_data(False)), encoding="utf-8")
    with open(src, encoding="utf-8") as f:
        snap_writer.publish_snapshot(json.load(f), tmp_path, source=src)
    sb = SnapBackend(tmp_path / "graph.snap", graph_json=src)
    sb.query("alpha")
    assert sb.stale_warning is None
    time.sleep(0.01)
    src.write_text(json.dumps(_data(False)) + " ", encoding="utf-8")
    sb.query("alpha")
    assert sb.stale_warning and "graph.json changed" in sb.stale_warning


def test_stale_when_partition_changes_without_graph_json(tmp_path):
    data = _data(False)
    current = {"hash": snap_writer.partition_hash(data["nodes"]).hex()}
    snap_writer.publish_snapshot(data, tmp_path)
    sb = SnapBackend(tmp_path / "graph.snap", partition=lambda: current["hash"])
    sb.graph_stats()
    assert sb.stale_warning is None
    moved = json.loads(json.dumps(data["nodes"]))
    moved[0]["community"] = 7
    current["hash"] = snap_writer.partition_hash(moved).hex()
    sb.graph_stats()
    assert sb.stale_warning and "communities changed" in sb.stale_warning


@pytest.mark.parametrize("damage", ["version", "fingerprint", "truncate"])
def test_reader_refuses_incompatible_snapshot(tmp_path, damage):
    snap_of(_data(False), tmp_path)
    p = tmp_path / "graph.snap"
    fp = norm_fingerprint()
    if damage == "version":
        with open(p, "r+b") as f:
            f.seek(8)
            f.write((99).to_bytes(4, "little"))
    elif damage == "fingerprint":
        fp = bytes(32)
    else:
        # A copy: the published file is still mapped, and a mapped file
        # cannot be truncated below its mapped size (README §1.3).
        cut = tmp_path / "cut.snap"
        cut.write_bytes(p.read_bytes()[:-10])
        p = cut
    with pytest.raises(ValueError, match="rebuild the snapshot"):
        graphify_snap.Snap(str(p), fp)
