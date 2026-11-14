"""Rebuild stability (fork delta, stable-incremental-rebuild).

Measured on INVINA_DB 2026-11-15: an incremental ``extract`` with no changed
document refused to write because the stored graph shrank on rebuild. Two
causes: the #1145 ghost pass in ``build_from_json`` merged only one ghost per
(source_file, label) key per build, and the #2543 heal re-queued a file whose
nodes are lost in the build on every run.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from graphify import cache as _cache
from graphify.build import build_from_json, take_shrink_accounting

DOC = "docs/dictionary.md"


def _sem(nid: str, label: str, loc=None, stub=False) -> dict:
    n = {"id": nid, "label": label, "file_type": "code", "source_file": DOC,
         "source_location": loc, "_origin": "semantic"}
    if stub:
        n["metadata"] = {"graphify_reference_stub": True}
    return n


def _pair_with_two_ghosts() -> dict:
    nodes = [
        _sem("docs_dictionary_pall", "dbo.PALL", loc="db/Tables/PALL.sql:1"),
        _sem("docs_dictionary_dbo_pall", "dbo.PALL"),
        _sem("docs_dictionary_ref_dbo_pall", "dbo.PALL", stub=True),
        _sem("docs_dictionary_cells", "dbo.cells", loc="db/Tables/cells.sql:1"),
    ]
    edges = [
        {"source": "docs_dictionary_cells", "target": "docs_dictionary_dbo_pall",
         "relation": "references", "source_file": DOC, "_origin": "semantic"},
        {"source": "docs_dictionary_cells", "target": "docs_dictionary_ref_dbo_pall",
         "relation": "references", "source_file": DOC, "_origin": "semantic"},
    ]
    return {"nodes": nodes, "edges": edges, "hyperedges": []}


def _as_extraction(G) -> dict:
    return {
        "nodes": [{"id": n, **a} for n, a in G.nodes(data=True)],
        "edges": [{"source": a.get("_src", u), "target": a.get("_tgt", v),
                   **{k: x for k, x in a.items() if k not in ("_src", "_tgt")}}
                  for u, v, a in G.edges(data=True)],
        "hyperedges": G.graph.get("hyperedges", []),
    }


def test_every_ghost_on_a_key_merges_in_one_build():
    G1 = build_from_json(copy.deepcopy(_pair_with_two_ghosts()), directed=True, dedup=True)
    G2 = build_from_json(_as_extraction(G1), directed=True, dedup=True)

    assert set(G1.nodes) == {"docs_dictionary_pall", "docs_dictionary_cells"}
    assert set(G2.nodes) == set(G1.nodes)
    assert {(a.get("_src", u), a.get("_tgt", v)) for u, v, a in G1.edges(data=True)} == {
        ("docs_dictionary_cells", "docs_dictionary_pall")
    }


def test_ghost_merges_are_counted_apart_from_dedup():
    G = build_from_json(copy.deepcopy(_pair_with_two_ghosts()), directed=True, dedup=True)

    dedup, pruned, ghosts = take_shrink_accounting(G)

    assert (dedup, pruned, ghosts) == (0, 0, 2)
    assert "_ghost_collapsed" not in G.graph


# --- #2543 heal must not re-queue a file whose nodes the build drops -----------


def _reset_stat_index() -> None:
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}


def _run(proj: Path, monkeypatch, capsys, *extra: str) -> tuple[int, str]:
    import graphify.__main__ as mainmod

    _reset_stat_index()
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", [
        "graphify", "extract", str(proj), "--code-only", "--no-cluster", *extra])
    try:
        mainmod.main()
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, capsys.readouterr().out


def test_nodeless_heal_is_recorded_and_not_repeated(tmp_path, monkeypatch, capsys):
    from graphify.cli import _healed_still_nodeless

    proj = tmp_path / "proj"
    (proj / "db" / "app").mkdir(parents=True)
    (proj / "db" / "app" / "app.slnx").write_text(
        '<Solution><Project Path="app.sqlproj" /></Solution>\n', encoding="utf-8")
    (proj / "main.py").write_text("x = 1\n", encoding="utf-8")
    assert _run(proj, monkeypatch, capsys)[0] == 0
    gpath = proj / "graphify-out" / "graph.json"
    graph = json.loads(gpath.read_text(encoding="utf-8"))
    # The state INVINA_DB was left in: the solution file's node is gone from the
    # stored graph although its manifest row is stamped.
    graph["nodes"] = [n for n in graph["nodes"]
                      if Path(str(n.get("source_file", ""))).as_posix() != "db/app/app.slnx"]
    gpath.write_text(json.dumps(graph), encoding="utf-8")

    monkeypatch.setattr("graphify.cli._healed_still_nodeless",
                        lambda healed, nodes, root: list(healed))
    code, out = _run(proj, monkeypatch, capsys)
    assert code == 0
    assert "re-queuing 1 manifest-stamped code file" in out
    assert "all lost in the build" in out and "app.slnx" in out
    manifest = json.loads((proj / "graphify-out" / "manifest.json").read_text(encoding="utf-8"))
    row = next(v for k, v in manifest.items() if k.replace("\\", "/").endswith("app.slnx"))
    assert row.get("ast_empty") is True

    monkeypatch.setattr("graphify.cli._healed_still_nodeless", _healed_still_nodeless)
    code, out = _run(proj, monkeypatch, capsys)
    assert code == 0
    assert "re-queuing" not in out

    (proj / "db" / "app" / "app.slnx").write_text(
        '<Solution><Project Path="app.sqlproj" />\n</Solution>\n', encoding="utf-8")
    code, out = _run(proj, monkeypatch, capsys)
    assert code == 0
    assert "1 code" in out  # a changed file is extracted again


def test_healed_still_nodeless_matches_relative_and_absolute(tmp_path):
    from graphify.cli import _healed_still_nodeless

    root = tmp_path
    (root / "a.slnx").write_text("x", encoding="utf-8")
    (root / "b.slnx").write_text("x", encoding="utf-8")
    healed = [str(root / "a.slnx"), str(root / "b.slnx")]

    lost = _healed_still_nodeless(healed, [{"source_file": "a.slnx"}], root)

    assert lost == [str(root / "b.slnx")]
