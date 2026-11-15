"""A node's id does not depend on which other files are extracted with it
(fork delta, batch-independent-node-ids).

Found on INVINA_DB 2026-11-15: `newinv.slnx` extracted alone minted its own
file node and its reference to `newinv.sqlproj` under one id; extracted together
with the `.sqlproj` the pair was salted apart. An incremental run (only the
changed `.slnx`) therefore lost the solution's node in the build.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify import cache as _cache

SLNX = "db/app/app.slnx"
SQLPROJ = "db/app/app.sqlproj"


def _reset_stat_index() -> None:
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "db" / "app").mkdir(parents=True)
    (proj / SLNX).write_text('<Solution><Project Path="app.sqlproj" /></Solution>\n', encoding="utf-8")
    (proj / SQLPROJ).write_text("<Project />\n", encoding="utf-8")
    (proj / "main.py").write_text("x = 1\n", encoding="utf-8")
    return proj


def _ids_by_source(nodes) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for n in nodes:
        out.setdefault(Path(str(n.get("source_file", ""))).as_posix(), set()).add(n["id"])
    return out


def test_solution_ids_do_not_depend_on_the_batch(tmp_path):
    from graphify.extract import extract

    proj = _project(tmp_path)
    _reset_stat_index()
    alone = extract([proj / SLNX], root=proj, cache_root=tmp_path / "c1")
    _reset_stat_index()
    together = extract([proj / SLNX, proj / SQLPROJ], root=proj, cache_root=tmp_path / "c2")

    a, t = _ids_by_source(alone["nodes"]), _ids_by_source(together["nodes"])
    assert a[SLNX] == t[SLNX]
    assert a[SQLPROJ] == t[SQLPROJ]
    assert a[SLNX].isdisjoint(a[SQLPROJ]), "the solution and its project must not share an id"


def test_taken_extension_id_is_salted_not_dropped():
    """x.py and x.html share `pkg_x`; x.html would move to `pkg_x_html`, but a
    document's node already holds it. The rename used to be cancelled and x.html's
    node was lost in the collapse; now it takes a path-salted id."""
    from graphify.dedup import deduplicate_entities

    nodes = [
        {"id": "pkg_x", "label": "x.py", "file_type": "code", "source_file": "pkg/x.py", "_origin": "ast"},
        {"id": "pkg_x", "label": "x.html", "file_type": "code", "source_file": "pkg/x.html", "_origin": "ast"},
        {"id": "pkg_x_html", "label": "x.html", "file_type": "document", "source_file": "docs/a.md",
         "_origin": "semantic"},
    ]
    out, _ = deduplicate_entities([dict(n) for n in nodes], [], communities={}, root=None)
    by_sf = {n["source_file"]: n["id"] for n in out}

    assert set(by_sf) == {"pkg/x.py", "pkg/x.html", "docs/a.md"}
    assert by_sf["pkg/x.py"] == "pkg_x" and by_sf["docs/a.md"] == "pkg_x_html"
    assert by_sf["pkg/x.html"].startswith("pkg_x_html_") and len(set(by_sf.values())) == 3
    again, _ = deduplicate_entities([dict(n) for n in nodes], [], communities={}, root=None)
    assert {n["source_file"]: n["id"] for n in again} == by_sf, "the salt is deterministic"


def _run(proj: Path, monkeypatch, capsys) -> int:
    import graphify.__main__ as mainmod

    _reset_stat_index()
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", [
        "graphify", "extract", str(proj), "--code-only", "--no-cluster"])
    try:
        mainmod.main()
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    capsys.readouterr()
    return code


def test_incremental_build_keeps_the_solution_node(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path)
    assert _run(proj, monkeypatch, capsys) == 0
    gpath = proj / "graphify-out" / "graph.json"
    full = _ids_by_source(json.loads(gpath.read_text(encoding="utf-8"))["nodes"])
    assert full.get(SLNX), "a full scan gives the solution a node"

    # The state INVINA_DB was in: a document's semantic node holds the id dedup
    # would fall back to for the solution. The document exists, so the node is
    # not pruned as a stale source.
    (proj / "docs").mkdir()
    (proj / "docs" / "mapping.md").write_text("# Mapping\n\napp.slnx\n", encoding="utf-8")
    graph = json.loads(gpath.read_text(encoding="utf-8"))
    graph["nodes"].append({"id": "db_app_app_slnx", "label": "app.slnx", "file_type": "document",
                           "source_file": "docs/mapping.md", "_origin": "semantic"})
    gpath.write_text(json.dumps(graph), encoding="utf-8")

    (proj / SLNX).write_text('<Solution>\n  <Project Path="app.sqlproj" />\n</Solution>\n', encoding="utf-8")
    assert _run(proj, monkeypatch, capsys) == 0

    nodes = json.loads(gpath.read_text(encoding="utf-8"))["nodes"]
    inc = _ids_by_source(nodes)
    assert inc.get(SLNX) == full[SLNX], "incremental and full scans give the solution the same id"
    assert "db_app_app_slnx" in {n["id"] for n in nodes}, "the semantic node stays"
