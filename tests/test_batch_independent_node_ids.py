"""A solution keeps its own node, with the same id on full and incremental runs
(fork delta, batch-independent-node-ids).

Found on INVINA_DB 2026-11-15: `db/newinv/newinv.slnx` had no node in the graph,
neither after a full run nor after an incremental one. The solution's file node and
its reference to `newinv.sqlproj` share the extension-stripped id; dedup moves the
solution to the extension-qualified id, but a document's semantic node already held
that id, the rename was cancelled and the solution's node was dropped.

`.sqlproj` is not a corpus file (no extractor), so it never enters an extraction
batch: references to it are canonicalised late and their collision with the
solution is settled by dedup over the whole graph — which is what keeps the ids
independent of the batch. These tests pin both halves.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify import cache as _cache

SLNX = "db/app/app.slnx"
ROOT_SLNX = "all.slnx"
SQLPROJ = "db/app/app.sqlproj"
SEMANTIC_ID = "db_app_app_slnx"


def _reset_stat_index() -> None:
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "db" / "app").mkdir(parents=True)
    (proj / SLNX).write_text('<Solution><Project Path="app.sqlproj" /></Solution>\n', encoding="utf-8")
    (proj / ROOT_SLNX).write_text('<Solution><Project Path="db/app/app.sqlproj" /></Solution>\n',
                                  encoding="utf-8")
    (proj / SQLPROJ).write_text("<Project />\n", encoding="utf-8")
    (proj / "main.py").write_text("x = 1\n", encoding="utf-8")
    (proj / "docs").mkdir()
    (proj / "docs" / "mapping.md").write_text("# Mapping\n\napp.slnx\n", encoding="utf-8")
    return proj


def _ids_by_source(nodes) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for n in nodes:
        out.setdefault(Path(str(n.get("source_file", ""))).as_posix(), set()).add(n["id"])
    return out


def test_reference_ids_do_not_depend_on_the_batch(tmp_path):
    """Both solutions reference app.sqlproj. Whichever of them a batch holds, the
    reference keeps one id: an early canonicalisation salted it only when both
    were in the batch, and a reference node is never replaced on merge, so the
    other spelling stayed in the graph as a duplicate."""
    from graphify.extract import extract

    proj = _project(tmp_path)
    seen = []
    for i, batch in enumerate(([SLNX], [ROOT_SLNX], [ROOT_SLNX, SLNX])):
        _reset_stat_index()
        out = extract([proj / f for f in batch], root=proj, cache_root=tmp_path / f"c{i}")
        seen.append(_ids_by_source(out["nodes"]))
    assert len({frozenset(s[SQLPROJ]) for s in seen}) == 1, [s[SQLPROJ] for s in seen]
    assert len({frozenset(s[SLNX]) for s in seen if SLNX in s}) == 1


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


def _run(proj: Path, monkeypatch, capsys, *extra: str) -> int:
    import graphify.__main__ as mainmod

    _reset_stat_index()
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    # The clustered path, as on INVINA_DB: the --no-cluster raw dump collapses
    # same-id nodes last-writer-wins (build.dedupe_nodes) without dedup.
    monkeypatch.setattr(mainmod.sys, "argv", [
        "graphify", "extract", str(proj), "--code-only", *extra])
    try:
        mainmod.main()
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    capsys.readouterr()
    return code


def test_solution_node_survives_a_taken_id_on_full_and_incremental_runs(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path)
    assert _run(proj, monkeypatch, capsys) == 0
    gpath = proj / "graphify-out" / "graph.json"

    # The state INVINA_DB was in: the solution has no node, and a document's
    # semantic node holds the extension-qualified id dedup falls back to.
    graph = json.loads(gpath.read_text(encoding="utf-8"))
    graph["nodes"] = [n for n in graph["nodes"]
                      if Path(str(n.get("source_file", ""))).as_posix() != SLNX]
    # Not a file label: a label naming the file alone is folded into the file's
    # AST node by the #3344 ghost pass. INVINA_DB's node read "Solution
    # BSProjects (warehouse + newinv) — newinv.slnx, не объект базы".
    graph["nodes"].append({"id": SEMANTIC_ID, "label": "Solution app — app.slnx",
                           "file_type": "document",
                           "source_file": "docs/mapping.md", "_origin": "semantic"})
    gpath.write_text(json.dumps(graph), encoding="utf-8")

    (proj / SLNX).write_text('<Solution>\n  <Project Path="app.sqlproj" />\n</Solution>\n', encoding="utf-8")
    assert _run(proj, monkeypatch, capsys) == 0
    inc_nodes = json.loads(gpath.read_text(encoding="utf-8"))["nodes"]
    inc = _ids_by_source(inc_nodes)

    assert _run(proj, monkeypatch, capsys, "--force") == 0
    full_nodes = json.loads(gpath.read_text(encoding="utf-8"))["nodes"]
    full = _ids_by_source(full_nodes)

    assert inc.get(SLNX), "the solution has its own node after an incremental run"
    assert inc[SLNX] == full[SLNX], "incremental and full runs give the solution the same id"
    assert inc[SQLPROJ] == full[SQLPROJ] and len(full[SQLPROJ]) == 1, "one node for the project"
    for nodes in (inc_nodes, full_nodes):
        assert SEMANTIC_ID in {n["id"] for n in nodes}, "the document's node stays"
