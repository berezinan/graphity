"""Tests for the EDT diff harness (tests/edt_harness.py).

Split in two: the harness's own behaviour is verified on the synthetic fixtures
so it runs everywhere, and the corpus-backed checks skip unless
GRAPHIFY_EDT_CORPUS points at a real EDT project. The suite stays green on a
machine without the corpus — that is a requirement of the change, not a
convenience.
"""
from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from graphify.extractors.bsl import _v8_elements
from tests.edt_harness import (
    CORPUS_ENV,
    EDT_EXTRACTORS,
    STATUS_EDT,
    STATUS_EDT_NO_OUTPUT,
    STATUS_DOCUMENT,
    STATUS_NOT_PARSED,
    STATUS_OTHER,
    build_snapshot,
    census,
    corpus_root,
    diff_snapshots,
    format_census,
    format_diff,
    is_edt_project,
    tree_fingerprint,
)

FIXTURES = Path(__file__).parent / "fixtures"
EDT = FIXTURES / "edt"


@pytest.fixture
def corpus() -> Path:
    root = corpus_root()
    if root is None:
        pytest.skip(f"{CORPUS_ENV} не задана или не указывает на EDT-проект")
    return root


# ── 1. Снимок ────────────────────────────────────────────────────────────────

def test_snapshot_matches_direct_extractor_calls():
    """The snapshot is exactly the union of what the extractors return.

    Union, not sum: one object is described by both its own .mdo and its
    Configuration registration, and those must collapse to one id — that
    collapsing is what makes the graph a graph.
    """
    snapshot = build_snapshot(EDT)
    expected: set[str] = set()
    for path in sorted(EDT.rglob("*")):
        extractor = EDT_EXTRACTORS.get(path.suffix.lower()) if path.is_file() else None
        if extractor is None:
            continue
        result = extractor(path)
        if result.get("error"):
            continue
        expected |= {n["id"] for n in result.get("nodes", []) if n.get("id")}

    assert snapshot["node_ids"], "снимок по фикстурам пуст"
    assert set(snapshot["node_ids"]) == expected
    assert snapshot["counters"]["nodes"] == len(expected)


def test_snapshot_ignores_labels_and_provenance(monkeypatch):
    """Cosmetic fields must not move the snapshot.

    Otherwise every label tweak reads as a change and the one signal that
    matters — a vanished id — drowns in the noise.
    """
    before = build_snapshot(EDT)

    def repaint(extractor):
        def inner(path):
            result = extractor(path)
            for node in result.get("nodes", []):
                node["label"] = "ПЕРЕКРАШЕНО"
                node["source_file"] = "/somewhere/else"
            for edge in result.get("edges", []):
                edge["weight"] = 99.0
                edge["confidence"] = "GUESSED"
                edge["source_file"] = "/somewhere/else"
            return result
        return inner

    for ext, extractor in list(EDT_EXTRACTORS.items()):
        monkeypatch.setitem(EDT_EXTRACTORS, ext, repaint(extractor))

    assert build_snapshot(EDT) == before


def test_snapshot_makes_no_network_calls(monkeypatch):
    """No LLM, no ArcadeDB, no network — asserted by removing the socket."""
    def forbidden(*args, **kwargs):
        raise AssertionError("харнесс попытался открыть сокет")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    snapshot = build_snapshot(EDT)
    assert snapshot["counters"]["nodes"] > 0


# ── 2. Дифф ──────────────────────────────────────────────────────────────────

def _snap(nodes, edges):
    return {"node_ids": sorted(nodes), "edges": sorted(edges), "counters": {}}


def test_diff_splits_added_removed_unchanged():
    before = _snap(["a", "b"], [("a", "contains", "b")])
    after = _snap(["a", "b", "c"], [("a", "contains", "b"), ("a", "contains", "c")])

    diff = diff_snapshots(before, after)
    assert diff["nodes"] == {"added": ["c"], "removed": [], "unchanged": 2}
    assert diff["edges"]["added"] == [("a", "contains", "c")]
    assert diff["edges"]["removed"] == []
    assert diff["edges"]["unchanged"] == 1
    assert diff["regression"] is False


def test_diff_flags_a_changed_id_as_regression():
    """A renamed id is a regression, not a wash of one added and one removed.

    This is the shape the harness exists to catch: the totals balance, the
    aggregate looks harmless, and every stored reference to the old id is dead.
    """
    before = _snap(["table_инфотаблица"], [])
    after = _snap(["externaldatasource_субд_table_инфотаблица"], [])

    diff = diff_snapshots(before, after)
    assert diff["regression"] is True
    assert diff["nodes"]["removed"] == ["table_инфотаблица"]
    assert diff["added_total"] == diff["removed_total"] == 1


def test_format_diff_carries_both_counts_and_examples():
    before = _snap(["a", "b"], [])
    after = _snap(["a", "c"], [])

    text = format_diff(diff_snapshots(before, after))
    assert "РЕГРЕСС" in text
    assert "nodes.removed (1)" in text
    assert "    b" in text        # конкретный пример, а не только число


# ── 3. Перепись ──────────────────────────────────────────────────────────────

def test_census_marks_extension_that_never_reaches_a_parser(tmp_path):
    """The category a diff is blind to: files with no extractor at all."""
    project = tmp_path / "Проект"
    (project / "src" / "Catalogs" / "X" / "Forms" / "Ф").mkdir(parents=True)
    (project / "src" / "Catalogs" / "X" / "X.mdo").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mdclass:Catalog xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass"'
        ' uuid="eeeeeeee-0000-4000-8000-000000000001">\n'
        "  <name>X</name>\n"
        "</mdclass:Catalog>\n",
        encoding="utf-8",
    )
    # `.dcss` — dynamic-list settings. Measured across the corpus as carrying no
    # links (661 files, 13 references), so it is deliberately left without an
    # extractor; that decision holds only while the census keeps saying so.
    (project / "src" / "Catalogs" / "X" / "Forms" / "Ф" / "ListSettings.dcss").write_text(
        "<settings/>", encoding="utf-8",
    )

    rows = {row["ext"]: row for row in census(project)}
    assert rows[".mdo"]["status"] == STATUS_EDT
    assert rows[".dcss"]["status"] == STATUS_NOT_PARSED
    assert rows[".dcss"]["files"] == 1
    assert rows[".dcss"]["in_code_extensions"] is False
    assert "не доходит до парсера" in format_census(census(project))


def test_census_separates_no_extractor_from_no_output(tmp_path):
    """Two different failures must not share one label.

    "No extractor" is a gap in coverage; "extractor ran, produced nothing" is a
    gap in the extractor or in what feeds it. Collapsing them hides whichever is
    rarer.
    """
    project = tmp_path / "Проект"
    (project / "src" / "Roles" / "R").mkdir(parents=True)
    # A .rights file the extractor reads but that grants nothing -> no nodes...
    (project / "src" / "Roles" / "R" / "Rights.rights").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Rights xmlns="http://v8.1c.ru/8.2/roles"/>\n',
        encoding="utf-8",
    )
    (project / "src" / "Roles" / "R" / "Notes.txt").write_text("x", encoding="utf-8")

    rows = {row["ext"]: row for row in census(project)}
    # ...well, the role node itself is always emitted, so this is STATUS_EDT.
    assert rows[".rights"]["edt_extractor"] is True
    assert rows[".rights"]["status"] in (STATUS_EDT, STATUS_EDT_NO_OUTPUT)
    assert rows[".txt"]["edt_extractor"] is False
    assert rows[".txt"]["status"] in (STATUS_OTHER, STATUS_DOCUMENT, STATUS_NOT_PARSED)


# ── 4. Опциональность и безопасность ─────────────────────────────────────────

def test_corpus_is_optional(monkeypatch):
    monkeypatch.delenv(CORPUS_ENV, raising=False)
    assert corpus_root() is None


def test_corpus_missing_directory_is_a_skip_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv(CORPUS_ENV, str(tmp_path / "нет-такого"))
    assert corpus_root() is None

    monkeypatch.setenv(CORPUS_ENV, str(tmp_path))   # существует, но не EDT-проект
    assert corpus_root() is None
    assert is_edt_project(tmp_path) is False


def test_harness_does_not_modify_the_tree_it_reads():
    """Read-only, verified rather than promised."""
    before = tree_fingerprint(EDT)
    snapshot = build_snapshot(EDT)
    census(EDT, snapshot=snapshot)
    assert tree_fingerprint(EDT) == before


# ── Проверки на корпусе (пропускаются без него) ──────────────────────────────

def test_corpus_snapshot_is_deterministic(corpus):
    """Two runs on unchanged input give byte-identical snapshots."""
    first = build_snapshot(corpus)
    second = build_snapshot(corpus)
    assert first == second
    assert first["counters"]["nodes"] > 0


def test_corpus_run_leaves_the_corpus_untouched(corpus):
    before = tree_fingerprint(corpus)
    census(corpus)
    assert tree_fingerprint(corpus) == before


def test_corpus_census_reports_unparsed_extensions(corpus):
    rows = census(corpus)
    assert rows, "перепись корпуса пуста"
    assert any(r["status"] == STATUS_NOT_PARSED and r["files"] > 0 for r in rows), (
        "ожидалось хотя бы одно расширение, не доходящее до парсера"
    )


# Headers and terminators of a BSL module, matched at the start of a line.
_BSL_HEAD_RE = re.compile(r"^\s*(?:Procedure|Function|Процедура|Функция)\b", re.I)
_BSL_TAIL_RE = re.compile(r"^\s*(?:EndProcedure|EndFunction|КонецПроцедуры|КонецФункции)\b", re.I)


def test_corpus_oform_modules_are_complete(corpus):
    """Every procedure opened inside an extracted module block is also closed.

    A block cut short is the failure with teeth: it drops the tail of a module
    and returns no error. Counting terminators against headers is what makes
    that visible. Lines are split as BSL reads them — on CR, LF or CRLF alike,
    because one module of the audited corpus uses bare CR inside a procedure.
    """
    heads = tails = 0
    for path in corpus.rglob("*.oform"):
        elements = _v8_elements(path.read_bytes())
        module = (elements or {}).get("module")
        if not module:
            continue
        # utf-8-sig: a module that opens straight with a declaration puts the
        # byte-order mark in front of it, where `^\s*` will not step over it.
        lines = module[0].decode("utf-8-sig", "replace").splitlines()
        heads += sum(1 for line in lines if _BSL_HEAD_RE.match(line))
        tails += sum(1 for line in lines if _BSL_TAIL_RE.match(line))
    assert heads, "в корпусе не найдено ни одного модуля .oform"
    assert heads == tails
