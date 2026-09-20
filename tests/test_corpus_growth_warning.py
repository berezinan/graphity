"""Corpus growth is announced, the mirror image of the mass-prune gate.

#2495 gates a corpus that collapses. Nothing watched the opposite direction, so
a widened scope — a deleted .graphifyignore, or the first rebuild after the
`.gitignore` default flipped — silently turned into a much larger graph and a
much larger semantic bill.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify.watch import warn_on_corpus_growth


def _manifest(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({name: {"mtime": 0, "ast_hash": "", "semantic_hash": ""} for name in names}),
        encoding="utf-8",
    )


def _detected(root: Path, names: list[str]) -> dict:
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n", encoding="utf-8")
    return {"files": {"code": [str(root / name) for name in names]}}


def test_growth_beyond_the_factor_warns_and_names_the_largest_dirs(tmp_path, capsys):
    manifest = tmp_path / "graphify-out" / "manifest.json"
    _manifest(manifest, ["app.py", "lib.py"])
    names = ["app.py", "lib.py"] + [f"vendor/v{i}.py" for i in range(6)] + ["docs/d1.py"]

    warn_on_corpus_growth(manifest, _detected(tmp_path, names), tmp_path)

    out = capsys.readouterr().out
    assert "corpus grew from 2 to 9 file(s)" in out
    assert "x4.5" in out
    assert "vendor/ (6)" in out
    assert "docs/ (1)" in out


def test_growth_within_the_factor_is_silent(tmp_path, capsys):
    manifest = tmp_path / "graphify-out" / "manifest.json"
    _manifest(manifest, ["app.py", "lib.py"])

    warn_on_corpus_growth(
        manifest, _detected(tmp_path, ["app.py", "lib.py", "extra.py", "more.py"]), tmp_path
    )

    assert capsys.readouterr().out == ""


def test_first_build_has_nothing_to_compare_against(tmp_path, capsys):
    """No manifest means no previous corpus — a first build never warns."""
    manifest = tmp_path / "graphify-out" / "manifest.json"

    warn_on_corpus_growth(
        manifest, _detected(tmp_path, [f"f{i}.py" for i in range(50)]), tmp_path
    )

    assert capsys.readouterr().out == ""


def test_unreadable_manifest_does_not_break_the_rebuild(tmp_path, capsys):
    manifest = tmp_path / "graphify-out" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{ not json", encoding="utf-8")

    warn_on_corpus_growth(
        manifest, _detected(tmp_path, [f"f{i}.py" for i in range(50)]), tmp_path
    )

    assert capsys.readouterr().out == ""
