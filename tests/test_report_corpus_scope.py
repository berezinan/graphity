"""GRAPH_REPORT.md states which rules shaped the corpus.

Working out how an existing graph was scoped used to mean digging through shell
history: nothing in the outputs recorded whether `.gitignore` had been honored.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GRAPHIFY_")}
    env["GRAPHIFY_BACKEND"] = "json"
    return subprocess.run(
        [PYTHON, "-m", "graphify", *args], cwd=str(cwd),
        capture_output=True, text=True, env=env,
    )


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def keep(): return 1\n", encoding="utf-8")
    return corpus


def test_report_names_the_mode_and_its_source(tmp_path):
    corpus = _corpus(tmp_path)

    _run(["update", ".", "--gitignore"], cwd=corpus)

    report = (corpus / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "- Corpus scope: .gitignore honored (recorded in .graphify_build.json)" in report


def test_report_names_the_default_when_nothing_was_recorded(tmp_path):
    corpus = _corpus(tmp_path)

    _run(["update", "."], cwd=corpus)

    report = (corpus / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "- Corpus scope: .gitignore not honored (default)" in report


def test_report_lists_persisted_excludes(tmp_path):
    from graphify.watch import _write_build_config

    corpus = _corpus(tmp_path)
    (corpus / "vendor").mkdir()
    (corpus / "vendor" / "lib.py").write_text("def vendored(): pass\n", encoding="utf-8")
    _write_build_config(corpus / "graphify-out", excludes=["vendor"])

    _run(["update", "."], cwd=corpus)

    report = (corpus / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "--exclude: vendor" in report
