"""Corpus mode (`--gitignore` / `--no-gitignore`) as a recorded graph setting.

The mode used to be an extract-only flag whose value was written after the
database probe, so `graphify update . --no-gitignore` died with "unknown update
option" and an extract that failed the probe left no trace of what the user
asked for — the next rebuild silently fell back to the default.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(args: list[str], cwd: Path, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GRAPHIFY_")}
    env["GRAPHIFY_BACKEND"] = "json"
    env.update(env_overrides)
    return subprocess.run(
        [PYTHON, "-m", "graphify", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def _build_config(corpus: Path) -> dict:
    path = corpus / "graphify-out" / ".graphify_build.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    generated = corpus / "generated"
    generated.mkdir(parents=True)
    (corpus / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (corpus / "app.py").write_text("def keep(): return 1\n", encoding="utf-8")
    (generated / "gen.py").write_text("def generated(): return 2\n", encoding="utf-8")
    return corpus


def test_update_accepts_the_mode_flags(tmp_path):
    """`update --gitignore` must run, not exit 2 on argument parsing."""
    corpus = _corpus(tmp_path)

    result = _run(["update", ".", "--gitignore", "--no-cluster"], cwd=corpus)

    assert "unknown update option" not in result.stderr
    assert result.returncode == 0, result.stderr


def test_update_records_the_mode_for_later_rebuilds(tmp_path):
    """The flag is the graph's mode, so every later rebuild reads it back."""
    corpus = _corpus(tmp_path)

    _run(["update", ".", "--gitignore", "--no-cluster"], cwd=corpus)
    assert _build_config(corpus)["gitignore"] is True

    graph = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    sources = {Path(str(n.get("source_file", ""))).as_posix() for n in graph["nodes"]}
    assert any(s.endswith("app.py") for s in sources)
    assert not any(s.endswith("generated/gen.py") for s in sources)

    # A flag-less rebuild keeps the recorded mode rather than reverting to the
    # default — the #1971 complaint, now in both directions.
    _run(["update", ".", "--no-cluster"], cwd=corpus)
    assert _build_config(corpus)["gitignore"] is True


def test_update_without_flag_keeps_the_default_mode(tmp_path):
    """No flag and no recorded mode means .gitignore is not honored."""
    corpus = _corpus(tmp_path)

    result = _run(["update", ".", "--no-cluster"], cwd=corpus)
    assert result.returncode == 0, result.stderr

    graph = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    sources = {Path(str(n.get("source_file", ""))).as_posix() for n in graph["nodes"]}
    assert any(s.endswith("generated/gen.py") for s in sources)


def test_mode_is_recorded_even_when_the_database_probe_fails(tmp_path):
    """The setting is written before the probe, so a failed run is not silent."""
    corpus = _corpus(tmp_path)

    result = _run(
        ["extract", ".", "--gitignore", "--code-only", "--no-cluster"],
        cwd=corpus,
        GRAPHIFY_BACKEND="arcadedb",
        GRAPHIFY_ARCADE_URL="http://127.0.0.1:1",  # nothing listens here
    )

    assert result.returncode != 0, "an unreachable database must still fail the run"
    assert _build_config(corpus)["gitignore"] is True


def test_missing_build_config_means_gitignore_is_not_honored(tmp_path):
    """Graphs predating the setting rebuild in the new default mode."""
    from graphify.watch import _read_build_gitignore

    assert _read_build_gitignore(tmp_path / "graphify-out") is False


def test_recorded_mode_survives_a_flagless_write(tmp_path):
    """`_write_build_config(gitignore=None)` leaves a recorded mode alone."""
    from graphify.watch import _read_build_gitignore, _write_build_config

    out = tmp_path / "graphify-out"
    _write_build_config(out, excludes=None, gitignore=True)
    _write_build_config(out, excludes=["vendor"], gitignore=None)

    assert _read_build_gitignore(out) is True
