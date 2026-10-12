"""Explicitly allowed external corpus roots, reached through directory links.

A consumer project mounts a shared knowledge tree with a directory link
(a Windows junction, or a symlink elsewhere) at the same relative path the
tree has in its source project. Files under an allowed external root are
admitted under their walked path, keep the source project's cache keys and
ids, and the walk never loops through links.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from graphify import cache as _cache
from graphify.cache import file_hash, load_cached, save_cached
from graphify.detect import detect
from graphify.ids import normalize_id


def _reset_stat_index() -> None:
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}


@pytest.fixture
def dir_link():
    """Create a directory link: a junction on Windows, a symlink elsewhere.

    Links are removed before tmp_path cleanup so a cyclic link can never send
    the cleanup walk into its own target.
    """
    made: list[Path] = []

    def _make(link: Path, target: Path) -> Path:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
        made.append(link)
        return link

    yield _make
    for link in reversed(made):
        _remove_link(link)


def _remove_link(link: Path) -> None:
    try:
        if sys.platform == "win32":
            os.rmdir(link)  # removes the junction itself, never its target
        else:
            link.unlink()
    except FileNotFoundError:
        pass


def _all_files(result: dict) -> list[str]:
    return [f for files in result["files"].values() for f in files]


def _rel(result: dict, root: Path) -> list[str]:
    return sorted(Path(f).relative_to(root).as_posix() for f in _all_files(result))


@pytest.fixture
def projects(tmp_path):
    """A source project holding a shared tree and an empty consumer."""
    src = tmp_path / "src_proj"
    (src / "db").mkdir(parents=True)
    (src / "db" / "a.md").write_text("# A\n\nshared document a\n", encoding="utf-8")
    mem = src / "_reversa_sdd" / "db" / "mem"
    mem.mkdir(parents=True)
    (mem / "design.md").write_text("# Design\n\nshared design\n", encoding="utf-8")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    (consumer / "main.py").write_text("x = 1\n", encoding="utf-8")
    return src, consumer


# --- walk containment ------------------------------------------------------


def test_junction_cycle_is_walked_once(tmp_path, dir_link):
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.md").write_text("# F\n\nbody\n", encoding="utf-8")
    dir_link(root / "a" / "loop", root)

    result = detect(root)

    assert [f for f in _rel(result, root) if f.endswith("f.md")] == ["a/f.md"]
    assert any(
        "loop" in item and "[link cycle]" in item for item in result["skipped_sensitive"]
    )


def test_cycle_inside_external_root_is_cut(projects, dir_link):
    src, consumer = projects
    dir_link(src / "db" / "back", src / "db")
    dir_link(consumer / "db", src / "db")

    result = detect(consumer, external_roots=[src])

    assert [f for f in _rel(result, consumer) if f.endswith("a.md")] == ["db/a.md"]
    assert any("[link cycle]" in item for item in result["skipped_sensitive"])


def test_external_diamond_keeps_first_sorted_link(projects, dir_link):
    src, consumer = projects
    for order in (("a_link", "b_link"), ("b_link", "a_link")):
        links = [dir_link(consumer / name, src / "db") for name in order]

        result = detect(consumer, external_roots=[src])

        assert [f for f in _rel(result, consumer) if f.endswith("a.md")] == ["a_link/a.md"]
        assert any(
            "b_link" in item and "[link duplicate]" in item
            for item in result["skipped_sensitive"]
        )
        for link in links:
            _remove_link(link)


# --- admission -------------------------------------------------------------


def test_external_root_admits_files_through_link(projects, dir_link):
    src, consumer = projects
    dir_link(consumer / "db", src / "db")

    result = detect(consumer, external_roots=[src])

    assert "db/a.md" in _rel(result, consumer)


def test_without_external_root_link_target_stays_out(projects, dir_link):
    src, consumer = projects
    dir_link(consumer / "db", src / "db")

    result = detect(consumer)

    assert "db/a.md" not in _rel(result, consumer)
    assert any(
        "db" in item and "[symlink target outside scan root]" in item
        for item in result["skipped_sensitive"]
    )


def test_link_outside_allowed_roots_is_cut(projects, dir_link, tmp_path):
    src, consumer = projects
    third = tmp_path / "third"
    third.mkdir()
    (third / "z.md").write_text("# Z\n\nforeign\n", encoding="utf-8")
    dir_link(consumer / "other", third)

    result = detect(consumer, external_roots=[src])

    assert not any(f.endswith("z.md") for f in _rel(result, consumer))
    assert any(
        "other" in item and "[symlink target outside scan root]" in item
        for item in result["skipped_sensitive"]
    )


def test_file_link_into_external_root_is_not_admitted(requires_symlinks, projects):
    src, consumer = projects
    (consumer / "note.md").symlink_to(src / "db" / "a.md")

    result = detect(consumer, external_roots=[src])

    assert "note.md" not in _rel(result, consumer)
    assert any(
        "note.md" in item and "[symlink target outside scan root]" in item
        for item in result["skipped_sensitive"]
    )


def test_junction_and_symlink_admit_the_same_files(requires_symlinks, tmp_path, dir_link):
    if sys.platform != "win32":
        pytest.skip("junctions exist only on Windows")
    sets = []
    for kind in ("junction", "symlink"):
        base = tmp_path / kind
        src = base / "src_proj"
        (src / "db").mkdir(parents=True)
        (src / "db" / "a.md").write_text("# A\n\nshared\n", encoding="utf-8")
        consumer = base / "consumer"
        consumer.mkdir()
        if kind == "junction":
            dir_link(consumer / "db", src / "db")
        else:
            (consumer / "db").symlink_to(src / "db", target_is_directory=True)
        sets.append(_rel(detect(consumer, follow_symlinks=False, external_roots=[src]), consumer))
    assert sets[0] == sets[1] == ["db/a.md"]


def test_mirror_mismatch_is_admitted_and_reported(projects, dir_link):
    src, consumer = projects
    (consumer / "shared").mkdir()
    dir_link(consumer / "shared" / "invina", src / "db")

    result = detect(consumer, external_roots=[src])

    assert "shared/invina/a.md" in _rel(result, consumer)
    assert any("shared/invina" in m for m in result["external_mirror_mismatches"])


def test_mirrored_links_report_no_mismatch(projects, dir_link):
    src, consumer = projects
    dir_link(consumer / "db", src / "db")
    (consumer / "_reversa_sdd").mkdir()
    dir_link(consumer / "_reversa_sdd" / "db", src / "_reversa_sdd" / "db")

    result = detect(consumer, external_roots=[src])

    assert result["external_mirror_mismatches"] == []
    assert result["external_roots"] == {src.resolve().as_posix(): 2}


def test_ignore_rules_apply_by_walked_path(projects, dir_link):
    src, consumer = projects
    (src / "db" / "archive").mkdir()
    (src / "db" / "archive" / "old.md").write_text("# Old\n\nold\n", encoding="utf-8")
    (consumer / ".graphifyignore").write_text("db/archive/\n", encoding="utf-8")
    dir_link(consumer / "db", src / "db")

    result = detect(consumer, external_roots=[src])

    rel = _rel(result, consumer)
    assert "db/a.md" in rel
    assert not any(f.startswith("db/archive/") for f in rel)


# --- identity --------------------------------------------------------------


def test_file_hash_matches_source_project(projects, dir_link):
    src, consumer = projects
    (consumer / "_reversa_sdd").mkdir()
    dir_link(consumer / "_reversa_sdd" / "db", src / "_reversa_sdd" / "db")
    rel = Path("_reversa_sdd/db/mem/design.md")

    _reset_stat_index()
    assert file_hash(consumer / rel, consumer) == file_hash(src / rel, src)


def test_semantic_entry_replays_across_source_and_consumer(projects, dir_link, tmp_path):
    src, consumer = projects
    (consumer / "_reversa_sdd").mkdir()
    dir_link(consumer / "_reversa_sdd" / "db", src / "_reversa_sdd" / "db")
    rel = Path("_reversa_sdd/db/mem/design.md")

    def payload(file_path: Path) -> dict:
        nid = normalize_id(str(file_path)) + "_intro"
        return {
            "nodes": [{"id": nid, "label": "Intro", "source_file": str(file_path)}],
            "edges": [],
        }

    for writer, reader in ((src, consumer), (consumer, src)):
        _reset_stat_index()
        cache_root = tmp_path / f"cache_{writer.name}"
        save_cached(writer / rel, payload(writer / rel), writer, kind="semantic", cache_root=cache_root)

        _reset_stat_index()
        got = load_cached(reader / rel, reader, kind="semantic", cache_root=cache_root)

        assert got is not None, f"{writer.name} -> {reader.name}: cache miss"
        assert got["nodes"][0]["id"] == payload(reader / rel)["nodes"][0]["id"]
        stored = json.dumps(
            json.loads(next(cache_root.rglob("*.json")).read_text(encoding="utf-8"))
        )
        assert "src_proj" not in stored and "consumer" not in stored


# --- collect_files ----------------------------------------------------------


def test_collect_files_cuts_junction_cycle(tmp_path, dir_link):
    from graphify.extract import collect_files

    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    dir_link(root / "pkg" / "loop", root)

    for follow in (False, True):
        files = collect_files(root, follow_symlinks=follow)
        assert [f for f in files if f.name == "mod.py"] == [root / "pkg" / "mod.py"]


def test_collect_files_admits_external_root(projects, dir_link):
    from graphify.extract import collect_files

    src, consumer = projects
    (src / "db" / "tool.py").write_text("y = 2\n", encoding="utf-8")
    dir_link(consumer / "db", src / "db")

    assert consumer / "db" / "tool.py" not in collect_files(consumer)
    assert consumer / "db" / "tool.py" in collect_files(consumer, external_roots=[src])


def test_normalize_external_roots(tmp_path, capsys):
    from graphify.detect import _normalize_external_roots

    scan = tmp_path / "scan"
    (scan / "sub").mkdir(parents=True)
    ext = tmp_path / "ext"
    ext.mkdir()
    missing = tmp_path / "missing"

    got = _normalize_external_roots([ext, str(ext), scan / "sub", missing], scan)

    assert got == [ext.resolve(), missing.resolve()]
    err = capsys.readouterr().err
    assert "inside the scan root" in err and "does not exist" in err


def test_extract_ids_match_source_project(projects, dir_link, tmp_path):
    from graphify.extract import extract

    src, consumer = projects
    (src / "db" / "a.md").write_text(
        "# A\n\nshared document a\n\n## Section\n\nsee [b](b.md)\n", encoding="utf-8"
    )
    (src / "db" / "b.md").write_text("# B\n\nother\n", encoding="utf-8")
    dir_link(consumer / "db", src / "db")

    def shape(root: Path, cache: str) -> set[tuple[str, str]]:
        _reset_stat_index()
        files = [root / "db" / "a.md", root / "db" / "b.md"]
        out = extract(files, root=root, cache_root=tmp_path / cache)
        return {(n["id"], n.get("source_file")) for n in out["nodes"]}

    from_src = shape(src, "c_src")
    from_consumer = shape(consumer, "c_consumer")

    assert from_src and from_src == from_consumer
    assert all(sf.startswith("db/") for _, sf in from_consumer)


# --- persistence, CLI, deletion, report ------------------------------------

_REPO = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path, **env_overrides: str):
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("GRAPHIFY_")}
    env["GRAPHIFY_BACKEND"] = "json"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), os.environ.get("PYTHONPATH", "")) if p
    )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "graphify", *args],
        cwd=str(cwd), capture_output=True, text=True, env=env,
    )


def _build_config(corpus: Path) -> dict:
    path = corpus / "graphify-out" / ".graphify_build.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _graph_sources(corpus: Path) -> set[str]:
    graph = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return {Path(str(n.get("source_file", ""))).as_posix() for n in graph["nodes"]}


def test_build_config_external_roots_replace_keep_clear(tmp_path):
    from graphify.watch import _read_build_external_roots, _write_build_config

    out = tmp_path / "graphify-out"
    assert _read_build_external_roots(out) == []
    _write_build_config(out, excludes=None, external_roots=["C:/A", "C:/B"])
    _write_build_config(out, excludes=None, gitignore=True)
    assert _read_build_external_roots(out) == ["C:/A", "C:/B"]
    _write_build_config(out, excludes=None, external_roots=["C:/C"])
    assert _read_build_external_roots(out) == ["C:/C"]
    _write_build_config(out, excludes=None, external_roots=[])
    assert _read_build_external_roots(out) == []


def test_update_records_external_root_and_indexes_through_link(projects, dir_link):
    src, consumer = projects
    (src / "db" / "tool.py").write_text("def shared(): return 1\n", encoding="utf-8")
    dir_link(consumer / "db", src / "db")

    result = _run(["update", ".", "--external-root", str(src), "--no-cluster"], cwd=consumer)

    assert "unknown update option" not in result.stderr
    assert result.returncode == 0, result.stderr
    assert _build_config(consumer)["external_roots"] == [src.resolve().as_posix()]
    assert "db/tool.py" in _graph_sources(consumer)

    # A flag-less rebuild keeps the recorded root...
    result = _run(["update", ".", "--no-cluster"], cwd=consumer)
    assert result.returncode == 0, result.stderr
    assert "db/tool.py" in _graph_sources(consumer)

    # ...a file deleted in the external tree leaves the graph like any other...
    (src / "db" / "tool.py").unlink()
    (src / "db" / "other.py").write_text("def other(): return 2\n", encoding="utf-8")
    result = _run(["update", ".", "--no-cluster"], cwd=consumer)
    assert result.returncode == 0, result.stderr
    sources = _graph_sources(consumer)
    assert "db/tool.py" not in sources and "db/other.py" in sources

    # ...and clearing the roots drops everything admitted through them.
    result = _run(["update", ".", "--no-external-roots", "--no-cluster"], cwd=consumer)
    assert result.returncode == 0, result.stderr
    assert _build_config(consumer)["external_roots"] == []
    assert not any(s.startswith("db/") for s in _graph_sources(consumer))


def test_removed_link_drops_its_files(projects, dir_link):
    src, consumer = projects
    (src / "db" / "tool.py").write_text("def shared(): return 1\n", encoding="utf-8")
    link = dir_link(consumer / "db", src / "db")
    assert _run(["update", ".", "--external-root", str(src), "--no-cluster"], cwd=consumer).returncode == 0
    assert "db/tool.py" in _graph_sources(consumer)

    _remove_link(link)
    result = _run(["update", ".", "--no-cluster"], cwd=consumer)

    assert result.returncode == 0, result.stderr
    assert not any(s.startswith("db/") for s in _graph_sources(consumer))


def test_external_root_flags_are_exclusive(projects):
    _, consumer = projects
    result = _run(
        ["update", ".", "--external-root", "x", "--no-external-roots"], cwd=consumer
    )
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_external_root_is_recorded_even_when_the_database_probe_fails(projects):
    src, consumer = projects
    result = _run(
        ["extract", ".", "--external-root", str(src), "--code-only", "--no-cluster"],
        cwd=consumer,
        GRAPHIFY_BACKEND="arcadedb",
        GRAPHIFY_ARCADE_URL="http://127.0.0.1:1",
    )
    assert result.returncode != 0
    assert _build_config(consumer)["external_roots"] == [src.resolve().as_posix()]


def test_external_root_inside_scan_root_is_not_recorded(projects):
    _, consumer = projects
    (consumer / "sub").mkdir()
    result = _run(["update", ".", "--external-root", "sub", "--no-cluster"], cwd=consumer)
    assert result.returncode == 0, result.stderr
    assert "inside the scan root" in result.stderr
    assert _build_config(consumer)["external_roots"] == []


def test_report_lists_external_roots_and_mirror_mismatches():
    import networkx as nx

    from graphify.report import generate

    detection = {
        "total_files": 3,
        "total_words": 50000,
        "external_roots": {"C:/AI/INVINA_DB": 2},
        "external_mirror_mismatches": ["shared/invina -> C:/AI/INVINA_DB/db"],
    }
    report = generate(
        nx.Graph(), {}, {}, {}, [], [], detection, {"input": 0, "output": 0}, "corpus",
        corpus_scope={"gitignore": False, "source": "recorded in .graphify_build.json"},
    )
    assert (
        "- External roots (recorded in .graphify_build.json): C:/AI/INVINA_DB (2 file(s))"
        in report
    )
    assert "- External link not mirrored (no shared cache): shared/invina -> C:/AI/INVINA_DB/db" in report


def test_skill_snippets_pass_recorded_external_roots():
    pkg = _REPO / "graphify"
    snippets = [
        line
        for md in pkg.rglob("*.md")
        for line in md.read_text(encoding="utf-8").splitlines()
        if line.startswith(("result = detect(Path('INPUT_PATH')", "result = detect_incremental(Path('INPUT_PATH')"))
    ]
    assert snippets
    assert all("external_roots=_read_build_external_roots(Path('graphify-out'))" in s for s in snippets)


def test_extract_without_external_roots_prunes_their_nodes(projects, dir_link):
    src, consumer = projects
    (src / "db" / "tool.py").write_text("def shared(): return 1\n", encoding="utf-8")
    dir_link(consumer / "db", src / "db")
    base = ["extract", ".", "--code-only", "--no-cluster"]

    assert _run(base + ["--external-root", str(src)], cwd=consumer).returncode == 0
    assert "db/tool.py" in _graph_sources(consumer)

    result = _run(base + ["--no-external-roots"], cwd=consumer)

    assert result.returncode == 0, result.stderr
    assert not any(s.startswith("db/") for s in _graph_sources(consumer))
