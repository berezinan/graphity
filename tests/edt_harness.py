"""Diff harness for the 1C:EDT extractors, measured against a real EDT project.

The fixture suite in ``tests/fixtures/edt/`` cannot detect an omission: a node
that is missing because the parser ignores a construct looks exactly like a node
that is missing because the fixture never contained that construct. Only a real
configuration answers "what are we not seeing".

Two blind spots need two modes, and they do not overlap:

* :func:`build_snapshot` + :func:`diff_snapshots` compare what the parsers
  produce before and after a change. This catches a regression — an id that used
  to exist and no longer does — which is invisible in aggregate counts.
* :func:`census` reports what never reaches a parser at all. A diff is blind to
  it by construction: files with no extractor are absent from both snapshots.

The harness holds no golden snapshot. Keeping one in the repository would mean
updating it with every change, and updating a golden file is exactly the
operation that silently blesses a regression. Both snapshots are taken in one
run, against two versions of the code.

The corpus is machine-local and proprietary: its path comes from
``GRAPHIFY_EDT_CORPUS`` and every corpus-backed test skips without it. Nothing
here writes to the corpus, and its ``graphify-out/`` is never read.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

from graphify.detect import CODE_EXTENSIONS, classify_file
from graphify.extractors.bsl import (
    extract_bsl,
    extract_edt_dcs,
    extract_edt_cmi,
    extract_edt_oform,
    extract_edt_form,
    extract_edt_mdo,
    extract_edt_rights,
)

CORPUS_ENV = "GRAPHIFY_EDT_CORPUS"

# The extractors that read EDT project files. `.bsl` is absent from the default
# set because it is not EDT-specific — not because it is expensive: measured on a
# 12 335-module configuration, `extract_bsl` costs 6.6 ms per file, ~1.4 min for
# the whole corpus. Pass `BSL_EXTRACTORS` to `build_snapshot` to include it; a
# change to the BSL grammar or extractor is invisible without it, since object,
# manager and common modules reach no EDT-specific extractor. The census counts
# `.bsl` files either way — it reports the whole corpus.
EDT_EXTRACTORS: dict[str, Callable[[Path], dict]] = {
    ".mdo": extract_edt_mdo,
    ".rights": extract_edt_rights,
    ".form": extract_edt_form,
    ".dcs": extract_edt_dcs,
    # An ordinary form carries its BSL module inside the container, so the
    # snapshot has to go through the extractor to see any of it.
    ".oform": extract_edt_oform,
    ".cmi": extract_edt_cmi,
}

# Opt-in addition for questions about BSL parsing itself:
# build_snapshot(root, extractors=EDT_EXTRACTORS | BSL_EXTRACTORS).
BSL_EXTRACTORS: dict[str, Callable[[Path], dict]] = {
    ".bsl": extract_bsl,
}

# Never descended into: build output, VCS and IDE state, and — required by the
# spec — the corpus's own graphify output directory.
_SKIP_DIRS = frozenset({
    "graphify-out", ".git", ".metadata", ".settings", "bin", "__pycache__",
})

# Census statuses. Two of them are the point of the whole mode.
STATUS_EDT = "edt"                      # EDT extractor ran and produced nodes
STATUS_EDT_NO_OUTPUT = "edt-no-output"  # EDT extractor exists, snapshot got nothing
STATUS_OTHER = "other-extractor"        # code, but handled outside this harness
STATUS_DOCUMENT = "document"            # classified, ingested by the non-code path
STATUS_NOT_PARSED = "not-parsed"        # classify_file says nothing — truly invisible


def corpus_root() -> Path | None:
    """Corpus path from the environment, or None when it is not usable.

    Returns None both when the variable is unset and when it points at something
    that is not an EDT project, so callers have a single skip condition.
    """
    raw = os.environ.get(CORPUS_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw)
    return root if is_edt_project(root) else None


def is_edt_project(root: Path) -> bool:
    """True when `root` looks like an EDT project (or a fixture tree of one)."""
    try:
        if not root.is_dir():
            return False
    except OSError:
        return False
    return (root / "src").is_dir() or (root / "DT-INF").is_dir() or any(
        (root / marker).is_dir() for marker in ("Configuration", "Catalogs")
    )


def _walk(root: Path) -> Iterator[Path]:
    """Every file under `root`, skipping derived/VCS directories. Read-only."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _node_kind(node_id: str) -> str:
    """Coarse kind of a node, taken from its id.

    Derived from the id rather than the label on purpose: the label is excluded
    from the snapshot, so counters must not depend on it either — otherwise a
    cosmetic label edit would show up as a snapshot change.
    """
    return node_id.split("_", 1)[0] or "?"


def build_snapshot(root: Path, extractors: "dict[str, Callable[[Path], dict]] | None" = None) -> dict:
    """Run the EDT extractors over `root` and reduce the result to identities.

    The snapshot compares *what exists*, not how it is rendered: labels,
    ``source_file``, ``weight`` and ``confidence`` are dropped. They churn with
    cosmetic edits and would drown the one signal worth having — an id that
    disappeared.

    `extractors` selects which file kinds are read; it defaults to
    :data:`EDT_EXTRACTORS`. Resolve it per call rather than at import time so a
    test can monkeypatch entries of the module-level map. Pass
    ``EDT_EXTRACTORS | BSL_EXTRACTORS`` when the question is about BSL parsing.
    """
    if extractors is None:
        extractors = EDT_EXTRACTORS
    node_ids: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    files_by_ext: Counter[str] = Counter()
    errors_by_ext: Counter[str] = Counter()
    nodes_by_ext: Counter[str] = Counter()

    for path in _walk(root):
        ext = path.suffix.lower()
        extractor = extractors.get(ext)
        if extractor is None:
            continue
        files_by_ext[ext] += 1
        try:
            result = extractor(path)
        except Exception:                                   # noqa: BLE001
            # A crashing extractor is a finding, not a reason to abort the run:
            # the whole point is to measure the corpus, including its bad files.
            errors_by_ext[ext] += 1
            continue
        if result.get("error"):
            errors_by_ext[ext] += 1
            continue
        produced = 0
        for node in result.get("nodes", []):
            nid = node.get("id")
            if nid:
                node_ids.add(nid)
                produced += 1
        nodes_by_ext[ext] += produced
        for edge in result.get("edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            rel = edge.get("relation")
            if src and tgt and rel:
                edges.add((src, rel, tgt))

    return {
        "node_ids": sorted(node_ids),
        "edges": sorted(edges),
        "counters": {
            "nodes": len(node_ids),
            "edges": len(edges),
            "nodes_by_kind": dict(Counter(_node_kind(n) for n in node_ids).most_common()),
            "edges_by_relation": dict(Counter(r for _, r, _ in edges).most_common()),
            "files_by_ext": dict(sorted(files_by_ext.items())),
            "nodes_by_ext": dict(sorted(nodes_by_ext.items())),
            "errors_by_ext": dict(sorted(errors_by_ext.items())),
        },
    }


def diff_snapshots(before: dict, after: dict) -> dict:
    """Compare two snapshots, separating nodes from edges.

    ``removed`` is the category that matters: an id that used to exist and no
    longer does means previously stored references and the code↔metadata merge
    break. It is surfaced as an explicit ``regression`` flag so a caller cannot
    read "N added, M removed" as a wash.
    """
    before_nodes, after_nodes = set(before["node_ids"]), set(after["node_ids"])
    before_edges = {tuple(e) for e in before["edges"]}
    after_edges = {tuple(e) for e in after["edges"]}

    nodes = {
        "added": sorted(after_nodes - before_nodes),
        "removed": sorted(before_nodes - after_nodes),
        "unchanged": len(before_nodes & after_nodes),
    }
    edges = {
        "added": sorted(after_edges - before_edges),
        "removed": sorted(before_edges - after_edges),
        "unchanged": len(before_edges & after_edges),
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "regression": bool(nodes["removed"] or edges["removed"]),
        "removed_total": len(nodes["removed"]) + len(edges["removed"]),
        "added_total": len(nodes["added"]) + len(edges["added"]),
    }


def format_diff(diff: dict, examples: int = 5) -> str:
    """Render a diff as text meant to be pasted into a change's tasks.md.

    Carries both the aggregate and concrete examples: a bare number is not
    evidence anybody can check, and a wall of ids is not a criterion.
    """
    lines: list[str] = []
    verdict = "РЕГРЕСС" if diff["regression"] else "регресса нет"
    lines.append(
        f"узлы: +{len(diff['nodes']['added'])} / -{len(diff['nodes']['removed'])} "
        f"/ без изменений {diff['nodes']['unchanged']}"
    )
    lines.append(
        f"рёбра: +{len(diff['edges']['added'])} / -{len(diff['edges']['removed'])} "
        f"/ без изменений {diff['edges']['unchanged']}"
    )
    lines.append(f"вердикт: {verdict} (удалено {diff['removed_total']})")
    for section in ("nodes", "edges"):
        for category in ("added", "removed"):
            items = diff[section][category]
            if not items:
                continue
            shown = items[:examples]
            lines.append(f"  {section}.{category} ({len(items)}), первые {len(shown)}:")
            for item in shown:
                lines.append(f"    {item}")
    return "\n".join(lines)


def census(root: Path, snapshot: dict | None = None) -> list[dict]:
    """Per-extension inventory of the corpus, with what reaches a parser.

    A diff cannot answer this: a file with no extractor produces nothing in
    either snapshot, so it is equally absent before and after. The census is the
    only mode that names what is invisible.
    """
    if snapshot is None:
        snapshot = build_snapshot(root)
    nodes_by_ext = snapshot["counters"]["nodes_by_ext"]

    counts: Counter[str] = Counter()
    classified: dict[str, str | None] = {}
    for path in _walk(root):
        ext = path.suffix.lower()
        counts[ext] += 1
        if ext not in classified:
            file_type = classify_file(path)
            classified[ext] = str(file_type) if file_type is not None else None

    rows: list[dict] = []
    for ext, n in counts.most_common():
        is_code = ext in CODE_EXTENSIONS
        if ext in EDT_EXTRACTORS:
            status = STATUS_EDT if nodes_by_ext.get(ext) else STATUS_EDT_NO_OUTPUT
        elif is_code:
            status = STATUS_OTHER
        elif classified.get(ext) is not None:
            # Classified as a document/image: graphify ingests it through the
            # non-code path. Lumping these in with the invisible files would
            # inflate the headline number with 2182 help pages and hide the
            # extensions that genuinely never reach anything.
            status = STATUS_DOCUMENT
        else:
            status = STATUS_NOT_PARSED
        rows.append({
            "ext": ext,
            "files": n,
            "classified": classified.get(ext),
            "in_code_extensions": is_code,
            "edt_extractor": ext in EDT_EXTRACTORS,
            "nodes": nodes_by_ext.get(ext, 0),
            "status": status,
        })
    return rows


def format_census(rows: list[dict], limit: int = 25) -> str:
    """Render a census as a table; the two blind-spot statuses are called out."""
    lines = [f"{'ext':<12}{'files':>7}  {'classified':<10}{'nodes':>8}  status"]
    for row in rows[:limit]:
        lines.append(
            f"{row['ext'] or '(нет)':<12}{row['files']:>7}  "
            f"{str(row['classified'] or '-'):<10}{row['nodes']:>8}  {row['status']}"
        )
    missed = [r for r in rows if r["status"] == STATUS_NOT_PARSED and r["files"] > 0]
    empty = [r for r in rows if r["status"] == STATUS_EDT_NO_OUTPUT]
    if missed:
        total = sum(r["files"] for r in missed)
        lines.append(f"не доходит до парсера: {total} файлов в "
                     f"{len(missed)} расширениях")
    if empty:
        lines.append("экстрактор есть, вклада в снимок нет: "
                     + ", ".join(r["ext"] for r in empty))
    return "\n".join(lines)


def tree_fingerprint(root: Path) -> str:
    """Hash of the corpus layout: relative path, size and mtime of every file.

    Used to prove the run did not touch the corpus. Contents are not read — the
    manifest already changes on any create, delete, resize or rewrite, and
    hashing thousands of files would make the check too slow to keep.
    """
    digest = hashlib.sha256()
    for path in sorted(_walk(root)):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()
