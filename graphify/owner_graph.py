"""Semantic layer of externally mounted files, taken from the owner's graph.

Fork delta (owner-first-external-roots). A file admitted through a directory
link into an allowed external root ``E`` is owned by the project rooted at
``E``. Its semantic (LLM) layer is extracted once, by the owner, and a consumer
copies that layer from ``E/graphify-out/graph.json`` instead of extracting it
again with its own prompt, chunk budget and chunk neighbours.

The owner's data for a file is usable only when the link is mirrored (the
owner's ``source_file`` is the consumer's walked path), the owner's manifest
stamps the file's current content as semantically extracted, and the owner's
graph holds a semantic node or hyperedge for it. The manifest is the freshness
signal, not the graph: a file the owner failed to extract keeps its old nodes
in the graph while its ``semantic_hash`` is blanked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

OWNER_OUT = "graphify-out"
SEMANTIC_KINDS = ("document", "paper", "image")
# Set by build/clustering on the owner's side; a consumer clusters on its own.
_CLUSTER_FIELDS = ("community", "community_name", "norm_label")

# Reason key -> wording in the refusal / warning, in reporting order.
REASONS = {
    "not_mirrored": "link not mirrored",
    "no_owner_output": "owner has no graphify-out/graph.json + manifest.json",
    "not_in_manifest": "not in the owner's manifest",
    "owner_failed": "the owner could not extract it",
    "changed": "content changed since the owner's extraction",
    "not_in_graph": "no semantic nodes in the owner's graph",
}


@dataclass
class ExternalFile:
    path: str        # as detect() listed it (walked, absolute)
    rel: str         # posix, relative to the scan root
    root: Path       # the external root it was admitted through
    owner_rel: str   # posix, relative to that root (resolved target)
    kind: str = ""   # detect() file type

    @property
    def mirrored(self) -> bool:
        return os.path.normcase(self.rel) == os.path.normcase(self.owner_rel)


@dataclass
class OwnerPlan:
    taken: list[ExternalFile] = field(default_factory=list)
    refused: dict[str, list[ExternalFile]] = field(default_factory=dict)
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    hyperedges: list[dict] = field(default_factory=list)

    def taken_keys(self) -> set[str]:
        return {path_key(f.path) for f in self.taken}

    def taken_sources(self) -> set[str]:
        return {_sf_key(f.rel) for f in self.taken}


def path_key(path: "str | Path") -> str:
    """Comparable form of a walked path (no resolve: links must stay walked)."""
    return os.path.normcase(os.path.abspath(str(path)))


def _sf_key(sf) -> str:
    return os.path.normcase(str(sf or "").replace("\\", "/"))


def external_files(files_by_type: dict, scan_root: Path, external_roots, kinds=None) -> list[ExternalFile]:
    """Files of ``files_by_type`` (``kinds``, default all) admitted through an
    external root, with their walked and owner-relative paths."""
    from graphify.detect import _key_under, _real_key

    root_abs = Path(os.path.abspath(scan_root))
    root_key = _real_key(scan_root)
    roots = [(os.path.normcase(str(e)), Path(e)) for e in external_roots]
    out: list[ExternalFile] = []
    for kind in (kinds or files_by_type.keys()):
        for f in files_by_type.get(kind, []) or []:
            real_key = _real_key(f)
            if _key_under(real_key, root_key):
                continue
            for ext_key, ext in roots:
                if not _key_under(real_key, ext_key):
                    continue
                try:
                    rel = Path(os.path.abspath(f)).relative_to(root_abs).as_posix()
                    owner_rel = Path(f).resolve().relative_to(ext).as_posix()
                except (ValueError, OSError, RuntimeError):
                    break
                out.append(ExternalFile(str(f), rel, ext, owner_rel, kind))
                break
    return out


def _manifest_lookup(owner_root: Path, manifest_path: Path) -> dict[str, object]:
    from graphify.detect import load_manifest

    return {
        os.path.normcase(k): v
        for k, v in load_manifest(str(manifest_path), root=owner_root).items()
    }


class _OwnerSink:
    """Keeps only what a consumer can use from the owner's graph.json."""

    def __init__(self, wanted: set[str], corpus: set[str]):
        self.wanted = wanted      # sf keys of the files whose semantics we take
        self.corpus = corpus      # sf keys of every mirrored external file in the consumer's corpus
        self.ids: dict[str, tuple[str, bool]] = {}   # node id -> (sf key, is_ast)
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.hyperedges: list[dict] = []

    def begin(self, counts):
        pass

    def not_an_object(self, raw):
        pass

    def value(self, k, v):
        pass

    def array(self, k):
        pass

    def items(self, k, batch):
        from graphify.build import _is_ast_tier

        if k == "nodes":
            for n in batch:
                if not isinstance(n, dict) or "id" not in n:
                    continue
                key = _sf_key(n.get("source_file"))
                if key not in self.corpus:
                    continue
                ast = _is_ast_tier(n)
                self.ids[n["id"]] = (key, ast)
                if not ast and key in self.wanted:
                    self.nodes.append({f: v for f, v in n.items() if f not in _CLUSTER_FIELDS})
        elif k in ("links", "edges"):
            for e in batch:
                if (isinstance(e, dict) and not _is_ast_tier(e)
                        and _sf_key(e.get("source_file")) in self.wanted):
                    self.edges.append(dict(e))
        elif k == "hyperedges":
            for h in batch:
                if isinstance(h, dict) and _sf_key(h.get("source_file")) in self.wanted:
                    self.hyperedges.append(dict(h))


def _read_owner_graph(graph_path: Path, wanted: set[str], corpus: set[str]) -> _OwnerSink:
    from graphify.graph_compare import _read

    return _read(graph_path, lambda: _OwnerSink(wanted, corpus))


def plan_owner_semantics(semantic: list[ExternalFile], corpus: list[ExternalFile]) -> OwnerPlan:
    """Decide, per external semantic file, whether the owner's data is usable,
    and collect that data. ``corpus`` is every external file of the consumer's
    corpus (any kind): the only nodes an owner edge may end at."""
    from graphify.detect import _md5_file, _to_absolute_from_storage

    plan = OwnerPlan()

    def refuse(reason: str, f: ExternalFile) -> None:
        plan.refused.setdefault(reason, []).append(f)

    by_root: dict[str, list[ExternalFile]] = {}
    for f in semantic:
        if not f.mirrored:
            refuse("not_mirrored", f)
        else:
            by_root.setdefault(f.root.as_posix(), []).append(f)

    for files in by_root.values():
        root = files[0].root
        out = root / OWNER_OUT
        manifest_path, graph_path = out / "manifest.json", out / "graph.json"
        if not (manifest_path.is_file() and graph_path.is_file()):
            for f in files:
                refuse("no_owner_output", f)
            continue
        manifest = _manifest_lookup(root, manifest_path)
        candidates: list[ExternalFile] = []
        for f in files:
            entry = manifest.get(os.path.normcase(_to_absolute_from_storage(f.owner_rel, root)))
            if entry is None:
                refuse("not_in_manifest", f)
                continue
            sem = entry.get("semantic_hash", "") if isinstance(entry, dict) else ""
            if not sem:
                refuse("owner_failed", f)
            elif sem != _md5_file(Path(f.path)):
                refuse("changed", f)
            else:
                candidates.append(f)
        if not candidates:
            continue
        wanted = {_sf_key(f.rel) for f in candidates}
        corpus_keys = {_sf_key(f.rel) for f in corpus if f.mirrored and f.root == root}
        sink = _read_owner_graph(graph_path, wanted, corpus_keys | wanted)
        present = {_sf_key(n.get("source_file")) for n in sink.nodes}
        present |= {_sf_key(h.get("source_file")) for h in sink.hyperedges}
        taken: set[str] = set()
        for f in candidates:
            if _sf_key(f.rel) in present:
                plan.taken.append(f)
                taken.add(_sf_key(f.rel))
            else:
                refuse("not_in_graph", f)
        # An edge end must be a node the consumer has: an AST node of a file in
        # its corpus, or a semantic node taken here. Anything else (the owner's
        # own scripts, files the consumer ignores) would dangle — or, for an
        # import relation, be minted as an external stub by build.
        usable = {i for i, (key, ast) in sink.ids.items() if ast or key in taken}
        plan.nodes += [n for n in sink.nodes if _sf_key(n.get("source_file")) in taken]
        plan.edges += [
            e for e in sink.edges
            if _sf_key(e.get("source_file")) in taken
            and e.get("source") in usable and e.get("target") in usable
        ]
        for h in sink.hyperedges:
            if _sf_key(h.get("source_file")) not in taken:
                continue
            members = [m for m in h.get("nodes", []) or [] if m in usable]
            if len(members) >= 2:
                plan.hyperedges.append({**h, "nodes": members})
    return plan


def refusal_lines(plan: OwnerPlan, *, allowed: bool) -> list[str]:
    """Refusal (or, with --allow-external-extract, warning) text per external root."""
    lines: list[str] = []
    roots = sorted({f.root.as_posix() for fs in plan.refused.values() for f in fs})
    for root in roots:
        total = sum(1 for fs in plan.refused.values() for f in fs if f.root.as_posix() == root)
        if allowed:
            lines.append(
                f"[graphify extract] WARNING: {total} file(s) under external root {root} have no "
                f"usable owner data; extracting them here (--allow-external-extract):"
            )
        else:
            lines.append(
                f"[graphify extract] error: {total} file(s) under external root {root} have no "
                f"usable owner data (owner-first external roots):"
            )
        for reason, text in REASONS.items():
            files = sorted(f.rel for f in plan.refused.get(reason, []) if f.root.as_posix() == root)
            if not files:
                continue
            more = f" (+{len(files) - 5} more)" if len(files) > 5 else ""
            lines.append(f"  {text}: {len(files)} - {', '.join(files[:5])}{more}")
        if not allowed:
            lines.append(
                f"  run `graphify extract .` in {root} first, or pass --allow-external-extract "
                f"to extract these files here"
            )
    return lines


def summary_lines(plan: OwnerPlan, extracted_here: list[ExternalFile]) -> list[str]:
    counts: dict[str, list[int]] = {}
    for f in plan.taken:
        counts.setdefault(f.root.as_posix(), [0, 0])[0] += 1
    for f in extracted_here:
        counts.setdefault(f.root.as_posix(), [0, 0])[1] += 1
    return [
        f"[graphify extract] external root {root}: {n} from owner graph / {m} extracted here"
        for root, (n, m) in sorted(counts.items())
    ]


def without_owner_sources(shrink, plan: "OwnerPlan | None"):
    """The #3203 shrink map minus files whose semantics came from the owner:
    those are authoritative data, not an unverified model answer."""
    if not shrink or plan is None or not plan.taken:
        return shrink
    taken = plan.taken_sources()
    left = {k: v for k, v in shrink.items() if _sf_key(k) not in taken}
    return left or None
