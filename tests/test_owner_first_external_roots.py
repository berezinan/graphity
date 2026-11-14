"""Owner-first external roots (fork delta, owner-first-external-roots).

A file mounted from an allowed external root is semantically extracted once,
by the project that owns that root. A consumer takes the owner's semantic layer
from ``<root>/graphify-out/graph.json`` and never sends the file to its own
LLM; when the owner's data is missing or stale, ``extract`` refuses unless
``--allow-external-extract`` is given for that run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from graphify import cache as _cache

DESIGN = "_reversa_sdd/db/mem/design.md"
DOC_A = "db/a.md"
TOOL = "db/tool.py"


def _reset_stat_index() -> None:
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}


@pytest.fixture
def dir_link():
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
        try:
            if sys.platform == "win32":
                os.rmdir(link)
            else:
                link.unlink()
        except FileNotFoundError:
            pass


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _ast_shared_id(src: Path) -> str:
    from graphify.extract import extract

    _reset_stat_index()
    out = extract([src / "db" / "tool.py"], root=src, cache_root=src / "graphify-out")
    return next(n["id"] for n in out["nodes"] if n.get("label", "").startswith("shared"))


def _sem_node(nid: str, label: str, sf: str, community: int = 99) -> dict:
    return {
        "id": nid, "label": label, "file_type": "document", "source_file": sf,
        "source_location": None, "_origin": "semantic",
        "community": community, "community_name": "Owner cluster", "norm_label": label.lower(),
    }


def _sem_edge(a: str, b: str, sf: str, relation: str = "references") -> dict:
    return {"source": a, "target": b, "relation": relation, "confidence": "EXTRACTED",
            "source_file": sf, "_origin": "semantic", "weight": 1.0}


class Owner:
    """Writes the owner's graph.json and manifest.json the way extract leaves them."""

    def __init__(self, src: Path, shared_id: str):
        self.src = src
        self.shared_id = shared_id
        self.design_nodes = [
            _sem_node("reversa_sdd_db_mem_design_alpha", "Alpha", DESIGN),
            _sem_node("reversa_sdd_db_mem_design_beta", "Beta", DESIGN),
        ]
        self.manifest_overrides: dict[str, dict | None] = {}

    def write(self) -> None:
        out = self.src / "graphify-out"
        out.mkdir(exist_ok=True)
        ids = [n["id"] for n in self.design_nodes]
        nodes = [
            *self.design_nodes,
            _sem_node("db_a_doc", "Doc A", DOC_A),
            {"id": self.shared_id, "label": "shared()", "file_type": "code",
             "source_file": TOOL, "source_location": "L1", "_origin": "ast", "community": 1},
            {"id": "tools_render", "label": "render.py", "file_type": "code",
             "source_file": "tools/render.py", "source_location": "L1", "_origin": "ast"},
        ]
        links: list = []
        hyper: list = []
        if ids:
            links = [
                _sem_edge(ids[0], ids[-1], DESIGN),
                _sem_edge(ids[0], self.shared_id, DESIGN, "describes"),
                _sem_edge(ids[0], "tools_render", DESIGN, "imports"),
                _sem_edge(ids[0], "db_a_doc", DESIGN),
            ]
            hyper = [{"id": "h1", "label": "group", "nodes": [*ids, "tools_render", "db_a_doc"],
                      "relation": "participate_in", "confidence": "INFERRED",
                      "confidence_score": 0.7, "source_file": DESIGN}]
        graph = {"directed": True, "multigraph": False, "graph": {},
                 "nodes": nodes, "links": links, "hyperedges": hyper}
        (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
        manifest = {}
        for rel in (DESIGN, DOC_A, TOOL, "tools/render.py"):
            p = self.src / rel
            h = _md5(p) if p.exists() else "0" * 32
            manifest[rel] = {"mtime": 0.0, "seen": 0.0, "ast_hash": h, "semantic_hash": h}
        for rel, row in self.manifest_overrides.items():
            if row is None:
                manifest.pop(rel, None)
            else:
                manifest[rel] = {**manifest.get(rel, {}), **row}
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def layout(tmp_path, dir_link):
    src = tmp_path / "src_proj"
    (src / "db").mkdir(parents=True)
    (src / "db" / "a.md").write_text("# A\n\nshared document a\n", encoding="utf-8")
    (src / "db" / "tool.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
    (src / "tools").mkdir()
    (src / "tools" / "render.py").write_text("def render():\n    return 2\n", encoding="utf-8")
    mem = src / "_reversa_sdd" / "db" / "mem"
    mem.mkdir(parents=True)
    (mem / "design.md").write_text("# Design\n\nshared design\n", encoding="utf-8")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    (consumer / "main.py").write_text("x = 1\n", encoding="utf-8")
    (consumer / "notes.md").write_text("# Notes\n\nconsumer own notes\n", encoding="utf-8")
    return src, consumer


@pytest.fixture
def mounted(layout, dir_link):
    src, consumer = layout
    (consumer / "_reversa_sdd").mkdir()
    dir_link(consumer / "_reversa_sdd" / "db", src / "_reversa_sdd" / "db")
    dir_link(consumer / "db", src / "db")
    owner = Owner(src, _ast_shared_id(src))
    owner.write()
    return src, consumer, owner


class Run:
    def __init__(self, consumer: Path, out: Path, src: Path, monkeypatch, capsys):
        import graphify.__main__ as mainmod

        self.mainmod = mainmod
        self.consumer = consumer
        self.out = out
        self.src = src
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        self.messages: list[str] = []
        self.ast_calls = 0
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
        monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
        monkeypatch.setattr("graphify.llm._call_claude", self._fake_claude)
        monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
        import graphify.extract as _ext

        real = _ext.extract

        def _counting(*a, **kw):
            self.ast_calls += 1
            return real(*a, **kw)

        monkeypatch.setattr(_ext, "extract", _counting)

    def _fake_claude(self, api_key, model, user_message, max_tokens=8192, *, deep_mode=False, images=None):
        self.messages.append(user_message)
        nodes = []
        for rel in ("notes.md", DESIGN, DOC_A):
            if f'path="{rel}"' in user_message:
                nodes.append({"id": "here_" + rel.replace("/", "_").replace(".", "_"),
                              "label": "Here " + rel, "file_type": "document", "source_file": rel})
        return {"nodes": nodes, "edges": [], "hyperedges": [],
                "input_tokens": 1, "output_tokens": 1, "finish_reason": "stop"}

    def __call__(self, *extra: str, external: bool = True) -> int:
        _reset_stat_index()
        argv = ["graphify", "extract", str(self.consumer), "--backend", "claude",
                "--out", str(self.out), *extra]
        if external:
            argv += ["--external-root", str(self.src)]
        self.monkeypatch.setattr(self.mainmod.sys, "argv", argv)
        try:
            self.mainmod.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    @property
    def gout(self) -> Path:
        return self.out / "graphify-out"

    def graph(self) -> dict:
        return json.loads((self.gout / "graph.json").read_text(encoding="utf-8"))

    def sent(self, text: str) -> bool:
        return any(text in m for m in self.messages)


@pytest.fixture
def run(mounted, tmp_path, monkeypatch, capsys):
    src, consumer, _owner = mounted
    return Run(consumer, tmp_path / "out", src, monkeypatch, capsys)


def _by_source(graph: dict, sf: str) -> list[dict]:
    return [n for n in graph["nodes"] if Path(str(n.get("source_file", ""))).as_posix() == sf]


def _sem_ids(graph: dict, sf: str) -> set[str]:
    return {n["id"] for n in _by_source(graph, sf) if n.get("_origin") != "ast"}


def _edges(graph: dict) -> list[dict]:
    return graph.get("links", graph.get("edges", []))


# --- semantics come from the owner's graph ---------------------------------


@pytest.mark.parametrize("cluster", [False, True], ids=["raw", "clustered"])
def test_external_document_taken_from_owner_without_llm(run, mounted, cluster):
    extra = () if cluster else ("--no-cluster",)

    assert run(*extra) == 0

    assert run.sent("consumer own notes")
    assert not run.sent("shared design") and not run.sent("shared document a")
    graph = run.graph()
    assert _sem_ids(graph, DESIGN) == {"reversa_sdd_db_mem_design_alpha", "reversa_sdd_db_mem_design_beta"}
    alpha = next(n for n in graph["nodes"] if n["id"] == "reversa_sdd_db_mem_design_alpha")
    assert alpha["label"] == "Alpha"
    assert alpha.get("community") != 99 and alpha.get("community_name") != "Owner cluster"
    cache_text = "".join(
        p.read_text(encoding="utf-8") for p in (run.gout / "cache").rglob("*.json")
        if "semantic" in p.parts
    )
    assert DESIGN not in cache_text and DOC_A not in cache_text
    manifest = json.loads((run.gout / "manifest.json").read_text(encoding="utf-8"))
    row = next(v for k, v in manifest.items() if k.replace("\\", "/").endswith(DESIGN))
    assert row["semantic_hash"]
    out = run.capsys.readouterr().out
    assert f"external root {mounted[0].resolve().as_posix()}: 2 from owner graph / 0 extracted here" in out


def test_owner_reextraction_reaches_unchanged_consumer_file(run, mounted):
    _src, _consumer, owner = mounted
    assert run("--no-cluster") == 0
    owner.design_nodes = [_sem_node(f"reversa_sdd_db_mem_design_v2_{i}", f"V2 {i}", DESIGN) for i in range(3)]
    owner.write()
    run.messages.clear()

    assert run("--no-cluster") == 0

    assert not run.sent("shared design")
    assert _sem_ids(run.graph(), DESIGN) == {"reversa_sdd_db_mem_design_v2_0", "reversa_sdd_db_mem_design_v2_1", "reversa_sdd_db_mem_design_v2_2"}


def test_owner_edges_are_cut_to_the_consumer_corpus(run, mounted):
    _src, _consumer, owner = mounted

    assert run("--no-cluster") == 0

    graph = run.graph()
    ids = {n["id"] for n in graph["nodes"]}
    assert "tools_render" not in ids  # owner's own script: no node, no import stub
    pairs = {(e["source"], e["target"]) for e in _edges(graph)}
    assert ("reversa_sdd_db_mem_design_alpha", "tools_render") not in pairs
    assert ("reversa_sdd_db_mem_design_alpha", owner.shared_id) in pairs  # to the consumer's AST node
    assert ("reversa_sdd_db_mem_design_alpha", "db_a_doc") in pairs       # to another taken file
    hyper = graph.get("hyperedges") or graph.get("graph", {}).get("hyperedges", [])
    members = next(h["nodes"] for h in hyper if h["id"] == "h1")
    assert "tools_render" not in members and "reversa_sdd_db_mem_design_alpha" in members


def test_code_under_external_root_is_extracted_by_ast(run, mounted):
    assert run("--no-cluster") == 0
    assert run.ast_calls == 1
    assert any(n.get("_origin") == "ast" for n in _by_source(run.graph(), TOOL))


# --- refusal ---------------------------------------------------------------


def _break(reason: str, src: Path, owner: Owner) -> None:
    if reason == "no_owner_output":
        (src / "graphify-out" / "graph.json").unlink()
    elif reason == "not_in_manifest":
        owner.manifest_overrides[DESIGN] = None
        owner.write()
    elif reason == "owner_failed":
        owner.manifest_overrides[DESIGN] = {"semantic_hash": ""}
        owner.write()
    elif reason == "changed":
        (src / DESIGN).write_text("# Design\n\nrevised shared design\n", encoding="utf-8")
    elif reason == "not_in_graph":
        owner.design_nodes = []
        owner.write()


_TEXT = {
    "no_owner_output": "owner has no graphify-out",
    "not_in_manifest": "not in the owner's manifest",
    "owner_failed": "the owner could not extract it",
    "changed": "content changed since the owner's extraction",
    "not_in_graph": "no semantic nodes in the owner's graph",
}


@pytest.mark.parametrize("reason", sorted(_TEXT))
def test_unusable_owner_data_refuses_before_any_work(run, mounted, reason):
    src, _consumer, owner = mounted
    assert run("--no-cluster") == 0
    before = {p: (run.gout / p).read_bytes() for p in ("graph.json", "manifest.json")}
    cache_before = sorted(str(p) for p in (run.gout / "cache").rglob("*.json") if "semantic" in p.parts)
    run.capsys.readouterr()
    _break(reason, src, owner)
    (run.consumer / "notes.md").write_text("# Notes\n\nedited notes\n", encoding="utf-8")
    run.messages.clear()
    run.ast_calls = 0

    code = run("--no-cluster")

    assert code == 1
    err = run.capsys.readouterr().err
    assert _TEXT[reason] in err and DESIGN in err
    assert "--allow-external-extract" in err and "graphify extract ." in err
    assert run.messages == [] and run.ast_calls == 0
    assert {p: (run.gout / p).read_bytes() for p in before} == before
    assert sorted(str(p) for p in (run.gout / "cache").rglob("*.json") if "semantic" in p.parts) == cache_before


def test_unmirrored_link_refuses(layout, dir_link, tmp_path, monkeypatch, capsys):
    src, consumer = layout
    (consumer / "shared").mkdir()
    dir_link(consumer / "shared" / "sdd", src / "_reversa_sdd" / "db")
    Owner(src, "unused").write()
    r = Run(consumer, tmp_path / "out", src, monkeypatch, capsys)

    assert r("--no-cluster") == 1

    err = capsys.readouterr().err
    assert "link not mirrored" in err and "shared/sdd/mem/design.md" in err
    assert not (r.gout / "graph.json").exists()


def test_runs_without_semantic_pass_do_not_check_the_owner(run, mounted):
    src, _consumer, _owner = mounted
    import shutil

    shutil.rmtree(src / "graphify-out")

    assert run("--code-only", "--no-cluster") == 0

    run.monkeypatch.setattr(run.mainmod.sys, "argv", [
        "graphify", "update", str(run.consumer), "--external-root", str(src), "--no-cluster"])
    _reset_stat_index()
    try:
        run.mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    assert "owner-first" not in run.capsys.readouterr().err


# --- --allow-external-extract ----------------------------------------------


def test_allow_flag_extracts_only_unusable_files_and_is_not_recorded(run, mounted):
    src, _consumer, _owner = mounted
    (src / DESIGN).write_text("# Design\n\nrevised shared design\n", encoding="utf-8")

    assert run("--no-cluster", "--allow-external-extract") == 0

    assert run.sent("revised shared design")
    assert not run.sent("shared document a")
    captured = run.capsys.readouterr()
    assert "content changed since the owner's extraction" in captured.err
    assert "1 from owner graph / 1 extracted here" in captured.out
    config = json.loads((run.gout / ".graphify_build.json").read_text(encoding="utf-8"))
    assert not any("allow" in k for k in config)

    assert run("--no-cluster") == 1


# --- token budget persistence ----------------------------------------------


def test_build_config_token_budget_replace_and_keep(tmp_path):
    from graphify.watch import (
        _read_build_external_roots,
        _read_build_token_budget,
        _write_build_config,
    )

    out = tmp_path / "graphify-out"
    assert _read_build_token_budget(out) is None
    _write_build_config(out, excludes=None, external_roots=["C:/E"])
    _write_build_config(out, excludes=None, token_budget=8000)
    assert _read_build_token_budget(out) == 8000
    assert _read_build_external_roots(out) == ["C:/E"]
    _write_build_config(out, excludes=None, gitignore=True)
    assert _read_build_token_budget(out) == 8000
    _write_build_config(out, excludes=None, token_budget=30000)
    assert _read_build_token_budget(out) == 30000


def test_extract_records_and_reuses_token_budget(tmp_path, monkeypatch, capsys):
    import graphify.__main__ as mainmod

    proj = tmp_path / "proj"
    proj.mkdir()
    doc = proj / "notes.md"
    doc.write_text("# Notes\n\nfirst\n", encoding="utf-8")
    budgets: list = []

    def _fake_corpus(files, **kw):
        budgets.append(kw.get("token_budget"))
        kw["on_chunk_done"](0, 1, {})
        nodes = [{"id": f"n_{Path(f).stem}", "label": "N", "file_type": "document",
                  "source_file": Path(f).relative_to(proj).as_posix()} for f in files]
        return {"nodes": nodes, "edges": [], "hyperedges": [], "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _fake_corpus)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)

    def _extract(*extra):
        _reset_stat_index()
        monkeypatch.setattr(mainmod.sys, "argv", [
            "graphify", "extract", str(proj), "--backend", "claude", "--no-cluster", *extra])
        try:
            mainmod.main()
        except SystemExit as exc:
            assert exc.code in (None, 0)

    cfg = proj / "graphify-out" / ".graphify_build.json"
    _extract("--token-budget", "8000", "--no-external-roots")
    assert budgets == [8000]
    assert json.loads(cfg.read_text(encoding="utf-8"))["token_budget"] == 8000
    capsys.readouterr()

    doc.write_text("# Notes\n\nsecond\n", encoding="utf-8")
    _extract()
    assert budgets[-1] == 8000
    out = capsys.readouterr().out
    assert "token budget 8000 (recorded in .graphify_build.json)" in out

    doc.write_text("# Notes\n\nthird\n", encoding="utf-8")
    _extract("--token-budget", "30000")
    assert budgets[-1] == 30000
    config = json.loads(cfg.read_text(encoding="utf-8"))
    assert config["token_budget"] == 30000 and config["external_roots"] == []


# --- owner_graph unit -------------------------------------------------------


def test_plan_reads_owner_graph_when_streaming_fails(mounted, monkeypatch):
    from graphify import graph_compare, owner_graph
    from graphify.detect import detect

    src, consumer, _owner = mounted
    _reset_stat_index()
    det = detect(consumer, external_roots=[src])
    roots = [src.resolve()]
    sem = owner_graph.external_files(det["files"], consumer, roots, kinds=owner_graph.SEMANTIC_KINDS)
    corpus = owner_graph.external_files(det["files"], consumer, roots)

    streamed = owner_graph.plan_owner_semantics(sem, corpus)

    def _boom(*a, **kw):
        raise RuntimeError("no element-wise read")

    monkeypatch.setattr(graph_compare, "_stream", _boom)
    fallback = owner_graph.plan_owner_semantics(sem, corpus)

    assert sorted(f.rel for f in streamed.taken) == sorted([DOC_A, DESIGN])
    assert streamed.nodes == fallback.nodes and streamed.edges == fallback.edges
    assert all("community" not in n and "norm_label" not in n for n in streamed.nodes)
    assert not streamed.refused
