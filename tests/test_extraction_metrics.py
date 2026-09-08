"""Extraction-quality metrics: one test per normative scenario.

Fixtures are synthetic on purpose. Each one embodies a failure mode that was
actually measured against real models, but the numbers under test are the
metric definitions, not any particular model's score.
"""
from __future__ import annotations

from graphify.extraction_metrics import (
    EDGE_RELATIONS,
    HYPEREDGE_RELATIONS,
    MIN_EDGES,
    MIN_RUNS,
    NODE_FILE_TYPES,
    GraphMetrics,
    aggregate_cell,
    completion_rate,
    measure_graph,
    strip_metadata_header,
)

ACT = """\
# Определение — Верховный Суд РФ

**Дело:** А47-10662/2024 · **Дата:** 2026-05-21 · **Инстанция:** вс · \
**Исход (словарь):** отменить
**Судья:** Завьялова Т.В.
**Источник:** https://sudact.ru/vsrf/doc/04cVGWlZMCUF/

---

ВЕРХОВНЫЙ СУД РОССИЙСКОЙ ФЕДЕРАЦИИ ОПРЕДЕЛЕНИЕ г. Москва Дело № А47-10662/2024 \
Судебная коллегия по экономическим спорам рассмотрела кассационную жалобу.
"""

# ~19 500 chars — one `_TEXT_SLICE_CHARS` slice, so `coverage` here is on the
# same scale as the measured table (deepseek 12–15, gpt-oss ~10, mistral 2).
RU_SOURCE = "Верховный Суд отменил постановление суда апелляционной инстанции. " * 300
EN_SOURCE = "The Supreme Court reversed the appellate ruling. " * 300


def _node(nid: str, label: str, file_type: str = "concept") -> dict:
    return {"id": nid, "label": label, "file_type": file_type}


def _edge(src: str, dst: str, relation: str = "references") -> dict:
    return {"source": src, "target": dst, "relation": relation}


# ── topology: a star is a mention list, not a graph ─────────────────────────

def test_star_output_scores_one_and_has_no_entity_to_entity_edges():
    fragment = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(9)],
        "edges": [_edge("n0", f"n{i}") for i in range(1, 9)],  # 8 edges, all from n0
        "hyperedges": [],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.star_score == 1.0
    assert m.ent_ent == 0


def test_real_graph_scores_below_one_and_links_non_hub_nodes():
    edges = [_edge("n0", f"n{i}") for i in range(1, 5)]
    edges += [_edge("n1", "n2"), _edge("n2", "n3"), _edge("n3", "n4"), _edge("n4", "n5")]
    fragment = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(6)],
        "edges": edges,
        "hyperedges": [],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.star_score == 0.5  # 4 of 8 edges leave the hub
    assert m.ent_ent == 4


def test_min_edges_threshold_does_not_decide_who_gets_scored():
    # Edge counts measured per model. The two stubs sit at 2 and 4 edges, every
    # model that built a graph sits at 13 or above — nothing lands near the
    # threshold, so 6, 8 and 12 select the same set. The default is not tuned.
    measured = {
        "deepseek-v4-flash": 24,
        "gpt-oss-20b": 20,
        "qwen3.6-35b-a3b": 24,
        "qwen3-8b": 13,
        "qwen2.5-7b-q5": 4,
        "mistral-nemo-12b": 2,
    }

    def scored(threshold: int) -> set[str]:
        return {
            name
            for name, edges in measured.items()
            if measure_graph(
                {
                    "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(edges + 1)],
                    "edges": [_edge("n0", f"n{i}") for i in range(1, edges + 1)],
                    "hyperedges": [],
                },
                RU_SOURCE,
                min_edges=threshold,
            ).star_score
            is not None
        }

    assert scored(6) == scored(MIN_EDGES) == scored(12)


def test_two_edge_graph_gets_no_topological_score():
    # The measured stub: 4 nodes, 2 edges. star_score would be a flattering 0.50
    # no matter what the model did, so it must not be scored at all.
    fragment = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(4)],
        "edges": [_edge("n0", "n1"), _edge("n0", "n2")],
        "hyperedges": [],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.edges < MIN_EDGES
    assert m.star_score is None
    assert m.coverage < 3.0  # the low coverage is what carries the verdict


# ── schema compliance ───────────────────────────────────────────────────────

def test_file_type_outside_the_schema_counts_as_a_violation():
    fragment = {
        "nodes": [_node("n0", "ООО Заря", file_type="organization")],
        "edges": [],
        "hyperedges": [],
    }
    assert measure_graph(fragment, RU_SOURCE).enum_compliance == 0.0


def test_cyrillic_identifiers_fail_id_compliance():
    fragment = {
        "nodes": [_node("верховный_суд", "Верховный Суд"), _node("ooo_zarya", "ООО Заря")],
        "edges": [],
        "hyperedges": [],
    }
    assert measure_graph(fragment, RU_SOURCE).id_compliance == 0.5


# ── label language ──────────────────────────────────────────────────────────

def test_translated_labels_are_recorded_as_a_loss():
    fragment = {
        "nodes": [_node("n0", "Supreme Court of Russia"), _node("n1", "Zarya LLC")],
        "edges": [],
        "hyperedges": [],
    }
    assert measure_graph(fragment, RU_SOURCE).label_language == 0.0


def test_labels_in_the_source_language_score_full():
    fragment = {
        "nodes": [_node("n0", "Верховный Суд"), _node("n1", "ООО Заря")],
        "edges": [],
        "hyperedges": [],
    }
    assert measure_graph(fragment, RU_SOURCE).label_language == 1.0


def test_latin_source_makes_label_language_inapplicable_not_perfect():
    fragment = {"nodes": [_node("n0", "Supreme Court")], "edges": [], "hyperedges": []}
    assert measure_graph(fragment, EN_SOURCE).label_language is None


# ── relation diversity ──────────────────────────────────────────────────────

def test_collapsed_relations_are_flagged_as_a_failure_mode():
    fragment = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(5)],
        "edges": [_edge(f"n{i}", f"n{i + 1}") for i in range(4)],  # all "references"
        "hyperedges": [],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.relation_diversity == 1
    assert m.relation_collapsed is True


def test_distinct_relations_are_not_flagged():
    fragment = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(4)],
        "edges": [
            _edge("n0", "n1", "references"),
            _edge("n1", "n2", "cites"),
            _edge("n2", "n3", "implements"),
        ],
        "hyperedges": [],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.relation_diversity == 3
    assert m.relation_collapsed is False


# ── hyperedges are scored, not dropped ──────────────────────────────────────

def test_group_relation_in_a_hyperedge_counts_and_is_not_penalised():
    # Five cartel members tied by one hyperedge instead of ten pairwise edges.
    members = [_node(f"org{i}", f"ООО Участник {i}") for i in range(5)]
    fragment = {
        "nodes": members,
        "edges": [],
        "hyperedges": [
            {
                "id": "cartel",
                "label": "Состав картеля",
                "nodes": [n["id"] for n in members],
                "relation": "participate_in",
            }
        ],
    }
    m = measure_graph(fragment, RU_SOURCE)
    assert m.hyper_count == 1
    assert m.hyper_enum_compliance == 1.0
    assert m.star_score is None  # no pairwise edges is not a topology failure


def test_hyperedge_relation_outside_its_own_schema_is_a_violation():
    fragment = {
        "nodes": [],
        "edges": [],
        "hyperedges": [{"id": "h", "nodes": ["a", "b", "c"], "relation": "references"}],
    }
    # "references" is a valid *pairwise* relation and still wrong here.
    assert "references" in EDGE_RELATIONS
    assert measure_graph(fragment, RU_SOURCE).hyper_enum_compliance == 0.0


# ── validity is a diagnostic, not a quality score ───────────────────────────

def test_completion_rate_is_not_part_of_the_quality_metrics():
    assert "completion_rate" not in GraphMetrics.__dataclass_fields__


def test_richer_graph_loses_on_completion_rate_and_still_wins_on_quality():
    stub = {
        "nodes": [_node(f"n{i}", "Zarya LLC") for i in range(4)],
        "edges": [_edge("n0", "n1"), _edge("n0", "n2")],
        "hyperedges": [],
    }
    rich = {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(21)],
        "edges": [_edge("n0", f"n{i}") for i in range(1, 6)]
        + [_edge(f"n{i}", f"n{i + 1}", "cites") for i in range(6, 20)],
        "hyperedges": [],
    }
    # The stub always fits the output budget; the rich graph gets truncated.
    assert completion_rate([("stop", stub), ("stop", stub)]) == 1.0
    assert completion_rate([("stop", rich), ("length", {})]) == 0.5

    a, b = measure_graph(stub, RU_SOURCE), measure_graph(rich, RU_SOURCE)
    assert a.star_score is None and b.star_score is not None
    assert b.coverage > a.coverage
    assert b.ent_ent > a.ent_ent
    assert b.label_language == 1.0 and a.label_language == 0.0


def test_completion_rate_rejects_a_stopped_but_empty_response():
    empty = {"nodes": [], "edges": [], "hyperedges": []}
    assert completion_rate([("stop", empty), ("stop", empty)]) == 0.0
    assert completion_rate([]) is None


# ── metrics run on the pipeline's own parser output ─────────────────────────

def test_metrics_read_a_fenced_response_through_the_pipeline_parser():
    from graphify import llm

    raw = (
        "Вот извлечённый граф:\n\n```json\n"
        '{"nodes":[{"id":"sud","label":"Верховный Суд","file_type":"document"}],'
        '"edges":[],"hyperedges":[]}\n```\n'
    )
    m = measure_graph(llm._parse_llm_json(raw), RU_SOURCE)
    assert m.nodes == 1
    assert m.id_compliance == 1.0
    assert m.enum_compliance == 1.0


# ── a cell is a distribution, not a point ───────────────────────────────────

def _run(edges: int, relations: list[str]) -> dict:
    return {
        "nodes": [_node(f"n{i}", f"Узел {i}") for i in range(edges + 1)],
        "edges": [
            _edge("n0", f"n{i}", relations[i % len(relations)])
            for i in range(1, edges + 1)
        ],
        "hyperedges": [],
    }


def test_cell_reports_the_spread_not_a_single_value():
    # The measured MoE case: three back-to-back runs collapsed to one relation,
    # an isolated run of the same model on the same input reached four.
    runs = [
        measure_graph(_run(24, ["references"]), RU_SOURCE),
        measure_graph(_run(24, ["references"]), RU_SOURCE),
        measure_graph(_run(24, ["references"]), RU_SOURCE),
        measure_graph(_run(24, ["references", "implements", "calls", "cites"]), RU_SOURCE),
    ]
    diversity = aggregate_cell(runs).stats["relation_diversity"]
    assert diversity.median == 1
    assert (diversity.low, diversity.high) == (1, 4)  # the spread is visible


def test_single_run_cell_is_marked_underpowered():
    one = [measure_graph(_run(24, ["references"]), RU_SOURCE)]
    assert aggregate_cell(one).underpowered is True
    assert aggregate_cell(one * MIN_RUNS).underpowered is False


def test_inapplicable_runs_are_skipped_per_metric_not_per_cell():
    # Two stub runs (no topology) and one real graph: star_score survives, and
    # says out loud that it rests on a single run.
    runs = [
        measure_graph(_run(2, ["references"]), RU_SOURCE),
        measure_graph(_run(2, ["references"]), RU_SOURCE),
        measure_graph(_run(24, ["references"]), RU_SOURCE),
    ]
    summary = aggregate_cell(runs)
    assert summary.stats["star_score"].defined == 1
    assert summary.stats["nodes"].defined == 3  # every run had a node count


# ── the header must not leak the manifest into the model's input ────────────

def test_header_is_stripped_before_the_document_is_fed_to_the_model():
    body = strip_metadata_header(ACT)
    # The fields the manifest will be checked against are gone from the input…
    assert "**Дело:**" not in body
    assert "2026-05-21" not in body
    assert "отменить" not in body
    # …while the act itself survives intact.
    assert body.startswith("ВЕРХОВНЫЙ СУД")
    assert "кассационную жалобу" in body


def test_case_number_still_reachable_from_the_body_itself():
    # Not every manifest field survives stripping as a literal: the case number
    # is restated in the act, the outcome and instance are not.
    assert "А47-10662/2024" in strip_metadata_header(ACT)


def test_document_without_a_rule_is_left_alone():
    text = "# Заголовок\n\nТело документа без горизонтальной черты.\n"
    assert strip_metadata_header(text) == text


def test_rule_deep_in_the_body_does_not_truncate_the_document():
    text = "# Заголовок\n\n" + "прозаический текст. " * 200 + "\n---\nхвост\n"
    assert strip_metadata_header(text) == text


# ── the copied schema must not drift from the prompt ────────────────────────

def test_schema_constants_still_match_the_extraction_prompt():
    from graphify import llm

    for value in NODE_FILE_TYPES | EDGE_RELATIONS | HYPEREDGE_RELATIONS:
        assert value in llm._EXTRACTION_SYSTEM, value
