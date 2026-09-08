"""Quality metrics for one semantic-extraction response.

Answers "how good is the graph this model returned", which `benchmark.py` does
not: that module measures token economy against a corpus, not extraction
quality. Everything here is computed from the model's own output, so no golden
graph is needed.

Two rules shape the design, both learned by measurement:

* **Validity is not quality.** A model that returns almost nothing always fits
  in the output budget and always parses; a model that builds a rich graph runs
  into the ceiling and gets truncated. Ranking by "share of valid responses"
  therefore inverts. `completion_rate` is kept as a *diagnostic* and is
  deliberately not part of `GraphMetrics`.
* **Topology needs enough edges to mean anything.** With two edges `star_score`
  is mechanically 0.50 whatever the model did, so below `MIN_EDGES` it is
  reported as ``None`` rather than as a good score.

Metrics cover all three arrays — ``nodes``, ``edges`` **and** ``hyperedges``.
Hyperedges carry the group semantics (who took part in what) that pairwise
edges cannot express, so a model that puts a group relation there must not be
scored as if it had produced nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

# The schema the extraction prompt declares (`llm._EXTRACTION_SYSTEM`). Copied
# rather than parsed out of the prompt text, and pinned by a drift test that
# fails if the prompt and this list stop agreeing.
NODE_FILE_TYPES = frozenset(
    {"code", "document", "paper", "image", "rationale", "concept"}
)
EDGE_RELATIONS = frozenset(
    {
        "calls",
        "implements",
        "references",
        "cites",
        "conceptually_related_to",
        "shares_data_with",
        "semantically_similar_to",
    }
)
# Hyperedges have their own, separate relation schema.
HYPEREDGE_RELATIONS = frozenset({"participate_in", "implement", "form"})

# Node ids the prompt asks for: lowercase snake_case stems.
_ID_RE = re.compile(r"^[a-z0-9_]+$")

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Fewest edges for which `star_score` carries information. Below this, the ratio
# is dictated by the edge count itself.
MIN_EDGES = 8

# Fewest runs a cell needs before its numbers may carry a verdict.
MIN_RUNS = 3

# A metadata header sits at the top of a document; a horizontal rule deeper than
# this is body punctuation, not the end of a header.
_HEADER_SCAN_CHARS = 2_000
_RULE_RE = re.compile(r"^-{3,}[ \t]*$", re.M)


@dataclass(frozen=True)
class GraphMetrics:
    """Quality of a single extracted graph fragment.

    ``None`` means *not applicable* (too little data to say anything), never
    "zero" — a metric that cannot be computed must not be read as a bad score.
    """

    nodes: int
    edges: int
    coverage: float           # nodes per 10 000 source chars
    star_score: float | None  # max out-degree / edges; None below MIN_EDGES
    ent_ent: int              # edges touching neither end of the hub
    id_compliance: float | None
    label_language: float | None
    relation_diversity: int
    relation_collapsed: bool
    enum_compliance: float | None
    hyper_count: int
    hyper_enum_compliance: float | None


def _ratio(ok: int, total: int) -> float | None:
    return ok / total if total else None


def _dominant_script(text: str) -> str | None:
    """The script a translated label would have left behind, or None.

    Only Cyrillic-vs-Latin is distinguished: that is the loss actually observed
    (Russian sources coming back as ``Supreme Court of Russia``). A Latin-script
    source returns None because this heuristic cannot see translation between
    two Latin-script languages — the metric is then not applicable, not perfect.
    """
    cyrillic = len(_CYRILLIC_RE.findall(text))
    return "cyrillic" if cyrillic > len(_LATIN_RE.findall(text)) else None


def measure_graph(
    fragment: dict,
    source_text: str,
    *,
    min_edges: int = MIN_EDGES,
) -> GraphMetrics:
    """Measure one parsed graph fragment against the text it was extracted from.

    ``fragment`` is the output of the pipeline's own parser
    (``llm._parse_llm_json``), not a strict JSON load: the parser tolerates a
    preamble and markdown fences, and scoring must not be stricter than the
    pipeline it stands in for.
    """
    nodes = [n for n in fragment.get("nodes") or [] if isinstance(n, dict)]
    edges = [e for e in fragment.get("edges") or [] if isinstance(e, dict)]
    hyper = [h for h in fragment.get("hyperedges") or [] if isinstance(h, dict)]

    # Topology: the hub is whichever node the model hung the most edges off.
    out_degree = Counter(e.get("source") for e in edges if e.get("source"))
    if out_degree and edges:
        hub, hub_out = out_degree.most_common(1)[0]
        star = hub_out / len(edges) if len(edges) >= min_edges else None
        ent_ent = sum(
            1 for e in edges if e.get("source") != hub and e.get("target") != hub
        )
    else:
        hub, star, ent_ent = None, None, 0

    relations = [e.get("relation") for e in edges if e.get("relation")]
    diversity = len(set(relations))

    # One enum-compliance figure over both paired arrays: a node's file_type and
    # an edge's relation are the two places the prompt's paired schema applies.
    enum_ok = sum(1 for n in nodes if n.get("file_type") in NODE_FILE_TYPES)
    enum_ok += sum(1 for e in edges if e.get("relation") in EDGE_RELATIONS)

    script = _dominant_script(source_text)
    if script == "cyrillic":
        labels = [str(n.get("label", "")) for n in nodes]
        label_language = _ratio(
            sum(1 for label in labels if _CYRILLIC_RE.search(label)), len(labels)
        )
    else:
        label_language = None

    return GraphMetrics(
        nodes=len(nodes),
        edges=len(edges),
        coverage=10_000 * len(nodes) / len(source_text) if source_text else 0.0,
        star_score=star,
        ent_ent=ent_ent,
        id_compliance=_ratio(
            sum(1 for n in nodes if _ID_RE.match(str(n.get("id", "")))), len(nodes)
        ),
        label_language=label_language,
        relation_diversity=diversity,
        relation_collapsed=diversity == 1,
        enum_compliance=_ratio(enum_ok, len(nodes) + len(edges)),
        hyper_count=len(hyper),
        hyper_enum_compliance=_ratio(
            sum(1 for h in hyper if h.get("relation") in HYPEREDGE_RELATIONS),
            len(hyper),
        ),
    )


@dataclass(frozen=True)
class CellStat:
    """One metric across the runs of one cell: the median, and the spread."""

    median: float | None
    low: float | None
    high: float | None
    defined: int  # runs where the metric was applicable


@dataclass(frozen=True)
class CellSummary:
    """A model × condition cell, aggregated over its runs.

    ``underpowered`` marks a cell measured in fewer than ``MIN_RUNS`` runs. Such
    a cell may be reported but must not carry a verdict: ``temperature = 0`` buys
    no determinism — the same call has been seen returning both a truncated
    non-graph and a 19-node graph, and a MoE model returned both a collapsed and
    a diverse graph on identical input.
    """

    stats: dict[str, CellStat]
    runs: int
    underpowered: bool


def aggregate_cell(runs: list[GraphMetrics]) -> CellSummary:
    """Median and spread per metric over the runs of one cell.

    Runs where a metric was not applicable are skipped for that metric only, so
    a cell where two runs of three produced too few edges still reports the
    third run's `star_score` — over ``defined = 1``, which the reader can see.
    """
    stats: dict[str, CellStat] = {}
    for field in GraphMetrics.__dataclass_fields__:
        values = sorted(
            float(v) for v in (getattr(r, field) for r in runs) if v is not None
        )
        if values:
            stats[field] = CellStat(
                median=median(values),
                low=values[0],
                high=values[-1],
                defined=len(values),
            )
        else:
            stats[field] = CellStat(median=None, low=None, high=None, defined=0)
    return CellSummary(stats=stats, runs=len(runs), underpowered=len(runs) < MIN_RUNS)


def strip_metadata_header(text: str) -> str:
    """Text after a leading metadata block, or the text unchanged.

    A corpus that ships a manifest usually repeats those very fields in a header
    above a horizontal rule (``**Дело:** … **Исход:** …``). Feed that header to
    the model and a manifest check stops measuring extraction and starts
    measuring transcription — every model scores near 100% by echoing the line.

    Only a rule inside the first ``_HEADER_SCAN_CHARS`` counts, so a rule used
    as body punctuation further down cannot truncate the document.
    """
    match = _RULE_RE.search(text, 0, _HEADER_SCAN_CHARS)
    return text[match.end():].lstrip("\n") if match else text


def completion_rate(responses: list[tuple[str, dict]]) -> float | None:
    """Share of responses that both stopped cleanly and parsed into a graph.

    A **diagnostic**, not a quality score: it is exactly the number that ranks
    a 4-node stub above a 21-node graph, because the stub never reaches the
    output ceiling. Report it beside the metrics, never instead of them.

    Each response is ``(finish_reason, parsed_fragment)``.
    """
    if not responses:
        return None
    good = sum(
        1
        for finish, fragment in responses
        if finish == "stop"
        and any((fragment or {}).get(k) for k in ("nodes", "edges", "hyperedges"))
    )
    return good / len(responses)
