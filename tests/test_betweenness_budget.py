"""suggest_questions caps the cost of sampled betweenness with a work budget:
unchanged (100 pivots) up to 500k nodes+edges, down to 10 pivots beyond that."""
from __future__ import annotations

import networkx as nx

import graphify.analyze as analyze
from graphify.analyze import _betweenness_pivots, suggest_questions


def test_pivot_count_follows_the_budget():
    assert _betweenness_pivots(1000, 5000) is None            # small graph: exact
    assert _betweenness_pivots(1001, 2000) == 100             # as before
    assert _betweenness_pivots(200_000, 300_000) == 100       # n + m = 500k: still 100
    assert _betweenness_pivots(400_000, 800_000) == 41
    assert _betweenness_pivots(2_019_923, 4_903_605) == 10    # ERP2: floor


def _graph(n: int) -> "tuple[nx.Graph, dict[int, list[str]]]":
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"n{i}", label=f"sym{i}", source_file=f"f{i % 50}.py", file_type="code")
    for i in range(n - 1):
        G.add_edge(f"n{i}", f"n{i + 1}", relation="calls", confidence="EXTRACTED")
    communities = {c: [f"n{i}" for i in range(n) if i % 4 == c] for c in range(4)}
    return G, communities


def test_suggest_questions_passes_the_budgeted_pivot_count(monkeypatch):
    G, communities = _graph(1500)
    seen = {}
    real = nx.betweenness_centrality

    def spy(graph, k=None, seed=None, **kw):
        seen["k"], seen["seed"] = k, seed
        return real(graph, k=k, seed=seed, **kw)

    monkeypatch.setattr(analyze.nx, "betweenness_centrality", spy)
    suggest_questions(G, communities, {})
    assert seen == {"k": 100, "seed": 42}

    monkeypatch.setattr(analyze, "_BETWEENNESS_BUDGET", 1)    # force the floor
    questions = suggest_questions(G, communities, {})
    assert seen == {"k": 10, "seed": 42}
    assert any(q["type"] == "bridge_node" for q in questions)


def test_suggest_questions_is_deterministic(monkeypatch):
    G, communities = _graph(1500)
    monkeypatch.setattr(analyze, "_BETWEENNESS_BUDGET", 1)
    assert suggest_questions(G, communities, {}) == suggest_questions(G, communities, {})
