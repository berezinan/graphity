"""2026-11-04 speed-ups: Rust tiers, zero-copy views, spike tokenisation."""
import pytest

import spike_patches
from graphify import serve
from graphify.serve import _find_node_tiers, _query_terms


def test_cyrillic_short_words_are_dropped():
    assert serve._is_searchable is spike_patches.is_searchable
    assert _query_terms("Товары в пути и на складе") == ["товары", "пути", "складе"]
    assert _query_terms("1с в") == ["1с"]
    # Unsearchable tokens never reach the stopword fallback, as upstream ("in of").
    assert _query_terms("в и") == []
    assert _query_terms("in of") == []
    assert _query_terms("in of form") == ["form"]     # Latin rule unchanged


@pytest.mark.parametrize("repos", [None, ("r1", "r1", "r2"), ("r1", "r2", "r3")])
@pytest.mark.parametrize("files", [("a.py", "b.py", "c.py"), ("a.py", "a.py", "c.py"), ("", "a.py", "")])
def test_linear_ambiguity_message_is_byte_identical(repos, files):
    import networkx as nx
    G = nx.DiGraph()
    for k, f in enumerate(files):
        d = {"label": "foo()", "source_file": f}
        if repos:
            d["repo"] = repos[k]
        G.add_node(f"n{k}", **d)
    rivals = list(G.nodes)
    assert spike_patches.ambiguity_message(G, "foo", rivals) == \
        spike_patches.UPSTREAM_AMBIGUITY(G, "foo", rivals)


def test_views_are_zero_copy(psn_snap):
    sb = psn_snap
    v = memoryview(sb.snap.view("l.src")).cast("I")
    assert v.readonly and len(v) == sb.snap.link_count
    assert v[0] == sb.snap.link_endpoints(0)[0]
    with pytest.raises(KeyError):
        sb.snap.view("no.such.block")


def test_tiers_match_find_node_tiers(graphs_psn, psn_snap, qset_psn):
    G, sb = graphs_psn, psn_snap
    labels = qset_psn["explain"] + qset_psn["query"] + [
        "Module.bsl", "Form", "ФормаУправляемая", "x", "()", "ПолучитьЗначениеСвойства",
        "Catalog.ДополнительныеОтчетыИОбработки", "form.form", "src"]
    for label in labels:
        if sb._special(label):
            continue
        want = [list(t) for t in _find_node_tiers(G, label)]
        got = [sb._ids_of(t) for t in sb._tiers(label, 2)]
        assert got == want, label
