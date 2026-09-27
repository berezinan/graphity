"""Полнота выборки кандидатов ArcadeDB для резолвера метки (без службы).

`ArcadeDBBackend` выбирает из базы узлы и отдаёт их эталонному
`_find_node_tiers`. Если выборка упустит узел, который эталон ставит в тир,
ArcadeDB ответит иначе, чем JSON. Здесь условия выборки вычисляются в Python
так же, как их вычислила бы база, и проверяется: первый этап
(`_label_upper_tier_filter`) покрывает тиры source_exact, exact и prefix, а
второй (`_label_substring_tier_filter`) — тир substring. Второй идёт, только
когда верхние тиры пусты, поэтому вместе они дают ответ полного графа.
"""
import re

import networkx as nx
from hypothesis import given, settings, strategies as st

from graphify.query_backend import _SEP, _label_substring_tier_filter, _label_upper_tier_filter
from graphify.serve import _find_node_tiers, _strip_diacritics

_NODES = [
    ("p1", "dbo_Price", "import/price_import.py"),
    ("p2", "dbo.Price", "sql/views/v_sales.sql"),
    ("p3", "Price.sql", "sql/tables/Price.sql"),
    ("p4", "ExternalDataSource.X.Table.dbo_Price", "src/X.mdo"),
    ("p5", "load_prices", "import/price_import.py"),
    ("p6", "load_prices()", "legacy/loader.py"),
    ("p7", "graph-first-guard.py", "hooks/graph-first-guard.py"),
    ("p8", "rust::module::Item", "src/lib.rs"),
    ("справочник_настройки_пользователей", "НастройкиПользователей", "src/Catalogs/НастройкиПользователей/Ёлка.bsl"),
    ("p10", "Café.résumé", "docs/Café.md"),
    ("p11", "ОбщийМодуль.Цены_Сервер", None),
    ("p12", r"C:\legacy\Module.bsl", "src/CommonModules/Цены/Ext/Module.bsl"),
]


def _graph(extra=()) -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, label, source in (*_NODES, *extra):
        attrs = {"label": label, "source_location": "L1"}
        if source is not None:
            attrs["source_file"] = source
        G.add_node(nid, **attrs)
    return G


def _field(nid: str, d: dict, expr: str) -> str | None:
    # Значения полей такими, какими их хранит загрузчик ArcadeDB.
    if expr == "norm_label":
        return _strip_diacritics(str(d.get("label", ""))).lower()
    if expr == "source_file":
        return d.get("source_file") or None
    if expr == "id":
        return nid
    raise AssertionError(expr)


def _holds(value: str | None, op: str, pattern: str) -> bool:
    if value is None:
        return False
    if op == "LIKE":
        # Семантика LIKE в ArcadeDB (проверена на службе): `%` — любая строка,
        # `?` — один символ, `_` — литерал, сравнение регистрозависимо.
        rx = "".join(".*" if c == "%" else "." if c == "?" else re.escape(c) for c in pattern)
    else:
        # MATCHES — Java-регулярка на всю строку. `\p{L}`/`\p{N}` и `\x{HH}` в
        # `re` нет: «не буква и не цифра» там `[\W_]`, символ — сам символ.
        assert op == "MATCHES", op
        rx = pattern.replace(_SEP, r"[\W_]")
        rx = re.sub(r"\\x\{([0-9A-F]+)\}", lambda m: re.escape(chr(int(m[1], 16))), rx)
    return re.fullmatch(rx, value) is not None


def _selected(G: nx.DiGraph, groups) -> set[str]:
    return {
        nid for nid, d in G.nodes(data=True)
        if any(all(_holds(_field(nid, d, e), op, p) for e, op, p in g) for g in groups)
    }


def _assert_complete(G: nx.DiGraph, label: str) -> None:
    tiers = _find_node_tiers(G, label)
    missing = set(tiers[3]) - _selected(G, _label_substring_tier_filter(label))
    assert not missing, f"{label!r}: второй этап упустил {sorted(missing)}"
    missing = {nid for t in tiers[:3] for nid in t} - _selected(G, _label_upper_tier_filter(label))
    assert not missing, f"{label!r}: первый этап упустил {sorted(missing)}"


def _variants(label: str, source: str | None) -> list[str]:
    out = [label, label.upper(), re.sub(r"[\W_]", " ", label), _strip_diacritics(label),
           label[:4], label[:-1]]
    if source:
        out += [source, source.upper(), _strip_diacritics(source), f"{source}::{label}"]
    return out


def test_every_tiered_node_is_selected():
    G = _graph()
    for nid, label, source in _NODES:
        for q in _variants(label, source) + [nid]:
            _assert_complete(G, q)


def test_punctuated_labels_are_found_at_all():
    G = _graph()
    for q in ("dbo_Price", "dbo.Price", "dbo Price", "Price.sql",
              "import/price_import.py::load_prices", "src/Catalogs/НастройкиПользователей/Ёлка.bsl",
              # Путь набран без диакритики: тир source_exact достижим только
              # через шаблон с одиночным символом на месте `е`/`и`.
              "src/Catalogs/НастроикиПользователеи/Елка.bsl",
              r"C:\legacy\Module.bsl"):
        assert any(_find_node_tiers(G, q)), q
        _assert_complete(G, q)


def test_matches_patterns_carry_no_dot():
    # ArcadeDB #5258: две точки в регулярке MATCHES роняют запрос («Nested
    # property access»), одна — компилирует шаблон на каждой строке.
    for nid, label, source in _NODES:
        for q in _variants(label, source):
            for f in (_label_substring_tier_filter, _label_upper_tier_filter):
                for g in f(q):
                    for _, op, p in g:
                        assert op != "MATCHES" or "." not in p, (q, p)


def test_upper_tier_filter_skips_substring_only_matches():
    # Смысл первого этапа: узел, найденный только по подстроке, он не берёт.
    G = _graph()
    assert _find_node_tiers(G, "price")[3]
    substring_only = set(_find_node_tiers(G, "price")[3])
    assert not substring_only & _selected(G, _label_upper_tier_filter("price"))


_ALPHABET = "abcdPriceDBO_.:/-\\ йёЁЙéÉприцеN0"
_text = st.text(alphabet=_ALPHABET, min_size=1, max_size=24)


@settings(max_examples=300, deadline=None)
@given(_text)
def test_random_labels_selection_is_complete(q):
    _assert_complete(_graph(), q)


@settings(max_examples=300, deadline=None)
@given(_text, _text, st.integers(0, 9))
def test_random_node_found_by_its_own_variants(label, source, which):
    # Случайные запросы почти не попадают в верхние тиры фиксированных узлов,
    # поэтому здесь запрос выводится из самого случайного узла.
    G = _graph(extra=[("rnd", label, source)])
    variants = _variants(label, source)
    _assert_complete(G, variants[which % len(variants)])
