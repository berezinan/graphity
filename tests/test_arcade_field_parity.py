"""What ArcadeDB readers show is what the loaders write (arcadedb-loader-field-parity).

`explain` selected `Rel.source_file`/`source_location`, `get_community` selected
`Node.community_name`, and neither loader wrote them: on the default backend every
connection lost its call site and every community its name, while the parity
suite compared neither. Measured on a live database: 0 of 33 796 edges carried a
file.

The edge file is stored only where it cannot be recovered from the edge's source
vertex (99.96 % of ERP2 edges repeat it). These tests pin both halves: the answer
matches the JSON reference, and the storage rule holds on a database of its own.
They skip when no server answers.
"""

import json

import pytest
from networkx.readwrite import json_graph

from graphify.query_backend import ArcadeDBBackend, JsonBackend, _edge_doc, _id_key

try:  # same credential discovery the parity tests use
    from tests.test_arcade_backend import _PW, _URL, _make_digraph, _parse_host_port
except Exception:  # pragma: no cover - import shape differs when run as a file
    from test_arcade_backend import _PW, _URL, _make_digraph, _parse_host_port  # type: ignore

_DB = "graphify_field_parity_test"


@pytest.fixture()
def arc():
    host, port = _parse_host_port(_URL)
    backend = ArcadeDBBackend(_DB, host=host, port=port, password=_PW)
    if not backend.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    yield backend
    try:
        backend._server(f"drop database {_DB}")
    except Exception:
        pass


def _load(backend, G, tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    backend.ensure_database(drop=True)
    backend.load_from_graph_json(str(p))


def _stored_edge_file(backend, src: str, tgt: str):
    """``Rel.source_file`` as stored: missing key -> None, so absence and "" differ."""
    rows = backend._run(
        "MATCH (a:Node {id_key:$s})-[r:Rel]->(b:Node {id_key:$t}) RETURN r.source_file AS sf",
        kind="query", language="cypher", params={"s": _id_key(src), "t": _id_key(tgt)})
    assert len(rows) == 1, rows
    return rows[0].get("sf")


def _at(backend, label, other):
    return next((c.source_file, c.source_location)
                for c in backend.explain(label).connections if c.label == other)


@pytest.mark.parametrize("edge,src_file,written,stored", [
    ({"source_file": "a.py"}, "a.py", True, None),      # = источник, записан тут же: не хранится
    ({"source_file": "b.py"}, "a.py", True, "b.py"),    # отличается от источника
    ({}, "a.py", True, ""),                              # файла нет: явный пустой маркер
    ({"source_file": None}, "a.py", True, ""),
    ({"source_file": "a.py"}, "a.py", False, "a.py"),   # источник не пишется: хранится всегда
])
def test_edge_doc_storage_rule(edge, src_file, written, stored):
    doc = _edge_doc({"relation": "calls", "source_location": "L1", **edge}, src_file, written)
    assert doc.get("source_file", None) == stored
    assert ("source_file" in doc) == (stored is not None)
    assert doc["source_location"] == "L1"


def test_full_load_stores_edge_file_only_where_needed(arc, tmp_path):
    _load(arc, _make_digraph(), tmp_path)
    assert _stored_edge_file(arc, "n1", "n2") is None               # = файл источника
    assert _stored_edge_file(arc, "n2", "n3") == "build.py"         # = файл цели
    assert _stored_edge_file(arc, "n3", "n4") == "glue/wiring.py"   # третий файл
    assert _stored_edge_file(arc, "p5", "p1") == ""                 # файла нет


def test_sync_edge_from_unchanged_node_keeps_its_file(arc, tmp_path):
    """Edge n1 -> n2 repeats its source's file. Only n2's file changes, so n1 is
    not rewritten: the edge must carry its own file instead of trusting whatever
    n1 holds in the database (a shared node's stored file can drift)."""
    G = _make_digraph()
    _load(arc, G, tmp_path)

    arc.sync_graph(G, changed_sources={"cluster.py"}, pruned_sources=set())

    assert _stored_edge_file(arc, "n1", "n2") == "extract.py"
    assert _at(arc, "extract", "cluster") == _at(JsonBackend(G), "extract", "cluster") \
        == ("extract.py", "L12")


def test_full_load_stores_one_name_per_community(arc, tmp_path):
    _load(arc, _make_digraph(), tmp_path)
    rows = arc._run("SELECT cid, name FROM Community")
    assert sorted((r["cid"], r["name"]) for r in rows) == \
        [(0, "Extraction"), (1, "Сборка")]            # сообщества 2 и 3 без имени
    assert arc.get_community(1).name == "Сборка"
    assert arc.get_community(2).name is None


def test_relabel_without_moves_reaches_get_community(arc, tmp_path):
    _load(arc, _make_digraph(), tmp_path)
    stats = arc.update_communities({}, {0: 2, 1: 2, 2: 1, 3: 7},
                                   names={0: "Извлечение", 1: "Сборка", 2: "Community 2", 3: "Цены"})
    assert stats["reconciled"]
    assert arc.get_community(0).name == "Извлечение"
    assert arc.get_community(3).name == "Цены"


def test_sync_leaves_community_names_alone(arc, tmp_path):
    G = _make_digraph()
    _load(arc, G, tmp_path)
    arc.sync_graph(G, changed_sources={"cluster.py"}, pruned_sources=set())
    assert arc.get_community(0).name == "Extraction"


def test_database_without_community_type_reads_no_name(arc, tmp_path):
    """A database loaded before `Community` existed answers without names
    instead of failing, until its one-off reload."""
    _load(arc, _make_digraph(), tmp_path)
    arc._run("DROP TYPE Community UNSAFE")
    assert arc.get_community(0).name is None
    assert {m.label for m in arc.get_community(0).members} == {"extract", "cluster"}
