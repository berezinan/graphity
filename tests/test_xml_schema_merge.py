"""One XML schema symbol, one node — attributed to the file that declares it
(xml-schema-types, D6).

Every global symbol carries an id built from (namespace, symbol space, name)
and `shared_anchor`; a referring file mints a stub under the same id and label.
The declaring record (`xml_defined`) must survive dedup whatever the arrival
order, take the stubs' missing attributes, and not be reported as a collision.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from graphify import cache as _cache
from graphify.extractors.xsd import symbol_id, symbol_label

NS = "http://example.com/common"
NID = symbol_id(NS, "type", "Party")
LABEL = symbol_label(NS, "type", "Party")
DEF = "schemas/common.xsd"


def _records() -> list[dict]:
    base = {"id": NID, "label": LABEL, "file_type": "code", "shared_anchor": True,
            "namespace": NS, "xml_kind": "type", "_origin": "ast"}
    return [
        {**base, "source_file": "schemas/orders.xsd", "source_location": "L12"},
        {**base, "source_file": DEF, "source_location": "L4", "xml_defined": True},
        {**base, "source_file": "src/WebServices/S/S.mdo", "source_location": "L1",
         "note": "from a stub"},
    ]


def test_declaration_survives_any_order(capsys):
    from graphify.dedup import deduplicate_entities

    results = []
    for perm in itertools.permutations(_records()):
        out, _ = deduplicate_entities([dict(n) for n in perm], [], communities={}, root=None)
        [node] = [n for n in out if n["id"] == NID]
        results.append(node)
    err = capsys.readouterr().err

    node = results[0]
    assert node["source_file"] == DEF and node["source_location"] == "L4"
    assert node["xml_defined"] is True
    assert node.get("note") == "from a stub", "stubs gap-fill the declaration"
    assert all(r == node for r in results)
    assert "minted by two different files" not in err


def test_without_declaration_the_stub_stays(capsys):
    from graphify.dedup import deduplicate_entities

    recs = [r for r in _records() if r["source_file"] != DEF]
    out, _ = deduplicate_entities([dict(n) for n in recs], [], communities={}, root=None)
    [node] = [n for n in out if n["id"] == NID]
    assert not node.get("xml_defined")
    assert "minted by two different files" not in capsys.readouterr().err


_COMMON = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://example.com/common">
  <xs:complexType name="Party"><xs:sequence><xs:element name="Name" type="xs:string"/></xs:sequence></xs:complexType>
</xs:schema>
"""

_ORDERS = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:c="http://example.com/common"
           targetNamespace="http://example.com/orders">
  <xs:import namespace="http://example.com/common"/>
  <xs:complexType name="Order"><xs:sequence><xs:element name="Buyer" type="c:Party"/></xs:sequence></xs:complexType>
</xs:schema>
"""

_INVOICES = _ORDERS.replace("orders", "invoices").replace("Order", "Invoice")


def _run(proj: Path, monkeypatch, capsys) -> str:
    import graphify.__main__ as mainmod

    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "extract", str(proj), "--code-only"])
    try:
        mainmod.main()
    except SystemExit as exc:
        assert not exc.code, capsys.readouterr()
    return capsys.readouterr().err


def test_extract_merges_references_onto_the_declaration(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "proj"
    (proj / "schemas").mkdir(parents=True)
    # Referrers sort before the declaration on purpose.
    (proj / "schemas" / "a_orders.xsd").write_text(_ORDERS, encoding="utf-8")
    (proj / "schemas" / "b_invoices.xsd").write_text(_INVOICES, encoding="utf-8")
    (proj / "schemas" / "z_common.xsd").write_text(_COMMON, encoding="utf-8")
    err = _run(proj, monkeypatch, capsys)

    graph = json.loads((proj / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    party = [n for n in graph["nodes"] if n.get("label") == LABEL]
    assert len(party) == 1, party
    assert Path(party[0]["source_file"]).name == "z_common.xsd"
    ns_nodes = [n for n in graph["nodes"] if n.get("label") == NS]
    assert len(ns_nodes) == 1 and Path(ns_nodes[0]["source_file"]).name == "z_common.xsd"
    links = graph.get("links") or graph.get("edges")
    into_party = {Path(e["source_file"]).name for e in links
                  if e.get("target") == party[0]["id"] and e.get("relation") == "references"}
    assert into_party == {"a_orders.xsd", "b_invoices.xsd"}
    assert "minted by two different files" not in err
