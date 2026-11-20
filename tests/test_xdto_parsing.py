"""1C XDTO packages (`Package.xdto`) on the shared XML schema model
(xml-schema-types, fork delta). Samples are trimmed from ERP2's
`XDTOPackages/AgentScripts` and `EnterpriseData_1_22_1`.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify import cache as _cache
from graphify.extractors.xdto import extract_xdto
from graphify.extractors.xsd import symbol_id

ED = "EnterpriseData/1.22"
XS = "{http://www.w3.org/2001/XMLSchema}"

_AGENT = """<package xmlns="http://v8.1c.ru/8.1/xdto" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" targetNamespace="http://v8.1c.ru/agent/scripts/1.0">
	<import namespace="http://v8.1c.ru/8.1/data/core"/>
	<objectType name="UpdateFile">
		<property xmlns:d3p1="http://v8.1c.ru/8.1/data/core" name="Id" type="d3p1:UUID"/>
		<property name="Extension" type="xs:string"/>
	</objectType>
</package>
"""

_ED = """<package xmlns="http://v8.1c.ru/8.1/xdto" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" targetNamespace="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22">
	<import namespace="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/Common"/>
	<property xmlns:d2p1="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22" name="Body" type="d2p1:Body"/>
	<valueType name="СтавкаНДС" base="xs:string" variety="Atomic" maxLength="20">
		<enumeration xsi:type="xs:string">БезНДС</enumeration>
		<enumeration xsi:type="xs:string">НДС20</enumeration>
	</valueType>
	<valueType name="ИНН" base="xs:string" variety="Atomic" minLength="10" maxLength="12">
		<pattern>[0-9]{10,12}</pattern>
	</valueType>
	<objectType name="КлючевыеСвойства" open="true" sequenced="true">
		<property name="Ссылка" type="xs:string"/>
	</objectType>
	<objectType xmlns:d2p1="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22" name="СправочникСсылка.Банки" base="d2p1:КлючевыеСвойства">
		<property name="БИК" type="d2p1:ИНН"/>
	</objectType>
	<objectType name="ЛюбаяСсылка">
		<property name="СсылкаНаОбъект">
			<typeDef xsi:type="ObjectType" ordered="false" sequenced="true">
				<property xmlns:d5p1="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22" name="БанкиСсылка" type="d5p1:СправочникСсылка.Банки" lowerBound="0"/>
			</typeDef>
		</property>
		<property name="Комментарий">
			<typeDef xsi:type="ValueType" base="xs:string" variety="Atomic" maxLength="100"/>
		</property>
		<property xmlns:d3p1="http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22" ref="d3p1:Body" lowerBound="0"/>
	</objectType>
	<objectType name="СправочникСсылка">
		<property name="Банки" type="xs:string"/>
	</objectType>
</package>
"""


def _write(tmp_path: Path, folder: str, text: str) -> Path:
    d = tmp_path / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / "Package.xdto"
    p.write_text(text, encoding="utf-8")
    return p


def _by_label(r: dict) -> dict[str, dict]:
    return {n["label"]: n for n in r["nodes"]}


def _edges(r: dict) -> set[tuple]:
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    return {(labels[e["source"]], labels[e["target"]], e["relation"], e.get("context"))
            for e in r["edges"]}


def test_object_type_properties_and_builtin_refs(tmp_path):
    r = extract_xdto(_write(tmp_path, "AgentScripts", _AGENT))
    nodes = _by_label(r)
    assert nodes["scripts/1.0 · UpdateFile"]["xml_kind"] == "type"
    assert nodes["scripts/1.0 · UpdateFile.Id"]["type_ref"] == "{http://v8.1c.ru/8.1/data/core}UUID"
    assert nodes["scripts/1.0 · UpdateFile.Extension"]["type_ref"] == f"{XS}string"
    edges = _edges(r)
    assert ("Package.xdto", "http://v8.1c.ru/agent/scripts/1.0", "contains", None) in edges
    assert ("http://v8.1c.ru/agent/scripts/1.0", "scripts/1.0 · UpdateFile", "contains", None) in edges
    # The platform namespace import yields neither an edge nor a node.
    assert not any(e[2] == "imports" for e in edges)
    assert not any("data/core" in n["label"] for n in r["nodes"])


def test_value_types_keep_facets_and_enumerations_as_attributes(tmp_path):
    nodes = _by_label(extract_xdto(_write(tmp_path, "ED", _ED)))
    vat = nodes[f"{ED} · СтавкаНДС"]
    assert vat["xml_enumeration"] == ["БезНДС", "НДС20"]
    assert vat["xml_maxLength"] == "20" and vat["base_ref"] == f"{XS}string"
    inn = nodes[f"{ED} · ИНН"]
    assert inn["xml_pattern"] == ["[0-9]{10,12}"] and inn["xml_minLength"] == "10"
    assert not any(label.endswith(("БезНДС", "НДС20")) for label in nodes)


def test_inheritance_references_and_global_property(tmp_path):
    r = extract_xdto(_write(tmp_path, "ED", _ED))
    edges = _edges(r)
    assert (f"{ED} · СправочникСсылка.Банки", f"{ED} · КлючевыеСвойства", "inherits", None) in edges
    assert (f"{ED} · СправочникСсылка.Банки.БИК", f"{ED} · ИНН", "references", "property-type") in edges
    assert (f"{ED} · ЛюбаяСсылка", f"{ED} · Body [element]", "references", "element-ref") in edges
    assert _by_label(r)[f"{ED} · Body [element]"]["xml_kind"] == "element"
    assert ("http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22",
            "http://v8.1c.ru/edi/edi_stnd/EnterpriseData/Common", "imports", None) in edges


def test_type_def_is_an_anonymous_type_under_its_property(tmp_path):
    r = extract_xdto(_write(tmp_path, "ED", _ED))
    edges = _edges(r)
    owner = f"{ED} · ЛюбаяСсылка.СсылкаНаОбъект"
    anon = f"{owner}#type"
    assert (owner, anon, "contains", None) in edges
    assert (anon, f"{anon}.БанкиСсылка", "contains", None) in edges
    assert (f"{anon}.БанкиСсылка", f"{ED} · СправочникСсылка.Банки", "references", "property-type") in edges
    nodes = _by_label(r)
    assert nodes[anon]["xml_kind"] == "anonymous-type"
    comment = nodes[f"{ED} · ЛюбаяСсылка.Комментарий#type"]
    assert comment["xml_maxLength"] == "100" and comment["base_ref"] == f"{XS}string"


def test_dotted_type_and_property_of_its_prefix_stay_apart(tmp_path):
    # Type `СправочникСсылка.Банки` and property `Банки` of type `СправочникСсылка`
    # share a label (the label format cannot tell them apart), but not an id:
    # make_id folds the dot, so without the member token they would be one node.
    r = extract_xdto(_write(tmp_path, "ED", _ED))
    same = [n for n in r["nodes"] if n["label"] == f"{ED} · СправочникСсылка.Банки"]
    assert sorted(n["xml_kind"] for n in same) == ["property", "type"]
    assert len({n["id"] for n in same}) == 2
    [dotted] = [n for n in same if n["xml_kind"] == "type"]
    assert dotted["id"] == symbol_id("http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22",
                                     "type", "СправочникСсылка.Банки")


def test_not_a_package_is_an_error(tmp_path):
    r = extract_xdto(_write(tmp_path, "X", "<root/>"))
    assert r["nodes"] == [] and "not an XDTO package" in r["error"]


_XSD_SAME_NS = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://v8.1c.ru/agent/scripts/1.0">
  <xs:complexType name="UpdateFile"><xs:sequence><xs:element name="Extension" type="xs:string"/></xs:sequence></xs:complexType>
</xs:schema>
"""


def test_xsd_and_xdto_of_one_namespace_share_nodes(tmp_path, monkeypatch, capsys):
    import graphify.__main__ as mainmod

    proj = tmp_path / "proj"
    _write(proj, "src/XDTOPackages/AgentScripts", _AGENT)
    (proj / "schemas").mkdir()
    (proj / "schemas" / "agent.xsd").write_text(_XSD_SAME_NS, encoding="utf-8")
    _cache._stat_index_root = None
    _cache._stat_index_anchor = None
    _cache._stat_index = {}
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "extract", str(proj), "--code-only"])
    try:
        mainmod.main()
    except SystemExit as exc:
        assert not exc.code, capsys.readouterr()
    err = capsys.readouterr().err

    graph = json.loads((proj / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    for label in ("scripts/1.0 · UpdateFile", "scripts/1.0 · UpdateFile.Extension",
                  "http://v8.1c.ru/agent/scripts/1.0"):
        assert len([n for n in graph["nodes"] if n.get("label") == label]) == 1, label
    assert "minted by two different files" not in err
