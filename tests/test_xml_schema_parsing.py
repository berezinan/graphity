"""XML schemas — W3C XSD, schemas inside WSDL <types> — as namespace / type /
property nodes (xml-schema-types, capability xml-schema-parsing).

The XDTO grammar and the cross-file merge have their own test files; this one
covers the shared model (ids, labels, built-in namespaces, QName scoping,
refusals) and the XSD/WSDL adapters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graphify.extractors import xsd as X
from graphify.extractors.xsd import (
    extract_xsd,
    is_builtin_namespace,
    namespace_prefix,
    symbol_id,
    symbol_label,
)

ED22 = "http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22"
ED21 = "http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.21"
XS_STRING = "{http://www.w3.org/2001/XMLSchema}string"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _by_label(result: dict) -> dict[str, dict]:
    return {n["label"]: n for n in result["nodes"]}


def _edges(result: dict) -> set[tuple[str, str, str, str | None]]:
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return {(labels[e["source"]], labels[e["target"]], e["relation"], e.get("context"))
            for e in result["edges"]}


# ── Model: prefixes, ids, built-ins ─────────────────────────────────────────

@pytest.mark.parametrize("uri,prefix", [
    (ED22, "EnterpriseData/1.22"),
    ("http://v8.1c.ru/agent/scripts/1.0", "scripts/1.0"),
    ("http://www.1c.ru/CustomerOrders/Exchange", "CustomerOrders/Exchange"),
    ("http://file.api.update.onec.ru", "file.api.update.onec.ru"),
    ("http://example.com/orders", "example.com/orders"),
    ("urn:example:orders:v1", "orders:v1"),
    ("", ""),
])
def test_namespace_prefix(uri, prefix):
    assert namespace_prefix(uri) == prefix


@pytest.mark.parametrize("uri,builtin", [
    ("http://www.w3.org/2001/XMLSchema", True),
    ("http://www.w3.org/XML/1998/namespace", True),
    ("http://v8.1c.ru/8.1/data/core", True),
    ("http://v8.1c.ru/8.3/data/ext", True),
    ("http://v8.1c.ru/8.1/data/enterprise/current-config", True),
    (ED22, False),
    ("http://v8.1c.ru/agent/scripts/1.0", False),
])
def test_builtin_namespaces(uri, builtin):
    assert is_builtin_namespace(uri) is builtin


def test_symbol_identity():
    assert symbol_id(ED22, "type", "Банки") != symbol_id(ED21, "type", "Банки")
    assert symbol_id(ED22, "type", "Order") != symbol_id(ED22, "element", "Order")
    assert symbol_label(ED22, "type", "СправочникСсылка.Банки") == \
        "EnterpriseData/1.22 · СправочникСсылка.Банки"
    assert symbol_label(ED22, "element", "Order") == "EnterpriseData/1.22 · Order [element]"


# ── QName scoping ────────────────────────────────────────────────────────────

_SCOPED = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:t:a">
  <xs:complexType name="Holder">
    <xs:sequence>
      <xs:element xmlns:p="urn:t:one" name="First" type="p:T"/>
      <xs:element xmlns:p="urn:t:two" name="Second" type="p:T"/>
      <xs:element name="Third" type="zz:T"/>
      <xs:element name="Fourth" type="xs:string"/>
    </xs:sequence>
  </xs:complexType>
</xs:schema>
"""


def test_prefix_declared_on_the_element_and_rebound(tmp_path):
    r = extract_xsd(_write(tmp_path, "scoped.xsd", _SCOPED))
    assert "error" not in r
    edges = _edges(r)
    assert ("t:a · Holder.First", "t:one · T", "references", "property-type") in edges
    assert ("t:a · Holder.Second", "t:two · T", "references", "property-type") in edges
    # Unknown prefix: no edge, no invented node.
    assert not any(src == "t:a · Holder.Third" for src, *_ in edges)
    assert not any(n["label"].endswith("· T") and "zz" in n["label"] for n in r["nodes"])
    # Built-in: attribute, not edge.
    props = _by_label(r)
    assert props["t:a · Holder.Fourth"]["type_ref"] == XS_STRING
    assert not any(n["label"].endswith("string") for n in r["nodes"])


# ── Refusals ─────────────────────────────────────────────────────────────────

def test_doctype_is_refused(tmp_path):
    p = _write(tmp_path, "dtd.xsd", '<?xml version="1.0"?>\n<!DOCTYPE xs:schema SYSTEM "x.dtd">\n'
               '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>')
    r = extract_xsd(p)
    assert r["nodes"] == [] and "DOCTYPE" in r["error"] and "dtd.xsd" in r["error"]


def test_malformed_is_refused(tmp_path):
    r = extract_xsd(_write(tmp_path, "bad.xsd", "<xs:schema xmlns:xs='x'><unclosed></xs:schema>"))
    assert r["nodes"] == [] and "bad.xsd" in r["error"] and "parse error" in r["error"]


def test_oversized_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "_PROJECT_XML_MAX_BYTES", 10)
    r = extract_xsd(_write(tmp_path, "big.xsd", _SCOPED))
    assert r["nodes"] == [] and "too large" in r["error"] and "big.xsd" in r["error"]


# ── XSD adapter ──────────────────────────────────────────────────────────────

_ORDERS = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           xmlns:tns="http://example.com/orders"
           xmlns:c="http://example.com/common"
           targetNamespace="http://example.com/orders">
  <xs:import namespace="http://example.com/common" schemaLocation="common.xsd"/>
  <xs:import namespace="http://www.w3.org/XML/1998/namespace"/>
  <xs:element name="Order" type="tns:Order"/>
  <xs:element name="SpecialOrder" type="tns:Special" substitutionGroup="tns:Order"/>
  <xs:complexType name="Order">
    <xs:sequence>
      <xs:element name="Id" type="xs:string"/>
      <xs:choice>
        <xs:element name="Customer" type="c:Party"/>
        <xs:element name="Guest">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="Email">
                <xs:simpleType>
                  <xs:restriction base="xs:string">
                    <xs:pattern value=".+@.+"/>
                    <xs:maxLength value="100"/>
                  </xs:restriction>
                </xs:simpleType>
              </xs:element>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:choice>
      <xs:element ref="tns:Note"/>
      <xs:group ref="tns:Audit"/>
    </xs:sequence>
    <xs:attribute name="Id" type="xs:int"/>
    <xs:attributeGroup ref="tns:Tracking"/>
  </xs:complexType>
  <xs:complexType name="Special">
    <xs:complexContent>
      <xs:extension base="tns:Order">
        <xs:sequence><xs:element name="Reason" type="xs:string"/></xs:sequence>
      </xs:extension>
    </xs:complexContent>
  </xs:complexType>
  <xs:simpleType name="Status">
    <xs:restriction base="xs:token">
      <xs:enumeration value="new"/>
      <xs:enumeration value="done"/>
    </xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="Statuses"><xs:list itemType="tns:Status"/></xs:simpleType>
  <xs:simpleType name="Code"><xs:union memberTypes="tns:Status xs:int"/></xs:simpleType>
  <xs:element name="Note" type="xs:string"/>
  <xs:group name="Audit"><xs:sequence><xs:element name="Created" type="xs:dateTime"/></xs:sequence></xs:group>
  <xs:attributeGroup name="Tracking"><xs:attribute name="Rev" type="xs:int"/></xs:attributeGroup>
</xs:schema>
"""

O = "example.com/orders"


@pytest.fixture
def orders(tmp_path):
    return extract_xsd(_write(tmp_path, "orders.xsd", _ORDERS))


def test_globals_and_symbol_spaces(orders):
    nodes = _by_label(orders)
    assert nodes[f"{O} · Order"]["xml_kind"] == "type"
    assert nodes[f"{O} · Order [element]"]["xml_kind"] == "element"
    assert nodes[f"{O} · Audit [group]"]["xml_kind"] == "group"
    assert nodes[f"{O} · Tracking [attributeGroup]"]["xml_kind"] == "attributeGroup"
    assert nodes["http://example.com/orders"]["xml_kind"] == "namespace"
    for label in (f"{O} · Order", f"{O} · Order [element]", "http://example.com/orders"):
        assert nodes[label]["shared_anchor"] is True and nodes[label]["xml_defined"] is True
        assert nodes[label]["namespace"] == "http://example.com/orders"
    edges = _edges(orders)
    assert ("orders.xsd", "http://example.com/orders", "contains", None) in edges
    assert ("http://example.com/orders", f"{O} · Order", "contains", None) in edges


def test_properties_through_compositors(orders):
    edges = _edges(orders)
    for prop in ("Id", "Customer", "Guest", "@Id"):
        assert (f"{O} · Order", f"{O} · Order.{prop}", "contains", None) in edges
    # Element and attribute of one name stay two properties.
    nodes = _by_label(orders)
    assert nodes[f"{O} · Order.Id"]["id"] != nodes[f"{O} · Order.@Id"]["id"]
    assert not any(n.get("label", "").endswith(("sequence", "choice")) for n in orders["nodes"])


def test_anonymous_types_nest(orders):
    edges = _edges(orders)
    assert (f"{O} · Order.Guest", f"{O} · Order.Guest#type", "contains", None) in edges
    assert (f"{O} · Order.Guest#type", f"{O} · Order.Guest#type.Email", "contains", None) in edges
    email_type = _by_label(orders)[f"{O} · Order.Guest#type.Email#type"]
    assert email_type["xml_kind"] == "anonymous-type"
    assert email_type["base_ref"] == XS_STRING
    assert email_type["xml_pattern"] == [".+@.+"] and email_type["xml_maxLength"] == "100"


def test_references_and_inheritance(orders):
    edges = _edges(orders)
    assert (f"{O} · Order [element]", f"{O} · Order", "references", "property-type") in edges
    assert (f"{O} · Order.Customer", "example.com/common · Party", "references", "property-type") in edges
    assert (f"{O} · Special", f"{O} · Order", "inherits", None) in edges
    assert (f"{O} · Special", f"{O} · Special.Reason", "contains", None) in edges
    assert (f"{O} · Order", f"{O} · Note [element]", "references", "element-ref") in edges
    assert (f"{O} · Order", f"{O} · Audit [group]", "references", "group-ref") in edges
    assert (f"{O} · Order", f"{O} · Tracking [attributeGroup]", "references", "attribute-group-ref") in edges
    assert (f"{O} · SpecialOrder [element]", f"{O} · Order [element]", "references", "substitution-group") in edges
    assert (f"{O} · Statuses", f"{O} · Status", "references", "list-item") in edges
    assert (f"{O} · Code", f"{O} · Status", "references", "union-member") in edges
    nodes = _by_label(orders)
    assert nodes[f"{O} · Code"]["member_type_refs"] == ["{http://www.w3.org/2001/XMLSchema}int"]
    assert nodes[f"{O} · Audit [group]"] and (f"{O} · Audit [group]", f"{O} · Audit [group].Created", "contains", None) in edges


def test_enumeration_is_an_attribute(orders):
    status = _by_label(orders)[f"{O} · Status"]
    assert status["xml_enumeration"] == ["new", "done"]
    assert not any(n["label"].endswith(("new", "done")) for n in orders["nodes"])


def test_imports_skip_builtin(orders):
    edges = _edges(orders)
    assert ("http://example.com/orders", "http://example.com/common", "imports", None) in edges
    assert not any("XML/1998" in n["label"] for n in orders["nodes"])


def test_referenced_symbol_is_a_stub(orders):
    party = _by_label(orders)["example.com/common · Party"]
    assert party["shared_anchor"] is True and not party.get("xml_defined")


def test_chameleon_schema_stays_in_no_namespace(tmp_path):
    r = extract_xsd(_write(tmp_path, "cham.xsd",
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">'
        '<xs:complexType name="Item"><xs:sequence><xs:element name="A" type="xs:string"/>'
        '</xs:sequence></xs:complexType></xs:schema>'))
    nodes = _by_label(r)
    assert nodes["Item"]["namespace"] == "" and nodes["Item"]["id"] == symbol_id("", "type", "Item")
    assert ("cham.xsd", "Item", "contains", None) in _edges(r)
    assert not any(n.get("xml_kind") == "namespace" for n in r["nodes"])


def test_dotted_type_name_does_not_collide_with_property(tmp_path):
    r = extract_xsd(_write(tmp_path, "dots.xsd",
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:t:d">'
        '<xs:complexType name="A"><xs:sequence><xs:element name="B" type="xs:string"/></xs:sequence></xs:complexType>'
        '<xs:complexType name="A.B"/></xs:schema>'))
    nodes = _by_label(r)
    assert nodes["t:d · A.B"]["xml_kind"] == "type"
    assert nodes["t:d · A.B"]["id"] != [n for n in r["nodes"] if n.get("xml_kind") == "property"][0]["id"]


# ── WSDL ─────────────────────────────────────────────────────────────────────

_WSDL = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://schemas.xmlsoap.org/wsdl/"
             xmlns:xsd1="http://file.api.update.onec.ru"
             targetNamespace="http://file.api.update.onec.ru">
  <types>
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://file.api.update.onec.ru">
      <xs:element name="getFiles" type="xsd1:getFiles"/>
      <xs:complexType name="getFiles"><xs:sequence><xs:element name="version" type="xs:string"/></xs:sequence></xs:complexType>
    </xs:schema>
  </types>
  <message name="getFilesRequest"><part name="p" element="xsd1:getFiles"/></message>
  <portType name="UpdatePort"/>
  <binding name="UpdateBinding" type="xsd1:UpdatePort"/>
</definitions>
"""


def test_wsdl_reads_only_types(tmp_path):
    r = extract_xsd(_write(tmp_path, "WsDefinitions.wsdl", _WSDL))
    labels = set(_by_label(r))
    p = "file.api.update.onec.ru"
    assert {f"{p} · getFiles", f"{p} · getFiles [element]", f"{p} · getFiles.version"} <= labels
    # The prefix xsd1 is declared on <definitions>, outside the schema.
    assert (f"{p} · getFiles [element]", f"{p} · getFiles", "references", "property-type") in _edges(r)
    assert not any(word in label for label in labels
                   for word in ("getFilesRequest", "UpdatePort", "UpdateBinding"))


# ── Registration ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["a.xsd", "b.wsdl", "Package.xdto"])
def test_schema_files_are_code(tmp_path, name):
    from graphify.detect import FileType, classify_file
    from graphify.extract import _get_extractor
    from graphify.extractors.xdto import extract_xdto

    p = tmp_path / name
    p.write_text("<x/>", encoding="utf-8")
    assert classify_file(p) == FileType.CODE
    assert _get_extractor(p) is (extract_xdto if name.endswith(".xdto") else extract_xsd)


def test_extract_on_a_corpus_needs_no_llm(tmp_path):
    from graphify.extract import extract

    p = _write(tmp_path, "orders.xsd", _ORDERS)
    out = extract([p], root=tmp_path, parallel=False)
    assert any(n.get("label") == f"{O} · Order" for n in out["nodes"])
    assert out.get("input_tokens", 0) == 0
