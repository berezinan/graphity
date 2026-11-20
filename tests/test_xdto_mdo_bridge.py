"""1C:EDT metadata linked to XML schema nodes (xml-schema-types, fork delta,
capability edt-metadata-parsing). Samples are trimmed from ERP2's
`WebServices/CustomerOrdersExchange` and `XDTOPackages/CustomerOrders`.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify import cache as _cache
from graphify.extractors.base import _make_id
from graphify.extractors.bsl import extract_edt_mdo
from graphify.extractors.xsd import namespace_id, symbol_id

NS = "http://www.1c.ru/CustomerOrders/Exchange"
P = "CustomerOrders/Exchange"

_WS = """<?xml version="1.0" encoding="UTF-8"?>
<mdclass:WebService xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:core="http://g5.1c.ru/v8/dt/mcore" xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass" uuid="c0aa0d5d-598d-474f-b169-5b289e06a605">
  <name>CustomerOrdersExchange</name>
  <namespace>http://www.1c.ru/CustomerOrders/Exchange</namespace>
  <xdtoPackages xsi:type="core:ReferenceValue">
    <value>XDTOPackage.CustomerOrders</value>
  </xdtoPackages>
  <xdtoPackages xsi:type="core:StringValue">
    <value>http://v8.1c.ru/8.3/data/ext</value>
  </xdtoPackages>
  <xdtoPackages xsi:type="core:StringValue">
    <value>http://www.1c.ru/dmil</value>
  </xdtoPackages>
  <operations uuid="0d4f5d4a-6f1b-4d34-a2d3-7f1b1c3bbf01">
    <name>GetSettings</name>
    <xdtoReturningValueType>
      <name>Settings</name>
      <nsUri>http://www.1c.ru/CustomerOrders/Exchange</nsUri>
    </xdtoReturningValueType>
    <procedureName>GetSettings</procedureName>
    <parameters uuid="6f41a242-7b68-4655-8caf-52b4cc935fce">
      <name>MobileDeviceID</name>
      <xdtoValueType>
        <name>string</name>
        <nsUri>http://www.w3.org/2001/XMLSchema</nsUri>
      </xdtoValueType>
    </parameters>
    <parameters uuid="7a41a242-7b68-4655-8caf-52b4cc935fce">
      <name>Filter</name>
      <xdtoValueType>
        <name>JobTargetSettings</name>
        <nsUri>http://www.1c.ru/CustomerOrders/Exchange</nsUri>
      </xdtoValueType>
    </parameters>
  </operations>
</mdclass:WebService>
"""

_PKG_MDO = """<?xml version="1.0" encoding="UTF-8"?>
<mdclass:XDTOPackage xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass" uuid="d3e0dd8f-886a-4441-8f05-fd629da80f07">
  <name>CustomerOrders</name>
  <namespace>http://www.1c.ru/CustomerOrders/Exchange</namespace>
</mdclass:XDTOPackage>
"""

_PKG_XDTO = """<package xmlns="http://v8.1c.ru/8.1/xdto" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" targetNamespace="http://www.1c.ru/CustomerOrders/Exchange">
	<objectType name="Settings">
		<property name="Interval" type="xs:decimal"/>
	</objectType>
	<objectType name="JobTargetSettings">
		<property name="Code" type="xs:string"/>
	</objectType>
</package>
"""

_MODULE = """Функция GetSettings(MobileDeviceID, Filter)
	Возврат Неопределено;
КонецФункции
"""


def _project(root: Path, *, with_schema: bool = True) -> Path:
    src = root / "src"
    ws = src / "WebServices" / "CustomerOrdersExchange"
    ws.mkdir(parents=True)
    (ws / "CustomerOrdersExchange.mdo").write_text(_WS, encoding="utf-8")
    (ws / "Module.bsl").write_text(_MODULE, encoding="utf-8-sig")
    pkg = src / "XDTOPackages" / "CustomerOrders"
    pkg.mkdir(parents=True)
    (pkg / "CustomerOrders.mdo").write_text(_PKG_MDO, encoding="utf-8")
    if with_schema:
        (pkg / "Package.xdto").write_text(_PKG_XDTO, encoding="utf-8")
    return root


def _edges(r: dict) -> set[tuple]:
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    return {(labels.get(e["source"], e["source"]), labels.get(e["target"], e["target"]),
             e["relation"], e.get("context")) for e in r["edges"]}


def test_operation_and_parameter_types(tmp_path):
    proj = _project(tmp_path / "p")
    r = extract_edt_mdo(proj / "src/WebServices/CustomerOrdersExchange/CustomerOrdersExchange.mdo")
    edges = _edges(r)
    assert ("GetSettings", f"{P} · Settings", "references", "xdto-return-type") in edges
    assert ("Filter", f"{P} · JobTargetSettings", "references", "xdto-param-type") in edges
    by_id = {n["id"]: n for n in r["nodes"]}
    stub = by_id[symbol_id(NS, "type", "Settings")]
    assert stub["shared_anchor"] is True and not stub.get("xml_defined")
    [param] = [n for n in r["nodes"] if n["label"] == "MobileDeviceID"]
    assert param["type_ref"] == "{http://www.w3.org/2001/XMLSchema}string"
    assert not any(n["label"].endswith("string") for n in r["nodes"])


def test_xdto_packages_by_reference_and_by_namespace(tmp_path):
    proj = _project(tmp_path / "p")
    r = extract_edt_mdo(proj / "src/WebServices/CustomerOrdersExchange/CustomerOrdersExchange.mdo")
    edges = _edges(r)
    assert ("WebService.CustomerOrdersExchange", "XDTOPackage.CustomerOrders",
            "references", "xdtoPackages") in edges
    assert ("WebService.CustomerOrdersExchange", "http://www.1c.ru/dmil",
            "references", "xdtoPackages") in edges
    # The platform namespace yields neither an edge nor a node.
    assert not any("8.3/data/ext" in n["label"] for n in r["nodes"])
    # The service's own <namespace> is not a package reference.
    assert not any(t == NS for _, t, *_ in edges)


def test_package_links_its_namespace_and_schema(tmp_path):
    proj = _project(tmp_path / "p")
    r = extract_edt_mdo(proj / "src/XDTOPackages/CustomerOrders/CustomerOrders.mdo")
    edges = _edges(r)
    assert ("XDTOPackage.CustomerOrders", NS, "references", "namespace") in edges
    schema = proj / "src/XDTOPackages/CustomerOrders/Package.xdto"
    assert any(e["relation"] == "defines" and e["target"] == _make_id(str(schema))
               for e in r["edges"])


def test_package_without_schema_file(tmp_path):
    proj = _project(tmp_path / "p", with_schema=False)
    r = extract_edt_mdo(proj / "src/XDTOPackages/CustomerOrders/CustomerOrders.mdo")
    assert not any(e["relation"] == "defines" for e in r["edges"])
    assert ("XDTOPackage.CustomerOrders", NS, "references", "namespace") in _edges(r)


def test_extract_joins_service_package_and_schema(tmp_path, monkeypatch, capsys):
    import graphify.__main__ as mainmod

    proj = _project(tmp_path / "p")
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
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = graph.get("links") or graph.get("edges")

    settings = nodes[symbol_id(NS, "type", "Settings")]
    assert Path(settings["source_file"]).name == "Package.xdto"
    assert settings.get("xml_defined") is True
    assert len([n for n in graph["nodes"] if n.get("label") == f"{P} · Settings"]) == 1
    ns_node = nodes[namespace_id(NS)]
    assert Path(ns_node["source_file"]).name == "Package.xdto"

    def has(src_label, tgt_id, relation):
        return any(nodes[e["source"]]["label"] == src_label and e["target"] == tgt_id
                   and e["relation"] == relation for e in links)

    assert has("GetSettings", settings["id"], "references")
    assert has("XDTOPackage.CustomerOrders", namespace_id(NS), "references")
    assert any(e["target"] == settings["id"] and e["source"] == namespace_id(NS)
               and e["relation"] == "contains" for e in links)
    assert "minted by two different files" not in err
