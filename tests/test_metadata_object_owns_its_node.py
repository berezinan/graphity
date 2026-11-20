"""A metadata object's node belongs to its own .mdo (fork delta,
metadata-object-owns-its-node).

Found on 1CConsole/Marketing 2026-11-15: `CommonCommand.ОбновитьИмпортеров` is
named by `Configuration.mdo` (registration), by its own
`CommonCommands/ОбновитьИмпортеров/ОбновитьИмпортеров.mdo` and by its command
module. All copies carry the FQN id and `shared_anchor`; dedup kept the one from
`Configuration.mdo`, never gap-filled across files, lost the own .mdo's `uuid` and
warned "minted by two different files". The own .mdo then had no node at all.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from graphify import cache as _cache

NID = "commoncommand_обновитьимпортеров"
LABEL = "CommonCommand.ОбновитьИмпортеров"
UUID = "7861e50b-1333-4834-b5a9-bbf3397bd85a"
OWN = "src/CommonCommands/ОбновитьИмпортеров/ОбновитьИмпортеров.mdo"


def _records() -> list[dict]:
    base = {"id": NID, "label": LABEL, "file_type": "code", "shared_anchor": True,
            "_origin": "ast"}
    return [
        {**base, "source_file": "src/Configuration/Configuration.mdo", "source_location": "L40"},
        {**base, "source_file": OWN, "source_location": "L1", "uuid": UUID,
         "object_belonging": "Native"},
        {**base, "source_file": "src/CommonModules/М/Module.bsl", "source_location": "L7"},
    ]


def test_own_mdo_survives_and_attributes_are_merged(capsys):
    from graphify.dedup import deduplicate_entities

    results = []
    for perm in itertools.permutations(_records()):
        out, _ = deduplicate_entities([dict(n) for n in perm], [], communities={}, root=None)
        [node] = [n for n in out if n["id"] == NID]
        results.append(node)
    err = capsys.readouterr().err

    node = results[0]
    assert node["source_file"] == OWN
    assert node["uuid"] == UUID and node["object_belonging"] == "Native"
    assert all(r == node for r in results), "arrival order does not matter"
    assert "minted by two different files" not in err


def test_without_own_mdo_attributes_still_merge(capsys):
    from graphify.dedup import deduplicate_entities

    recs = [r for r in _records() if r["source_file"] != OWN]
    recs[1]["note"] = "from the module"
    out, _ = deduplicate_entities([dict(n) for n in recs], [], communities={}, root=None)
    [node] = [n for n in out if n["id"] == NID]
    assert node.get("note") == "from the module"
    assert "minted by two different files" not in capsys.readouterr().err


def test_same_named_object_of_another_kind_is_not_the_own_mdo(capsys):
    # ERP2 (xml-schema-types acceptance): WebService.EquipmentService references
    # XDTOPackage.EquipmentService. Both own .mdo files end in
    # `EquipmentService/EquipmentService.mdo`; only the kind folder tells them apart.
    from graphify.dedup import deduplicate_entities

    nid, label = "xdtopackage_equipmentservice", "XDTOPackage.EquipmentService"
    own = "src/XDTOPackages/EquipmentService/EquipmentService.mdo"
    base = {"id": nid, "label": label, "file_type": "code", "shared_anchor": True,
            "_origin": "ast"}
    recs = [
        {**base, "source_file": "src/WebServices/EquipmentService/EquipmentService.mdo"},
        {**base, "source_file": own, "uuid": "u-pkg"},
    ]
    for perm in itertools.permutations(recs):
        out, _ = deduplicate_entities([dict(n) for n in perm], [], communities={}, root=None)
        [node] = [n for n in out if n["id"] == nid]
        assert node["source_file"] == own and node["uuid"] == "u-pkg"


def test_unmarked_copy_and_nested_subsystem(capsys):
    """A form names the object by its FQN without the mark — still one entity; a
    nested subsystem `Subsystem.A.B` is owned by `A/Subsystems/B/B.mdo`, not by
    the parent subsystem that lists it."""
    from graphify.dedup import deduplicate_entities

    recs = _records()
    recs[2] = {k: v for k, v in recs[2].items() if k != "shared_anchor"}
    recs[2]["source_file"] = "src/CommonForms/Ф/Form.form"
    sid, slabel = "subsystem_а_б", "Subsystem.А.Б"
    sub = {"id": sid, "label": slabel, "file_type": "code", "shared_anchor": True, "_origin": "ast"}
    recs += [{**sub, "source_file": "src/Subsystems/А/А.mdo"},
             {**sub, "source_file": "src/Subsystems/А/Subsystems/Б/Б.mdo", "uuid": "u-б"}]
    out, _ = deduplicate_entities([dict(n) for n in recs], [], communities={}, root=None)
    by_id = {n["id"]: n for n in out}
    assert by_id[NID]["source_file"] == OWN and by_id[NID]["uuid"] == UUID
    assert by_id[sid]["source_file"].endswith("А/Subsystems/Б/Б.mdo") and by_id[sid]["uuid"] == "u-б"
    assert "minted by two different files" not in capsys.readouterr().err


def test_nested_member_named_by_journal_column(capsys):
    """UNF 2026-11-16: the document's own .mdo names its attribute bare
    (`Архивный`, with uuid), a document journal column references it by an FQN
    stub carrying the mark. Same id by construction — one entity, no warning,
    the own .mdo survives. A common form named bare by its Form.form is a
    different label shape (two segments) and stays a collision (D2/D4)."""
    from graphify.dedup import deduplicate_entities

    def pair(nid, fqn, bare, own, ref):
        return [
            {"id": nid, "label": bare, "file_type": "code", "source_file": own,
             "source_location": "L1", "uuid": "u-" + bare, "_origin": "ast"},
            {"id": nid, "label": fqn, "file_type": "code", "source_file": ref,
             "source_location": "L1", "shared_anchor": True, "_origin": "ast"},
        ]

    nid = "document_чекккмвозврат_attribute_архивный"
    own = "src/Documents/ЧекККМВозврат/ЧекККМВозврат.mdo"
    xid = "extension_volkova_document_д_attribute_р"
    xown = "Volkova/src/Documents/Д/Д.mdo"
    recs = (pair(nid, "Document.ЧекККМВозврат.Attribute.Архивный", "Архивный", own,
                 "src/DocumentJournals/ЧекиККМ/ЧекиККМ.mdo")
            + pair(xid, "Extension.Volkova.Document.Д.Attribute.Р", "Р", xown,
                   "Volkova/src/DocumentJournals/Ж/Ж.mdo"))
    edges = [
        {"source": "document_чекккмвозврат", "target": nid, "relation": "contains",
         "confidence": "EXTRACTED", "source_file": own},
        {"source": "documentjournal_чекиккм_column_архивный", "target": nid,
         "relation": "references", "confidence": "EXTRACTED",
         "source_file": "src/DocumentJournals/ЧекиККМ/ЧекиККМ.mdo"},
    ]
    for perm in itertools.permutations(recs):
        out, out_edges = deduplicate_entities([dict(n) for n in perm], [dict(e) for e in edges],
                                              communities={}, root=None)
        by_id = {n["id"]: n for n in out}
        assert by_id[nid]["source_file"] == own and by_id[nid]["uuid"] == "u-Архивный"
        assert by_id[xid]["source_file"] == xown and by_id[xid]["uuid"] == "u-Р"
        assert {e["relation"] for e in out_edges if e["target"] == nid} == {"contains", "references"}
    assert "minted by two different files" not in capsys.readouterr().err

    form = pair("commonform_выбортовара", "CommonForm.ВыборТовара", "ВыборТовара",
                "src/CommonForms/ВыборТовара/Form.form",
                "src/CommonForms/ВыборТовара/ВыборТовара.mdo")
    deduplicate_entities(form, [], communities={}, root=None)
    assert "minted by two different files" in capsys.readouterr().err


def test_non_anchor_collisions_keep_the_upstream_rule(capsys):
    from graphify.dedup import deduplicate_entities

    recs = [{k: v for k, v in r.items() if k != "shared_anchor"} for r in _records()]
    out, _ = deduplicate_entities([dict(n) for n in recs], [], communities={}, root=None)
    [node] = [n for n in out if n["id"] == NID]
    assert "uuid" not in node or node["source_file"] == OWN
    assert "minted by two different files" in capsys.readouterr().err


# ── end to end ────────────────────────────────────────────────────────────────

_CONF = """<?xml version="1.0" encoding="UTF-8"?>
<mdclass:Configuration xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass" uuid="c1c77aa8-3d02-4024-9c86-167a14702ff0">
  <name>Конфа</name>
  <commonCommands>CommonCommand.ОбновитьИмпортеров</commonCommands>
</mdclass:Configuration>
"""
_CMD = f"""<?xml version="1.0" encoding="UTF-8"?>
<mdclass:CommonCommand xmlns:mdclass="http://g5.1c.ru/v8/dt/metadata/mdclass" uuid="{UUID}">
  <name>ОбновитьИмпортеров</name>
  <group>ActionsPanelTools</group>
</mdclass:CommonCommand>
"""
_MODULE = """&НаКлиенте
Процедура ОбработкаКоманды(ПараметрКоманды, ПараметрыВыполненияКоманды)
	Сообщить("x");
КонецПроцедуры
"""


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "src" / "Configuration").mkdir(parents=True)
    (proj / "src" / "Configuration" / "Configuration.mdo").write_text(_CONF, encoding="utf-8")
    cmd = proj / "src" / "CommonCommands" / "ОбновитьИмпортеров"
    cmd.mkdir(parents=True)
    (cmd / "ОбновитьИмпортеров.mdo").write_text(_CMD, encoding="utf-8")
    (cmd / "CommandModule.bsl").write_text(_MODULE, encoding="utf-8-sig")
    return proj


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


def test_extract_attributes_the_command_to_its_own_mdo(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path)
    err = _run(proj, monkeypatch, capsys)
    graph = json.loads((proj / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    nodes = [n for n in graph["nodes"] if n.get("label") == LABEL]
    assert len(nodes) == 1, nodes
    node = nodes[0]
    assert Path(node["source_file"]).as_posix().endswith("ОбновитьИмпортеров/ОбновитьИмпортеров.mdo")
    assert node.get("uuid") == UUID
    links = graph.get("links") or graph.get("edges")
    rels = {(e.get("relation"), Path(str(e.get("source_file", ""))).name)
            for e in links if node["id"] in (e.get("source"), e.get("target"))}
    assert any(r == "contains" and f == "Configuration.mdo" for r, f in rels), rels
    assert any(r == "defines" for r, _ in rels), rels
    assert "minted by two different files" not in err

    # Nothing changed: the own .mdo has a node, so the #2543 heal leaves it alone.
    err = _run(proj, monkeypatch, capsys)
    assert "ОбновитьИмпортеров.mdo" not in err, err
    again = json.loads((proj / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert [n for n in again["nodes"] if n.get("label") == LABEL] == nodes

    # Only the registration changes: the incremental merge keeps the node on its own .mdo.
    conf = proj / "src" / "Configuration" / "Configuration.mdo"
    conf.write_text(_CONF.replace("<name>Конфа</name>", "<name>Конфа2</name>"), encoding="utf-8")
    _run(proj, monkeypatch, capsys)
    third = json.loads((proj / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    [n3] = [n for n in third["nodes"] if n.get("label") == LABEL]
    assert Path(n3["source_file"]).as_posix().endswith("ОбновитьИмпортеров/ОбновитьИмпортеров.mdo")
    assert n3.get("uuid") == UUID
