"""BSL (1C / OneScript) and 1C:EDT metadata extractors.

Moved verbatim from graphify/extract.py (fork delta: tree-sitter-bsl language
support plus 1C:EDT .mdo / .rights / Form.form / .dcs ingestion).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from graphify.extractors.base import (
    _LANGUAGE_BUILTIN_GLOBALS,
    _PROJECT_XML_MAX_BYTES,
    _file_stem,
    _make_id,
    _project_xml_is_safe,
    _read_text,
)

# ── BSL (1C / OneScript) extractor (custom walk) ──────────────────────────────

# OneScript `#Использовать <lib>` / `#Использовать "path"` (RU) and the English
# `#Use` synonym. The tree-sitter-bsl grammar does not model this directive (it
# parses as an ERROR node), so imports are recovered with a regex pass, mirroring
# the Lua/Svelte fallbacks elsewhere in this module.
_BSL_USE_RE = re.compile(
    r'#\s*(?:Использовать|Use)\s+(?:"(?P<path>[^"]+)"|(?P<name>[\w.]+))',
    re.IGNORECASE,
)


def extract_bsl(path: Path) -> dict:
    """Extract procedures, functions, call graph, `Новый <Тип>` references, and
    OneScript `#Использовать` imports from a .bsl/.os/.osl file via tree-sitter.

    1C has no class or import constructs: a module is a flat list of procedures
    and functions that call each other (and procedures in other modules) by bare
    name. Cross-module calls are emitted as unresolved `raw_calls` and resolved
    later by symbol_resolution.resolve_cross_file_raw_calls. Procedure/function
    names are bilingual (Процедура/Procedure) but the grammar already normalises
    both to the same node types, so no language-specific handling is needed.
    """
    try:
        import tree_sitter_bsl as tsbsl
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-bsl not installed"}

    try:
        language = Language(tsbsl.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []  # (caller_nid, definition node)

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0,
                 context: str | None = None) -> None:
        edge = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def ensure_named_node(name: str, line: int) -> str:
        nid = _make_id(stem, name)
        if nid in seen_ids:
            return nid
        nid = _make_id(name)
        if nid not in seen_ids:
            add_node(nid, name, line)
        return nid

    def _named_field(node, field: str):
        """child_by_field_name with a first-identifier fallback."""
        n = node.child_by_field_name(field)
        if n is not None:
            return n
        for child in node.children:
            if child.type == "identifier":
                return child
        return None

    def walk(node) -> None:
        t = node.type
        if t in ("procedure_definition", "function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                func_nid = _make_id(stem, name)
                add_node(func_nid, f"{name}()", line)
                add_edge(file_nid, func_nid, "contains", line)
                # The definition node holds statements as direct children (no body
                # wrapper), so the whole node is queued; walk_calls recurses its
                # children. Definitions cannot nest in BSL, so nothing is missed.
                function_bodies.append((func_nid, node))
            return
        # Recurse — definitions may be nested inside `#Область` (preprocessor) nodes.
        for child in node.children:
            walk(child)

    walk(root)

    # OneScript imports (regex; see _BSL_USE_RE).
    try:
        text = source.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    for m in _BSL_USE_RE.finditer(text):
        raw = (m.group("path") or m.group("name") or "").strip()
        if not raw:
            continue
        line = text.count("\n", 0, m.start()) + 1
        if m.group("path") and (raw.startswith(".") or "/" in raw or "\\" in raw):
            resolved = Path(os.path.normpath(path.parent / raw))
            tgt_nid = _make_id(str(resolved))
        else:
            tgt_nid = _make_id(raw)
        add_edge(file_nid, tgt_nid, "imports", line, context="import")

    label_to_nid: dict[str, str] = {}
    for n in nodes:
        normalised = n["label"].strip("()").lstrip(".")
        label_to_nid[normalised] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    seen_meta_refs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def _manager_object_ref(node) -> tuple[str, str] | None:
        """If `node` is `<Collection>.<Name>` (e.g. `Справочники.Клиенты`), return
        (Kind, Name) for the referenced metadata object, else None. Both
        `property_access` (bare expression) and the chained `access` (receiver of a
        further `.method()`) share the shape `access(identifier) '.' property`."""
        base = prop = None
        for c in node.children:
            if c.type == "access" and base is None:
                base = c
            elif c.type == "property" and prop is None:
                prop = c
        if base is None or prop is None:
            return None
        ident = [c for c in base.children if c.type == "identifier"]
        if len(base.children) != 1 or not ident:
            return None  # base must be a bare identifier, not a deeper chain
        kind = _EDT_MANAGER_TO_KIND.get(_read_text(ident[0], source))
        if not kind:
            return None
        return kind, _read_text(prop, source)

    def walk_calls(node, caller_nid: str) -> None:
        t = node.type
        if t in ("procedure_definition", "function_definition"):
            return  # defensive: BSL has no nested definitions

        if t == "method_call":
            name_node = _named_field(node, "name")
            if name_node is not None:
                callee_name = _read_text(name_node, source)
                # `Объект.Метод()` nests the method_call inside a call_expression /
                # property_access after an `access` receiver — treat as a member
                # call so it is not resolved cross-module (it's a platform/object
                # method, not a free procedure).
                is_member_call = (
                    node.parent is not None
                    and node.parent.type in ("call_expression", "property_access")
                )
                if callee_name and callee_name not in _LANGUAGE_BUILTIN_GLOBALS:
                    tgt_nid = label_to_nid.get(callee_name)
                    if tgt_nid and tgt_nid != caller_nid:
                        pair = (caller_nid, tgt_nid)
                        if pair not in seen_call_pairs:
                            seen_call_pairs.add(pair)
                            add_edge(caller_nid, tgt_nid, "calls",
                                     node.start_point[0] + 1, context="call")
                    elif not tgt_nid:
                        raw_calls.append({
                            "caller_nid": caller_nid,
                            "callee": callee_name,
                            "is_member_call": is_member_call,
                            "source_file": str_path,
                            "source_location": f"L{node.start_point[0] + 1}",
                        })

        elif t == "new_expression":
            # `Новый <Тип>(...)` — platform type instantiation. Record a reference
            # so the graph shows which modules use which platform types.
            type_node = node.child_by_field_name("type") or _named_field(node, "type")
            if type_node is not None:
                type_name = _read_text(type_node, source)
                if type_name:
                    line = node.start_point[0] + 1
                    tgt_nid = ensure_named_node(type_name, line)
                    if tgt_nid != caller_nid:
                        add_edge(caller_nid, tgt_nid, "references", line, context="new")

        elif t in ("access", "property_access"):
            # `Справочники.Клиенты` / `Documents.Order` — a reference to a metadata
            # object. Emit it with the same canonical id extract_edt_mdo uses, so
            # code and the .mdo backbone merge into one node at build time.
            ref = _manager_object_ref(node)
            if ref:
                kind, obj_name = ref
                if obj_name:
                    meta_id = _make_id(kind, obj_name)
                    pair = (caller_nid, meta_id)
                    if meta_id != caller_nid and pair not in seen_meta_refs:
                        seen_meta_refs.add(pair)
                        line = node.start_point[0] + 1
                        add_node(meta_id, f"{kind}.{obj_name}", line)
                        add_edge(caller_nid, meta_id, "references", line,
                                 context="metadata")

        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, def_node in function_bodies:
        for child in def_node.children:
            walk_calls(child, caller_nid)

    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] == "imports"):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── 1C:EDT metadata (.mdo) extractor ──────────────────────────────────────────
#
# A 1C:EDT project export is more than its .bsl modules: the structural backbone
# lives in *.mdo files (one per metadata object, plus Configuration.mdo). This
# extractor turns that backbone into graph nodes so the call graph from extract_bsl
# connects to the catalogs/documents/registers the code actually operates on.
#
# The canonical node id of a metadata object is `_make_id(<EnglishKind>, <Name>)`
# — e.g. `_make_id("Catalog", "Контрагенты")`. That same id is produced by
# extract_bsl when it sees `Справочники.Контрагенты`, so the two halves merge into
# one node at build time. The object's root .mdo element local name IS the English
# singular kind prefix (`<mdclass:Catalog>` -> "Catalog"), which is why no
# kind-name translation table is needed on the .mdo side.

# English singular type prefixes (skill: edt-structures §3 type-prefix table).
# Used to recognise `Kind.Name` registration values inside Configuration.mdo.
_EDT_KIND_PREFIXES: frozenset[str] = frozenset({
    "Catalog", "Document", "Enum", "Constant", "InformationRegister",
    "AccumulationRegister", "AccountingRegister", "CalculationRegister",
    "CommonModule", "CommonForm", "CommonCommand", "CommonTemplate",
    "CommonPicture", "CommonAttribute", "Subsystem", "Role", "ScheduledJob",
    "SessionParameter", "DefinedType", "ExchangePlan",
    "ChartOfCharacteristicTypes", "ChartOfAccounts", "ChartOfCalculationTypes",
    "BusinessProcess", "Task", "Report", "DataProcessor", "HTTPService",
    "WebService", "XDTOPackage", "EventSubscription", "FilterCriterion",
    "FunctionalOption", "FunctionalOptionsParameter", "Language", "Sequence",
    # Kinds documented by the skill but previously unrecognised: the first ten
    # come from the mdclasses fixture (edt-structures §4.0), PaletteColor from
    # the live-audited §4.22, which the fixture does not carry.
    "DocumentJournal", "DocumentNumerator", "CommandGroup", "SettingsStorage",
    "StyleItem", "Style", "PaletteColor", "ExternalDataSource", "WSReference",
    "Bot", "IntegrationService", "WebSocketClient",
})

# Manager-collection identifiers (RU + EN) -> English kind prefix. A BSL access
# `<Collection>.<Name>` (e.g. `Справочники.Клиенты`, `Documents.Order`) is a
# reference to the metadata object `_make_id(<Kind>, <Name>)`.
_EDT_MANAGER_TO_KIND: dict[str, str] = {
    "Справочники": "Catalog", "Catalogs": "Catalog",
    "Документы": "Document", "Documents": "Document",
    "Перечисления": "Enum", "Enums": "Enum",
    "Константы": "Constant", "Constants": "Constant",
    "РегистрыСведений": "InformationRegister", "InformationRegisters": "InformationRegister",
    "РегистрыНакопления": "AccumulationRegister", "AccumulationRegisters": "AccumulationRegister",
    "РегистрыБухгалтерии": "AccountingRegister", "AccountingRegisters": "AccountingRegister",
    "РегистрыРасчета": "CalculationRegister", "CalculationRegisters": "CalculationRegister",
    "ПланыВидовХарактеристик": "ChartOfCharacteristicTypes", "ChartsOfCharacteristicTypes": "ChartOfCharacteristicTypes",
    "ПланыСчетов": "ChartOfAccounts", "ChartsOfAccounts": "ChartOfAccounts",
    "ПланыВидовРасчета": "ChartOfCalculationTypes", "ChartsOfCalculationTypes": "ChartOfCalculationTypes",
    "ПланыОбмена": "ExchangePlan", "ExchangePlans": "ExchangePlan",
    "БизнесПроцессы": "BusinessProcess", "BusinessProcesses": "BusinessProcess",
    "Задачи": "Task", "Tasks": "Task",
    "Обработки": "DataProcessor", "DataProcessors": "DataProcessor",
    "Отчеты": "Report", "Reports": "Report",
}

# Object-level BSL modules that may sit next to a <Name>.mdo, by their fixed
# filename. Each existing one is linked object -> module with a `defines` edge.
_EDT_OBJECT_MODULES: tuple[str, ...] = (
    "ObjectModule.bsl", "ManagerModule.bsl", "RecordSetModule.bsl",
    "ValueManagerModule.bsl", "Module.bsl",
)

# Configuration.mdo registration value, e.g. a catalog FQN. Restricted to
# known kind prefixes so non-FQN children (e.g. <compatibilityMode>8.3.24) are
# ignored.
_EDT_FQN_RE = re.compile(r'^([A-Za-z]+)\.(.+)$')

# A ref-type in a form, e.g. "CatalogRef.Клиенты" -> kind "Catalog". The prefix
# minus the "Ref" suffix is the English singular kind.
_EDT_REF_TYPE_RE = re.compile(r'^([A-Za-z]+)Ref\.(.+)$')

# On-disk plural folder name (src/<KindPlural>/) -> English singular kind. Used to
# rebuild a form's owner-object node id from its file path so the id matches the
# `<forms>` child id that extract_edt_mdo emits from the owner .mdo.
_EDT_PLURAL_TO_KIND: dict[str, str] = {
    "Catalogs": "Catalog", "Documents": "Document", "Enums": "Enum",
    "Constants": "Constant", "InformationRegisters": "InformationRegister",
    "AccumulationRegisters": "AccumulationRegister",
    "AccountingRegisters": "AccountingRegister",
    "CalculationRegisters": "CalculationRegister",
    "ChartsOfCharacteristicTypes": "ChartOfCharacteristicTypes",
    "ChartsOfAccounts": "ChartOfAccounts",
    "ChartsOfCalculationTypes": "ChartOfCalculationTypes",
    "ExchangePlans": "ExchangePlan", "BusinessProcesses": "BusinessProcess",
    "Tasks": "Task", "DataProcessors": "DataProcessor", "Reports": "Report",
    "ExternalDataSources": "ExternalDataSource",
}

# Common (owner-less) plural folders -> singular kind, for path-based owner
# resolution of artifacts that live under them (e.g. CommonTemplates/<Name>).
_EDT_COMMON_FOLDER_TO_KIND: dict[str, str] = {
    "CommonForms": "CommonForm", "CommonTemplates": "CommonTemplate",
    "CommonModules": "CommonModule", "CommonCommands": "CommonCommand",
    "CommonPictures": "CommonPicture", "CommonAttributes": "CommonAttribute",
}

# Query-language (SDBL) source-table prefixes -> English kind. These are the
# SINGULAR RU/EN names used in a query FROM/ИЗ clause (e.g.
# `РегистрНакопления.Взаиморасчеты.Остатки`), distinct from the plural manager
# collections in _EDT_MANAGER_TO_KIND.
_EDT_QUERY_TABLE_TO_KIND: dict[str, str] = {
    "Справочник": "Catalog", "Catalog": "Catalog",
    "Документ": "Document", "Document": "Document",
    "Перечисление": "Enum", "Enum": "Enum",
    "Константа": "Constant", "Constant": "Constant",
    "РегистрСведений": "InformationRegister", "InformationRegister": "InformationRegister",
    "РегистрНакопления": "AccumulationRegister", "AccumulationRegister": "AccumulationRegister",
    "РегистрБухгалтерии": "AccountingRegister", "AccountingRegister": "AccountingRegister",
    "РегистрРасчета": "CalculationRegister", "CalculationRegister": "CalculationRegister",
    "ПланВидовХарактеристик": "ChartOfCharacteristicTypes", "ChartOfCharacteristicTypes": "ChartOfCharacteristicTypes",
    "ПланСчетов": "ChartOfAccounts", "ChartOfAccounts": "ChartOfAccounts",
    "ПланВидовРасчета": "ChartOfCalculationTypes", "ChartOfCalculationTypes": "ChartOfCalculationTypes",
    "ПланОбмена": "ExchangePlan", "ExchangePlan": "ExchangePlan",
    "БизнесПроцесс": "BusinessProcess", "BusinessProcess": "BusinessProcess",
    "Задача": "Task", "Task": "Task",
}

# A `Prefix.Name` reference in query text. group(2) is the metadata object name;
# any trailing `.VirtualTable`/`.field` segment is left for the caller to ignore.
_EDT_QUERY_REF_RE = re.compile(r'([A-Za-zА-Яа-яЁё]+)\.([A-Za-zА-Яа-яЁё0-9_]+)')


def _edt_localname(tag: str) -> str:
    """Strip the `{namespace}` prefix ElementTree puts on every tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _edt_child_text(elem, localname: str) -> str | None:
    """Text of the first direct child of `elem` whose local tag is `localname`."""
    for c in elem:
        if _edt_localname(c.tag) == localname and c.text:
            return c.text.strip()
    return None


def _edt_subsystem_chain(path: Path) -> list[str]:
    """Parent->child names of the subsystem `path` describes, from its location.

    A subsystem nests on disk as `.../Subsystems/<S1>/Subsystems/<S2>/.../<Sn>.mdo`
    (skill: edt-structures §4.8 — nested subsystems are real folders, registered in
    the parent by bare name). Each folder name immediately after a `Subsystems`
    segment is one chain link. Two subsystems sharing a bare name under different
    parents thus get distinct ids (`Subsystem.A.Настройки` vs `Subsystem.B.Настройки`),
    while a top-level subsystem keeps its single-part id unchanged.
    """
    parts = path.parts
    return [parts[i + 1] for i, seg in enumerate(parts[:-1])
            if seg == "Subsystems" and i + 1 < len(parts)]


# Container folder -> the SubKind its children carry in an FQN. Unlike the
# homogeneous `Subsystems` marker (one kind, repeated), these are heterogeneous:
# the parent and the child are different kinds, so the FQN keeps both
# (`CalculationRegister.X.Recalculation.R`). Skill: edt-structures §4.5, §4.21.
_EDT_NESTED_MARKERS: dict[str, str] = {
    "Recalculations": "Recalculation",
    "Tables": "Table",
    "Cubes": "Cube",
    "Functions": "Function",
    "DimensionTables": "DimensionTable",
}


def _edt_nested_owner_chain(path: Path) -> list[str] | None:
    """FQN parts of the nested object `path` describes, or None if not nested.

    Walks up from the file consuming `<Marker>/<Name>` pairs (`Tables/T`,
    `Cubes/C`, `Recalculations/R`) and stops at the top-level `<KindPlural>/<Owner>`
    pair, yielding `[Kind, Owner, SubKind, SubName, ...]` — the recursive grammar
    `Kind.Name.SubKind.SubName[...]` verified against the 1c-syntax/mdclasses
    fixtures. Returns None when the path is not under a marker folder, or when the
    chain does not bottom out in a known kind folder, so callers fall back to the
    flat id rather than invent an FQN.
    """
    dirs = path.parts[:-1]                      # drop the .mdo file name
    chain: list[str] = []
    i = len(dirs) - 2                           # index of the innermost marker
    while i >= 0 and dirs[i] in _EDT_NESTED_MARKERS:
        chain[:0] = [_EDT_NESTED_MARKERS[dirs[i]], dirs[i + 1]]
        i -= 2
    if not chain or i < 0:
        return None
    kind = _EDT_PLURAL_TO_KIND.get(dirs[i])
    if not kind:
        return None
    return [kind, dirs[i + 1]] + chain


def extract_edt_mdo(path: Path) -> dict:
    """Extract a 1C:EDT metadata object (or the configuration root) from a .mdo file.

    Object .mdo  -> one object node (`Kind.Name`) plus `contains` edges to its
    attributes, tabular sections, enum values, forms and commands, and `defines`
    edges to the sibling .bsl modules (ObjectModule/ManagerModule/…) that exist
    on disk. Configuration.mdo -> a Configuration node with `contains` edges to
    every registered child object.
    """
    import xml.etree.ElementTree as ET

    try:
        src = path.read_bytes()
    except OSError:
        return {"nodes": [], "edges": [], "error": f"cannot read {path}"}
    if len(src) > _PROJECT_XML_MAX_BYTES:
        return {"nodes": [], "edges": [], "error": "mdo file too large"}
    if not _project_xml_is_safe(src):
        return {"nodes": [], "edges": [],
                "error": "refusing XML with DOCTYPE/ENTITY declaration"}
    try:
        root = ET.fromstring(src)
    except ET.ParseError as e:
        return {"nodes": [], "edges": [], "error": f"XML parse error: {e}"}

    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(nid: str, label: str, file_type: str = "code",
                 uuid: str | None = None) -> None:
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            node = {"id": nid, "label": label, "file_type": file_type,
                    "source_file": str_path, "source_location": "L1"}
            # EDT object/child uuid (skill: edt-structures §3). Stored as an
            # attribute only — the node id stays name-based so code↔metadata
            # nodes still merge by FQN. The uuid disambiguates genuine name
            # collisions and anchors uuid-keyed lookups downstream.
            if uuid:
                node["uuid"] = uuid
            nodes.append(node)

    def add_edge(src_id: str, tgt_id: str, relation: str) -> None:
        key = (src_id, tgt_id, relation)
        if not src_id or not tgt_id or src_id == tgt_id or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": src_id, "target": tgt_id, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0})

    def child_text(elem, localname: str) -> str | None:
        for c in elem:
            if _edt_localname(c.tag) == localname and c.text:
                return c.text.strip()
        return None

    kind = _edt_localname(root.tag)

    # ── Configuration root: register every contained child object ──────────────
    if kind == "Configuration":
        conf_id = _make_id("Configuration")
        add_node(conf_id, child_text(root, "name") or "Configuration", "concept",
                 uuid=root.get("uuid"))
        for child in root:
            text = (child.text or "").strip()
            m = _EDT_FQN_RE.match(text)
            if not m or m.group(1) not in _EDT_KIND_PREFIXES:
                continue
            obj_id = _make_id(m.group(1), m.group(2))
            add_node(obj_id, text)
            add_edge(conf_id, obj_id, "contains")
        return {"nodes": nodes, "edges": edges}

    # ── Object root: the object plus its child artifacts and modules ───────────
    name = child_text(root, "name")
    if not name:
        return {"nodes": nodes, "edges": edges}

    obj_id = _make_id(kind, name)
    obj_label = f"{kind}.{name}"
    id_parts: list[str] = [kind, name]
    parent_id: str | None = None
    sub_chain: list[str] = []
    if kind == "Subsystem":
        sub_chain = _edt_subsystem_chain(path)
        # Only nested subsystems (chain length > 1) change id; top-level keeps the
        # bare `Subsystem.Name` id so existing references/merge are untouched.
        if len(sub_chain) > 1 and sub_chain[-1] == name:
            obj_id = _make_id(kind, *sub_chain)
            obj_label = f"{kind}." + ".".join(sub_chain)
    else:
        # Heterogeneous nesting. The root tag of a nested standalone .mdo IS the
        # SubKind, so the flat `_make_id(kind, name)` above both invents a kind
        # that does not exist in 1C (`Table.X`) and collides between parents.
        # Requiring the tag to match the marker map means an unrecognised layout
        # keeps the flat id rather than getting an invented FQN.
        chain = _edt_nested_owner_chain(path)
        if chain and chain[-2] == kind and chain[-1] == name:
            id_parts = chain
            obj_id = _make_id(*chain)
            obj_label = ".".join(chain)
            parent_id = _make_id(*chain[:-2])
    add_node(obj_id, obj_label, uuid=root.get("uuid"))
    if parent_id:
        # ExternalDataSource children are registered by folder only — the parent
        # .mdo carries no inline block to emit this edge from (skill §4.21), so it
        # is synthesised here. A Recalculation also gets it from the parent's
        # inline <recalculations>; the duplicate collapses on the shared id.
        add_node(parent_id, ".".join(id_parts[:-2]))
        add_edge(parent_id, obj_id, "contains")

    # Child artifacts: attributes / tabular sections / enum values / forms / commands.
    _CHILD_KINDS = {
        "attributes": "Attribute", "tabularSections": "TabularSection",
        "enumValues": "EnumValue", "forms": "Form", "commands": "Command",
        # A CalculationRegister registers its recalculations inline, so this is
        # where the parent -> child edge comes from (the child .mdo also
        # synthesises it from its path; the two share an id and collapse).
        "recalculations": "Recalculation",
        # External-table fields are SubKind `Field`, not `Attribute` — and the
        # container tag differs by owner: <tableFields> on a Table, <fields> on a
        # DimensionTable (skill: edt-structures §4.21).
        "tableFields": "Field", "fields": "Field",
    }
    for child in root:
        sub = _CHILD_KINDS.get(_edt_localname(child.tag))
        if not sub:
            continue
        child_name = child_text(child, "name")
        if not child_name:
            continue
        child_id = _make_id(*id_parts, sub, child_name)
        add_node(child_id, child_name, uuid=child.get("uuid"))
        add_edge(obj_id, child_id, "contains")
        # A form/command owns a BSL module folder next to the .mdo.
        if sub == "Form":
            mod = path.parent / "Forms" / child_name / "Module.bsl"
        elif sub == "Command":
            mod = path.parent / "Commands" / child_name / "CommandModule.bsl"
        else:
            mod = None
        if mod is not None and mod.is_file():
            add_edge(child_id, _make_id(str(mod)), "defines")

    # A subsystem registers the objects it groups via <content>Kind.Name</content>,
    # and its child subsystems via <subsystems>BareName</subsystems> (skill §4.8).
    if kind == "Subsystem":
        base_chain = sub_chain or [name]
        for child in root:
            ln = _edt_localname(child.tag)
            text = (child.text or "").strip()
            if ln == "content":
                m = _EDT_FQN_RE.match(text)
                if not m or m.group(1) not in _EDT_KIND_PREFIXES:
                    continue
                member_id = _make_id(m.group(1), m.group(2))
                add_node(member_id, text)
                add_edge(obj_id, member_id, "contains")
            elif ln == "subsystems" and text:
                # Child registered by bare name; its id is the full parent chain +
                # child, matching what the child's own .mdo emits, so they merge.
                child_chain = base_chain + [text]
                child_sid = _make_id("Subsystem", *child_chain)
                add_node(child_sid, "Subsystem." + ".".join(child_chain))
                add_edge(obj_id, child_sid, "contains")

    # Object-level modules sitting beside the .mdo.
    for mod_name in _EDT_OBJECT_MODULES:
        mod = path.parent / mod_name
        if mod.is_file():
            add_edge(obj_id, _make_id(str(mod)), "defines")

    return {"nodes": nodes, "edges": edges}


def _edt_parse_xml(path: Path):
    """Shared, guarded parse for .rights/.form XML. Returns (root, error_result).

    Exactly one of the two is non-None.
    """
    import xml.etree.ElementTree as ET

    try:
        src = path.read_bytes()
    except OSError:
        return None, {"nodes": [], "edges": [], "error": f"cannot read {path}"}
    if len(src) > _PROJECT_XML_MAX_BYTES:
        return None, {"nodes": [], "edges": [], "error": "xml file too large"}
    if not _project_xml_is_safe(src):
        return None, {"nodes": [], "edges": [],
                      "error": "refusing XML with DOCTYPE/ENTITY declaration"}
    try:
        return ET.fromstring(src), None
    except ET.ParseError as e:
        return None, {"nodes": [], "edges": [], "error": f"XML parse error: {e}"}


def extract_edt_rights(path: Path) -> dict:
    """Extract a 1C:EDT role's access rights from a Rights.rights file.

    The role is identified by its folder (`src/Roles/<Role>/Rights.rights`). Each
    granted `<object>` (one whose rights include a `<value>true</value>`) yields a
    `secures` edge Role -> object, collapsing attribute-level FQNs
    (`Catalog.X.Attribute.Y`) onto the owning object (`Catalog.X`).
    """
    root, err = _edt_parse_xml(path)
    if err is not None:
        return err

    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(nid: str, label: str) -> None:
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"})

    role_name = path.parent.name
    role_id = _make_id("Role", role_name)
    add_node(role_id, f"Role.{role_name}")

    for obj in root.iter():
        if _edt_localname(obj.tag) != "object":
            continue
        fqn = _edt_child_text(obj, "name")
        if not fqn:
            continue
        granted = any(_edt_localname(r.tag) == "right"
                      and _edt_child_text(r, "value") == "true" for r in obj)
        if not granted:
            continue
        parts = fqn.split(".")
        if len(parts) < 2 or parts[0] not in _EDT_KIND_PREFIXES:
            continue
        member_id = _make_id(parts[0], parts[1])
        if member_id == role_id or member_id in seen_edges:
            continue
        seen_edges.add(member_id)
        add_node(member_id, f"{parts[0]}.{parts[1]}")
        edges.append({"source": role_id, "target": member_id, "relation": "secures",
                      "confidence": "EXTRACTED", "source_file": str_path, "weight": 1.0})

    return {"nodes": nodes, "edges": edges}


def _edt_form_owner_id(path: Path) -> str | None:
    """Rebuild the node id of the form that `Form.form` describes, from its path.

    Object-owned: `.../<KindPlural>/<Owner>/Forms/<FormName>/Form.form`
                  -> `_make_id(<Kind>, <Owner>, "Form", <FormName>)`.
    Common form:  `.../CommonForms/<Name>/Form.form` -> `_make_id("CommonForm", <Name>)`.
    """
    parts = path.parts
    if "Forms" in parts:
        i = parts.index("Forms")
        if 0 < i - 1 and i + 1 < len(parts):
            kind = _EDT_PLURAL_TO_KIND.get(parts[i - 2])
            if kind:
                return _make_id(kind, parts[i - 1], "Form", parts[i + 1])
    elif "CommonForms" in parts:
        j = parts.index("CommonForms")
        if j + 1 < len(parts):
            return _make_id("CommonForm", parts[j + 1])
    return None


def _edt_owner_dir(path: Path) -> Path | None:
    """Folder of the metadata object that owns the artifact at `path`.

    Returns the first `.../<KindPlural>/<Owner>` (or common-folder) directory on
    the way down, e.g. `.../Catalogs/Контрагенты` for a form, or
    `.../Reports/X` for a DCS template.
    """
    parts = path.parts
    for i, seg in enumerate(parts[:-1]):
        if seg in _EDT_PLURAL_TO_KIND or seg in _EDT_COMMON_FOLDER_TO_KIND:
            return Path(*parts[:i + 2])
    return None


def _edt_parent_uuid(path: Path, form_name: str | None = None) -> str | None:
    """Best-effort uuid of the artifact's parent object, read from the owner .mdo.

    Form attributes and DCS fields carry no uuid of their own (skill: edt-structures
    §3, §5 — Form.form has no uuid). Per the agreed design, such children inherit
    the parent's uuid: for a form, the `<forms uuid>` matching `form_name` (the
    form's identity, which the Form.form file itself lacks); otherwise the owner
    object's root uuid. Returns None if the .mdo can't be located/parsed — the node
    id already encodes the parent path, so this uuid is provenance, not identity.
    """
    owner_dir = _edt_owner_dir(path)
    if owner_dir is None:
        return None
    mdo = owner_dir / f"{owner_dir.name}.mdo"
    if not mdo.is_file():
        return None
    root, err = _edt_parse_xml(mdo)
    if err is not None:
        return None
    if form_name is not None:
        for c in root:
            if _edt_localname(c.tag) == "forms" and _edt_child_text(c, "name") == form_name:
                return c.get("uuid")
        return None
    return root.get("uuid")


def extract_edt_form(path: Path) -> dict:
    """Extract data references from a 1C:EDT managed form (Form.form).

    Anchors on the form's own node (the same id extract_edt_mdo emits for it) and
    emits `references` edges to the metadata objects the form binds: the dynamic
    list `<mainTable>Kind.Name</mainTable>` and every ref-typed attribute
    `<types>KindRef.Name</types>`.
    """
    root, err = _edt_parse_xml(path)
    if err is not None:
        return err

    form_id = _edt_form_owner_id(path)
    if not form_id:
        return {"nodes": [], "edges": []}

    str_path = str(path)
    form_name = path.parent.name
    form_uuid = _edt_parent_uuid(path, form_name=form_name)
    anchor = {"id": form_id, "label": form_name, "file_type": "code",
              "source_file": str_path, "source_location": "L1"}
    if form_uuid:
        anchor["uuid"] = form_uuid
    nodes: list[dict] = [anchor]
    edges: list[dict] = []
    seen_targets: set[str] = set()

    # Form attributes (реквизиты формы) become their own nodes. They carry no
    # uuid of their own, so the id is the form path + attribute name (unique per
    # form) and the form's uuid is recorded as parent_uuid.
    seen_attrs: set[str] = set()
    for elem in root:
        if _edt_localname(elem.tag) != "attributes":
            continue
        attr_name = _edt_child_text(elem, "name")
        if not attr_name:
            continue
        attr_id = _make_id(form_id, "FormAttribute", attr_name)
        if attr_id == form_id or attr_id in seen_attrs:
            continue
        seen_attrs.add(attr_id)
        node = {"id": attr_id, "label": attr_name, "file_type": "code",
                "source_file": str_path, "source_location": "L1"}
        if form_uuid:
            node["parent_uuid"] = form_uuid
        nodes.append(node)
        edges.append({"source": form_id, "target": attr_id, "relation": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0})

    def add_ref(kind: str, name: str) -> None:
        tgt = _make_id(kind, name)
        if tgt == form_id or tgt in seen_targets:
            return
        seen_targets.add(tgt)
        nodes.append({"id": tgt, "label": f"{kind}.{name}", "file_type": "code",
                      "source_file": str_path, "source_location": "L1"})
        edges.append({"source": form_id, "target": tgt, "relation": "references",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0, "context": "form"})

    for elem in root.iter():
        text = (elem.text or "").strip()
        if not text:
            continue
        ln = _edt_localname(elem.tag)
        if ln == "mainTable":
            m = _EDT_FQN_RE.match(text)
            if m and m.group(1) in _EDT_KIND_PREFIXES:
                add_ref(m.group(1), m.group(2))
        elif ln == "types":
            m = _EDT_REF_TYPE_RE.match(text)
            if m and m.group(1) in _EDT_KIND_PREFIXES:
                add_ref(m.group(1), m.group(2))

    # An event/command handler binds the form to a procedure in its Module.bsl.
    # `<handlers><event>E</event><name>Proc</name>` (item/form events) and
    # `<handler><name>Proc</name>` (command actions) both name the procedure via a
    # direct <name> child. extract_bsl ids that procedure _make_id(stem, Proc) with
    # stem == _file_stem(Module.bsl), so the edge lands on the real node.
    module = path.parent / "Module.bsl"
    if module.is_file():
        mod_stem = _file_stem(module)
        seen_handlers: set[str] = set()
        for elem in root.iter():
            if _edt_localname(elem.tag) not in ("handlers", "handler"):
                continue
            proc = _edt_child_text(elem, "name")
            if not proc:
                continue
            proc_id = _make_id(mod_stem, proc)
            if proc_id == form_id or proc_id in seen_handlers:
                continue
            seen_handlers.add(proc_id)
            edges.append({"source": form_id, "target": proc_id, "relation": "references",
                          "confidence": "EXTRACTED", "source_file": str_path,
                          "weight": 1.0, "context": "form-handler"})

    return {"nodes": nodes, "edges": edges}


def _edt_owner_id_from_path(path: Path) -> tuple[str, str] | None:
    """Resolve the owning metadata object from an artifact's file path.

    Walks the path for the first `<KindPlural>/<Owner>` segment pair — either an
    object kind folder (`Reports/<Name>`) or a common folder
    (`CommonTemplates/<Name>`) — and returns `(node_id, "Kind.Name")`.
    """
    parts = path.parts
    for i, seg in enumerate(parts[:-1]):
        kind = _EDT_COMMON_FOLDER_TO_KIND.get(seg) or _EDT_PLURAL_TO_KIND.get(seg)
        if kind:
            return _make_id(kind, parts[i + 1]), f"{kind}.{parts[i + 1]}"
    return None


def extract_edt_dcs(path: Path) -> dict:
    """Extract data lineage from a 1C:EDT data composition schema (Template.dcs).

    Each `<query>` is SDBL; its FROM/ИЗ source tables (`Справочник.X`,
    `РегистрНакопления.Y.Остатки`, …) are the metadata objects the schema reads.
    Anchors on the owning object (the report/data-processor/common-template the
    template belongs to) and emits `references` edges to each source object.
    """
    root, err = _edt_parse_xml(path)
    if err is not None:
        return err

    owner = _edt_owner_id_from_path(path)
    if not owner:
        return {"nodes": [], "edges": []}
    owner_id, owner_label = owner

    str_path = str(path)
    nodes: list[dict] = [{"id": owner_id, "label": owner_label, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen_targets: set[str] = set()

    # DCS fields (`<field><dataPath>…`) become their own nodes. Like form
    # attributes they have no uuid; id is owner path + dataPath, with the owner
    # object's uuid recorded as parent_uuid.
    owner_uuid = _edt_parent_uuid(path)
    seen_fields: set[str] = set()
    for elem in root.iter():
        if _edt_localname(elem.tag) != "field":
            continue
        data_path = _edt_child_text(elem, "dataPath")
        if not data_path:
            continue
        field_id = _make_id(owner_id, "DcsField", data_path)
        if field_id == owner_id or field_id in seen_fields:
            continue
        seen_fields.add(field_id)
        node = {"id": field_id, "label": data_path, "file_type": "code",
                "source_file": str_path, "source_location": "L1"}
        if owner_uuid:
            node["parent_uuid"] = owner_uuid
        nodes.append(node)
        edges.append({"source": owner_id, "target": field_id, "relation": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0})

    for elem in root.iter():
        if _edt_localname(elem.tag) != "query" or not elem.text:
            continue
        for m in _EDT_QUERY_REF_RE.finditer(elem.text):
            kind = _EDT_QUERY_TABLE_TO_KIND.get(m.group(1))
            if not kind:
                continue
            tgt = _make_id(kind, m.group(2))
            if tgt == owner_id or tgt in seen_targets:
                continue
            seen_targets.add(tgt)
            nodes.append({"id": tgt, "label": f"{kind}.{m.group(2)}", "file_type": "code",
                          "source_file": str_path, "source_location": "L1"})
            edges.append({"source": owner_id, "target": tgt, "relation": "references",
                          "confidence": "EXTRACTED", "source_file": str_path,
                          "weight": 1.0, "context": "dcs"})

    return {"nodes": nodes, "edges": edges}


