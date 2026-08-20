"""BSL (1C / OneScript) and 1C:EDT metadata extractors.

Moved verbatim from graphify/extract.py (fork delta: tree-sitter-bsl language
support plus 1C:EDT .mdo / .rights / Form.form / .dcs ingestion).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
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


def extract_bsl(path: Path, source: bytes | None = None,
                line_offset: int = 0) -> dict:
    """Extract procedures, functions, call graph, `Новый <Тип>` references, and
    OneScript `#Использовать` imports from a .bsl/.os/.osl file via tree-sitter.

    `source` lets a caller hand in module text that is not the whole file — an
    ordinary form keeps its module inside the .oform container, where it has no
    file of its own. `line_offset` is added to every reported line so positions
    point into the real file rather than into the extracted fragment: a link to
    `Form.oform:15890` opens where the procedure actually is.

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
        if source is None:
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
                "source_location": f"L{line + line_offset}",
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
            "source_location": f"L{line + line_offset}",
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    file_nid = _make_id(str(path))
    # The file node stands for the whole file, so it sits at line 1 even when the
    # fragment it was parsed from starts further down.
    add_node(file_nid, path.name, 1 - line_offset)

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
                            "source_location":
                                f"L{node.start_point[0] + 1 + line_offset}",
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

# Configuration-level BSL modules, which live beside Configuration.mdo rather
# than beside an object (skill: edt-structures §1). All five are listed even
# though a given configuration carries at most a couple: the is_file() check
# below filters, and hard-coding the observed subset would silently drop the
# rest on the next project.
_EDT_CONFIGURATION_MODULES: tuple[str, ...] = (
    "ApplicationModule.bsl", "ManagedApplicationModule.bsl",
    "OrdinaryApplicationModule.bsl", "SessionModule.bsl",
    "ExternalConnectionModule.bsl",
)

# Inline children that own children of their own. Kept as an explicit set for
# the same reason as every other whitelist here: descending into everything
# would mint nodes for blocks that own nothing — `<attributes>` holds a
# `<type><types>String</types>`, and an unconditional walk would turn that into
# a node. Measured on the audited corpus: nothing nests three levels deep, so
# one level of descent is the whole requirement.
_EDT_CONTAINER_CHILD_KINDS: frozenset[str] = frozenset({
    "TabularSection", "Operation", "URLTemplate",
})

# Where a SubKind comes from. CONFIRMED means 1C names this FQN form itself
# (skill: edt-structures §3, §4.20, §4.22), so the id addresses something the
# platform addresses too. CONVENTION means the name is graphify's own and 1C
# makes no such statement — `HTTPService.X.URLTemplate.Y` is our spelling, not a
# claim about 1C's nomenclature. The origin is kept in the map rather than in a
# comment so it stays checkable six months from now.
_EDT_SUBKIND_CONFIRMED = "skill"
_EDT_SUBKIND_CONVENTION = "graphify"

# Inline child blocks of a .mdo: XML tag -> (SubKind, origin of that SubKind).
_EDT_CHILD_KINDS: dict[str, tuple[str, str]] = {
    "attributes": ("Attribute", _EDT_SUBKIND_CONFIRMED),
    "tabularSections": ("TabularSection", _EDT_SUBKIND_CONFIRMED),
    "enumValues": ("EnumValue", _EDT_SUBKIND_CONFIRMED),
    "forms": ("Form", _EDT_SUBKIND_CONFIRMED),
    "commands": ("Command", _EDT_SUBKIND_CONFIRMED),
    # A CalculationRegister registers its recalculations inline, so this is
    # where the parent -> child edge comes from (the child .mdo also
    # synthesises it from its path; the two share an id and collapse).
    "recalculations": ("Recalculation", _EDT_SUBKIND_CONFIRMED),
    # External-table fields are SubKind `Field`, not `Attribute` — and the
    # container tag differs by owner: <tableFields> on a Table, <fields> on a
    # DimensionTable (skill: edt-structures §4.21).
    "tableFields": ("Field", _EDT_SUBKIND_CONFIRMED),
    "fields": ("Field", _EDT_SUBKIND_CONFIRMED),
    "standardAttributes": ("StandardAttribute", _EDT_SUBKIND_CONFIRMED),
    "resources": ("Resource", _EDT_SUBKIND_CONFIRMED),
    "dimensions": ("Dimension", _EDT_SUBKIND_CONFIRMED),
    "templates": ("Template", _EDT_SUBKIND_CONFIRMED),
    "addressingAttributes": ("AddressingAttribute", _EDT_SUBKIND_CONFIRMED),
    "accountingFlags": ("AccountingFlag", _EDT_SUBKIND_CONFIRMED),
    "items": ("Predefined", _EDT_SUBKIND_CONVENTION),
    "columns": ("Column", _EDT_SUBKIND_CONVENTION),
    "operations": ("Operation", _EDT_SUBKIND_CONVENTION),
    "parameters": ("Parameter", _EDT_SUBKIND_CONVENTION),
    "urlTemplates": ("URLTemplate", _EDT_SUBKIND_CONVENTION),
    "methods": ("Method", _EDT_SUBKIND_CONVENTION),
    "integrationServiceChannels": ("Channel", _EDT_SUBKIND_CONVENTION),
}

# Wrappers that are not entities: they carry neither a name nor a uuid, and the
# things worth a node are their children. `<predefined>` is the only one in the
# audited corpus — 44 wrappers holding 559 items — and it is exactly the
# container-versus-leaf trap the skill names (§3): counting the wrapper answers
# "44 predefined items" for a configuration that has 559.
_EDT_TRANSPARENT_CHILD_TAGS: frozenset[str] = frozenset({"predefined"})

# Blocks whose identifier is not spelled `uuid`. A standard attribute has no
# identifier at all, and a predefined item's is `id` — the item's identity in
# user data (skill §3). Reading `uuid` unconditionally returns None for both,
# which loses the second silently.
_EDT_NO_IDENTIFIER_TAGS: frozenset[str] = frozenset({"standardAttributes"})
_EDT_PLAIN_ID_TAGS: frozenset[str] = frozenset({"items"})

# Reference-bearing tags of an object .mdo: tag -> (value shape, context label).
#
# An explicit map, not "anything shaped like Kind.Name is a reference". That
# shortcut breaks immediately: <name> can contain a dot, <synonym> and <comment>
# hold free text, and <version>3.2.7.38 parses as kind "3". The cost is manual
# upkeep; the benefit is that no edge is ever invented.
#
# Shapes:
#   object    Kind.Name                     -> the object node
#   member    Kind.Name.SubKind.SubName     -> the member node (stub if absent)
#   method    CommonModule.Name.Method      -> the module node, method in context
#   type      CatalogRef.Name / CatalogObject.Name -> the object node
_EDT_REF_TAGS: dict[str, tuple[str, str]] = {
    "registerRecords": ("object", "register-records"),
    "basedOn": ("object", "based-on"),
    "owners": ("object", "owner"),
    "sequences": ("object", "sequence"),
    "characteristicExtValues": ("object", "characteristic-ext-values"),
    "chartOfAccounts": ("object", "chart-of-accounts"),
    "registeredDocuments": ("object", "registered-document"),
    "task": ("object", "task"),
    "addressing": ("object", "addressing-register"),
    "location": ("object", "functional-option-location"),
    "defaultRoles": ("object", "default-role"),
    "defaultLanguage": ("object", "default-language"),
    "mainDataCompositionSchema": ("member", "main-dcs"),
    "defaultObjectForm": ("member", "default-object-form"),
    "defaultListForm": ("member", "default-list-form"),
    "defaultChoiceForm": ("member", "default-choice-form"),
    "inputByString": ("member", "input-by-string"),
    "mainAddressingAttribute": ("member", "main-addressing-attribute"),
    "addressingDimension": ("member", "addressing-dimension"),
    "handler": ("method", "event-handler"),
    "methodName": ("method", "scheduled-job-method"),
}

# Container-shaped tags: the FQN sits in a named child, not in the tag's text.
# `<content>` is BOTH — a leaf on Subsystem and FunctionalOption, a container on
# ExchangePlan (`<content><mdObject>Catalog.X</mdObject></content>`). Reading
# only `.text` reports zero for 562 real references; this is the container/leaf
# trap the skill calls out (edt-structures §3). Keying on the child tag also
# makes Catalog's unrelated <content> blocks (name/description/code) a no-op.
_EDT_REF_CONTAINER_TAGS: dict[str, tuple[str, str, str]] = {
    "content": ("mdObject", "object", "exchange-plan-content"),
    "source": ("types", "type", "event-source"),
    "commandParameterType": ("types", "type", "command-parameter"),
}


def _edt_ref_target(shape: str, text: str) -> tuple[list[str], str | None] | None:
    """FQN parts of a reference value, plus a trailing method name if any.

    Returns None when the value is not a reference of that shape — the caller
    then emits nothing rather than inventing a node.
    """
    parts = [p for p in text.split(".") if p]
    if len(parts) < 2:
        return None
    if shape == "type":
        kind = _edt_type_kind(parts[0])
        return ([kind, parts[1]], None) if kind else None
    if parts[0] not in _EDT_KIND_PREFIXES:
        return None
    if shape == "object":
        return ([parts[0], parts[1]], None)
    if shape == "method":
        # `CommonModule.X.Метод` — the graph has no method nodes, so the edge
        # lands on the module and the method name rides in the context.
        return ([parts[0], parts[1]], ".".join(parts[2:]) or None)
    if shape == "member":
        # A bare `Kind.Name` in a member slot is still the object itself.
        return (parts, None)
    return None


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

# A typed value in a form, e.g. "CatalogRef.Клиенты" or "CatalogObject.Клиенты".
# group(1) is the whole prefix; _edt_type_kind below strips the flavour suffix.
_EDT_TYPE_RE = re.compile(r'^([A-Za-z]+)\.(.+)$')

# Type flavours from the prefix table (skill: edt-structures §3). A flavour says
# in WHICH CAPACITY an object is used, not which object — every flavour of
# `Справочник.Клиенты` denotes the same `Catalog.Клиенты`, so they all resolve to
# one node. Ordered longest-first: `InformationRegisterRecordManager` ends in
# both `RecordManager` and `Manager`, and taking the shorter one leaves
# `InformationRegisterRecord`, a kind that does not exist.
_EDT_TYPE_FLAVOURS: tuple[str, ...] = (
    "RecordManager", "RecordSet", "RecordKey", "Selection", "Manager",
    "Object", "List", "Ref",
)


def _edt_type_kind(prefix: str) -> str | None:
    """English kind behind a type prefix, or None when it is not one.

    The remainder after stripping a flavour must itself be a known kind. Without
    that check `ChartOfCharacteristicTypesObject` would yield the non-existent
    kind `ChartOfCharacteristicTypes` + leftovers, and any word ending in "List"
    would look like a metadata reference.
    """
    if prefix in _EDT_KIND_PREFIXES:
        return prefix
    for flavour in _EDT_TYPE_FLAVOURS:
        if prefix.endswith(flavour):
            kind = prefix[: -len(flavour)]
            if kind in _EDT_KIND_PREFIXES:
                return kind
    return None

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
    # Everything else that owns a src/<KindPlural>/ folder (skill §4.0). Spelled
    # out rather than derived by appending "s": the irregular plurals below are
    # exactly the ones a rule would get wrong — ChartsOf* is not ChartOf*s,
    # FilterCriteria comes from FilterCriterion, and Sequences / WSReferences /
    # XDTOPackages / CommonPictures each break a different guess. A folder that
    # is not in this map resolves to no owner, which is the safe outcome: an
    # invented FQN is worse than a missing one.
    "DocumentJournals": "DocumentJournal",
    "DocumentNumerators": "DocumentNumerator",
    "Constants": "Constant", "Sequences": "Sequence",
    "CommonForms": "CommonForm", "CommonCommands": "CommonCommand",
    "CommonTemplates": "CommonTemplate", "CommonPictures": "CommonPicture",
    "CommonAttributes": "CommonAttribute", "CommonModules": "CommonModule",
    "CommandGroups": "CommandGroup",
    "Subsystems": "Subsystem", "Roles": "Role",
    "ScheduledJobs": "ScheduledJob", "SessionParameters": "SessionParameter",
    "SettingsStorages": "SettingsStorage", "DefinedTypes": "DefinedType",
    "FunctionalOptions": "FunctionalOption",
    "FunctionalOptionsParameters": "FunctionalOptionsParameter",
    "EventSubscriptions": "EventSubscription",
    "FilterCriteria": "FilterCriterion",
    "HTTPServices": "HTTPService", "WebServices": "WebService",
    "WSReferences": "WSReference", "XDTOPackages": "XDTOPackage",
    "StyleItems": "StyleItem", "Styles": "Style",
    "PaletteColors": "PaletteColor", "Languages": "Language",
    "Bots": "Bot", "IntegrationServices": "IntegrationService",
    "WebSocketClients": "WebSocketClient",
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
                 uuid: str | None = None, predefined_id: str | None = None) -> None:
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
            # A predefined item's `id` is a different thing from a uuid — it is
            # the item's identity in user data — and it keeps its own field so
            # the two stay tellable apart.
            if predefined_id:
                node["predefined_id"] = predefined_id
            nodes.append(node)

    def add_edge(src_id: str, tgt_id: str, relation: str) -> None:
        key = (src_id, tgt_id, relation)
        if not src_id or not tgt_id or src_id == tgt_id or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": src_id, "target": tgt_id, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0})

    def add_ref_edge(src_id: str, tgt_id: str, context: str) -> None:
        """A declared reference. Deduped WITH the context, so one object naming
        another through two different tags yields two edges, not one."""
        key = (src_id, tgt_id, "references", context)
        if not src_id or not tgt_id or src_id == tgt_id or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": src_id, "target": tgt_id, "relation": "references",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "weight": 1.0, "context": context})

    def add_ref(src_id: str, shape: str, text: str, context: str) -> None:
        parsed = _edt_ref_target(shape, text)
        if not parsed:
            return
        target_parts, method = parsed
        target_id = _make_id(*target_parts)
        # The target may not be produced by any extractor yet — a register
        # dimension, a standard attribute. A stub carrying the id the real node
        # will get keeps the reference instead of dropping it, and merges the
        # moment that extraction lands. Configuration.mdo registrations already
        # work this way.
        add_node(target_id, ".".join(target_parts))
        add_ref_edge(src_id, target_id, f"{context}:{method}" if method else context)

    def add_procedure_ref(src_id: str, proc: str | None, context: str) -> None:
        """Link a BARE procedure name to the procedure in the sibling Module.bsl.

        A service names its handler this way — `<handler>PostEntriesPOST</handler>`
        on an HTTPService method, `<procedureName>` on a WebService operation,
        both implemented in the Module.bsl beside the .mdo (skill §4.22). That is
        a different value form from an EventSubscription's `<handler>`, which is
        a full method FQN in someone ELSE's module; the skill spells out the
        contrast, and reading only the FQN form covered 8 of 98 handlers.

        The target id matches what extract_bsl emits for that procedure, so the
        two halves merge — the same wiring extract_edt_form already uses for
        form event handlers.
        """
        if not proc or "." in proc:
            return
        module = path.parent / "Module.bsl"
        if module.is_file():
            add_ref_edge(src_id, _make_id(_file_stem(module), proc), context)

    def type_refs(elem, src_id: str, context: str = "attribute-type") -> None:
        """Emit references for the `<type><types>…</types></type>` of one element."""
        for t in elem:
            if _edt_localname(t.tag) != "type":
                continue
            for tt in t:
                if _edt_localname(tt.tag) == "types" and (tt.text or "").strip():
                    add_ref(src_id, "type", tt.text.strip(), context)

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
        # The five configuration modules sit beside Configuration.mdo. Linked
        # here rather than by moving the early return: the return exists so a
        # Configuration.mdo never runs through the object path (inline children,
        # subsystems), and relaxing it to reach the shared module loop would
        # cost more than repeating three lines.
        for mod_name in _EDT_CONFIGURATION_MODULES:
            mod = path.parent / mod_name
            if mod.is_file():
                add_edge(conf_id, _make_id(str(mod)), "defines")
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

    # Child artifacts: attributes, tabular sections, enum values, forms, commands,
    # resources, dimensions, templates, predefined items and the rest of the map.
    def emit_child(elem, owner_id: str, owner_parts: list[str]):
        """One inline child: its node plus `contains` from its owner.

        Returns `(node id, SubKind, name)`, or None when the block is not in the
        map or carries no name — a block outside the map yields nothing rather
        than a guessed node.
        """
        tag = _edt_localname(elem.tag)
        entry = _EDT_CHILD_KINDS.get(tag)
        if entry is None:
            return None
        sub = entry[0]
        name = child_text(elem, "name")
        if not name:
            return None
        nid = _make_id(*owner_parts, sub, name)
        # The identifier is read the way its own block spells it, not uniformly.
        if tag in _EDT_NO_IDENTIFIER_TAGS:
            add_node(nid, name)
        elif tag in _EDT_PLAIN_ID_TAGS:
            add_node(nid, name, predefined_id=elem.get("id"))
        else:
            add_node(nid, name, uuid=elem.get("uuid"))
        add_edge(owner_id, nid, "contains")
        # A typed member references the object it is typed by, and the edge
        # belongs to the member, not to its owner: the aggregate follows from
        # the precise fact, never the other way round.
        type_refs(elem, nid)
        return nid, sub, name

    for child in root:
        if _edt_localname(child.tag) in _EDT_TRANSPARENT_CHILD_TAGS:
            # A wrapper that is not an entity: no name, no uuid. Its children
            # hang off the object itself, and they are what gets counted.
            for item in child:
                emit_child(item, obj_id, id_parts)
            continue
        emitted = emit_child(child, obj_id, id_parts)
        if emitted is None:
            continue
        child_id, sub, child_name = emitted
        # A container child owns children of its own, and the FQN grammar is
        # recursive (skill §3): a tabular-section column is
        # `Kind.Name.TabularSection.T.Attribute.A`, hanging off the section
        # rather than off the object. Only the kinds the skill documents as
        # containers descend — an unconditional recursion would turn <type>,
        # <synonym> and every other non-owning block into nodes.
        if sub in _EDT_CONTAINER_CHILD_KINDS:
            for grandchild in child:
                emit_child(grandchild, child_id, [*id_parts, sub, child_name])
        # A form/command owns a BSL module folder next to the .mdo.
        if sub == "Form":
            mod = path.parent / "Forms" / child_name / "Module.bsl"
        elif sub == "Command":
            mod = path.parent / "Commands" / child_name / "CommandModule.bsl"
        else:
            mod = None
        if mod is not None and mod.is_file():
            add_edge(child_id, _make_id(str(mod)), "defines")

    # Declared references, read by the explicit tag map. A tag outside the map
    # is ignored and a value that does not parse yields nothing — the same
    # "never invent" rule the kind whitelist follows.
    #
    # An EventSubscription only means something as a triple: <source> names the
    # type, <event> the platform event, <handler> the module method. The event
    # is read up front so it can ride in the handler edge's context; on its own
    # it names nothing to point at.
    event_name = child_text(root, "event") if kind == "EventSubscription" else None
    # The object's own <type>: a Constant, DefinedType, SessionParameter or
    # FilterCriterion is defined BY the type it holds, so the reference belongs
    # to the object itself rather than to any member of it.
    type_refs(root, obj_id, "object-type")
    for child in root:
        ln = _edt_localname(child.tag)
        text = (child.text or "").strip()

        entry = _EDT_REF_TAGS.get(ln)
        if entry and text:
            shape, context = entry
            if ln == "handler" and event_name:
                context = f"{context}:{event_name}"
            add_ref(obj_id, shape, text, context)

        # `<content>` is a leaf on FunctionalOption (membership, not ownership —
        # an option does not own the document whose visibility it drives, so
        # this stays `references`) and a container on ExchangePlan. Subsystem
        # keeps its own `contains` branch below and is skipped here.
        if ln == "content" and text and kind != "Subsystem":
            add_ref(obj_id, "object", text, "functional-option-content")

        container = _EDT_REF_CONTAINER_TAGS.get(ln)
        if container:
            child_tag, shape, context = container
            for c in child:
                if _edt_localname(c.tag) == child_tag and (c.text or "").strip():
                    add_ref(obj_id, shape, c.text.strip(), context)

        # References that belong to an inline child rather than to the object.
        # The child's node and its `contains` edge come from the child map above;
        # only the id is rebuilt here, to anchor the reference where it belongs.
        # A handler edge hanging off the service instead of off the method would
        # have to be re-anchored later, and a moved edge reads as a deleted one.
        if ln == "urlTemplates":
            tpl_name = child_text(child, "name")
            if tpl_name:
                for method in child:
                    if _edt_localname(method.tag) != "methods":
                        continue
                    m_name = child_text(method, "name")
                    if not m_name:
                        continue
                    m_id = _make_id(*id_parts, "URLTemplate", tpl_name,
                                    "Method", m_name)
                    add_procedure_ref(m_id, child_text(method, "handler"),
                                      "service-handler")

        if ln == "operations":
            op_name = child_text(child, "name")
            if op_name:
                add_procedure_ref(_make_id(*id_parts, "Operation", op_name),
                                  child_text(child, "procedureName"),
                                  "service-procedure")

        if ln == "columns":
            col_name = child_text(child, "name")
            if col_name:
                col_id = _make_id(*id_parts, "Column", col_name)
                for c in child:
                    if _edt_localname(c.tag) == "references" and (c.text or "").strip():
                        add_ref(col_id, "member", c.text.strip(),
                                "journal-column-source")

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

    # A CommonCommand keeps its CommandModule.bsl in its own folder (skill:
    # edt-structures §4.15). Bound here rather than added to _EDT_OBJECT_MODULES,
    # because an object's command keeps that same filename in Commands/<C>/ and
    # is already linked from the child loop above — an unconditional entry would
    # give every object a second, wrong edge to a file it does not own.
    if kind == "CommonCommand":
        mod = path.parent / "CommandModule.bsl"
        if mod.is_file():
            add_edge(obj_id, _make_id(str(mod)), "defines")

    return {"nodes": nodes, "edges": edges}


# ── 1C:EDT ordinary form (.oform) ─────────────────────────────────────────────
#
# An ordinary (non-managed) form is a V8 container — the same envelope as
# .cf/.epf/.erf — holding two named elements: `form` (the bracket tree of the
# layout) and `module` (the BSL, stored as a plain UTF-8 fragment, uncompressed).
# The format has NO official description: neither EDT's documentation nor
# edt.1c.ru specifies it. What is encoded here was measured across the 904
# ordinary forms of a real configuration and cross-checked against the
# edt-structures skill §5.3.
#
# A block header is the ASCII line `\r\nDDDDDDDD BBBBBBBB NNNNNNNN \r\n`:
# document size, this block's size, and the offset of the block that continues
# it (0x7fffffff = none). Data can be chained across blocks, so a reader that
# takes only the first block truncates the larger forms.
_V8_BLOCK_HEADER = re.compile(rb"\r\n([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) \r\n")
_V8_NO_NEXT = 0x7FFFFFFF
_V8_TOC_OFFSET = 16
_V8_NAME_OFFSET = 20          # element header: 8+8+4 bytes, then the UTF-16 name


def _v8_read_block(data: bytes, offset: int) -> bytes | None:
    """Payload of the block at `offset`, following its continuation chain."""
    m = _V8_BLOCK_HEADER.match(data, offset)
    if not m:
        return None
    doc_size, block_size, nxt = (int(g, 16) for g in m.groups())
    chunks = [data[m.end():m.end() + block_size]]
    seen = {offset}
    while nxt != _V8_NO_NEXT and nxt not in seen:
        seen.add(nxt)
        m = _V8_BLOCK_HEADER.match(data, nxt)
        if not m:
            break
        _, block_size, nxt = (int(g, 16) for g in m.groups())
        chunks.append(data[m.end():m.end() + block_size])
    payload = b"".join(chunks)
    return payload[:doc_size] if doc_size else payload


def _v8_elements(data: bytes) -> dict[str, tuple[bytes, int]] | None:
    """`{element name: (payload, byte offset of its data)}`, or None if unreadable.

    Selecting by NAME rather than by position or by sniffing the first byte:
    the container names its elements, and the physical order of the two data
    blocks is not fixed — measured 578 files with the tree first and 326 with
    the module first, so any position-based rule is wrong on a third of them.
    """
    toc = _v8_read_block(data, _V8_TOC_OFFSET)
    if not toc:
        return None
    out: dict[str, tuple[bytes, int]] = {}
    for i in range(0, len(toc) - 11, 12):
        head = int.from_bytes(toc[i:i + 4], "little")
        body = int.from_bytes(toc[i + 4:i + 8], "little")
        if head == _V8_NO_NEXT or body == _V8_NO_NEXT:
            continue
        header = _v8_read_block(data, head)
        if not header or len(header) <= _V8_NAME_OFFSET:
            continue
        try:
            name = header[_V8_NAME_OFFSET:].decode("utf-16-le").rstrip("\x00")
        except UnicodeDecodeError:
            continue
        payload = _v8_read_block(data, body)
        if payload is not None:
            out[name] = (payload, body)
    return out or None


def _v8_payload_line(data: bytes, body_offset: int) -> int:
    """1-based line in the .oform where an element's payload starts."""
    m = _V8_BLOCK_HEADER.match(data, body_offset)
    start = m.end() if m else body_offset
    return data.count(b"\n", 0, start) + 1


# The layout half of the container is 1C's bracket notation: nested `{…}` lists
# of quoted strings, uuids and numbers. It is tokenised rather than scanned with
# a regex over the file, so that a match inside a caption or inside the
# container's service bytes cannot become a reference.
_OFORM_TOKEN_RE = re.compile(
    r'"(?:[^"]|"")*"'                       # string; "" escapes a quote
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"   # uuid
    r"|-?\d+(?:\.\d+)?"                     # number
    r"|[{},]"                               # structure
    r'|[^\s{},"]+'                          # bare word
)
_OFORM_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def _oform_tree_uuids(text: str) -> list[str]:
    """Uuid tokens of a form layout, in order of first appearance.

    Only tokens that stand on their own in the stream: a uuid written inside a
    quoted string is data — a caption, a stored setting — not a reference.
    """
    seen: dict[str, None] = {}
    for m in _OFORM_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token[0] != '"' and _OFORM_UUID_RE.fullmatch(token):
            seen.setdefault(token, None)
    return list(seen)


def _oform_element_names(text: str) -> list[str]:
    """Names of the form's controls, in order of first appearance.

    The layout grammar is positional and undocumented, so this reads the one
    record shape that measurement pinned down: `{14,"<Name>",…` opens a named
    control. Verified on the corpus — every one of the 904 layouts yields names,
    23 039 in total, and not one of them fails to be a valid 1C identifier.

    Incomplete by construction: a few control kinds are named in a different
    record (`…,1,"CatalogList",{"Pattern"…`) and are not read here. Cross-checked
    against the `Controls.<Name>` uses in the modules, 94.5% of the forms that
    can be checked yield every name their own code refers to.
    """
    names: dict[str, None] = {}
    pending = 0        # 1 after `{`, 2 after `{14`, 3 after `{14,`
    for m in _OFORM_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token == "{":
            pending = 1
        elif pending == 1 and token == "14":
            pending = 2
        elif pending == 2 and token == ",":
            pending = 3
        elif pending == 3 and token[0] == '"':
            names.setdefault(token[1:-1].replace('""', '"'), None)
            pending = 0
        else:
            pending = 0
    return list(names)


def _edt_metadata_root(path: Path) -> Path | None:
    """The folder holding the project's metadata kind folders (`…/src`)."""
    parts = path.parts
    for i, seg in enumerate(parts[:-1]):
        if seg in _EDT_PLURAL_TO_KIND or seg in _EDT_COMMON_FOLDER_TO_KIND:
            return Path(*parts[:i]) if i else None
    return None


@lru_cache(maxsize=4)
def _edt_uuid_index(root: Path) -> dict[str, tuple[str, str]]:
    """`{uuid: (node id, label)}` for every object and member of the project.

    An ordinary form names nothing it refers to: the layout carries uuids, and
    only the `.mdo` files say what they are. Measured on the audited corpus,
    899 of 904 forms hold at least one uuid that resolves here, while the
    `Kind.Name` tokens the design first expected occur exactly zero times.

    Built by running the `.mdo` extractor rather than by re-deriving ids from
    the XML: an id assembled twice is an id that eventually disagrees with
    itself, and a reference to a node that does not exist is worse than none.
    """
    index: dict[str, tuple[str, str]] = {}
    for mdo in sorted(root.rglob("*.mdo")):
        result = extract_edt_mdo(mdo)
        if result.get("error"):
            continue
        object_id: tuple[str, str] | None = None
        for node in result.get("nodes", []):
            uuid = node.get("uuid")
            if not uuid:
                continue
            index.setdefault(uuid, (node["id"], node["label"]))
            if object_id is None:
                object_id = (node["id"], node["label"])
        # `<producedTypes>` are the platform types the object generates
        # (`CatalogRef.X` and friends). A control typed by one of them refers to
        # the object itself, which is how most object-level references appear.
        if object_id is None:
            continue
        try:
            head = mdo.read_bytes()[:4096].decode("utf-8", "replace")
        except OSError:
            continue
        produced = re.search(r"<producedTypes>(.*?)</producedTypes>", head, re.S)
        if produced:
            for uuid in _OFORM_UUID_RE.findall(produced.group(1)):
                index.setdefault(uuid, object_id)
    return index


def extract_edt_oform(path: Path) -> dict:
    """Extract an ordinary form (Form.oform): its module, controls and references.

    An ordinary form's module has no file of its own — it lives inside the
    container — which is why it was invisible to a scan that only looked at
    `.bsl`. On the audited configuration that is 16 045 procedures, roughly a
    third of all application logic.

    The module is handed to the shared `extract_bsl` rather than parsed here:
    a second BSL parser would drift from the first. Ordinary-form BSL has no
    compilation directives and reaches the form through `Controls.<Name>`
    instead of `Элементы`, so the extractor has to accept it as-is.

    The layout half yields the form's controls and its references, the latter by
    uuid — an ordinary form names nothing it refers to. Everything hangs off the
    form node the parent `.mdo` already emitted.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return {"nodes": [], "edges": [], "error": f"cannot read {path}"}
    if len(data) > _PROJECT_XML_MAX_BYTES:
        return {"nodes": [], "edges": [], "error": "oform file too large"}

    elements = _v8_elements(data)
    if not elements:
        return {"nodes": [], "edges": [], "error": "not a readable V8 container"}

    module = elements.get("module")
    result: dict = {"nodes": [], "edges": []}
    has_module = module is not None and module[0].strip() != b""
    if has_module:
        payload, body_offset = module
        if payload[:3] == b"\xef\xbb\xbf":
            payload = payload[3:]
        result = extract_bsl(path, source=payload,
                             line_offset=_v8_payload_line(data, body_offset) - 1)
        if result.get("error"):
            return result

    # The form node already exists, emitted from the <forms> block of the
    # parent .mdo; this fills it in rather than creating a second one.
    form_id = _edt_form_owner_id(path)
    if not form_id:
        return result
    str_path = str(path)
    result["nodes"].append({
        "id": form_id, "label": path.parent.name, "file_type": "code",
        "source_file": str_path, "source_location": "L1",
    })
    if has_module:
        # Only when the module was actually parsed: six forms of the corpus carry
        # an empty module element, and pointing `defines` at a file node that was
        # never emitted would leave the edge dangling.
        result["edges"].append({
            "source": form_id, "target": _make_id(str_path),
            "relation": "defines", "confidence": "EXTRACTED",
            "source_file": str_path, "weight": 1.0,
        })

    tree = elements.get("form")
    if tree is None:
        return result
    layout = tree[0].decode("utf-8", "replace")

    # Controls of the form. Like a managed form's attributes they carry no uuid,
    # so the id is the form's id plus the name, and the form's uuid is recorded
    # as parent_uuid.
    form_uuid = _edt_parent_uuid(path, form_name=path.parent.name)
    for name in _oform_element_names(layout):
        element_id = _make_id(form_id, "FormElement", name)
        if element_id == form_id:
            continue
        node = {"id": element_id, "label": name, "file_type": "code",
                "source_file": str_path, "source_location": "L1"}
        if form_uuid:
            node["parent_uuid"] = form_uuid
        result["nodes"].append(node)
        result["edges"].append({
            "source": form_id, "target": element_id, "relation": "contains",
            "confidence": "EXTRACTED", "source_file": str_path, "weight": 1.0,
        })

    root = _edt_metadata_root(path)
    if root is None:
        return result
    index = _edt_uuid_index(root)
    seen = {form_id}
    for uuid in _oform_tree_uuids(layout):
        target = index.get(uuid)
        if target is None or target[0] in seen:
            continue
        seen.add(target[0])
        result["nodes"].append({
            "id": target[0], "label": target[1], "file_type": "code",
            "source_file": str_path, "source_location": "L1",
        })
        result["edges"].append({
            "source": form_id, "target": target[0], "relation": "references",
            "confidence": "EXTRACTED", "source_file": str_path, "weight": 1.0,
            "context": "oform",
        })
    return result


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
            m = _EDT_TYPE_RE.match(text)
            if m:
                kind = _edt_type_kind(m.group(1))
                if kind:
                    add_ref(kind, m.group(2))

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


# ── 1C:EDT command interface (.cmi) ───────────────────────────────────────────
#
# The command interface of the configuration or of one subsystem: which commands
# it shows, in what order, and to which roles. Plain XML, and — unlike the
# ordinary form beside it — it names what it refers to. Measured on the audited
# corpus: 49 files, 1653 references, every one of them a `Kind.Name` FQN.


def _edt_cmi_owner(path: Path) -> tuple[str, str] | None:
    """`(node id, label)` of the configuration or subsystem the file belongs to."""
    chain = _edt_subsystem_chain(path)
    if chain:
        return _make_id("Subsystem", *chain), "Subsystem." + ".".join(chain)
    if "Configuration" in path.parts:
        return _make_id("Configuration"), "Configuration"
    return None


def _edt_cmi_target(text: str) -> tuple[str, str] | None:
    """`(node id, label)` for one command-interface value.

    Three shapes occur, and only these: `Kind.Name`; `Kind.Name.Command.C`, an
    object's own command; and `Kind.Name.StandardCommand.C`, a command the
    platform provides. A standard command is not a metadata object and has no
    node of its own, so that edge lands on the object it acts on.
    """
    parts = [p for p in text.split(".") if p]
    if len(parts) < 2 or parts[0] not in _EDT_KIND_PREFIXES:
        return None
    if len(parts) == 2:
        return _make_id(*parts), text
    if len(parts) == 4:
        if parts[0] == "Subsystem" and parts[2] == "Subsystem":
            # Nested subsystem: the repeated marker is the FQN's, not the id's.
            return (_make_id("Subsystem", parts[1], parts[3]),
                    f"Subsystem.{parts[1]}.{parts[3]}")
        if parts[2] == "Command":
            return _make_id(*parts), text
        if parts[2] == "StandardCommand":
            return _make_id(parts[0], parts[1]), f"{parts[0]}.{parts[1]}"
    return None


def extract_edt_cmi(path: Path) -> dict:
    """Extract the references a 1C:EDT command interface (.cmi) declares.

    Denser in links than any other auxiliary file of an EDT project — 34 per
    file against 6.8 for an ordinary form — because that is all it is: an
    ordering of commands, the subsystems that show them, and the roles that see
    them. Anchors on the configuration or the subsystem the file sits under.
    """
    root, err = _edt_parse_xml(path)
    if err is not None:
        return err
    owner = _edt_cmi_owner(path)
    if not owner:
        return {"nodes": [], "edges": []}
    owner_id, owner_label = owner

    str_path = str(path)
    nodes: list[dict] = [{"id": owner_id, "label": owner_label, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen: set[str] = {owner_id}
    for elem in root.iter():
        target = _edt_cmi_target((elem.text or "").strip())
        if target is None or target[0] in seen:
            continue
        seen.add(target[0])
        nodes.append({"id": target[0], "label": target[1], "file_type": "code",
                      "source_file": str_path, "source_location": "L1"})
        edges.append({"source": owner_id, "target": target[0],
                      "relation": "references", "confidence": "EXTRACTED",
                      "source_file": str_path, "weight": 1.0,
                      "context": "command-interface"})
    return {"nodes": nodes, "edges": edges}


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


