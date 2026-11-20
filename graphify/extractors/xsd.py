"""XML Schema extractor: W3C XSD files and the schemas inside a WSDL's <types>.

Every schema grammar is reduced to one model — namespace → global symbol →
local property → anonymous type — so a type is ONE node no matter which file
declares it and how many files refer to it:

- a global symbol's id is built from (namespace URI, symbol space, name) only,
  never from the file path, so a reference from another schema, another
  grammar or a 1C `.mdo` lands on the declaring file's node;
- its label carries a short prefix derived from the namespace URI alone, so a
  stub minted by a referring file and the declaration agree on the label too;
- references to built-in types (XML Schema itself, the 1C platform's
  `http://v8.1c.ru/8.N/...` namespaces) are kept as node attributes, not edges,
  so `xs:string` does not become a node with tens of thousands of edges.

Parsing uses expat directly: QName values (`type="tns:T"`) are resolved against
the prefix declarations in scope at the element that carries them — including
declarations on that very element — and ElementTree discards those.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphify.extractors.base import _PROJECT_XML_MAX_BYTES, _make_id, _project_xml_is_safe

XS_NS = "http://www.w3.org/2001/XMLSchema"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XML_NS = "http://www.w3.org/XML/1998/namespace"
_WSDL_NAMESPACES = frozenset({"http://schemas.xmlsoap.org/wsdl/", "http://www.w3.org/ns/wsdl"})
_BUILTIN_NAMESPACES = frozenset({XS_NS, XSI_NS, XML_NS})
_ONEC_PLATFORM_NS = re.compile(r"^http://v8\.1c\.ru/8\.\d+/")

# Id tokens between an owner and its member. make_id folds every non-word
# character to `_`, so without them property `P` of type `T` would collide with
# a type named `T.P` (EnterpriseData has thousands: `СправочникСсылка.Банки`),
# and `#type` / `@id` with a property literally named `type` / `id`.
_PROP_TOKEN = "xmlprop"
_ANON_TOKEN = "xmlanon"
_ATTR_TOKEN = "xmlattr"

FACETS = ("length", "minLength", "maxLength", "totalDigits", "fractionDigits",
          "minInclusive", "maxInclusive", "minExclusive", "maxExclusive", "whiteSpace")


# ── Identity and labels (shared with the XDTO adapter and the .mdo bridge) ────

def is_builtin_namespace(ns: str) -> bool:
    """XML Schema / XML / XSI and the 1C platform namespaces: never nodes."""
    return ns in _BUILTIN_NAMESPACES or bool(_ONEC_PLATFORM_NS.match(ns))


def namespace_prefix(ns: str) -> str:
    """Short, human label prefix computed from the namespace URI alone.

    `http://v8.1c.ru/edi/edi_stnd/EnterpriseData/1.22` -> `EnterpriseData/1.22`,
    `http://file.api.update.onec.ru` -> `file.api.update.onec.ru`,
    `urn:a:b:c` -> `b:c`, the empty namespace -> ``""``.
    """
    if not ns:
        return ""
    if ns.lower().startswith("urn:"):
        parts = [p for p in ns.split(":")[1:] if p]
        return ":".join(parts[-2:])
    m = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]*)([^?#]*)", ns)
    if not m:
        segs = [s for s in re.split(r"[/:]", ns) if s]
        return "/".join(segs[-2:]) or ns
    host, path = m.group(1), m.group(2)
    segs = [s for s in path.split("/") if s]
    if len(segs) >= 2:
        return "/".join(segs[-2:])
    if segs:
        return f"{host}/{segs[0]}"
    return host


def namespace_id(ns: str) -> str:
    return _make_id("xmlns", ns)


def _label(ns: str, tail: str) -> str:
    prefix = namespace_prefix(ns)
    return f"{prefix} · {tail}" if prefix else tail


def _symbol_tail(kind: str, name: str) -> str:
    return name if kind == "type" else f"{name} [{kind}]"


def symbol_id(ns: str, kind: str, name: str) -> str:
    """Id of a global symbol; `kind` is its symbol space (type, element, ...)."""
    return _make_id("xml", ns, kind, name)


def symbol_label(ns: str, kind: str, name: str) -> str:
    return _label(ns, _symbol_tail(kind, name))


# ── Parsing ───────────────────────────────────────────────────────────────────

@dataclass
class XmlElement:
    ns: str
    tag: str
    attrib: dict[str, str]
    line: int
    nsmap: dict[str, str]
    children: list["XmlElement"] = field(default_factory=list)
    _text: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self._text).strip()


def parse_xml(src: bytes) -> XmlElement:
    """Parse with expat, keeping line numbers and the prefix map in scope at
    every element. Raises ``xml.parsers.expat.ExpatError`` on malformed input.
    Namespaced attributes are keyed ``{uri}local``, plain ones by local name."""
    import xml.parsers.expat as expat

    sep = "\x01"
    parser = expat.ParserCreate(namespace_separator=sep)
    roots: list[XmlElement] = []
    stack: list[XmlElement] = []
    pending: dict[str, str] = {}

    def start_ns(prefix, uri):
        pending[prefix or ""] = uri or ""

    def start(name, attrs):
        ns, _, local = name.rpartition(sep)
        parent = stack[-1] if stack else None
        nsmap = parent.nsmap if parent else {"xml": XML_NS}
        if pending:
            nsmap = {**nsmap, **pending}
            pending.clear()
        attrib = {}
        for k, v in attrs.items():
            kns, _, kl = k.rpartition(sep)
            attrib[f"{{{kns}}}{kl}" if kns else kl] = v
        el = XmlElement(ns, local, attrib, parser.CurrentLineNumber, nsmap)
        (parent.children if parent else roots).append(el)
        stack.append(el)

    def end(_name):
        stack.pop()

    def chars(data):
        if stack:
            stack[-1]._text.append(data)

    parser.StartNamespaceDeclHandler = start_ns
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = chars
    parser.Parse(src, True)
    return roots[0]


def resolve_qname(el: XmlElement, value: str) -> tuple[str, str] | None:
    """`p:Local` -> (uri of p in scope at `el`, Local); unprefixed -> default
    namespace in scope (or no namespace). None when the prefix is unknown."""
    value = value.strip()
    if not value:
        return None
    prefix, sep, local = value.rpartition(":")
    if sep:
        if prefix not in el.nsmap or not local:
            return None
        return el.nsmap[prefix], local
    return el.nsmap.get("", ""), local


def read_schema_source(path: Path, what: str) -> tuple[XmlElement | None, str | None]:
    """Read and parse one schema file under the shared project-XML guards.
    Returns (root, None) or (None, reason) — a refusal never yields nodes."""
    import xml.parsers.expat as expat

    try:
        src = path.read_bytes()
    except OSError as e:
        return None, f"cannot read {path}: {e}"
    if len(src) > _PROJECT_XML_MAX_BYTES:
        return None, f"{what} file too large ({len(src)} bytes > {_PROJECT_XML_MAX_BYTES}): {path}"
    if not _project_xml_is_safe(src):
        return None, f"refusing XML with DOCTYPE/ENTITY declaration: {path}"
    try:
        return parse_xml(src), None
    except expat.ExpatError as e:
        return None, f"XML parse error in {path}: {e}"


# ── Emission ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Owner:
    """A node that can own properties: its id, id parts, namespace, label tail."""
    nid: str
    parts: tuple[str, ...]
    ns: str
    tail: str


class SchemaEmitter:
    """Collects the nodes and edges of one schema file."""

    def __init__(self, path: Path):
        self.str_path = str(path)
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._edge_keys: set[tuple] = set()
        self.file_id = _make_id(str(path))
        self._node(self.file_id, path.name, 1)

    def _node(self, nid: str, label: str, line: int, **attrs: Any) -> dict:
        node = self._by_id.get(nid)
        if node is None:
            node = {"id": nid, "label": label, "file_type": "code",
                    "source_file": self.str_path, "source_location": f"L{line}"}
            self._by_id[nid] = node
            self.nodes.append(node)
        for k, v in attrs.items():
            node.setdefault(k, v)
        return node

    def _define(self, nid: str, line: int) -> None:
        node = self._by_id[nid]
        if not node.get("xml_defined"):
            node["xml_defined"] = True
            node["source_location"] = f"L{line}"

    def node_attrs(self, nid: str) -> dict:
        return self._by_id[nid]

    def edge(self, src: str, tgt: str, relation: str, line: int,
             context: str | None = None) -> None:
        key = (src, tgt, relation, context)
        if not src or not tgt or src == tgt or key in self._edge_keys:
            return
        self._edge_keys.add(key)
        e = {"source": src, "target": tgt, "relation": relation,
             "confidence": "EXTRACTED", "source_file": self.str_path,
             "source_location": f"L{line}", "weight": 1.0}
        if context:
            e["context"] = context
        self.edges.append(e)

    def namespace(self, ns: str, line: int, *, defined: bool = False) -> str | None:
        """Node of a non-built-in namespace (None for built-in or empty)."""
        if not ns or is_builtin_namespace(ns):
            return None
        nid = namespace_id(ns)
        self._node(nid, ns, line, namespace=ns, xml_kind="namespace", shared_anchor=True)
        if defined:
            self._define(nid, line)
        return nid

    def symbol(self, ns: str, kind: str, name: str, line: int, *, defined: bool) -> Owner:
        parts = ("xml", ns, kind, name)
        nid = _make_id(*parts)
        self._node(nid, symbol_label(ns, kind, name), line,
                   namespace=ns, xml_kind=kind, shared_anchor=True)
        if defined:
            self._define(nid, line)
        return Owner(nid, parts, ns, _symbol_tail(kind, name))

    def declare_global(self, container: str, ns: str, kind: str, name: str,
                       line: int) -> Owner:
        owner = self.symbol(ns, kind, name, line, defined=True)
        self.edge(container, owner.nid, "contains", line)
        return owner

    def property(self, owner: Owner, name: str, line: int, *,
                 attribute: bool = False) -> Owner:
        parts = owner.parts + (_ATTR_TOKEN if attribute else _PROP_TOKEN, name)
        # `/` cannot occur in an XML name, while `.` can and does: EnterpriseData
        # names a tabular-section row type `Документ.X.Строки`, which a `.`
        # separator would print exactly like property `Строки` of `Документ.X`.
        tail = f"{owner.tail}/{'@' if attribute else ''}{name}"
        nid = _make_id(*parts)
        self._node(nid, _label(owner.ns, tail), line, namespace=owner.ns,
                   xml_kind="property", shared_anchor=True)
        self._define(nid, line)
        self.edge(owner.nid, nid, "contains", line)
        return Owner(nid, parts, owner.ns, tail)

    def anonymous(self, owner: Owner, line: int) -> Owner:
        parts = owner.parts + (_ANON_TOKEN,)
        tail = f"{owner.tail}#type"
        nid = _make_id(*parts)
        self._node(nid, _label(owner.ns, tail), line, namespace=owner.ns,
                   xml_kind="anonymous-type", shared_anchor=True)
        self._define(nid, line)
        self.edge(owner.nid, nid, "contains", line)
        return Owner(nid, parts, owner.ns, tail)

    def ref(self, src: Owner, el: XmlElement, value: str | None, *, kind: str,
            relation: str, context: str | None, attr: str | None,
            many: bool = False) -> None:
        """Reference from `src` to the global symbol named by QName `value`.
        A built-in target goes to attribute `attr` (a list when `many`)."""
        if not value:
            return
        q = resolve_qname(el, value)
        if q is None:
            return
        ns, local = q
        if is_builtin_namespace(ns):
            if attr:
                ref = f"{{{ns}}}{local}"
                node = self._by_id[src.nid]
                if many:
                    node.setdefault(attr, [])
                    if ref not in node[attr]:
                        node[attr].append(ref)
                else:
                    node.setdefault(attr, ref)
            return
        target = self.symbol(ns, kind, local, el.line, defined=False)
        self.edge(src.nid, target.nid, relation, el.line, context)

    def imports(self, container: str, own_ns: str, ns: str, line: int) -> None:
        if ns == own_ns:
            return
        target = self.namespace(ns, line)
        if target:
            self.edge(container, target, "imports", line)

    def facet(self, owner: Owner, name: str, value: str) -> None:
        node = self._by_id[owner.nid]
        if name in ("enumeration", "pattern"):
            node.setdefault(f"xml_{name}", []).append(value)
        else:
            node.setdefault(f"xml_{name}", value)

    def result(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges,
                "input_tokens": 0, "output_tokens": 0}


# ── W3C XSD adapter ───────────────────────────────────────────────────────────

_GLOBAL_KINDS = {"complexType": "type", "simpleType": "type", "element": "element",
                 "attribute": "attribute", "group": "group",
                 "attributeGroup": "attributeGroup"}


def _xs(el: XmlElement) -> list[XmlElement]:
    return [c for c in el.children if c.ns == XS_NS]


def _xsd_schema(em: SchemaEmitter, schema: XmlElement) -> None:
    tns = schema.attrib.get("targetNamespace", "")
    ns_id = em.namespace(tns, schema.line, defined=True)
    container = ns_id or em.file_id
    if ns_id:
        em.edge(em.file_id, ns_id, "contains", schema.line)
    _xsd_globals(em, schema, tns, container)


def _xsd_globals(em: SchemaEmitter, parent: XmlElement, tns: str, container: str) -> None:
    for c in _xs(parent):
        if c.tag == "import":
            em.imports(container, tns, c.attrib.get("namespace", ""), c.line)
        elif c.tag in ("redefine", "override"):
            _xsd_globals(em, c, tns, container)
        elif c.tag in _GLOBAL_KINDS and c.attrib.get("name"):
            owner = em.declare_global(container, tns, _GLOBAL_KINDS[c.tag],
                                      c.attrib["name"], c.line)
            _xsd_declaration(em, owner, c)


def _xsd_declaration(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    if el.tag == "complexType":
        _xsd_complex(em, owner, el)
    elif el.tag == "simpleType":
        _xsd_simple(em, owner, el)
    elif el.tag in ("element", "attribute"):
        em.ref(owner, el, el.attrib.get("type"), kind="type", relation="references",
               context="property-type", attr="type_ref")
        em.ref(owner, el, el.attrib.get("substitutionGroup"), kind="element",
               relation="references", context="substitution-group", attr=None)
        for c in _xs(el):
            if c.tag in ("complexType", "simpleType"):
                _xsd_declaration(em, em.anonymous(owner, c.line), c)
    elif el.tag == "group":
        _xsd_particles(em, owner, el)
    elif el.tag == "attributeGroup":
        _xsd_complex(em, owner, el)


def _xsd_complex(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    for c in _xs(el):
        if c.tag in ("sequence", "choice", "all"):
            _xsd_particles(em, owner, c)
        elif c.tag == "group":
            em.ref(owner, c, c.attrib.get("ref"), kind="group", relation="references",
                   context="group-ref", attr=None)
        elif c.tag in ("attribute", "attributeGroup"):
            _xsd_attribute_item(em, owner, c)
        elif c.tag in ("simpleContent", "complexContent"):
            for d in _xs(c):
                if d.tag in ("extension", "restriction"):
                    em.ref(owner, d, d.attrib.get("base"), kind="type", relation="inherits",
                           context=None, attr="base_ref")
                    _xsd_complex(em, owner, d)
                    _xsd_facets(em, owner, d)


def _xsd_particles(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    for c in _xs(el):
        if c.tag == "element":
            if c.attrib.get("name"):
                _xsd_declaration(em, em.property(owner, c.attrib["name"], c.line), c)
            else:
                em.ref(owner, c, c.attrib.get("ref"), kind="element", relation="references",
                       context="element-ref", attr=None)
        elif c.tag in ("sequence", "choice", "all"):
            _xsd_particles(em, owner, c)
        elif c.tag == "group":
            em.ref(owner, c, c.attrib.get("ref"), kind="group", relation="references",
                   context="group-ref", attr=None)


def _xsd_attribute_item(em: SchemaEmitter, owner: Owner, c: XmlElement) -> None:
    if c.tag == "attribute":
        if c.attrib.get("name"):
            prop = em.property(owner, c.attrib["name"], c.line, attribute=True)
            _xsd_declaration(em, prop, c)
        else:
            em.ref(owner, c, c.attrib.get("ref"), kind="attribute", relation="references",
                   context="attribute-ref", attr=None)
    else:
        em.ref(owner, c, c.attrib.get("ref"), kind="attributeGroup", relation="references",
               context="attribute-group-ref", attr=None)


def _xsd_simple(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    for c in _xs(el):
        if c.tag == "restriction":
            em.ref(owner, c, c.attrib.get("base"), kind="type", relation="inherits",
                   context=None, attr="base_ref")
            _xsd_facets(em, owner, c)
        elif c.tag == "list":
            em.ref(owner, c, c.attrib.get("itemType"), kind="type", relation="references",
                   context="list-item", attr="item_type_ref")
        elif c.tag == "union":
            for member in (c.attrib.get("memberTypes") or "").split():
                em.ref(owner, c, member, kind="type", relation="references",
                       context="union-member", attr="member_type_refs", many=True)


def _xsd_facets(em: SchemaEmitter, owner: Owner, restriction: XmlElement) -> None:
    for f in _xs(restriction):
        if (f.tag in FACETS or f.tag in ("enumeration", "pattern")) and "value" in f.attrib:
            em.facet(owner, f.tag, f.attrib["value"])


def extract_xsd(path: Path) -> dict:
    """Extract an XSD file, or the schemas inside a WSDL's `<types>`."""
    root, error = read_schema_source(path, "schema")
    if root is None:
        return {"nodes": [], "edges": [], "error": error}
    em = SchemaEmitter(path)
    if root.ns == XS_NS and root.tag == "schema":
        _xsd_schema(em, root)
    elif root.ns in _WSDL_NAMESPACES:
        for types in root.children:
            if types.ns == root.ns and types.tag == "types":
                for schema in _xs(types):
                    if schema.tag == "schema":
                        _xsd_schema(em, schema)
    return em.result()
