"""1C XDTO package extractor (`XDTOPackages/<Name>/Package.xdto`) — fork delta.

An XDTO package is an XML schema in 1C's own grammar, mapped onto the shared
model of `graphify.extractors.xsd`: `objectType`/`valueType` are types, a
package-level `property` is a global element, a `property` inside a type is a
property, and a `<typeDef>` is an anonymous type. Ids and labels come from the
same functions as for W3C XSD, so an XSD and an XDTO package describing one
namespace yield one set of nodes.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.xsd import (
    FACETS,
    XSI_NS,
    Owner,
    SchemaEmitter,
    XmlElement,
    read_schema_source,
)

XDTO_NS = "http://v8.1c.ru/8.1/xdto"


def _children(el: XmlElement, tag: str | None = None) -> list[XmlElement]:
    return [c for c in el.children if c.ns == XDTO_NS and (tag is None or c.tag == tag)]


def _object_type(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    em.ref(owner, el, el.attrib.get("base"), kind="type", relation="inherits",
           context=None, attr="base_ref")
    for c in _children(el, "property"):
        if c.attrib.get("name"):
            _property(em, em.property(owner, c.attrib["name"], c.line), c)
        else:
            em.ref(owner, c, c.attrib.get("ref"), kind="element", relation="references",
                   context="element-ref", attr=None)


def _value_type(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    em.ref(owner, el, el.attrib.get("base"), kind="type", relation="inherits",
           context=None, attr="base_ref")
    em.ref(owner, el, el.attrib.get("itemType"), kind="type", relation="references",
           context="list-item", attr="item_type_ref")
    for member in (el.attrib.get("memberTypes") or "").split():
        em.ref(owner, el, member, kind="type", relation="references",
               context="union-member", attr="member_type_refs", many=True)
    for name in FACETS:
        if name in el.attrib:
            em.facet(owner, name, el.attrib[name])
    for c in _children(el):
        if c.tag in ("enumeration", "pattern"):
            em.facet(owner, c.tag, c.text)
        elif c.tag == "typeDef":
            _type_def(em, owner, c)


def _type_def(em: SchemaEmitter, owner: Owner, el: XmlElement) -> None:
    anon = em.anonymous(owner, el.line)
    if el.attrib.get(f"{{{XSI_NS}}}type", "").rpartition(":")[2] == "ObjectType":
        _object_type(em, anon, el)
    else:
        _value_type(em, anon, el)


def _property(em: SchemaEmitter, prop: Owner, el: XmlElement) -> None:
    em.ref(prop, el, el.attrib.get("type"), kind="type", relation="references",
           context="property-type", attr="type_ref")
    for c in _children(el, "typeDef"):
        _type_def(em, prop, c)


def extract_xdto(path: Path) -> dict:
    """Extract a 1C XDTO package into namespace / type / property nodes."""
    root, error = read_schema_source(path, "XDTO package")
    if root is None:
        return {"nodes": [], "edges": [], "error": error}
    if root.ns != XDTO_NS or root.tag != "package":
        return {"nodes": [], "edges": [], "error": f"not an XDTO package: {path}"}
    em = SchemaEmitter(path)
    tns = root.attrib.get("targetNamespace", "")
    ns_id = em.namespace(tns, root.line, defined=True)
    container = ns_id or em.file_id
    if ns_id:
        em.edge(em.file_id, ns_id, "contains", root.line)
    for c in _children(root):
        name = c.attrib.get("name")
        if c.tag == "import":
            em.imports(container, tns, c.attrib.get("namespace", ""), c.line)
        elif c.tag == "objectType" and name:
            _object_type(em, em.declare_global(container, tns, "type", name, c.line), c)
        elif c.tag == "valueType" and name:
            _value_type(em, em.declare_global(container, tns, "type", name, c.line), c)
        elif c.tag == "property" and name:
            _property(em, em.declare_global(container, tns, "element", name, c.line), c)
    return em.result()
