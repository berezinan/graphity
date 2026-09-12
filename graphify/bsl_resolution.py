"""Cross-file resolution for 1C/BSL calls that name their module.

In 1C every call across a module boundary is written with a receiver —
``ОплатыОбщееСервер.Метод()``, ``Документы.Счет.Метод()``, or a variable bound to
a module by ``ОбщегоНазначения.ОбщийМодуль("Имя")``. That is the same shape other
languages use for object-method calls, so the shared name-based pass skips it as
a member call and 1C graphs ended up with intra-module edges only (measured:
6 081 of 6 238 ``calls`` edges inside one file, no cross-module call at all).

This pass resolves those calls the only way that is safe: through the module the
receiver names. The extractor carries the receiver on each raw_call
(``receiver`` — a common-module name, ``receiver_object`` — a metadata object
whose ManagerModule owns the method); here the module is turned into a file and
the method looked up in THAT file. No match in that file means no edge — never a
same-named definition elsewhere.

Bare (unqualified) calls are resolved too, but only into a global common module
(``<global>true</global>``): that is the one form in which 1C lets an
unqualified call leave its own module. Everything else stays unresolved.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from graphify.extractors.bsl import _EDT_KIND_TO_PLURAL

_BSL_SUFFIXES = (".bsl", ".os", ".osl")


def _norm(path: str) -> str:
    return str(path).replace("\\", "/")


def _bsl_raw_calls(per_file: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for result in per_file:
        if not isinstance(result, dict):
            continue
        for rc in result.get("raw_calls", []):
            if isinstance(rc, dict) and rc.get("lang") == "bsl":
                calls.append(rc)
    return calls


def _definition_key(label: str) -> str:
    """Key of a procedure/function node label (``Имя()`` -> ``имя``)."""
    return label.strip().rstrip("()").lower()


def resolve_bsl_module_calls(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Emit ``calls`` edges for BSL calls whose receiver names a module.

    Additive: every edge here is one the shared pass skipped. Each emission needs
    a single module file for the receiver AND a single definition of that name
    inside it, so an ambiguous name resolves to nothing rather than to a guess.
    """
    # module name (lower) -> module file; a name owned by two files is dropped,
    # because two projects in one corpus can each define `ОбщегоНазначения`.
    common_modules: dict[str, str | None] = {}
    # (kind, object name) lower -> ManagerModule file.
    manager_modules: dict[tuple[str, str], str | None] = {}
    # source file -> definition key -> [node ids]; only procedures and functions,
    # so a call can never land on an .mdo attribute or a form element.
    defs_by_file: dict[str, dict[str, list[str]]] = {}
    # Common modules whose .mdo carries <global>true</global>.
    global_modules: set[str] = set()

    for node in all_nodes:
        nid, label = node.get("id"), str(node.get("label", ""))
        source_file = _norm(node.get("source_file", ""))
        if node.get("bsl_global") and label:
            global_modules.add(label.rsplit(".", 1)[-1].lower())
        if not nid or not source_file:
            continue
        if source_file.endswith(_BSL_SUFFIXES) and label.endswith(")"):
            defs_by_file.setdefault(source_file, {}).setdefault(
                _definition_key(label), []).append(str(nid))

    for source_file in defs_by_file:
        parts = PurePosixPath(source_file).parts
        if len(parts) < 3:
            continue
        folder, owner, filename = parts[-3], parts[-2], parts[-1]
        if folder == "CommonModules" and filename.lower() == "module.bsl":
            key = owner.lower()
            common_modules[key] = None if key in common_modules else source_file
        elif filename.lower() == "managermodule.bsl":
            for kind, plural in _EDT_KIND_TO_PLURAL.items():
                if folder == plural:
                    mkey = (kind.lower(), owner.lower())
                    manager_modules[mkey] = (
                        None if mkey in manager_modules else source_file)
                    break

    existing_pairs = {(e.get("source"), e.get("target")) for e in all_edges}

    def _unique_definition(module_file: str | None, callee: str) -> str | None:
        if not module_file:
            return None
        nids = defs_by_file.get(module_file, {}).get(_definition_key(callee), [])
        return nids[0] if len(nids) == 1 else None

    def _emit(rc: dict[str, Any], target: str, context: str,
              confidence: str, score: float) -> None:
        caller = str(rc.get("caller_nid", ""))
        if not caller or not target or caller == target:
            return
        if (caller, target) in existing_pairs:
            return
        existing_pairs.add((caller, target))
        all_edges.append({
            "source": caller,
            "target": target,
            "relation": "calls",
            "context": context,
            "confidence": confidence,
            "confidence_score": score,
            "source_file": rc.get("source_file", ""),
            "source_location": rc.get("source_location"),
            "weight": 1.0,
        })

    for rc in _bsl_raw_calls(per_file):
        callee = str(rc.get("callee", "")).strip()
        if not callee:
            continue

        if not rc.get("is_member_call"):
            # A bare call leaves its own module only into a global common module.
            for name in global_modules:
                target = _unique_definition(common_modules.get(name), callee)
                if target is not None:
                    _emit(rc, target, "global_module_call", "INFERRED", 0.8)
                    break
            continue

        receiver_object = rc.get("receiver_object")
        if receiver_object:
            kind, obj_name = receiver_object
            module_file = manager_modules.get((str(kind).lower(), str(obj_name).lower()))
        else:
            receiver = rc.get("receiver")
            if not receiver:
                continue
            module_file = common_modules.get(str(receiver).lower())

        target = _unique_definition(module_file, callee)
        if target is None:
            continue
        # A receiver bound by `ОбщийМодуль("Имя")` is as explicit as a written
        # module name — the literal and the call sit in one body — but the two
        # are told apart so a consumer can see which form produced the edge.
        context = ("module_alias_call" if rc.get("receiver_bound")
                   else "module_qualified_call")
        _emit(rc, target, context, "EXTRACTED", 1.0)
