"""Spike-only change of upstream query tokenisation (decision 2026-11-04).

`serve._is_searchable` drops short Latin words only ("in", "of"), so
one- and two-letter Cyrillic words ("в", "и", "на") become query terms and,
matched as substrings, hit almost every node of a 1C graph (2.0 M of 2.0 M
candidates on ERP2). The spike drops them too: a token of at most two
characters, all Cyrillic (U+0400..U+052F), is not searchable. Mixed tokens
("1с") and other scripts keep the upstream rule.

The package is not touched. `apply()` patches the module attribute, which
`serve._query_terms` reads at call time, so JsonBackend and SnapBackend in
the same process both see the change. Every parity test and benchmark child
applies it before comparing.
"""
from __future__ import annotations

from graphify import serve

UPSTREAM = serve._is_searchable


def _is_cyrillic_short(term: str) -> bool:
    return 0 < len(term) <= 2 and all("Ѐ" <= ch <= "ԯ" for ch in term)


def is_searchable(term: str) -> bool:
    if all("a" <= ch <= "z" for ch in term):
        return len(term) > 2
    if _is_cyrillic_short(term):
        return False
    return True


UPSTREAM_AMBIGUITY = serve._ambiguity_message


def ambiguity_message(G, label: str, rivals: list[str]) -> str:
    """`serve._ambiguity_message` with its one quadratic line made linear.

    Upstream tests `any(sources.count(src) > 1 for src in sources if src)`:
    O(n^2) in the number of rivals, 6 s of 7 for a name 20 266 ERP2 nodes
    share. A Counter answers the same question. Every other line is
    upstream's, so the message is byte-identical (tests/test_fastpaths.py).
    """
    from collections import Counter

    several_repos = len({str(G.nodes[r].get("repo") or "") for r in rivals}) > 1

    def where(r: str) -> str:
        source = G.nodes[r].get("source_file") or ""
        repo = G.nodes[r].get("repo")
        return f"{repo}/{source}" if several_repos and repo and source else source

    listing = "\n".join(f"  {where(r) or r}\n    id: {r}" for r in rivals)
    symbol = G.nodes[rivals[0]].get("label") or label
    sources = [str(G.nodes[r].get("source_file") or "") for r in rivals]
    if any(c > 1 for src, c in Counter(sources).items() if src):
        retry = ("Retry with the full node id: the same path exists in more than one repo, "
                 "so a path cannot tell them apart.")
    else:
        retry = (f"Retry with path::symbol using one of the paths above (e.g. "
                 f"<path>::{symbol}) or the full node id.")
    return (
        f"{serve._AMBIGUOUS}'{label}' matches {len(rivals)} nodes in different files.\n"
        f"{listing}\n"
        f"{retry}"
    )


def apply() -> None:
    serve._is_searchable = is_searchable
    serve._ambiguity_message = ambiguity_message


def revert() -> None:
    serve._is_searchable = UPSTREAM
    serve._ambiguity_message = UPSTREAM_AMBIGUITY
