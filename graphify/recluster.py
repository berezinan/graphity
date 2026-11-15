"""What a re-clustering actually changed (fork-owned; upstream has no such file).

`cluster-only` on a 2M-node graph used to rewrite 11 GB of graph.json (45 GB peak
inside to_json) even when the partition came out identical. `community_changes`
tells the CLI whether there is anything to write; `sync_community_changes`
republishes the graph store after a write.
"""
from __future__ import annotations


def community_changes(
    raw: dict,
    G,
    communities: "dict[int, list[str]]",
    labels: "dict[int, str] | None",
) -> "dict[str, tuple[int | None, str | None]] | None":
    """``{node_id: (community, community_name)}`` for every node whose community
    or community name differs from what *raw* (the loaded graph.json) records.

    Empty dict = the re-clustering reproduced the stored partition, so rewriting
    graph.json would reproduce the stored file. ``None`` = can't vouch for that:
    the rebuilt graph differs from the file in shape (nodes merged or dropped,
    edges or hyperedges lost at build), so the caller must write as before.
    """
    nodes = raw.get("nodes", [])
    links = raw.get("links", raw.get("edges", []))
    if (G.number_of_nodes() != len(nodes)
            or G.number_of_edges() != len(links)
            or len(G.graph.get("hyperedges", [])) != len(raw.get("hyperedges", []))):
        return None

    node_community = {n: cid for cid, members in communities.items() for n in members}
    names = {int(k): v for k, v in (labels or {}).items()}
    changed: "dict[str, tuple[int | None, str | None]]" = {}
    for node in nodes:
        nid = node.get("id")
        cid = node_community.get(nid)
        # Mirrors to_json: community_name is only (re)written for a clustered
        # node when labels were supplied; otherwise the stored value survives.
        if cid is not None and names:
            name = names.get(cid, f"Community {cid}")
        else:
            name = node.get("community_name")
        if node.get("community") != cid or node.get("community_name") != name:
            changed[nid] = (cid, name)
    return changed


def needs_write(raw: dict, changed: "dict | None", built_at_commit: "str | None") -> bool:
    """Whether to_json would produce a file different from the stored one: some
    node changed (or the diff can't be vouched for), or the commit stamp would."""
    return changed != {} or bool(built_at_commit and raw.get("built_at_commit") != built_at_commit)


def sync_community_changes(graph_json_path: str, raw: dict, changed: "dict | None",
                           communities: "dict[int, list[str]]",
                           labels: "dict[int, str] | None" = None) -> None:
    """After graph.json is written: republish the graph store. No-op for the JSON
    store.

    Called only after graph.json was rewritten (an idle re-clustering never gets
    here), so the snapshot is rebuilt from the file just written. The other
    arguments are kept for the call site in cli.py, which upstream merges must
    not have to touch. Not guarded, like the write in `extract`/`update`: the
    store is required, so a failed publish raises and fails the command.
    """
    from graphify.query_backend import resolve_backend_config
    cfg = resolve_backend_config(graph_json_path)
    if cfg["kind"] == "snap":
        from graphify.snap_store import publish_snapshot_for
        publish_snapshot_for(graph_json_path)
