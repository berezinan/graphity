"""What a re-clustering actually changed (fork-owned; upstream has no such file).

`cluster-only` on a 2M-node graph used to rewrite 11 GB of graph.json (45 GB peak
inside to_json) and then need a 66-minute `arcade reload`, even when the partition
came out identical. `community_changes` tells the CLI whether there is anything
to write, and hands the ArcadeDB backend exactly the nodes to update.
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
                           communities: "dict[int, list[str]]") -> None:
    """After graph.json is written: bring the ArcadeDB copy of the partition up to
    date by updating only the nodes that moved. No-op for the JSON backend.

    Not guarded, like the sync in `extract`/`update`: the graph database is
    required, so a failed or short update raises and fails the command.
    """
    from graphify.query_backend import resolve_backend_config, require_backend
    cfg = resolve_backend_config(graph_json_path)
    if cfg["kind"] != "arcadedb":
        return
    db = require_backend(config=cfg, require_database=False)
    reload_hint = (f"[graphify db] ArcadeDB '{cfg['database']}' not updated - "
                   f"run `graphify arcade reload` to load the new partition.")
    if changed is None or not db.is_populated() or db.has_recorded_shortfall():
        # Shape changed, first load, or a database already known to be short:
        # patching communities is not enough (or not safe) - a reload is.
        print(reload_hint)
        return
    stored = {n.get("id"): n.get("community") for n in raw.get("nodes", [])}
    moved = {nid: cid for nid, (cid, _name) in changed.items() if stored.get(nid) != cid}
    if not moved:
        return  # only community names changed; this backend does not store them
    stats = db.update_communities(moved, {cid: len(m) for cid, m in communities.items()})
    print(f"[graphify db] updated ArcadeDB '{cfg['database']}' ({stats['updated_nodes']} nodes moved).")
    if not stats["reconciled"]:
        raise RuntimeError("ArcadeDB community update did not reconcile: per-community node "
                           f"counts differ from graph.json.\n{reload_hint}")
