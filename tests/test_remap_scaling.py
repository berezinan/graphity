"""remap_communities_to_previous must scale with the node count, not with
(old communities x new communities), and keep the pairwise result exactly.

On a 2M-node graph with 55 665 communities the pairwise set-intersection loop
took 37 minutes. The reference below is a verbatim copy of that implementation.
"""
from __future__ import annotations

import random
import time

from graphify.cluster import remap_communities_to_previous


def _ref_remap(communities, previous_node_community):
    if not communities:
        return {}
    new_sets = {cid: set(nodes) for cid, nodes in communities.items()}
    old_sets: dict[int, set[str]] = {}
    for node, old_cid in previous_node_community.items():
        old_sets.setdefault(old_cid, set()).add(node)
    overlaps = []
    for old_cid, old_nodes in old_sets.items():
        for new_cid, new_nodes in new_sets.items():
            overlap = len(old_nodes & new_nodes)
            if overlap > 0:
                overlaps.append((overlap, old_cid, new_cid))
    overlaps.sort(key=lambda x: (-x[0], x[1], x[2]))
    new_to_final: dict[int, int] = {}
    used_old_ids: set[int] = set()
    matched_new_ids: set[int] = set()
    for _overlap, old_cid, new_cid in overlaps:
        if old_cid in used_old_ids or new_cid in matched_new_ids:
            continue
        new_to_final[new_cid] = old_cid
        used_old_ids.add(old_cid)
        matched_new_ids.add(new_cid)
    unmatched = [cid for cid in communities if cid not in matched_new_ids]
    unmatched.sort(key=lambda cid: (-len(communities[cid]), tuple(sorted(communities[cid]))))
    next_id = 0
    for new_cid in unmatched:
        while next_id in used_old_ids:
            next_id += 1
        new_to_final[new_cid] = next_id
        used_old_ids.add(next_id)
        next_id += 1
    remapped = {}
    for new_cid, nodes in communities.items():
        remapped[new_to_final[new_cid]] = sorted(nodes)
    return dict(sorted(remapped.items(), key=lambda kv: kv[0]))


def _same(a, b):
    # Equal as mappings AND in key order / member order.
    return a == b and list(a) == list(b)


def _random_case(rng):
    nodes = [f"n{i}" for i in range(rng.randint(1, 120))]
    n_new = rng.randint(1, 12)
    communities: dict[int, list[str]] = {}
    for n in nodes:
        if rng.random() < 0.9:                      # some nodes only in `previous`
            communities.setdefault(rng.randrange(n_new) * 3, []).append(n)
    previous = {
        n: rng.randrange(rng.randint(1, 12)) * 2
        for n in nodes + [f"gone{i}" for i in range(5)]
        if rng.random() < 0.8                       # some nodes only in `communities`
    }
    return communities, previous


def test_matches_pairwise_reference_on_generated_partitions():
    rng = random.Random(822)
    for _ in range(300):
        communities, previous = _random_case(rng)
        if not communities:
            continue
        assert _same(remap_communities_to_previous(communities, previous),
                     _ref_remap(communities, previous))


def test_matches_reference_on_edges():
    assert remap_communities_to_previous({}, {"a": 1}) == {}
    comms = {0: ["b", "a"], 1: ["c"]}
    assert _same(remap_communities_to_previous(comms, {}), _ref_remap(comms, {}))
    # previous ids that collide with the fresh ids handed to unmatched communities
    prev = {"a": 0, "c": 0, "zzz": 1}
    assert _same(remap_communities_to_previous(comms, prev), _ref_remap(comms, prev))


def test_many_communities_is_not_quadratic():
    # 20 000 x 20 000 communities: 4e8 set intersections pairwise (minutes).
    n = 20_000
    communities = {cid: [f"n{cid}_{j}" for j in range(3)] for cid in range(n)}
    previous = {node: (cid + 1) % n for cid, nodes in communities.items() for node in nodes}
    t = time.perf_counter()
    out = remap_communities_to_previous(communities, previous)
    elapsed = time.perf_counter() - t
    assert len(out) == n
    assert out[1] == sorted(communities[0])
    assert elapsed < 20, f"remap took {elapsed:.1f}s - quadratic again?"
