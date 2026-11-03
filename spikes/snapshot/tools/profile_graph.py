"""Attribute census of a graph.json: which keys, which value types, flags.

Usage: python profile_graph.py <graph.json>
"""
import collections
import json
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("top-level keys:", sorted(data))
print("directed:", data.get("directed"), "multigraph:", data.get("multigraph"))
links = data.get("links") or data.get("edges") or []
nodes = data["nodes"]
print("nodes:", len(nodes), "links:", len(links))


def census(items, label):
    keys = collections.Counter()
    types = collections.defaultdict(collections.Counter)
    for it in items:
        for k, v in it.items():
            keys[k] += 1
            types[k][type(v).__name__] += 1
    print(f"\n{label} keys (count, types):")
    for k, c in keys.most_common():
        print(f"  {k:28s} {c:9d}  {dict(types[k])}")


census(nodes, "node")
census(links, "link")

loc = re.compile(r"L[1-9]\d*")
bad_node_loc = sum(1 for n in nodes if "source_location" in n and not (isinstance(n["source_location"], str) and loc.fullmatch(n["source_location"])))
bad_edge_loc = sum(1 for e in links if "source_location" in e and not (isinstance(e["source_location"], str) and loc.fullmatch(e["source_location"])))
print("\nsource_location not L<n>: nodes", bad_node_loc, "edges", bad_edge_loc)
pairs = collections.Counter()
for e in links:
    pairs[(e["source"], e["target"])] += 1
print("parallel same-direction links:", sum(c - 1 for c in pairs.values() if c > 1))
mutual = sum(1 for (s, t) in pairs if s != t and (t, s) in pairs) // 2
print("mutual arc pairs:", mutual, "self-loops:", sum(1 for (s, t) in pairs if s == t))
for k in ("relation", "confidence", "context", "file_type"):
    src = links if k != "file_type" else nodes
    vals = collections.Counter(repr(x.get(k, "<missing>")) for x in src)
    print(f"{k}: {len(vals)} distinct", dict(vals.most_common(15)))
for k in ("confidence_score", "weight"):
    vals = collections.Counter(repr(x.get(k, "<missing>")) for x in links)
    print(f"{k}: {len(vals)} distinct", dict(vals.most_common(8)))
upper_ids = sum(1 for n in nodes if n["id"] != n["id"].lower())
print("ids with uppercase:", upper_ids)
print("sample node:", json.dumps(nodes[0], ensure_ascii=False)[:600])
print("sample link:", json.dumps(links[0], ensure_ascii=False)[:600])
