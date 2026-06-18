"""Minimal ArcadeDB query backend for the Graphify scaling spike.

This is a proof-of-concept, NOT production code. It exists to measure whether
moving the query path off the in-memory NetworkX/JSON model and into a server
graph DB is worth it for large / global graphs.

Design mirrors the proposed `GraphBackend` split:
  - load_from_graph_json(): one-time sync of graph.json -> ArcadeDB (a
    materialized view; graph.json stays the build artifact).
  - query() / shortest_path(): push candidate resolution + traversal DOWN into
    the DB so the Python client never holds the whole graph in RAM.

Talks to ArcadeDB over its HTTP API (requests) - no native driver needed, which
is itself a finding: the Python story is HTTP-first, not a mature bolt-style
driver like Neo4j's.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests


class ArcadeDBBackend:
    def __init__(
        self,
        database: str = "graphify_spike",
        host: str = "127.0.0.1",
        port: int = 2480,
        user: str = "root",
        password: str = "playwithdata",
    ) -> None:
        self.base = f"http://{host}:{port}"
        self.db = database
        self.auth = (user, password)
        self.s = requests.Session()
        self.s.auth = self.auth

    # ---- low-level HTTP helpers -------------------------------------------------
    def _server(self, command: str) -> dict:
        r = self.s.post(f"{self.base}/api/v1/server", json={"command": command}, timeout=60)
        r.raise_for_status()
        return r.json() if r.text else {}

    def _command(self, command: str, *, language: str = "sql", params: dict | None = None) -> list[dict]:
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = self.s.post(f"{self.base}/api/v1/command/{self.db}", json=body, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"ArcadeDB {r.status_code}: {r.text[:500]}\n  cmd: {command[:200]}")
        return r.json().get("result", [])

    def _query(self, command: str, *, language: str = "sql", params: dict | None = None) -> list[dict]:
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = self.s.post(f"{self.base}/api/v1/query/{self.db}", json=body, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"ArcadeDB {r.status_code}: {r.text[:500]}\n  q: {command[:200]}")
        return r.json().get("result", [])

    # ---- lifecycle --------------------------------------------------------------
    def ready(self) -> bool:
        try:
            r = self.s.get(f"{self.base}/api/v1/ready", timeout=5)
            return r.status_code == 204 or r.ok
        except requests.RequestException:
            return False

    def ensure_database(self, *, drop: bool = False) -> None:
        existing = self._server("list databases").get("result", [])
        names = existing if isinstance(existing, list) else []
        if self.db in names and drop:
            self._server(f"drop database {self.db}")
            names = []
        if self.db not in names:
            self._server(f"create database {self.db}")

    # ---- load (graph.json -> ArcadeDB) -----------------------------------------
    def load_from_graph_json(self, path: str, *, batch: int = 2000) -> dict[str, Any]:
        data = json.loads(open(path, encoding="utf-8").read())
        ek = "links" if "links" in data else "edges"
        nodes, edges = data["nodes"], data[ek]

        # Schema: single Node vertex type + single Rel edge type (relation kept as
        # a property so variable-length traversal stays one cheap edge type).
        # ArcadeDB SQL DDL has no IF NOT EXISTS on PROPERTY/INDEX, so each step is
        # best-effort (a fresh --drop database makes the "already exists" path moot).
        def _ddl(cmd: str) -> None:
            try:
                self._command(cmd)
            except RuntimeError as exc:
                if "already exist" not in str(exc).lower():
                    raise

        _ddl("CREATE VERTEX TYPE Node")
        _ddl("CREATE EDGE TYPE Rel")
        _ddl("CREATE PROPERTY Node.id STRING")
        _ddl("CREATE PROPERTY Node.norm_label STRING")
        _ddl("CREATE INDEX ON Node (id) UNIQUE")
        # Full-text index over norm_label so candidate resolution is an index
        # lookup, not a full scan (the whole point of the spike).
        _ddl("CREATE INDEX ON Node (norm_label) FULL_TEXT")

        t0 = time.perf_counter()
        keep_n = ("id", "label", "source_file", "source_location", "file_type", "community")
        n_pushed = 0
        for i in range(0, len(nodes), batch):
            stmts = []
            for n in nodes[i:i + batch]:
                doc = {k: n[k] for k in keep_n if k in n and isinstance(n[k], (str, int, float, bool))}
                # Derive norm_label (the full-text key) so the file can stay lean;
                # prefer an existing one when the real graph already carries it.
                doc["norm_label"] = (n.get("norm_label") or str(n.get("label", ""))).lower()
                stmts.append("INSERT INTO Node CONTENT " + json.dumps(doc, ensure_ascii=False))
            self._command(";".join(stmts), language="sqlscript")
            n_pushed += len(nodes[i:i + batch])

        keep_e = ("relation", "confidence", "weight", "context")
        e_pushed = 0
        for i in range(0, len(edges), batch):
            stmts = []
            for e in edges[i:i + batch]:
                doc = {k: e[k] for k in keep_e if k in e and isinstance(e[k], (str, int, float, bool))}
                src = _sql_str(e["source"])
                tgt = _sql_str(e["target"])
                stmts.append(
                    f"CREATE EDGE Rel FROM (SELECT FROM Node WHERE id={src}) "
                    f"TO (SELECT FROM Node WHERE id={tgt}) CONTENT {json.dumps(doc, ensure_ascii=False)}"
                )
            self._command(";".join(stmts), language="sqlscript")
            e_pushed += len(edges[i:i + batch])

        return {
            "nodes": n_pushed,
            "edges": e_pushed,
            "load_seconds": round(time.perf_counter() - t0, 2),
        }

    # ---- query path -------------------------------------------------------------
    def resolve_seeds(self, terms: list[str], *, max_k: int = 3) -> list[dict]:
        """Candidate resolution pushed to the DB: one indexed lookup per term,
        union, then cheap tier scoring on the small candidate set (not all N)."""
        cand: dict[str, dict] = {}
        for t in terms:
            rows = self._query(
                "SELECT id, label, norm_label, source_file, community FROM Node "
                "WHERE norm_label LIKE :p LIMIT 200",
                params={"p": f"%{t.lower()}%"},
            )
            for row in rows:
                cand[row["id"]] = row
        scored = []
        joined = " ".join(terms).lower()
        for row in cand.values():
            nl = (row.get("norm_label") or "").lower()
            score = 0.0
            if nl == joined:
                score += 1000
            elif nl.startswith(joined):
                score += 100
            for t in terms:
                tl = t.lower()
                if nl == tl:
                    score += 1000
                elif nl.startswith(tl):
                    score += 100
                elif tl in nl:
                    score += 1
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: (-x[0], len(x[1].get("label") or "")))
        return [r for _, r in scored[:max_k]]

    def query(self, terms: list[str], *, depth: int = 2) -> dict[str, Any]:
        seeds = self.resolve_seeds(terms)
        if not seeds:
            return {"seeds": [], "nodes": [], "node_count": 0}
        seed_ids = [s["id"] for s in seeds]
        rows = self._query(
            "SELECT id, label, source_file, community FROM "
            f"(TRAVERSE both('Rel') FROM (SELECT FROM Node WHERE id IN :seeds) MAXDEPTH {depth}) "
            "WHERE @type = 'Node'",
            params={"seeds": seed_ids},
        )
        return {
            "seeds": [s["label"] for s in seeds],
            "node_count": len(rows),
            "nodes": rows[:50],
        }

    def shortest_path(self, src_terms: list[str], tgt_terms: list[str], *, max_hops: int = 8) -> dict[str, Any]:
        src = self.resolve_seeds(src_terms, max_k=1)
        tgt = self.resolve_seeds(tgt_terms, max_k=1)
        if not src or not tgt:
            return {"found": False, "reason": "endpoint not resolved"}
        s_id, t_id = src[0]["id"], tgt[0]["id"]
        if s_id == t_id:
            return {"found": False, "reason": "same node"}
        # openCypher shortestPath - this also exercises whether the Cypher that
        # push_to_neo4j already emits can be reused for reads on ArcadeDB.
        rows = self._query(
            f"MATCH (a:Node {{id:$s}}), (b:Node {{id:$t}}), "
            f"p = shortestPath((a)-[:Rel*..{max_hops}]-(b)) RETURN length(p) AS hops",
            language="cypher",
            params={"s": s_id, "t": t_id},
        )
        if not rows or rows[0].get("hops") is None:
            return {"found": False, "reason": "no path", "src": src[0]["label"], "tgt": tgt[0]["label"]}
        return {"found": True, "hops": rows[0]["hops"], "src": src[0]["label"], "tgt": tgt[0]["label"]}


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"
