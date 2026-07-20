"""CLI query/path/explain must run to the END of the command on the ArcadeDB
branch, not just the JSON branch.

Regression guard for the c147cde class of breakage: upstream 0.9.20 appended
`_touch_query_stamp(gp)` to the shared command tail, but `gp` is bound only in
the JSON branch — every arcadedb-backed query/path/explain died with
UnboundLocalError AFTER printing results, and no test noticed. The backend is
stubbed with an in-memory JsonBackend so the whole command body (including the
tail after the branch merge point) executes without a server."""
import sys

import networkx as nx
import pytest

from graphify import cli
from graphify import query_backend as qb


_ARCADE_CFG = {"kind": "arcadedb", "database": "proj_t",
               "host": "127.0.0.1", "port": 2480, "user": "root", "password": "pw"}


@pytest.fixture
def arcadedb_cli(monkeypatch, tmp_path):
    """Route resolve/open through a fake arcadedb backend; cwd -> tmp."""
    G = nx.DiGraph()
    G.add_node("alpha_node", label="Alpha()", source_file="a.py", file_type="code")
    G.add_node("beta_node", label="Beta()", source_file="b.py", file_type="code")
    G.add_edge("alpha_node", "beta_node", relation="calls", confidence="EXTRACTED")
    backend = qb.JsonBackend(G)
    monkeypatch.setattr(qb, "resolve_backend_config", lambda gp=None: dict(_ARCADE_CFG))
    monkeypatch.setattr(qb, "open_backend", lambda **kw: backend)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["graphify", *argv])
        cli.dispatch_command(argv[0])

    return run, tmp_path


def _stamp(tmp_path):
    return tmp_path / "graphify-out" / "cache" / "last_query_stamp"


def test_query_arcadedb_branch_runs_to_completion(arcadedb_cli, capsys):
    run, tmp_path = arcadedb_cli
    run("query", "Alpha")
    assert "Alpha()" in capsys.readouterr().out
    # The stamp is written by the command TAIL — its existence proves the code
    # after the JSON/arcadedb branch merge point survived (c147cde regression).
    assert _stamp(tmp_path).exists()


def test_path_arcadedb_branch_runs_to_completion(arcadedb_cli, capsys):
    run, tmp_path = arcadedb_cli
    run("path", "Alpha", "Beta")
    out = capsys.readouterr().out
    assert "Shortest path" in out
    assert _stamp(tmp_path).exists()


def test_explain_arcadedb_branch_runs_to_completion(arcadedb_cli, capsys):
    run, tmp_path = arcadedb_cli
    run("explain", "Alpha")
    assert "Alpha()" in capsys.readouterr().out
    assert _stamp(tmp_path).exists()
