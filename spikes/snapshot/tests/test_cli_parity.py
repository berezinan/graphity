"""Task 4.4, CLI level: the cases of tests/test_path_explain_cli.py (plus
`query`) run through the real `graphify` CLI twice, once as is and once with
the CLI's `JsonBackend` swapped for `SnapBackend` over a snapshot of the same
--graph file. stdout, stderr and the exit code must be byte-equal."""
import json

import networkx as nx
import pytest
from networkx.readwrite import json_graph

import graphify.__main__ as mainmod
import graphify.query_backend as qb
from snap_backend import SnapBackend
from snap_writer import publish_snapshot


def _graph(tmp_path, island=False):
    # Copy of tests/test_path_explain_cli.py::_write_graph (+ its island variant).
    G = nx.DiGraph()
    G.add_node("n1", label="extract", source_file="extract.py", source_location="L10", community=0)
    G.add_node("n2", label="cluster", source_file="cluster.py", source_location="L5", community=0)
    G.add_node("n3", label="build", source_file="build.py", source_location="L1", community=1)
    G.add_edge("n1", "n2", relation="calls", confidence="EXTRACTED", context="call")
    G.add_edge("n2", "n3", relation="imports", confidence="EXTRACTED", context="import")
    data = json_graph.node_link_data(G, edges="links")
    if island:
        data["nodes"].append({"id": "n9", "label": "island", "source_file": "x.py", "community": 5})
    gp = tmp_path / "graph.json"
    gp.write_text(json.dumps(data))
    return gp


OPENED: list = []


def _run(monkeypatch, capsys, argv, snap_path=None):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", argv)
    if snap_path is not None:
        def _factory(G, *a, **k):
            OPENED.append(snap_path)
            return SnapBackend(snap_path)
        monkeypatch.setattr(qb, "JsonBackend", _factory)
    code = 0
    try:
        mainmod.main()
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    monkeypatch.undo()
    return out.out, out.err, code


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")


CASES = [
    (False, ["path", "extract", "build"]),
    (False, ["path", "zzznope", "build"]),
    (False, ["path", "build", "extract"]),
    (False, ["path", "build", "extract", "--undirected"]),
    (True, ["path", "extract", "island"]),
    (False, ["explain", "cluster"]),
    (False, ["explain", "zzznope"]),
    (False, ["explain", "clust"]),
    (False, ["query", "extract"]),
    (False, ["query", "who calls cluster"]),
    (False, ["query", "build", "--dfs"]),
    (False, ["query", "zzznope"]),
]


@pytest.mark.parametrize("island,args", CASES, ids=[" ".join(a) for _, a in CASES])
def test_cli_parity(monkeypatch, capsys, tmp_path, island, args):
    gp = _graph(tmp_path, island)
    with open(gp, encoding="utf-8") as f:
        publish_snapshot(json.load(f), tmp_path)
    argv = ["graphify", *args, "--graph", str(gp)]
    want = _run(monkeypatch, capsys, argv)
    # The parity harness re-applies the env; monkeypatch.undo() above cleared it.
    monkeypatch.setenv("GRAPHIFY_GRAPH_STORE", "json")
    OPENED.clear()
    got = _run(monkeypatch, capsys, argv, snap_path=tmp_path / "graph.snap")
    assert OPENED, "the CLI did not go through SnapBackend"
    assert got == want
