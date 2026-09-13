"""Incremental sync removes sources that left the graph (prune-stale-sources).

Until this landed, deletion followed only the run's own delta
(``changed | pruned``), so a source that vanished from the graph any other way
stayed in the database forever. Directories were the everyday case: the graph
stops minting nodes for them, and they never reach ``deleted_files`` because a
directory is not a file. Measured on a real database: 103 such sources / 128
nodes, a figure that does not shrink on its own.

These run against a live ArcadeDB on a database of their own, and skip when no
server answers.
"""

import json
import os

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from graphify.query_backend import ArcadeDBBackend

_MACHINE = {}
try:  # same credential discovery the parity tests use
    from tests.test_arcade_backend import _MACHINE as _M, _PW, _URL, _parse_host_port
    _MACHINE = _M
except Exception:  # pragma: no cover - import shape differs when run as a file
    from test_arcade_backend import _PW, _URL, _parse_host_port  # type: ignore

_DB = "graphify_prune_stale_test"


def _graph(sources=("a.py", "b.py"), fileless=0) -> nx.DiGraph:
    """Small graph: one node per source, plus optional source-less nodes."""
    G = nx.DiGraph()
    for i, sf in enumerate(sources):
        G.add_node(f"n{i}", label=f"sym{i}", source_file=sf, source_location="L1", community=0)
    for j in range(fileless):
        G.add_node(f"f{j}", label=f"free{j}", community=0)
    if len(sources) >= 2:
        G.add_edge("n0", "n1", relation="calls", confidence="EXTRACTED")
    return G


@pytest.fixture()
def arc(tmp_path_factory):
    host, port = _parse_host_port(_URL)
    backend = ArcadeDBBackend(_DB, host=host, port=port, password=_PW)
    if not backend.ready():
        pytest.skip(f"ArcadeDB not reachable at {_URL}")
    yield backend
    try:
        backend._server(f"drop database {_DB}")
    except Exception:
        pass


def _load(backend, G, tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    backend.ensure_database(drop=True)
    backend.load_from_graph_json(str(p))


def _db_sources(backend) -> set:
    return backend._scan_nodes()[0]


# --- 4.1: the source left the graph -----------------------------------------

def test_source_absent_from_graph_is_pruned(arc, tmp_path):
    """The case the change exists for: no file was deleted this run, yet a
    source is gone from the graph. The old delta-only path never removed it."""
    _load(arc, _graph(("a.py", "b.py", "LOGISTICS")), tmp_path)
    assert "LOGISTICS" in _db_sources(arc)

    stats = arc.sync_graph(_graph(("a.py", "b.py")), changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 1
    assert "LOGISTICS" not in _db_sources(arc)
    assert {"a.py", "b.py"} <= _db_sources(arc)


def test_directory_that_exists_on_disk_is_still_pruned(arc, tmp_path):
    """Staleness is read from the graph, never from the disk.

    30 of the 103 measured stale entries DO exist on disk — they are directories
    the graph stopped minting nodes for. A disk check would keep them forever and
    solve less than a third of the problem.
    """
    real_dir = tmp_path / "EGAIS"
    real_dir.mkdir()
    _load(arc, _graph(("a.py", str(real_dir))), tmp_path)

    stats = arc.sync_graph(_graph(("a.py",)), changed_sources=set(), pruned_sources=set())

    assert real_dir.exists(), "the test's premise: the path is still on disk"
    assert stats["pruned_stale_sources"] == 1
    assert str(real_dir) not in _db_sources(arc)


def test_source_present_in_graph_survives_an_unreadable_file(arc, tmp_path):
    """The inverse error, and the expensive one: a file that cannot be read right
    now (unmounted share, wrong case) is still in the graph, so its nodes stay."""
    _load(arc, _graph(("a.py", "//unreachable-share/x.py")), tmp_path)

    G = _graph(("a.py", "//unreachable-share/x.py"))
    stats = arc.sync_graph(G, changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 0
    assert "//unreachable-share/x.py" in _db_sources(arc)


# --- 4.2: a short read must not read as "these sources are gone" -------------

def test_incomplete_source_read_blocks_the_prune(arc, tmp_path, monkeypatch):
    """A truncated page looks exactly like "the graph no longer has these"."""
    _load(arc, _graph(("a.py", "b.py", "GONE")), tmp_path)
    monkeypatch.setattr(type(arc), "_scan_nodes",
                        lambda self, **kw: ({"a.py", "b.py", "GONE"}, {"n0", "n1"}, False))

    stats = arc.sync_graph(_graph(("a.py", "b.py")), changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 0
    assert "incomplete" in stats.get("prune_blocked", "")
    assert "GONE" in _db_sources(arc), "nothing may be deleted on a short read"


# --- 4.3: the guard ----------------------------------------------------------

def test_empty_graph_prunes_nothing(arc, tmp_path):
    """An empty graph marks EVERY source stale; the sync would wipe the database
    and report success."""
    _load(arc, _graph(("a.py", "b.py")), tmp_path)

    stats = arc.sync_graph(nx.DiGraph(), changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 0
    assert "no nodes" in stats.get("prune_blocked", "")
    assert _db_sources(arc) == {"a.py", "b.py"}


def test_share_above_the_ceiling_prunes_nothing(arc, tmp_path):
    """A graph that lost most of its sources is not a cleanup, it is a ruin.

    Shaped after the one this project suffered: `graphify update` after an
    ignore-rule change rebuilt the graph as 431 nodes from 78 123.
    """
    _load(arc, _graph(tuple(f"s{i}.py" for i in range(10))), tmp_path)

    stats = arc.sync_graph(_graph(("s0.py", "s1.py")),
                           changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 0
    blocked = stats.get("prune_blocked", "")
    assert "80.0%" in blocked and "ceiling" in blocked, blocked
    assert len(_db_sources(arc)) == 10, "the ruin must leave the database alone"


def test_share_below_the_ceiling_prunes(arc, tmp_path):
    """The ordinary accumulation the change is for stays well under the ceiling:
    0.90% and 5.13% on the two measured databases."""
    _load(arc, _graph(tuple(f"s{i}.py" for i in range(10))), tmp_path)

    stats = arc.sync_graph(_graph(tuple(f"s{i}.py" for i in range(8))),
                           changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_sources"] == 2
    assert not stats.get("prune_blocked")
    assert len(_db_sources(arc)) == 8


def test_fileless_nodes_are_guarded_too(arc, tmp_path):
    """The guard covers both halves of the reconciliation.

    Source-less nodes were 20.6% of one measured database (15 324 of 74 457).
    Guarding only the sourced half would make an empty-graph wipe smaller, not
    prevent it.
    """
    _load(arc, _graph(("a.py",), fileless=4), tmp_path)
    before = int(arc._run(
        "SELECT count() AS c FROM Node WHERE source_file IS NULL OR source_file = ''")[0]["c"])
    assert before == 4

    arc.sync_graph(nx.DiGraph(), changed_sources=set(), pruned_sources=set())

    after = int(arc._run(
        "SELECT count() AS c FROM Node WHERE source_file IS NULL OR source_file = ''")[0]["c"])
    assert after == 4, "an empty graph must not delete source-less nodes either"


# --- 4.4: keep the implementation off the paged aggregate --------------------

def test_source_list_is_not_built_from_a_paged_group_by():
    """Measured on a live database: a paged ``GROUP BY`` returned 11 498 sources
    against 11 499 from one un-paged query, losing a source that exists —
    ``SKIP``/``LIMIT`` apply to an aggregate whose order is not guaranteed
    between requests. The loss is not destructive but it IS undetectable, so the
    defect would come back silently. Pin the implementation, not just the result.
    """
    import inspect

    src = inspect.getsource(ArcadeDBBackend._scan_nodes)
    body = src.split('"""')[-1]  # ignore the docstring, which explains the ban
    lowered = body.lower()
    assert "group by" not in lowered, "the source list must not come from an aggregate"
    assert "skip" in lowered and "limit" in lowered, "it must still be paged"


def test_both_spellings_of_fileless_are_excluded_from_sources(arc, tmp_path):
    """198 nodes carry no ``source_file`` at all and 15 126 carry an empty one on
    the measured corpus; neither is a source, and a predicate covering one of
    them finds half the set."""
    _load(arc, _graph(("a.py",), fileless=3), tmp_path)

    sources, _ids, complete = arc._scan_nodes()

    assert complete
    assert sources == {"a.py"}
    assert not any(s in ("", "None", None) for s in sources)


# --- the third category, found by verifying this change on a real database ---

def test_node_gone_from_a_live_source_is_pruned(arc, tmp_path):
    """A node the graph no longer holds, whose SOURCE is still alive.

    Neither reconciliation saw it: not a stale source (the file is still in the
    graph), not source-less. It survives because the source was not dirty in the
    run that dropped it. Measured on a real database after the source prune: 18
    such nodes remained out of 144 surplus.
    """
    G_before = _graph(("a.py",))
    G_before.add_node("gone", label="dropped", source_file="a.py",
                      source_location="L9", community=0)
    _load(arc, G_before, tmp_path)
    assert int(arc._run("SELECT count() AS c FROM Node")[0]["c"]) == 2

    stats = arc.sync_graph(_graph(("a.py",)), changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_nodes"] == 1
    assert stats["pruned_stale_sources"] == 0, "the source itself is not stale"
    assert "a.py" in _db_sources(arc), "the live source must stay"
    assert int(arc._run("SELECT count() AS c FROM Node")[0]["c"]) == 1


def test_stale_node_prune_is_guarded_too(arc, tmp_path):
    """The guard covers this half as well — an empty graph makes every node look
    dropped, which is the same wipe by another route."""
    G_before = _graph(("a.py", "b.py"))
    _load(arc, G_before, tmp_path)

    stats = arc.sync_graph(nx.DiGraph(), changed_sources=set(), pruned_sources=set())

    assert stats["pruned_stale_nodes"] == 0
    assert "no nodes" in stats.get("prune_blocked", "")
    assert int(arc._run("SELECT count() AS c FROM Node")[0]["c"]) == 2


def test_reconcile_reports_a_surplus(arc, tmp_path):
    """`_reconcile` clamped the surplus to zero, which is why 144 extra nodes
    passed as a clean bill of health on every run."""
    G_before = _graph(("a.py",))
    G_before.add_node("extra", label="extra", source_file="a.py", community=0)
    _load(arc, G_before, tmp_path)

    smaller = _graph(("a.py",))
    stats = arc._reconcile(smaller.number_of_nodes(), smaller.number_of_edges(),
                           node_ids=list(smaller.nodes()), edges=[])

    assert stats["extra_nodes"] == 1
    assert stats["missing_nodes"] == 0, "a surplus is not a shortfall"
