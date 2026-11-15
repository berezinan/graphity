"""Source deletion in ``sync_graph`` is batched (batch-dirty-source-delete-in-sync-graph).

The dirty-source DELETE used to carry the whole ``changed | pruned`` list in one
query. When a big directory vanished (HomeAssistant ``core/``: ~27.7k sources,
388 024 nodes) that single transaction outran the client's 300 s read timeout,
rolled back, and ``extract`` died before writing manifest.json.

These run against a live ArcadeDB on a database of their own, and skip when no
server answers. ``_run`` is wrapped, not replaced: the deletes still execute.
"""

import json

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from graphify.query_backend import ArcadeDBBackend

try:  # same credential discovery the other arcade tests use
    from tests.test_arcade_backend import _PW, _URL, _parse_host_port
except Exception:  # pragma: no cover - import shape differs when run as a file
    from test_arcade_backend import _PW, _URL, _parse_host_port  # type: ignore

_DB = "graphify_batch_dirty_delete_test"
_BATCH = 200
_DELETE_BY_SOURCE = "DELETE VERTEX FROM Node WHERE source_file IN"


def _graph(sources) -> nx.DiGraph:
    """One node per source; n0 -> n1 so the surviving pair keeps an edge."""
    G = nx.DiGraph()
    for i, sf in enumerate(sources):
        G.add_node(f"n{i}", label=f"sym{i}", source_file=sf, source_location="L1", community=0)
    if len(sources) >= 2:
        G.add_edge("n0", "n1", relation="calls", confidence="EXTRACTED")
    return G


@pytest.fixture()
def arc():
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


def _spy_source_deletes(backend, monkeypatch) -> list:
    """Record the source list of every DELETE-by-source, then run it for real."""
    calls: list = []
    real = backend._run

    def spy(command, *args, **kwargs):
        if command.startswith(_DELETE_BY_SOURCE):
            params = kwargs.get("params") or {}
            calls.append(list(next(iter(params.values()))))
        return real(command, *args, **kwargs)

    monkeypatch.setattr(backend, "_run", spy)
    return calls


def _dirty_scenario(arc, tmp_path, extras: int):
    """DB holds a.py, b.py and ``extras`` more sources; the new graph keeps only
    the first two. 99% of the sources leave, so the stale-prune ceiling blocks
    that route and only the dirty-source DELETE can remove them."""
    gone = [f"x{i:04d}.py" for i in range(extras)]
    _load(arc, _graph(["a.py", "b.py", *gone]), tmp_path)
    return gone, _graph(["a.py", "b.py"])


# --- 1.2: more dirty sources than one batch ----------------------------------

def test_large_dirty_set_is_deleted_in_batches(arc, tmp_path, monkeypatch):
    gone, G = _dirty_scenario(arc, tmp_path, 450)
    calls = _spy_source_deletes(arc, monkeypatch)

    arc.sync_graph(G, changed_sources=set(), pruned_sources=set(gone))

    assert len(calls) > 1, "450 dirty sources must not travel in one query"
    assert all(len(c) <= _BATCH for c in calls), [len(c) for c in calls]
    flat = [s for c in calls for s in c]
    assert sorted(flat) == sorted(gone), "every dirty source exactly once, none extra"
    assert arc._scan_nodes()[0] == {"a.py", "b.py"}


# --- 1.3: a small set stays one query ----------------------------------------

def test_small_dirty_set_is_one_query(arc, tmp_path, monkeypatch):
    gone, G = _dirty_scenario(arc, tmp_path, 50)
    calls = _spy_source_deletes(arc, monkeypatch)

    arc.sync_graph(G, changed_sources=set(), pruned_sources=set(gone))

    assert len(calls) == 1
    assert sorted(calls[0]) == sorted(gone)


# --- 1.4: stale sources were already batched; pin it --------------------------

def test_large_stale_set_is_deleted_in_batches(arc, tmp_path, monkeypatch):
    """The stale-source route already used ``_delete_by_sources``. This pins it so
    nobody "fixes" the dirty route by loosening the shared helper."""
    kept = [f"k{i:04d}.py" for i in range(5000)]
    gone = [f"x{i:04d}.py" for i in range(250)]  # ~4.8%, under the prune ceiling
    _load(arc, _graph([*kept, *gone]), tmp_path)
    calls = _spy_source_deletes(arc, monkeypatch)

    stats = arc.sync_graph(_graph(kept), changed_sources=set(), pruned_sources=set())

    assert not stats.get("prune_blocked"), stats
    assert stats["pruned_stale_sources"] == 250
    assert len(calls) > 1
    assert all(len(c) <= _BATCH for c in calls), [len(c) for c in calls]
    assert sorted(s for c in calls for s in c) == sorted(gone)


# --- 1.5: deterministic composition -------------------------------------------

def test_batch_composition_is_deterministic(arc, tmp_path, monkeypatch):
    """``dirty`` is a set; the batches must not inherit its iteration order."""
    gone, G = _dirty_scenario(arc, tmp_path, 450)
    calls = _spy_source_deletes(arc, monkeypatch)

    arc.sync_graph(G, changed_sources=set(), pruned_sources=set(gone))

    assert [s for c in calls for s in c] == sorted(gone)
