import json
import sys
from pathlib import Path

import pytest
from networkx.readwrite import json_graph

SPIKE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIKE))

import spike_patches  # noqa: E402

# Same tokenisation for JsonBackend and SnapBackend in every test (2026-11-04).
spike_patches.apply()

PSN = Path(r"C:\_dev\ANB\AI\psn_diadoc_4_67_3_Pravka\graphify-out\graph.json")


def snap_of(data_or_graph, tmp_path, name="graph.snap"):
    """Publish a snapshot of a graph.json dict (or an nx graph) and open it."""
    from snap_backend import SnapBackend
    from snap_writer import publish_snapshot

    data = data_or_graph
    if not isinstance(data, dict):
        data = json_graph.node_link_data(data, edges="links")
    publish_snapshot(data, tmp_path, name=name)
    return SnapBackend(tmp_path / name)


@pytest.fixture(scope="session")
def psn_data():
    if not PSN.exists():
        pytest.skip(f"{PSN} not present")
    with open(PSN, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def psn_snap(psn_data, tmp_path_factory):
    d = tmp_path_factory.mktemp("psn")
    return snap_of(psn_data, d)


@pytest.fixture(scope="session")
def graphs_psn(psn_data):
    import ref_loaders
    return ref_loaders.query_graph(psn_data)


@pytest.fixture(scope="session")
def qset_psn(psn_data):
    import query_set
    import ref_loaders
    from graphify.query_backend import JsonBackend
    gods = [g.label for g in JsonBackend(ref_loaders.mcp_graph(psn_data)).god_nodes(top_n=10)]
    return query_set.build(query_set.labels_of(psn_data), PSN, gods)
