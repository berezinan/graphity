import json
import sys
from pathlib import Path

import pytest
from networkx.readwrite import json_graph

SPIKE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIKE))

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
