from __future__ import annotations

from typing import Any

import pytest

# Test files that are ABOUT the ArcadeDB backend and therefore choose their own
# GRAPHIFY_BACKEND. Everything else runs against the JSON backend.
_ARCADE_TEST_FILES = frozenset({
    "test_arcade_backend.py",
    "test_arcade_server.py",
    "test_arcade_service.py",
    "test_cli_arcadedb_branch.py",
})

_ANALYZE_WARNING_FILTERS = (
    "ignore:Tensorflow not installed; ParametricUMAP will be unavailable:ImportWarning:umap",
    "ignore:Please import `random` from the `scipy\\.sparse` namespace.*:"
    "DeprecationWarning:hyppo\\.independence\\.hhg",
    "ignore:The keyword argument 'nopython=False' was supplied.*:Warning:numba\\.core\\.decorators",
)


@pytest.fixture(autouse=True)
def _pin_json_backend(request: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``GRAPHIFY_BACKEND=json`` for every test not about ArcadeDB.

    The default backend is ArcadeDB and an unreachable server is a fatal error,
    so without this the whole suite would demand a running service. Setting the
    variable (rather than deleting it) also stops whatever the developer happens
    to have exported from leaking in — the suite is deterministic either way.

    A test that wants the real default still gets it: its own monkeypatch runs
    after this fixture and wins.
    """
    if request.path.name in _ARCADE_TEST_FILES:
        return
    monkeypatch.setenv("GRAPHIFY_BACKEND", "json")


def pytest_collection_modifyitems(items: list[Any]) -> None:
    for item in items:
        if item.path.name != "test_analyze.py":
            continue
        for warning_filter in _ANALYZE_WARNING_FILTERS:
            item.add_marker(pytest.mark.filterwarnings(warning_filter))
