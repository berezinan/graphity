from __future__ import annotations

import tempfile
from pathlib import Path
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

@pytest.fixture(scope="session")
def _can_symlink() -> bool:
    """Whether this machine can create symlinks at all (#2642).

    Probed rather than inferred from ``sys.platform``: Windows *can* create
    symlinks from an elevated shell or with Developer Mode enabled, and those
    runs should still get the coverage. A plain non-elevated Windows shell
    raises ``OSError: [WinError 1314] A required privilege is not held by the
    client``, which pytest reports as a FAILURE — 15 of them, drowning out real
    defects — when what it means is "unsupported here".

    One file symlink is enough to probe: Windows gates file and directory
    symlinks behind the same ``SeCreateSymbolicLinkPrivilege`` check.
    """
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe-src"
        src.write_text("x", encoding="utf-8")
        try:
            (Path(d) / "probe-link").symlink_to(src)
        except (OSError, NotImplementedError):
            return False
        return True


@pytest.fixture
def requires_symlinks(_can_symlink) -> None:
    """Skip a test that must create symlinks when the platform won't allow it.

    Take this as a parameter rather than wrapping each ``symlink_to()`` call in
    try/except: the guard then sits in the signature where it is visible, and
    an OSError from the code UNDER test is still a real failure instead of
    being swallowed into a skip.
    """
    if not _can_symlink:
        pytest.skip(
            "symlink creation unavailable on this machine "
            "(Windows requires an elevated shell or Developer Mode)"
        )


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path_factory, monkeypatch):
    """Every test gets a throwaway HOME so installers/uninstallers can never
    touch the developer's real ~/.claude, ~/.gemini, ~/.codebuddy, ~/.copilot,
    ~/.config, ~/.agents (issue #2168).

    Allocated via tmp_path_factory (not inside tmp_path) so tests that assert
    the exact contents of their own tmp_path are unaffected."""
    home = tmp_path_factory.mktemp("sandbox-home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))              # Windows ntpath.expanduser
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)     # escape hatch that bypasses Path.home
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home

@pytest.fixture(autouse=True)
def _isolate_backend_env(monkeypatch):
    """detect_backend() probes the developer's real shell: with GOOGLE_API_KEY (or
    any of a dozen others) exported, the backend-detection tests picked that
    backend instead of the one the test set up (#3481). Clear every variable it
    reads before each test; a test that wants one sets it with monkeypatch."""
    from graphify.llm import backend_detection_env_vars

    for key in backend_detection_env_vars():
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _scrub_graphify_env(monkeypatch):
    """Clear every ``GRAPHIFY_*`` variable the developer happens to have exported.

    The suite patches module-level defaults (e.g. ``security._MAX_GRAPH_FILE_BYTES``)
    while the production code resolves the same setting from an env var first, so an
    exported ``GRAPHIFY_MAX_GRAPH_BYTES=32GB`` silently wins and ten size-cap tests
    fail with "DID NOT RAISE". The code reads ~40 such variables, so scrub the whole
    prefix rather than the one that happened to bite; a test that wants a variable
    sets it with monkeypatch, which runs after this fixture.
    """
    import os

    for key in [k for k in os.environ if k.startswith("GRAPHIFY_")]:
        monkeypatch.delenv(key, raising=False)


_ANALYZE_WARNING_FILTERS = (
    "ignore:Tensorflow not installed; ParametricUMAP will be unavailable:ImportWarning:umap",
    "ignore:Please import `random` from the `scipy\\.sparse` namespace.*:"
    "DeprecationWarning:hyppo\\.independence\\.hhg",
    "ignore:The keyword argument 'nopython=False' was supplied.*:Warning:numba\\.core\\.decorators",
)


@pytest.fixture(autouse=True)
def _pin_json_backend(
    request: Any, monkeypatch: pytest.MonkeyPatch, _scrub_graphify_env: None
) -> None:
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
