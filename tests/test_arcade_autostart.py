"""Unit tests for the opt-in ArcadeDB auto-start in the extract/update sync path.

The logic lives in fork-owned `arcade_server` (`autostart_enabled` /
`maybe_autostart`); churn files (cli.py, watch.py) carry only a one-line call
via `query_backend.open_backend_for_sync`. These tests are the regression guard
the openspec spec `arcadedb-backend` requires: if an upstream merge severs the
helper or the call sites, this file turns red instead of the loss being silent.

The lifecycle helper is mocked — no JVM is launched and `time.sleep` never fires
because the mocked server reports ready on the first in-loop poll."""
from pathlib import Path

import pytest

from graphify import arcade_server as a
from graphify import query_backend as qb


# ---- flag parsing -------------------------------------------------------------

def test_autostart_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_ARCADE_AUTOSTART", raising=False)
    assert a.autostart_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", " yes "])
def test_autostart_enabled_truthy(monkeypatch, val):
    monkeypatch.setenv("GRAPHIFY_ARCADE_AUTOSTART", val)
    assert a.autostart_enabled() is True


@pytest.mark.parametrize("val", ["", "0", "no", "off", "false"])
def test_autostart_disabled_falsy(monkeypatch, val):
    monkeypatch.setenv("GRAPHIFY_ARCADE_AUTOSTART", val)
    assert a.autostart_enabled() is False


# ---- maybe_autostart ----------------------------------------------------------

@pytest.fixture
def autostart_on(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_ARCADE_AUTOSTART", "1")


def _stub_status(monkeypatch, ready_sequence):
    """Make status() return ready=<next value in the sequence> on each call."""
    seq = iter(ready_sequence)
    monkeypatch.setattr(a, "status", lambda *args, **kw: {"ready": next(seq)})


def test_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_ARCADE_AUTOSTART", raising=False)
    monkeypatch.setattr(a, "status", lambda *args, **kw: pytest.fail("status must not be called"))
    a.maybe_autostart({"host": "127.0.0.1", "port": 2480, "password": "pw"})


def test_starts_when_not_ready(monkeypatch, autostart_on):
    # flag on + server down -> start is called, then ready is polled
    _stub_status(monkeypatch, [False, True])  # down, then up after launch
    calls = []
    monkeypatch.setattr(a, "start", lambda pw, **kw: calls.append((pw, kw)) or {"already_running": False})
    a.maybe_autostart({"host": "127.0.0.1", "port": 2480, "password": "pw"})
    assert len(calls) == 1


def test_noop_when_ready(monkeypatch, autostart_on):
    # flag on + server already up -> start NOT called
    _stub_status(monkeypatch, [True])
    calls = []
    monkeypatch.setattr(a, "start", lambda pw, **kw: calls.append(pw) or {"already_running": True})
    a.maybe_autostart({"host": "127.0.0.1", "port": 2480, "password": "pw"})
    assert calls == []


def test_start_failure_is_swallowed(monkeypatch, autostart_on, capsys):
    # start raises -> warning printed, no exception propagates
    _stub_status(monkeypatch, [False])

    def _boom(pw, **kw):
        raise RuntimeError("ArcadeDB needs Java >= 21")

    monkeypatch.setattr(a, "start", _boom)
    a.maybe_autostart({"host": "127.0.0.1", "port": 2480, "password": "pw"})
    err = capsys.readouterr().err
    assert "auto-start failed" in err
    assert "Java >= 21" in err


def test_passes_host_port_password_from_cfg(monkeypatch, autostart_on):
    # host/port/password are taken from the resolved cfg
    seen = {}
    monkeypatch.setattr(a, "status", lambda host, port, password, *a_, **kw: seen.update(
        host=host, port=port, password=password) or {"ready": True})
    monkeypatch.setattr(a, "start", lambda *a_, **kw: {"already_running": True})
    a.maybe_autostart({"host": "10.0.0.5", "port": 9999, "password": "s3cret"})
    assert seen == {"host": "10.0.0.5", "port": 9999, "password": "s3cret"}


def test_password_falls_back_to_default(monkeypatch, autostart_on):
    # empty cfg password + no env -> "playwithdata"
    monkeypatch.delenv("GRAPHIFY_ARCADE_PASSWORD", raising=False)
    seen = {}
    monkeypatch.setattr(a, "status", lambda host, port, password, *a_, **kw: seen.update(
        password=password) or {"ready": True})
    monkeypatch.setattr(a, "start", lambda *a_, **kw: {"already_running": True})
    a.maybe_autostart({"host": "127.0.0.1", "port": 2480, "password": ""})
    assert seen["password"] == "playwithdata"


# ---- call-site wiring (the anti-clobber guard) --------------------------------

def test_open_backend_for_sync_autostarts_arcadedb(monkeypatch):
    calls = []
    monkeypatch.setattr(a, "maybe_autostart", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(qb, "ArcadeDBBackend", lambda db, **kw: ("stub", db))
    cfg = {"kind": "arcadedb", "database": "proj_x", "host": "h", "port": 1, "user": "u", "password": "p"}
    assert qb.open_backend_for_sync(cfg) == ("stub", "proj_x")
    assert calls == [cfg]


def test_open_backend_for_sync_skips_json(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "maybe_autostart", lambda cfg: pytest.fail("must not autostart for json"))
    gp = tmp_path / "graph.json"
    gp.write_text('{"nodes": [], "links": []}', encoding="utf-8")
    backend = qb.open_backend_for_sync({"kind": "json", "graph_path": str(gp)})
    assert isinstance(backend, qb.JsonBackend)


def test_sync_call_sites_use_autostart_wrapper():
    """An upstream merge that rewrites the sync blocks in the churn files must
    turn this red rather than silently reverting to plain open_backend
    (= silently dropping auto-start, the 2026-06 incident)."""
    src_dir = Path(a.__file__).parent
    for mod in ("cli.py", "watch.py"):
        src = (src_dir / mod).read_text(encoding="utf-8")
        assert "open_backend_for_sync" in src, f"{mod}: sync call site lost open_backend_for_sync"
