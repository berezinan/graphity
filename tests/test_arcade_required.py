"""The graph database is required: absence is fatal, never a silent fallback.

Guards the helper every call site funnels through (`require_backend`) and the
two messages it produces — service not answering, and project database absent.
Both must name what to do next, because there is nothing else the user gets.
"""
import pytest

from graphify.query_backend import (
    GraphDBUnavailable,
    JsonBackend,
    SERVICE_NAME,
    _await_ready,
    require_backend,
)


class _Clock:
    """Fake clock the fake sleep advances, so a timed wait stays instant."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class _Backend:
    """Minimal stand-in: `_await_ready` only ever touches `.ready()` and `.base`."""

    def __init__(self, readiness):
        self.base = "http://127.0.0.1:2480"
        self.db = "proj_hotel_deadbeef"
        self._readiness = list(readiness)
        self.probes = 0

    def ready(self):
        self.probes += 1
        return self._readiness.pop(0) if self._readiness else False


def test_ready_immediately_does_not_wait(capsys):
    be = _Backend([True])
    slept = []
    _await_ready(be, sleep=slept.append)
    assert be.probes == 1 and slept == []
    assert capsys.readouterr().err == ""


def test_ready_on_the_second_probe_waits_and_proceeds():
    """The window right after boot: not ready, then ready."""
    be = _Backend([False, True])
    slept = []
    errs = []

    class _Stream:
        def write(self, s):
            errs.append(s)

        def flush(self):
            pass

    _await_ready(be, sleep=slept.append, stream=_Stream())
    assert be.probes == 2 and len(slept) == 1
    assert SERVICE_NAME in "".join(errs), "the wait must be announced, not silent"


def test_never_ready_is_fatal_and_says_how_to_check():
    be = _Backend([])
    clock = _Clock()
    with pytest.raises(GraphDBUnavailable) as exc:
        _await_ready(be, timeout=3.0, sleep=clock.advance, stream=_devnull(),
                     clock=clock.now)
    msg = str(exc.value)
    assert be.base in msg
    assert SERVICE_NAME in msg
    assert "graphify arcade status" in msg
    assert "does not fall back" in msg


def test_the_wait_honours_the_budget_it_announced():
    """A probe against a dead service costs seconds, not nothing.

    Counting attempts (`timeout / poll` of them) spends the whole budget on the
    sleeps alone and the probes on top: measured 91s of waiting behind a
    "waiting up to 30s" message. The budget is wall-clock.
    """
    clock = _Clock()

    class _Slow(_Backend):
        def ready(self):
            clock.advance(2.0)  # a connection refused, timed
            return super().ready()

    with pytest.raises(GraphDBUnavailable):
        _await_ready(_Slow([]), timeout=30.0, poll=1.0, sleep=clock.advance,
                     stream=_devnull(), clock=clock.now)
    assert clock.now() <= 32.0, f"waited {clock.now():.0f}s against a 30s budget"


def test_a_rejected_credential_says_so_instead_of_blaming_the_service():
    """`/ready` is open to anonymous callers but turns away a WRONG password.

    A stale `GRAPHIFY_ARCADE_PASSWORD` therefore reads as "the service did not
    answer" — pointing at the service, which is up and healthy — and burns the
    whole wait budget on a rejection that will never turn into an acceptance.
    """
    be = _Backend([])
    be.last_ready_status = 403
    slept = []

    with pytest.raises(GraphDBUnavailable) as exc:
        _await_ready(be, sleep=slept.append, stream=_devnull())

    msg = str(exc.value)
    assert "rejected the credentials" in msg and "403" in msg
    assert "GRAPHIFY_ARCADE_PASSWORD" in msg
    assert slept == [], "a rejected credential is final — nothing to wait for"


def test_missing_database_names_it_and_the_remedy():
    cfg = {"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
           "database": "proj_hotel_deadbeef", "user": "root", "password": "x",
           "project_path": r"C:\work\Hotel"}

    backend_seen = {}

    def _fake_open(**kw):
        be = _Backend([True])
        be._server = lambda cmd: {"result": ["proj_other_11111111"]}
        backend_seen["be"] = be
        return be

    import graphify.query_backend as qb
    real_open = qb.open_backend
    qb.open_backend = _fake_open
    try:
        with pytest.raises(GraphDBUnavailable) as exc:
            require_backend(config=cfg)
    finally:
        qb.open_backend = real_open

    msg = str(exc.value)
    assert "proj_hotel_deadbeef" in msg, "must name the database it computed"
    assert r"C:\work\Hotel" in msg, "must name the project it computed it from"
    assert "graphify extract" in msg and "graphify arcade reload" in msg


def test_json_backend_is_never_probed(monkeypatch, tmp_path):
    """`GRAPHIFY_BACKEND=json` must not touch the service at all."""
    monkeypatch.setenv("GRAPHIFY_BACKEND", "json")
    import networkx as nx
    be = require_backend(graph=nx.DiGraph(), graph_path=str(tmp_path / "graph.json"))
    assert isinstance(be, JsonBackend)


# --- the sync paths: fatal, not a warning with exit 0 ---------------------- #

_ARCADE_CFG = {"kind": "arcadedb", "host": "127.0.0.1", "port": 2480,
               "database": "proj_t_deadbeef", "user": "root", "password": "pw",
               "project_path": r"C:\work\t"}


@pytest.fixture
def _dead_service(monkeypatch):
    """Point the backend selection at a service that never answers, fast."""
    import graphify.query_backend as qb
    monkeypatch.setattr(qb, "_READY_TIMEOUT", 0.01)
    monkeypatch.setattr(qb, "_READY_POLL", 0.01)
    monkeypatch.setattr(qb, "resolve_backend_config", lambda *a, **kw: dict(_ARCADE_CFG))
    monkeypatch.setattr(qb, "open_backend", lambda **kw: _Backend([]))
    return qb


def test_global_sync_propagates_instead_of_warning(_dead_service, monkeypatch):
    """`global add/remove` used to print a warning and carry on, leaving the
    global graph silently one merge behind."""
    from graphify import global_graph
    with pytest.raises(GraphDBUnavailable):
        global_graph._sync_global_db()


def test_extract_checks_before_any_llm_work(_dead_service, monkeypatch, tmp_path, capsys):
    """The check sits at the top of extract, so an unreachable database costs
    milliseconds rather than a whole paid extraction."""
    import sys

    from graphify import cli, detect

    called = []
    monkeypatch.setattr(detect, "detect", lambda *a, **kw: called.append("detect"))
    (tmp_path / "a.py").write_text("def f(): pass\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["graphify", "extract", str(tmp_path)])

    with pytest.raises(GraphDBUnavailable):
        cli.dispatch_command("extract")
    assert called == [], "detect/extraction must not start before the DB check"


def test_watch_rebuild_reports_failure_when_the_service_goes_away(
        _dead_service, tmp_path, monkeypatch):
    """The service restarting under a running `watch`.

    The rebuild cannot succeed without the database, so it reports failure —
    which is what `graphify update` turns into a non-zero exit, and what a watch
    loop prints on every attempt until the service is back. It does NOT report a
    rebuild that quietly skipped the sync.
    """
    from pathlib import Path

    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lib.py").write_text("def f(): pass\n", encoding="utf-8")
    monkeypatch.chdir(corpus)

    assert _rebuild_code(Path("."), acquire_lock=False) is False


def test_update_checks_before_rebuilding(_dead_service, monkeypatch, tmp_path):
    """`update` asks at its entry, like extract — not at the sync that closes it.

    Checking at the sync means the whole AST rebuild, clustering and report
    regeneration run first, only to end on a database that was never there.
    """
    import sys

    from graphify import cli, watch

    rebuilt = []
    monkeypatch.setattr(watch, "_rebuild_code", lambda *a, **kw: rebuilt.append("rebuild"))
    (tmp_path / "a.py").write_text("def f(): pass\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["graphify", "update", str(tmp_path)])

    with pytest.raises(GraphDBUnavailable):
        cli.dispatch_command("update")
    assert rebuilt == [], "the rebuild must not start before the DB check"


def test_watch_refuses_to_start_without_the_service(_dead_service, monkeypatch, tmp_path):
    """`watch` that cannot reach the database never rebuilds anything, so it
    says so at startup rather than sitting there looking like it is working."""
    import sys

    from graphify import cli, watch

    watched = []
    monkeypatch.setattr(watch, "watch", lambda *a, **kw: watched.append("watch"))
    monkeypatch.setattr(sys, "argv", ["graphify", "watch", str(tmp_path)])

    with pytest.raises(GraphDBUnavailable):
        cli.dispatch_command("watch")
    assert watched == [], "the watcher must not start before the DB check"


def test_churn_files_do_not_swallow_a_failed_sync():
    """Structural guard: the swallow this change removed must not come back.

    An upstream merge rewrites these blocks wholesale, and a returning
    `except Exception` around the sync would restore exit-0-on-data-loss without
    any behavioural test noticing.
    """
    from pathlib import Path

    import graphify

    src_dir = Path(graphify.__file__).parent
    for mod, banned in (("cli.py", "ArcadeDB sync failed"),
                        ("watch.py", "ArcadeDB sync skipped"),
                        ("global_graph.py", "ArcadeDB global sync failed")):
        src = (src_dir / mod).read_text(encoding="utf-8")
        assert banned not in src, f"{mod}: the swallowed-sync warning is back"
        assert "require_backend" in src, f"{mod}: sync call site lost require_backend"


def _devnull():
    class _N:
        def write(self, s):
            pass

        def flush(self):
            pass

    return _N()
