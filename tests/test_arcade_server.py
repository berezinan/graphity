"""Unit tests for the ArcadeDB lifecycle helper (pure parts only — no download
or JVM launch). Start/stop are exercised live in manual/integration runs."""
import os
from pathlib import Path

import pytest

from graphify import arcade_server as a


def test_download_url():
    assert a.download_url("26.6.1") == (
        "https://github.com/ArcadeData/arcadedb/releases/download/26.6.1/arcadedb-26.6.1-minimal.tar.gz"
    )


def test_arcade_home_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    assert a.arcade_home() == tmp_path
    assert a.dist_dir() == tmp_path / "dist" / f"arcadedb-{a.ARCADE_VERSION}"


def test_arcade_home_defaults_to_programdata(monkeypatch):
    """Machine-wide, not a user profile: one service serves every session."""
    monkeypatch.delenv("GRAPHIFY_ARCADE_HOME", raising=False)
    monkeypatch.setenv("PROGRAMDATA", r"C:\ProgramData")
    assert a.arcade_home() == Path(r"C:\ProgramData") / "graphify" / "arcadedb"


def test_databases_live_outside_the_versioned_distribution(monkeypatch, tmp_path):
    """A distribution upgrade must not orphan the hive."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    assert a.databases_dir() == tmp_path / "databases"
    assert a.ARCADE_VERSION not in str(a.databases_dir())
    assert a.databases_dir() not in a.dist_dir().parents


def test_java_executable_respects_java_home(monkeypatch, tmp_path):
    name = "java.exe" if os.name == "nt" else "java"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    assert a.java_executable() == str(tmp_path / "bin" / name)


def test_java_executable_fallback(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    assert a.java_executable() == "java"


def test_server_command_has_key_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    cmd = a.server_command("s3cret", heap="4G", port=2490)
    assert cmd[-1] == "com.arcadedb.server.ArcadeDBServer"
    assert "-Darcadedb.server.rootPassword=s3cret" in cmd
    assert "-Darcadedb.server.httpIncomingPort=2490" in cmd
    assert f"-Darcadedb.server.databaseDirectory={tmp_path / 'databases'}" in cmd
    assert "-Xmx4G" in cmd
    assert cmd[cmd.index("-cp") + 1] == "lib/*"
    assert "jdk.incubator.vector" in cmd


def test_pick_java_prefers_adequate(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)  # java_executable() -> "java"
    monkeypatch.setattr(a, "_java_major", lambda exe: 25)
    assert a.pick_java() == "java"


def test_pick_java_falls_back_when_java_home_too_old(monkeypatch, tmp_path):
    name = "java.exe" if os.name == "nt" else "java"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    jh_exe = str(tmp_path / "bin" / name)
    monkeypatch.setattr(a, "_java_major", lambda exe: 17 if exe == jh_exe else 25)
    assert a.pick_java() == "java"  # JAVA_HOME's 17 rejected, PATH's 25 chosen


def test_pick_java_raises_when_all_too_old(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(a, "_java_major", lambda exe: 17)
    with pytest.raises(RuntimeError, match="Java >= 21"):
        a.pick_java()


def test_is_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    assert a.is_installed() is False
    lib = a.dist_dir() / "lib"
    lib.mkdir(parents=True)
    (lib / f"arcadedb-server-{a.ARCADE_VERSION}.jar").write_text("", encoding="utf-8")
    assert a.is_installed() is True


def test_status_not_running(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    st = a.status(host="127.0.0.1", port=1, password="x")  # nothing listens on :1
    assert st["ready"] is False
    assert st["installed"] is False
    assert st["service"] in ("running", "stopped", "not installed", "unknown")


def test_default_heap_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_ARCADE_HEAP", raising=False)
    assert a.default_heap() == "2G"


def test_default_heap_env_override(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HEAP", "8G")
    assert a.default_heap() == "8G"


@pytest.mark.parametrize("bad", ["8 GB", "8gb", "; rm -rf /", "-XX:+Evil", "big", ""])
def test_default_heap_rejects_non_heap_values(monkeypatch, bad):
    """The value is interpolated straight into the java argv, so anything that is
    not a JVM heap size must be refused rather than passed through. An empty/
    whitespace value is treated as unset (the documented way to opt out)."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HEAP", bad)
    if bad.strip() == "":
        assert a.default_heap() == "2G"
    else:
        with pytest.raises(ValueError, match="GRAPHIFY_ARCADE_HEAP"):
            a.default_heap()


def test_server_command_reads_heap_env_at_call_time(monkeypatch):
    """Resolved per call, not at import: a heap set in-process must still apply."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HEAP", "12G")
    assert "-Xmx12G" in a.server_command("s3cret")
    monkeypatch.setenv("GRAPHIFY_ARCADE_HEAP", "6G")
    assert "-Xmx6G" in a.server_command("s3cret")


def test_explicit_heap_argument_still_wins(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HEAP", "8G")
    assert "-Xmx4G" in a.server_command("s3cret", heap="4G")
