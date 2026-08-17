"""ArcadeDB as a Windows service: layout, definition, config, install guards.

Pure parts only — nothing here downloads, registers or starts anything. The
live acceptance (reboot without login, two sessions, a stopped service) is a
manual pass, because it needs an elevated prompt and a real reboot.
"""
import json

import pytest

from graphify import arcade_server as a


def test_winsw_url_is_pinned():
    assert a.winsw_url("2.12.0") == (
        "https://github.com/winsw/winsw/releases/download/v2.12.0/WinSW.NET461.exe"
    )


# --- the service definition ------------------------------------------------ #

def test_service_definition_pins_an_absolute_java(monkeypatch, tmp_path):
    """The service account has no JAVA_HOME, so the definition may not need one."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    xml = a.service_definition(java_exe=r"C:\jdk-21\bin\java.exe", password="pw", port=2480)
    assert r"<executable>C:\jdk-21\bin\java.exe</executable>" in xml
    assert "JAVA_HOME" not in xml
    assert "%" not in xml, "no environment expansion may survive into the definition"


def test_service_definition_starts_automatically_and_restarts(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    xml = a.service_definition(java_exe="java.exe", password="pw", port=2480)
    assert "<startmode>Automatic</startmode>" in xml
    assert 'onfailure action="restart"' in xml
    assert f"<id>{a.SERVICE_NAME}</id>" in xml


def test_service_definition_carries_the_heap_and_the_hive(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    xml = a.service_definition(java_exe="java.exe", password="pw", port=2480, heap="8G")
    assert "-Xmx8G" in xml
    assert str(tmp_path / "databases") in xml


def test_service_definition_escapes_the_password_field(monkeypatch, tmp_path):
    """The generated password is urlsafe, but the --password flag is free text."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    xml = a.service_definition(java_exe="java.exe", password="a<b>&c", port=2480)
    assert "a<b>&c" not in xml
    assert "&lt;b&gt;&amp;c" in xml


# --- the machine config ---------------------------------------------------- #

def test_machine_config_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    a.write_machine_config(url="http://127.0.0.1:2480", password="s3cret")
    cfg = a.read_machine_config()
    assert cfg["url"] == "http://127.0.0.1:2480" and cfg["password"] == "s3cret"
    assert json.loads(a.config_path().read_text(encoding="utf-8"))["user"] == "root"


def test_missing_machine_config_is_normal(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    assert a.read_machine_config() == {}


def test_corrupt_machine_config_is_loud(monkeypatch, tmp_path):
    """Falling back to defaults here would report 'cannot connect' for a
    problem that is actually a broken file."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    a.config_path().parent.mkdir(parents=True, exist_ok=True)
    a.config_path().write_text("{not json", encoding="utf-8")
    with pytest.raises(a.GraphDBUnavailable, match="machine config"):
        a.read_machine_config()


def test_generated_password_is_not_a_constant():
    assert a.generate_password() != a.generate_password()
    assert len(a.generate_password()) >= 24


def test_no_constant_password_left_in_the_package():
    """`playwithdata` was fine for a personal, disposable server."""
    from pathlib import Path

    import graphify

    pkg = Path(graphify.__file__).parent
    hits = [p.name for p in pkg.glob("*.py")
            if "playwithdata" in p.read_text(encoding="utf-8", errors="ignore")]
    assert hits == [], f"a constant connection password is still in {hits}"


# --- install guards -------------------------------------------------------- #

def test_install_refuses_without_elevation_before_touching_anything(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    monkeypatch.setattr(a.os, "name", "nt")
    monkeypatch.setattr(a, "is_admin", lambda: False)
    monkeypatch.setattr(a, "download", lambda *args, **kw: pytest.fail("must not download"))
    with pytest.raises(RuntimeError, match="elevated"):
        a.install_service()
    assert list(tmp_path.iterdir()) == [], "a refused install must leave nothing behind"


def test_install_refuses_a_password_the_hive_will_not_honour(monkeypatch, tmp_path):
    """ArcadeDB only applies the root password while creating server-users.jsonl."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    monkeypatch.setattr(a.os, "name", "nt")
    monkeypatch.setattr(a, "is_admin", lambda: True)
    monkeypatch.setattr(a, "download", lambda *args, **kw: pytest.fail("must not download"))
    a.server_users_file().parent.mkdir(parents=True, exist_ok=True)
    a.server_users_file().write_text('{"name":"root"}\n', encoding="utf-8")
    a.write_machine_config(url="http://127.0.0.1:2480", password="the-old-one")

    with pytest.raises(RuntimeError) as exc:
        a.install_service(password="a-new-one")
    msg = str(exc.value)
    assert "uninstall-service" in msg and "server-users.jsonl" in msg


def test_install_accepts_the_password_already_in_the_config(monkeypatch, tmp_path):
    """Re-running the installer with the same credential is not a conflict."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    a.server_users_file().parent.mkdir(parents=True, exist_ok=True)
    a.server_users_file().write_text('{"name":"root"}\n', encoding="utf-8")
    a.write_machine_config(url="http://127.0.0.1:2480", password="same")
    assert a._existing_password_differs("same") is False


def test_uninstall_without_an_install_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    assert a.uninstall_service() is False


def test_uninstall_keeps_the_hive(monkeypatch, tmp_path):
    """Databases are the user's data; removing the service is not removing them."""
    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    a.databases_dir().mkdir(parents=True)
    (a.databases_dir() / "proj_x_deadbeef").mkdir()
    a.service_exe().write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(a.subprocess, "run",
                        lambda argv, **kw: calls.append(argv) or _completed())

    assert a.uninstall_service() is True
    assert [c[1] for c in calls] == ["stop", "uninstall"]
    assert (a.databases_dir() / "proj_x_deadbeef").is_dir()


# --- the CLI surface ------------------------------------------------------- #

@pytest.mark.parametrize("verb", ["start", "stop", "download"])
def test_removed_verbs_point_at_the_service(monkeypatch, tmp_path, capsys, verb):
    """The lifecycle verbs are gone: duplicating the SCM would mean a second way
    to launch a server, hence a second hive."""
    import sys

    from graphify import cli

    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["graphify", "arcade", verb])
    with pytest.raises(SystemExit) as exc:
        cli.dispatch_command("arcade")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "install-service" in err and a.SERVICE_NAME in err


def test_usage_lists_only_the_surviving_verbs(monkeypatch, tmp_path, capsys):
    import sys

    from graphify import cli

    monkeypatch.setenv("GRAPHIFY_ARCADE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["graphify", "arcade", "nonsense"])
    with pytest.raises(SystemExit):
        cli.dispatch_command("arcade")
    err = capsys.readouterr().err
    assert "install-service" in err and "uninstall-service" in err
    assert "status" in err and "reload" in err


def _completed():
    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    return _P()
