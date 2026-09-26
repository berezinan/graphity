"""Install scope of the Claude Code skill and PreToolUse hooks (OpenSpec
``user-scope-claude-hooks``, capability ``agent-install-scope``).

A user-level install puts the skill AND the hooks in the user's Claude config
dir; a project install puts both in the project. Claude Code merges user and
project hooks rather than letting one shadow the other, so hooks written into
every project a user-level command ran in printed each hint twice once a
user-level pair existed.

The conftest ``_sandbox_home`` fixture makes ``Path.home()`` a throwaway dir and
clears ``CLAUDE_CONFIG_DIR``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from graphify.install import __version__

_EXE = r"C:\Users\installer\graphify.EXE"


@pytest.fixture
def project(tmp_path, monkeypatch):
    proj = tmp_path / "project"
    proj.mkdir()
    monkeypatch.chdir(proj)
    # Resolution would otherwise find this machine's real graphify.
    monkeypatch.setattr("shutil.which", lambda _name: _EXE)
    return proj


def _run(*argv: str) -> None:
    from graphify.__main__ import main

    with patch("sys.argv", ["graphify", *argv]):
        main()


def _graphify_hooks(settings: Path) -> list:
    pre = json.loads(settings.read_text(encoding="utf-8")).get("hooks", {}).get("PreToolUse", [])
    return [h for h in pre if "graphify" in str(h)]


def _commands(hooks: list) -> list:
    return [h["hooks"][0]["command"] for h in hooks]


def _user_settings() -> Path:
    return Path.home() / ".claude" / "settings.json"


# ---------------------------------------------------------------- hook scope


@pytest.mark.parametrize("platform", ["claude", "windows"])
def test_user_install_writes_hooks_to_user_settings(project, platform):
    _run("install", "--platform", platform)

    hooks = _graphify_hooks(_user_settings())
    assert sorted(h["matcher"] for h in hooks) == ["Bash|Grep", "Read|Glob"]
    assert not (project / ".claude").exists()


def test_project_install_writes_hooks_to_project_only(project):
    _run("install", "--project", "--platform", "claude")

    hooks = _graphify_hooks(project / ".claude" / "settings.json")
    assert sorted(h["matcher"] for h in hooks) == ["Bash|Grep", "Read|Glob"]
    assert not _user_settings().exists()


def test_hook_command_form_follows_scope(project):
    _run("install", "--platform", "claude")
    _run("install", "--project", "--platform", "claude")

    user_cmds = _commands(_graphify_hooks(_user_settings()))
    project_cmds = _commands(_graphify_hooks(project / ".claude" / "settings.json"))
    assert all(c.startswith("C:/Users/installer/graphify.EXE hook-guard ") for c in user_cmds), user_cmds
    assert all(c.startswith("graphify hook-guard ") for c in project_cmds), project_cmds


def test_reinstall_keeps_one_pair_and_foreign_hooks(project):
    foreign = {"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}
    settings = _user_settings()
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": {"PreToolUse": [foreign]}}), encoding="utf-8")

    _run("install", "--platform", "claude")
    _run("install", "--platform", "claude")

    pre = json.loads(settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]
    assert foreign in pre
    assert len(_graphify_hooks(settings)) == 2


def test_claude_install_in_two_projects_leaves_one_user_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: _EXE)
    for name in ("p1", "p2"):
        proj = tmp_path / name
        proj.mkdir()
        monkeypatch.chdir(proj)
        _run("claude", "install")

    for name in ("p1", "p2"):
        assert "## graphify" in (tmp_path / name / "CLAUDE.md").read_text(encoding="utf-8")
        assert not (tmp_path / name / ".claude" / "settings.json").exists()
    assert len(_graphify_hooks(_user_settings())) == 2


def test_claude_config_dir_is_honored(project, tmp_path, monkeypatch):
    config_dir = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    _run("install", "--platform", "claude")

    assert len(_graphify_hooks(config_dir / "settings.json")) == 2
    assert not _user_settings().exists()


# ---------------------------------------------------------------- skill scope


def test_user_install_puts_skill_in_user_dir(project):
    _run("install", "--platform", "claude")

    skill_dir = Path.home() / ".claude" / "skills" / "graphify"
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "references").is_dir()
    assert (skill_dir / ".graphify_version").read_text(encoding="utf-8").strip() == __version__
    assert not (project / ".claude" / "skills").exists()


def test_project_install_puts_skill_in_project(project):
    _run("install", "--project", "--platform", "claude")

    assert (project / ".claude" / "skills" / "graphify" / "SKILL.md").is_file()
    assert not (Path.home() / ".claude" / "skills").exists()


def test_reinstall_refreshes_skill_references_and_stamp(project):
    _run("install", "--platform", "claude")
    skill_dir = Path.home() / ".claude" / "skills" / "graphify"
    (skill_dir / ".graphify_version").write_text("0.0.1-old", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("stale\n", encoding="utf-8")
    (skill_dir / "references" / "stale.md").write_text("stale\n", encoding="utf-8")

    _run("install", "--platform", "claude")

    assert (skill_dir / ".graphify_version").read_text(encoding="utf-8").strip() == __version__
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") != "stale\n"
    # The stamp covers references/ too: the sidecar is replaced wholesale.
    assert not (skill_dir / "references" / "stale.md").exists()


# ---------------------------------------------------------------- duplicates


def test_warns_about_project_hooks(project, capsys):
    _run("install", "--project", "--platform", "claude")
    capsys.readouterr()

    _run("install", "--platform", "claude")

    err = capsys.readouterr().err
    assert str(project / ".claude" / "settings.json") in err
    assert "2 graphify PreToolUse hook" in err
    assert "graphify claude uninstall --project" in err


def test_warns_about_project_skill_copy(project, capsys):
    skill_dir = project / ".claude" / "skills" / "graphify"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("old\n", encoding="utf-8")
    (skill_dir / ".graphify_version").write_text("0.1.2", encoding="utf-8")

    _run("install", "--platform", "claude")

    err = capsys.readouterr().err
    assert str(skill_dir) in err
    assert "0.1.2" in err
    assert "graphify claude uninstall --project" in err


def test_no_warning_without_duplicates(project, capsys):
    _run("install", "--platform", "claude")

    assert "graphify claude uninstall --project" not in capsys.readouterr().err


def test_install_from_home_is_not_a_duplicate(project, monkeypatch, capsys):
    monkeypatch.chdir(Path.home())
    _run("install", "--platform", "claude")
    _run("install", "--platform", "claude")

    assert "graphify claude uninstall --project" not in capsys.readouterr().err


# ---------------------------------------------------------------- uninstall


@pytest.mark.parametrize("argv", [("uninstall",), ("claude", "uninstall")])
def test_user_uninstall_removes_user_hooks_only(project, argv):
    settings = _user_settings()
    settings.parent.mkdir(parents=True)
    foreign = {
        "theme": "dark",
        "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}]},
    }
    settings.write_text(json.dumps(foreign), encoding="utf-8")
    _run("install", "--platform", "claude")

    _run(*argv)

    assert json.loads(settings.read_text(encoding="utf-8")) == foreign


def test_project_uninstall_leaves_user_settings(project):
    _run("install", "--platform", "claude")
    _run("install", "--project", "--platform", "claude")
    before = _user_settings().read_bytes()

    _run("uninstall", "--project")

    assert _user_settings().read_bytes() == before
    project_settings = project / ".claude" / "settings.json"
    assert not project_settings.exists() or not _graphify_hooks(project_settings)


# ---------------------------------------------------------------- --strict


@pytest.mark.parametrize("argv", [("install", "--strict"), ("claude", "install", "--strict")])
def test_strict_without_project_is_refused(project, argv, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(*argv)

    assert exc.value.code not in (0, None)
    assert "--project" in capsys.readouterr().err
    assert not _user_settings().exists()
    assert not (project / ".claude").exists()
    assert not (project / "CLAUDE.md").exists()


def test_strict_with_project_lands_on_read_hook(project):
    _run("install", "--project", "--strict", "--platform", "claude")

    hooks = _graphify_hooks(project / ".claude" / "settings.json")
    read = [h for h in hooks if h["matcher"] == "Read|Glob"]
    assert _commands(read) == ["graphify hook-guard read --strict"]
