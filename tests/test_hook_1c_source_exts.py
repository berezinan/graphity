"""The Read hook treats the fork's 1C files as sources (fork delta).

`detect.CODE_EXTENSIONS` and the extractors know BSL/OneScript modules and the
1C:EDT files, but `_HOOK_SOURCE_EXTS` did not, so reading a module of a
configuration graph drew no nudge at all. The fix first lived only as an
uncommitted edit of the production clone, where an upstream merge would have
dropped it without trace.
"""
import pytest

from tests.test_hook_guard import _invoke

_1C_EXTS = (".bsl", ".os", ".osl", ".mdo", ".form", ".rights", ".dcs", ".oform", ".cmi")


def test_1c_exts_in_hook_source_exts():
    from graphify.cli import _HOOK_SOURCE_EXTS
    assert set(_1C_EXTS) <= set(_HOOK_SOURCE_EXTS)


@pytest.mark.parametrize("path", [
    "src/CommonModules/X/Module.bsl",
    "src/Catalogs/X/X.mdo",
    "src/Catalogs/X/Forms/F/Form.form",
    "Module.BSL",                      # регистр расширения не важен
])
def test_read_of_1c_file_nudges(path, tmp_path, monkeypatch):
    out = _invoke("read", {"tool_input": {"file_path": path}}, tmp_path, monkeypatch)
    assert "graphify query" in out, f"{path!r} should nudge"
