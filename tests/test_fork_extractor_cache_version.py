"""The fork's own extractors version their AST cache entries (fork delta,
xml-schema-types, task 5.4).

The AST cache key is graphify's version + upstream's cache schema + the grammar
set + the file's bytes. The fork's extractor code is not in it, so after the
`.mdo` extractor learned the XDTO edges an unchanged `.mdo` kept being served
from the old entry — even under `--force`, which skips only the semantic cache.
`cache_salt` now folds a per-extension fork version into the key, and the
manifest (which stores that salt) re-queues such files on a plain update.
"""
from __future__ import annotations

from pathlib import Path

from graphify.rcfile import _FORK_EXTRACTOR_VERSIONS, cache_salt, set_language_overrides

_PKG = """<package xmlns="http://v8.1c.ru/8.1/xdto" xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:t:p">
\t<objectType name="T"><property name="P" type="xs:string"/></objectType>
</package>
"""


def test_fork_extensions_carry_a_version_and_others_do_not():
    for ext in (".mdo", ".xsd", ".wsdl", ".xdto"):
        assert cache_salt(Path(f"a{ext}")) == f"fork={_FORK_EXTRACTOR_VERSIONS[ext]}"
    assert cache_salt(Path("a.py")) is None
    assert cache_salt(Path("a.bsl")) is None


def test_language_override_and_fork_version_combine():
    try:
        set_language_overrides({".inc": ".php", ".schema": ".xsd"})
        assert cache_salt(Path("a.inc")) == "language=.php"
        # The extractor that runs is the remapped one, so its version applies.
        assert cache_salt(Path("a.schema")) == f"language=.xsd;fork={_FORK_EXTRACTOR_VERSIONS['.xsd']}"
    finally:
        set_language_overrides(None)


def test_entry_written_without_the_fork_version_is_not_served(tmp_path):
    from graphify.cache import load_cached, save_cached
    from graphify.extract import extract

    pkg = tmp_path / "XDTOPackages" / "P"
    pkg.mkdir(parents=True)
    src = pkg / "Package.xdto"
    src.write_text(_PKG, encoding="utf-8")
    # What the pre-fix code left in the cache: same bytes, unsalted key, old labels.
    stale = {"nodes": [{"id": "stale", "label": "t:p · T.P", "file_type": "code",
                        "source_file": str(src)}], "edges": []}
    save_cached(src, stale, root=tmp_path, cache_root=tmp_path, kind="ast", salt=None)
    assert load_cached(src, tmp_path, cache_root=tmp_path, salt=None) is not None

    r = extract([src], root=tmp_path, cache_root=tmp_path, parallel=False)
    labels = {n["label"] for n in r["nodes"]}
    assert "t:p · T/P" in labels and "t:p · T.P" not in labels


def test_manifest_requeues_a_file_whose_fork_version_changed(tmp_path, monkeypatch):
    import graphify.rcfile as rc
    from graphify.detect import detect_incremental, save_manifest

    src = tmp_path / "Package.xdto"
    src.write_text(_PKG, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setitem(rc._FORK_EXTRACTOR_VERSIONS, ".xdto", "0")
    save_manifest({"code": [str(src)]}, str(manifest), kind="ast")
    monkeypatch.setitem(rc._FORK_EXTRACTOR_VERSIONS, ".xdto", "1")

    det = detect_incremental(tmp_path, manifest_path=str(manifest), kind="ast")
    changed = {Path(p).name for files in det.get("new_files", {}).values() for p in files}
    assert "Package.xdto" in changed
