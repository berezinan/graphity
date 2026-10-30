"""A stale ``GRAPHIFY_BACKEND=arcadedb`` does not select an LLM backend.

Before the 0.9.97 merge the fork read ``GRAPHIFY_BACKEND`` as the graph store;
upstream (#4361) reads it as the LLM provider. The store moved to
``GRAPHIFY_GRAPH_STORE``, and the old store values are filtered out of the LLM
selection so a setting left in a user's environment does not break extraction.
"""
from __future__ import annotations

import pytest

from graphify import llm
from graphify.query_backend import GRAPH_STORE_VALUES, resolve_backend_config


@pytest.fixture(autouse=True)
def _fresh_warning(monkeypatch):
    monkeypatch.setattr(llm, "_stale_graph_store_warned", False)


@pytest.mark.parametrize("value", sorted(GRAPH_STORE_VALUES) + ["ArcadeDB"])
def test_stale_store_value_falls_through_to_autodetect(monkeypatch, capsys, value):
    monkeypatch.setenv("GRAPHIFY_BACKEND", value)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    backend, source = llm.detect_backend_with_source()
    assert backend == "deepseek"
    assert source.startswith("auto-detected")
    assert not llm.backend_disabled_by_env()
    err = capsys.readouterr().err
    assert err.count(f"GRAPHIFY_BACKEND={value}") == 1
    assert "remove the variable" in err and "GRAPHIFY_GRAPH_STORE" in err


def test_stale_store_warning_is_printed_once_per_process(monkeypatch, capsys):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "arcadedb")
    for _ in range(3):
        llm.detect_backend_with_source()
        llm.backend_disabled_by_env()
    assert capsys.readouterr().err.count("no longer selects the graph store") == 1


def test_provider_name_keeps_its_meaning(monkeypatch, capsys):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "deepseek")
    monkeypatch.delenv("GRAPHIFY_GRAPH_STORE", raising=False)
    assert llm.detect_backend_with_source() == ("deepseek", "GRAPHIFY_BACKEND=deepseek")
    assert "no longer selects the graph store" not in capsys.readouterr().err
    assert resolve_backend_config("g.json")["kind"] == "arcadedb"


def test_upstream_disable_switch_keeps_its_meaning(monkeypatch, capsys):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    monkeypatch.delenv("GRAPHIFY_GRAPH_STORE", raising=False)
    assert llm.backend_disabled_by_env()
    assert llm.detect_backend_with_source()[0] is None
    assert "no longer selects the graph store" not in capsys.readouterr().err
    assert resolve_backend_config("g.json")["kind"] == "arcadedb"
