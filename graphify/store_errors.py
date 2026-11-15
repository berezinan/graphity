"""Errors of the graph store (fork-owned; upstream has no such file).

Defined in a stdlib-only module so the CLI entry point can catch them without
importing the query stack (networkx and friends) on every command.
"""
from __future__ import annotations


class GraphDBUnavailable(RuntimeError):
    """The selected graph store cannot serve this command.

    Carries the complete user-facing text: the store is required, so there is no
    fallback to weigh up and the only decision left to the caller is where to
    print it. The snapshot's own refusal (``snap_store.SnapshotUnavailable``) is
    a subclass.
    """
