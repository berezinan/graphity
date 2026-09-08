"""Intra-file slicing for oversized text documents (#1369).

The extraction packer (`_pack_chunks_by_tokens`) treats each file as atomic and
`_read_files` caps every file at its per-category char cap (`_char_cap_for`), so
a sliceable document larger than the slice cap would have everything past it
silently dropped — the model never saw it, and nothing in the adaptive-retry
path could recover it ("a single file larger than the budget ... packing can't
shrink one big file").

This module splits an oversized *splittable text* document (Markdown, plain
text, reStructuredText) into contiguous ``FileSlice`` units at heading /
paragraph / line boundaries so the whole file gets extracted across several
units. Every slice of a file reports the **parent file path** as its source, so
the resulting nodes are never fragmented per-slice — they merge by source_file
exactly as if the file had been extracted in one pass.

Plain-text documents and PDFs are sliced (PDFs via their text extractor); code
files are left whole because they need whole-symbol context, and raster images
are handled by the vision path, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Plain-text document types where boundary-based slicing is meaningful and where
# `_file_to_text` is a straight ``read_text`` (so a char range matches the bytes
# the model is shown). Deliberately excludes code (.py, .ts, ...) and binary
# docs (.pdf) — those are never sliced.
_SPLITTABLE_TEXT_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".txt", ".rst"})

# Boundary preferences, strongest first. A Markdown heading (``\n#``) keeps a
# section with its title; a blank line keeps a paragraph intact; a bare newline
# avoids cutting mid-line; a sentence end is the only boundary prose written
# without newlines offers. If none is found in the window we hard-cut.
_BOUNDARY_SEPARATORS = ("\n#", "\n\n", "\n", ". ", "? ", "! ")

# Least of the window a boundary must fill to be taken. Without it the strongest
# separator wins from anywhere in the window, so a lone blank line after a
# document header cuts the header off as a slice of its own and leaves the body
# — which may hold no newline at all — to hard cuts.
_MIN_FILL = 0.5


@dataclass(frozen=True)
class FileSlice:
    """A contiguous ``[start, end)`` character range of a splittable text file.

    ``index``/``total`` are for logging only. ``path`` is the real file on disk;
    the slice always reports ``path`` as its source so slices don't fragment the
    graph.
    """

    path: Path
    start: int
    end: int
    index: int
    total: int


# A unit of extraction work: either a whole file (``Path``) or one slice of one.
Unit = "Path | FileSlice"


def unit_path(unit: "Path | FileSlice") -> Path:
    """The on-disk path a unit belongs to (the parent file for a slice)."""
    return unit.path if isinstance(unit, FileSlice) else unit


def is_splittable_text(path: Path) -> bool:
    """True for plain-text document types that may be sliced."""
    return path.suffix.lower() in _SPLITTABLE_TEXT_SUFFIXES


# Sliceable suffixes whose text comes from a dedicated extractor rather than a
# straight ``read_text`` (see ``_unit_text``). PDFs become plain text after
# extraction, so they slice exactly like a Markdown/plain-text document.
_SLICEABLE_EXTRA_SUFFIXES = frozenset({".pdf"})


def is_sliceable(path: Path) -> bool:
    """True for documents that may be split into ``FileSlice``s when oversized.

    Plain-text docs (read as-is) plus PDFs (read via text extraction). Code and
    structured data are deliberately excluded — they are read whole to keep
    whole-symbol context.
    """
    return is_splittable_text(path) or path.suffix.lower() in _SLICEABLE_EXTRA_SUFFIXES


def _unit_text(path: Path) -> str:
    """Full text of a file for slicing/reading, dispatched by file type.

    PDFs are binary, so route them through the pdf text extractor; everything
    else is read as UTF-8. Mirrors ``llm._file_to_text`` so a slice's character
    range matches the bytes the model is shown.
    """
    if path.suffix.lower() == ".pdf":
        from graphify.detect import extract_pdf_text
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _best_cut(text: str, start: int, end: int) -> int:
    """Return a cut index in ``(start, end]`` at the strongest nearby boundary.

    Searches the window ``text[start:end]`` for the latest heading, then blank
    line, then newline, then sentence end, and returns the index just *after* it
    (a heading cuts just *before* the ``#`` so the heading leads the next slice).
    A boundary counts only when the slice it leaves fills at least ``_MIN_FILL``
    of the window, so a strong separator near the start loses to a weaker one
    near the limit. Falls back to a hard cut at ``end`` when no boundary
    qualifies, which still makes forward progress because ``end > start``.
    """
    window = text[start:end]
    floor = len(window) * _MIN_FILL
    for sep in _BOUNDARY_SEPARATORS:
        idx = window.rfind(sep)
        if idx <= 0:  # no boundary strictly inside the window (non-empty prev slice)
            continue
        # a heading cuts before the ``#``, keeping the newline with this slice
        cut = idx + 1 if sep == "\n#" else idx + len(sep)
        if cut >= floor:
            return start + cut
    return end


def slice_boundaries(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` ranges covering all of ``text``, each ≤ max_chars.

    Ranges are gap-free and non-overlapping, so concatenating the slices
    reproduces ``text`` exactly — no content is dropped.
    """
    n = len(text)
    if n <= max_chars:
        return [(0, n)]
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        hard = min(pos + max_chars, n)
        end = _best_cut(text, pos, hard) if hard < n else n
        if end <= pos:  # defensive: never stall
            end = hard
        bounds.append((pos, end))
        pos = end
    return bounds


def expand_oversized_files(
    files: list[Path], max_chars: int
) -> list["Path | FileSlice"]:
    """Replace each oversized splittable-text file with a list of ``FileSlice``s.

    Files at or below ``max_chars`` (and all non-sliceable files like code) pass
    through unchanged as ``Path``, so behaviour is identical for everything that
    already fit. Unreadable files pass through untouched (the reader handles the
    error).
    """
    out: list["Path | FileSlice"] = []
    for f in files:
        if not is_sliceable(f):
            out.append(f)
            continue
        try:
            text = _unit_text(f)
        except OSError:
            out.append(f)
            continue
        if len(text) <= max_chars:
            out.append(f)
            continue
        ranges = slice_boundaries(text, max_chars)
        total = len(ranges)
        for i, (s, e) in enumerate(ranges):
            out.append(FileSlice(path=f, start=s, end=e, index=i, total=total))
    return out


def read_slice_text(fs: FileSlice) -> str:
    """Read just this slice's characters from its parent file."""
    text = _unit_text(fs.path)
    return text[fs.start:fs.end]


def bisect_slice(fs: FileSlice) -> tuple[FileSlice, FileSlice] | None:
    """Split a slice into two halves at a newline near its midpoint, or None.

    Used by the adaptive-retry path when a single slice still overflows the
    model's output: halving it produces a smaller response. Returns None when the
    slice is already too small to split meaningfully.
    """
    if fs.end - fs.start <= 1:
        return None
    try:
        text = _unit_text(fs.path)
    except OSError:
        return None
    mid = (fs.start + fs.end) // 2
    nl = text.find("\n", mid, fs.end)
    cut = nl + 1 if (nl != -1 and fs.start < nl + 1 < fs.end) else mid
    if not (fs.start < cut < fs.end):
        return None
    left = FileSlice(fs.path, fs.start, cut, fs.index, fs.total)
    right = FileSlice(fs.path, cut, fs.end, fs.index, fs.total)
    return left, right


def bisect_path(path: Path) -> "tuple[FileSlice, FileSlice] | None":
    """Emergency-slice a whole non-sliced file (code / structured data).

    Used by the adaptive-retry path when a single whole file overflows the
    model's output or context (A-hybrid): wrap the whole file as one slice and
    bisect it at a newline near the midpoint, so the file is retried in halves
    instead of dropped or kept partial. Returns None when the file is unreadable
    or too small to split. The resulting slices report ``path`` as their source,
    so nodes still merge by ``source_file`` exactly as for a whole-file pass.
    """
    try:
        text = _unit_text(path)
    except OSError:
        return None
    n = len(text)
    if n <= 1:
        return None
    return bisect_slice(FileSlice(path=path, start=0, end=n, index=0, total=1))
