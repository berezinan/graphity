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
#
# This set has to keep pace with ``detect.DOC_EXTENSIONS``: anything classified
# as a document reaches the semantic pass, and anything the pass sees that is
# NOT listed here is silently cut at ``_FILE_CHAR_CAP`` by ``_read_files``. The
# two lists drifted as DOC_EXTENSIONS grew — .qmd, .skill, .html, .yaml and .yml
# were documents that never got sliced, so a 38k-character one reached the model
# as its first 20k with no warning and no partial marker (#2900).
# ``tests/test_oversized_document_slicing.py`` pins the relationship so a future
# addition to DOC_EXTENSIONS fails loudly instead of quietly losing content.
_SPLITTABLE_TEXT_SUFFIXES = frozenset({
    ".md", ".mdx", ".markdown", ".txt", ".rst",
    ".qmd", ".skill", ".html", ".yaml", ".yml",
})

# Document types whose BYTES are not what the model is shown. `llm._file_to_text`
# routes these through a converter, so a character range has to be taken over the
# converted text, never over the file. Kept separate from the set above because
# they are splittable for a different reason and via a different reader.
_CONVERTED_TEXT_SUFFIXES = frozenset({".pdf"})


def _pdf_text(path: Path) -> str:
    """Extracted text of a PDF — the same string `llm._file_to_text` builds.

    Imported lazily from ``detect`` so this module keeps no import-time
    dependency on the extraction stack (``llm`` imports *this* module, so the
    reverse direction would be a cycle).
    """
    from graphify.detect import extract_pdf_text
    return extract_pdf_text(path)


# Slicing a PDF means extracting its text, and the slicing pass asks for the same
# file several times: once to measure it, then once per slice as the prompt is
# built. Memoised on (path, size, mtime_ns) so a corpus of papers is parsed once
# rather than once per slice, and so a file rewritten mid-run is re-read instead
# of served a stale body. Bounded: the entries are whole documents, and a large
# corpus should not pin all of them in memory.
_CONVERTED_TEXT_CACHE: "dict[tuple, str]" = {}
_CONVERTED_TEXT_CACHE_MAX = 64


def unit_source_text(path: Path) -> str:
    """The text a unit contributes to the prompt, whatever its container.

    Plain-text files are read directly; converted types (PDF) go through their
    converter. Both `expand_oversized_files` and `read_slice_text` use this, so
    the offsets a slice carries always index the same string the model sees.
    """
    if path.suffix.lower() not in _CONVERTED_TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return _pdf_text(path)
    hit = _CONVERTED_TEXT_CACHE.get(key)
    if hit is not None:
        return hit
    text = _pdf_text(path)
    if len(_CONVERTED_TEXT_CACHE) >= _CONVERTED_TEXT_CACHE_MAX:
        _CONVERTED_TEXT_CACHE.clear()
    _CONVERTED_TEXT_CACHE[key] = text
    return text

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
    """True for document types that may be sliced.

    Covers plain text read straight off disk and converted types (PDF) whose
    text is produced by a converter. Both are sliceable because
    :func:`unit_source_text` gives the slicing pass the same string the prompt
    will carry; what disqualifies a type is having no text at all (an image) or
    text the reader cannot address by character offset.
    """
    suffix = path.suffix.lower()
    return suffix in _SPLITTABLE_TEXT_SUFFIXES or suffix in _CONVERTED_TEXT_SUFFIXES


# Prose-like documents for the fork's per-category char cap (`_char_cap_for`).
# Deliberately NARROWER than `_SPLITTABLE_TEXT_SUFFIXES`, which #2900 widened to
# .qmd/.skill/.html/.yaml/.yml so an oversized one is sliced rather than cut.
# That answers a different question: those are structured data, and this fork
# reads them WHOLE (`_UNSLICED_CHAR_CAP`) so a config or template is never split
# across prompts. PDFs join because their extracted text is prose.
_PROSE_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".txt", ".rst"})
_SLICEABLE_EXTRA_SUFFIXES = frozenset({".pdf"})


def is_sliceable(path: Path) -> bool:
    """True for documents that get the slice-sized char cap rather than the
    whole-file one. Prose plus PDFs (read via text extraction); code and
    structured data are read whole to keep whole-symbol / whole-document context.
    """
    suffix = path.suffix.lower()
    return suffix in _PROSE_SUFFIXES or suffix in _SLICEABLE_EXTRA_SUFFIXES


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
            # The CONVERTED text for a PDF, so the boundaries below index the
            # same string read_slice_text will later slice and the prompt will
            # carry — not the container's bytes (#2906).
            text = unit_source_text(f)
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
    """Read just this slice's characters from its parent file.

    Goes through :func:`unit_source_text`, so a PDF slice indexes the extracted
    text rather than the container's bytes — the offsets `expand_oversized_files`
    computed and the string the prompt carries are then the same string (#2906).
    """
    return unit_source_text(fs.path)[fs.start:fs.end]


def bisect_slice(fs: FileSlice) -> tuple[FileSlice, FileSlice] | None:
    """Split a slice into two halves at a newline near its midpoint, or None.

    Used by the adaptive-retry path when a single slice still overflows the
    model's output: halving it produces a smaller response. Returns None when the
    slice is already too small to split meaningfully.
    """
    if fs.end - fs.start <= 1:
        return None
    # Index the SAME string the slice offsets were computed against and the
    # prompt carries: for a PDF that is the extracted text via unit_source_text,
    # not the raw container bytes. Reading the container here (the old behavior)
    # searched for the newline cut in binary coordinates, so a compressed PDF
    # slice could cut mid-line or past the text end (#2906). Any converter/read
    # failure means we cannot split, so fall back to None (treated as atomic).
    try:
        text = unit_source_text(fs.path)
    except Exception:
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
