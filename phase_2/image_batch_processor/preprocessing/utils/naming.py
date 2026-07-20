"""Deterministic output naming for the Image Preprocessing Pipeline.

Output names must keep the book's global reading order intact under the sorted
discovery the downstream batch processor performs, while never colliding. This
module provides two layers:

- A **pure per-image** name function, :func:`derive_output_name` (with the
  stem-only helper :func:`derive_output_stem`), that maps a source name plus an
  ordered lineage to an output name. It is a pure function of those two inputs,
  so the identical inputs always produce the identical name (Requirement 8.1),
  and lineage tokens emitted in reading order (left page ``a`` before right page
  ``b``) yield names whose byte-wise lexicographic order matches reading order
  (Requirement 8.2).
- A **batch** assignment function, :func:`assign_output_names`, that names an
  entire final working set. It guarantees that, across the whole dataset, every
  output derived from an earlier source sorts before every output derived from a
  later source (Requirement 8.3), that all assigned names are pairwise distinct
  (Requirement 8.4), and that any collision halts naming with a
  :class:`ConfigurationError` rather than silently overwriting (Requirement 8.5).

Global ordering (Req 8.3)
-------------------------
The requirement is stated relative to the pipeline's *source processing order*
(the order discovery yields sources in). The raw cookbook stems ``pages-1`` …
``pages-224`` do **not** sort into a shape that reliably preserves an arbitrary
processing order under plain lexicographic comparison — for example
``pages-10`` sorts before ``pages-2`` lexicographically, and a source stem that
is a prefix of another (``pages-1`` vs ``pages-10``) only sorts correctly for
particular separator/character combinations.

Rather than depend on the shape of the source stems, :func:`assign_output_names`
receives the working images in processing order and prefixes each output name
with a **fixed-width, zero-padded global source index** derived from that order.
Because every index shares the same width and is purely numeric, comparing two
names from *different* sources reduces to comparing their indices, so an
earlier-processed source's outputs always sort before a later one's, regardless
of stem content. Within a single source the shared index prefix is identical, so
the pure lineage suffix (``a`` before ``b``) decides the order and left-before-
right is preserved. The index is a deterministic function of the (deterministic)
discovery order, so the full names remain stable run-to-run for identical
source-directory contents, consistent with Requirement 8.1's determinism intent;
the source stem is retained in the name for provenance and human readability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..core.models import WorkingImage
from ..exceptions import ConfigurationError

# Lineage tokens the page-split stage assigns so the left (first-read) page name
# sorts before the right (second-read) page name (Requirement 8.2). ``"a" < "b"``
# byte-wise, so ``pages-N-a`` precedes ``pages-N-b``.
LEFT_PAGE_TOKEN = "a"
RIGHT_PAGE_TOKEN = "b"

# Separator between the source stem and each lineage token (e.g. ``pages-12-a``).
_LINEAGE_SEP = "-"
# Separator between the global source index prefix and the pure per-image name.
_INDEX_SEP = "-"
# Minimum width of the zero-padded global index prefix.
_MIN_INDEX_WIDTH = 1


def _stem_of(source_name: str) -> str:
    """Return the extension-free stem of a source name or path."""
    return Path(source_name).stem


def _extension_for(source_name: str, output_format: Optional[str]) -> str:
    """Resolve the output file extension (including the leading dot).

    Uses ``output_format`` when provided (e.g. ``"png"`` or ``".JPG"``),
    otherwise falls back to the source name's own extension. The result is
    lower-cased so naming stays deterministic regardless of input casing.
    """
    if output_format:
        ext = output_format.strip().lower().lstrip(".")
        return f".{ext}" if ext else ""
    return Path(source_name).suffix.lower()


def derive_output_stem(source_name: str, lineage: Sequence[str]) -> str:
    """Derive the extension-free output stem from a source name and lineage.

    Pure function of ``source_name`` and ``lineage`` (Requirement 8.1): the
    source stem followed by each lineage token, joined with ``-``. Lineage
    tokens supplied in reading order (left page ``a`` before right page ``b``)
    produce stems whose byte-wise lexicographic order matches reading order
    (Requirement 8.2).

    Args:
        source_name: Original source filename (or path) the image descends from.
        lineage: Ordered lineage tokens describing how the image was produced.

    Returns:
        The output stem, e.g. ``"pages-12-a"`` for ``("pages-12.jpg", ["a"])``
        or ``"pages-12"`` for ``("pages-12.jpg", [])``.
    """
    parts = [_stem_of(source_name), *(str(token) for token in lineage)]
    return _LINEAGE_SEP.join(parts)


def derive_output_name(
    source_name: str,
    lineage: Sequence[str],
    output_format: Optional[str] = None,
) -> str:
    """Derive the per-image output name (stem + lineage + extension).

    Pure function of ``source_name`` and ``lineage`` (plus the optional, fixed
    ``output_format``); identical inputs always yield the identical name
    (Requirement 8.1). Within a split, tokens ``a``/``b`` keep the left page's
    name ahead of the right page's (Requirement 8.2).

    Args:
        source_name: Original source filename (or path).
        lineage: Ordered lineage tokens.
        output_format: Optional output extension/format (e.g. ``"png"``). When
            omitted, the source name's extension is preserved.

    Returns:
        The output file name, e.g. ``"pages-12-a.jpg"``.
    """
    return derive_output_stem(source_name, lineage) + _extension_for(
        source_name, output_format
    )


def assign_output_names(
    working_images: Iterable[WorkingImage],
    output_format: Optional[str] = None,
) -> List[Tuple[WorkingImage, str]]:
    """Assign collision-free, order-preserving output names to a working set.

    ``working_images`` must be supplied in the pipeline's processing order:
    sources in discovery order, and within each source the working set in
    reading order (left page before right page). The returned assignments are in
    that same order, and each assigned name embeds a fixed-width, zero-padded
    global source index so that:

    - every output derived from an earlier source sorts before every output
      derived from a later source, byte-wise (Requirement 8.3);
    - within a source, the shared index prefix defers to the pure lineage suffix,
      keeping the left page's name before the right page's (Requirement 8.2).

    All assigned names are verified pairwise distinct (Requirement 8.4); if two
    working images would resolve to the same name, naming halts and no name is
    returned (Requirement 8.5).

    Args:
        working_images: The final working set, in processing/reading order.
        output_format: Optional output extension/format applied to every name.

    Returns:
        A list of ``(WorkingImage, output_name)`` pairs, in input order.

    Raises:
        ConfigurationError: If two working images resolve to the same output
            name (naming collision); no partial result is returned.
    """
    images = list(working_images)
    if not images:
        return []

    # Rank sources by first appearance, i.e. the pipeline's processing order.
    source_index: dict[str, int] = {}
    for image in images:
        if image.source_name not in source_index:
            source_index[image.source_name] = len(source_index)

    # A single fixed width for every prefix so that byte-wise comparison of the
    # zero-padded index equals numeric comparison (Requirement 8.3).
    index_width = max(_MIN_INDEX_WIDTH, len(str(len(source_index) - 1)))

    assignments: List[Tuple[WorkingImage, str]] = []
    assigned: dict[str, WorkingImage] = {}
    for image in images:
        prefix = f"{source_index[image.source_name]:0{index_width}d}"
        pure_name = derive_output_name(image.source_name, image.lineage, output_format)
        name = f"{prefix}{_INDEX_SEP}{pure_name}"

        existing = assigned.get(name)
        if existing is not None:
            raise ConfigurationError(
                "Output naming collision: "
                f"{name!r} would be produced by both "
                f"(source={existing.source_name!r}, lineage={list(existing.lineage)!r}) "
                f"and (source={image.source_name!r}, lineage={list(image.lineage)!r}); "
                "no output was written."
            )
        assigned[name] = image
        assignments.append((image, name))

    return assignments
