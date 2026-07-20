"""Image I/O utility layer for the Image Preprocessing Pipeline.

Encapsulates all filesystem/image-decode concerns so the orchestrator and
stages stay free of I/O. Provides:

- :func:`load_image` — decode a source file into an in-memory image.
- :func:`get_dimensions` — report an image's ``(width, height)``.
- :func:`crop_region` — crop a rectangular region into a new image.
- :func:`save_image` — write an in-memory image to a destination path.
- :func:`discover_source_images` — find eligible source files in a directory.

Stages themselves perform no file I/O; loading and saving are the
orchestrator's responsibility via this module. All operations use Pillow for
image handling and :mod:`pathlib` for filesystem access.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError

from ..exceptions import ImageLoadError, ImageSaveError

# Splits a filename into alternating text/digit runs, e.g. "pages-12.jpg" ->
# ["pages-", "12", ".jpg"], so digit runs can be compared numerically below.
_DIGITS_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> List[object]:
    """Sort key that orders embedded digit runs numerically.

    Plain lexicographic sort compares filenames byte-by-byte, so
    "pages-100.jpg" would sort before "pages-2.jpg" (character '1' < '2'),
    which breaks the cookbook's true page order and the first/last cover-page
    skip logic that relies on it. This key splits each filename into
    text/digit runs and compares digit runs as integers, so "pages-2" <
    "pages-10" < "pages-99" < "pages-100" as expected.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _DIGITS_RE.split(path.name)
    ]


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image file into memory.

    The returned image is fully loaded (decoded) into memory so the caller does
    not depend on the file handle remaining open.

    Args:
        path: Path to the source image file.

    Returns:
        An in-memory :class:`PIL.Image.Image`.

    Raises:
        ImageLoadError: If the file does not exist, is not a file, or cannot be
            decoded as an image.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise ImageLoadError(str(image_path), "file does not exist")
    if not image_path.is_file():
        raise ImageLoadError(str(image_path), "path is not a file")

    try:
        with Image.open(image_path) as img:
            # Force decode while the file handle is open, then detach a copy so
            # the returned image is independent of the (now closed) file.
            img.load()
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(str(image_path), str(exc)) from exc


def get_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Report an image's pixel dimensions.

    Args:
        image: An in-memory image.

    Returns:
        A ``(width, height)`` tuple in pixels.
    """
    return image.width, image.height


def crop_region(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop a rectangular region from an image into a new image.

    Does not mutate the input image; Pillow's ``crop`` returns a new image.

    Args:
        image: The source in-memory image.
        box: A ``(left, upper, right, lower)`` box in pixel coordinates, where
            ``left``/``upper`` are inclusive and ``right``/``lower`` are
            exclusive (Pillow convention).

    Returns:
        A new :class:`PIL.Image.Image` containing the cropped region.

    Raises:
        ValueError: If the box is degenerate (``right <= left`` or
            ``lower <= upper``) or falls outside the image bounds.
    """
    left, upper, right, lower = box

    if right <= left or lower <= upper:
        raise ValueError(f"Degenerate crop box (zero or negative area): {box}")
    if left < 0 or upper < 0 or right > image.width or lower > image.height:
        raise ValueError(
            f"Crop box {box} exceeds image bounds {(image.width, image.height)}"
        )

    return image.crop((left, upper, right, lower))


def save_image(
    image: Image.Image,
    destination: Union[str, Path],
    image_format: Optional[str] = None,
) -> Path:
    """Save an in-memory image to a destination path.

    Creates the destination's parent directory (and any missing ancestors) if
    needed before writing.

    Args:
        image: The in-memory image to write.
        destination: Path where the image should be written.
        image_format: Optional explicit Pillow format (e.g. ``"JPEG"``,
            ``"PNG"``). When omitted, the format is inferred from the
            destination file extension.

    Returns:
        The :class:`pathlib.Path` the image was written to.

    Raises:
        ImageSaveError: If the output directory cannot be created or the image
            cannot be written.
    """
    dest_path = Path(destination)

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ImageSaveError(str(dest_path), f"could not create output directory: {exc}") from exc

    try:
        if image_format is not None:
            image.save(dest_path, format=image_format)
        else:
            image.save(dest_path)
    except (OSError, ValueError, KeyError) as exc:
        raise ImageSaveError(str(dest_path), str(exc)) from exc

    return dest_path


def discover_source_images(
    source_dir: Union[str, Path],
    supported_extensions: List[str],
) -> List[Path]:
    """Discover eligible source images directly within a directory.

    Discovery is:

    - **Non-recursive**: only files located directly in ``source_dir`` are
      considered; subdirectories and their contents are excluded.
    - **Case-insensitive** on extension: a file matches when its extension
      equals any configured supported extension, compared case-insensitively.
    - **Deterministic**: results are returned in a stable, natural
      (numeric-aware) order by filename, so repeating discovery over identical
      directory contents yields an identical sequence, and "pages-2" sorts
      before "pages-10" rather than after it (plain lexicographic order would
      place "pages-10" before "pages-2"). This matches the sort order the
      downstream batch processor uses, so a reading-order-preserving output
      naming scheme keeps global reading order intact.

    Args:
        source_dir: Directory to search (non-recursively).
        supported_extensions: Extensions to include, e.g. ``[".jpg", ".png"]``.
            Comparison is case-insensitive; a leading dot is expected but a
            missing one is tolerated.

    Returns:
        A naturally-sorted list of matching file paths. Empty when nothing
        matches.

    Raises:
        ImageLoadError: If the source directory does not exist or cannot be
            read (a domain error indicating the source directory is
            inaccessible). No discovered set is returned in this case.
    """
    directory = Path(source_dir)

    if not directory.exists():
        raise ImageLoadError(str(directory), "source directory does not exist")
    if not directory.is_dir():
        raise ImageLoadError(str(directory), "source path is not a directory")

    # Normalize configured extensions to lowercase with a leading dot.
    normalized_extensions = {
        (ext if ext.startswith(".") else f".{ext}").lower()
        for ext in supported_extensions
    }

    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise ImageLoadError(str(directory), f"source directory is not readable: {exc}") from exc

    matches = [
        entry
        for entry in entries
        if entry.is_file() and entry.suffix.lower() in normalized_extensions
    ]

    return sorted(matches, key=_natural_sort_key)
