"""Unit tests for preprocessing.utils.image_io's discovery ordering.

Covers the natural (numeric-aware) sort fix: numbered cookbook filenames must
sort in true page order, not plain lexicographic order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from preprocessing.utils.image_io import discover_source_images


def _make_image(path: Path, size=(20, 10)) -> Path:
    Image.new("RGB", size, (120, 60, 30)).save(path)
    return path


def test_discovery_sorts_numbered_pages_naturally(tmp_path):
    for name in [
        "pages-1.jpg", "pages-2.jpg", "pages-10.jpg",
        "pages-99.jpg", "pages-100.jpg", "pages-224.jpg",
    ]:
        _make_image(tmp_path / name)

    found = discover_source_images(tmp_path, [".jpg"])

    assert [p.name for p in found] == [
        "pages-1.jpg", "pages-2.jpg", "pages-10.jpg",
        "pages-99.jpg", "pages-100.jpg", "pages-224.jpg",
    ]
