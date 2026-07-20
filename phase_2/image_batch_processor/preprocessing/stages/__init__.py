"""Preprocessing stage implementations.

Each stage is an interchangeable implementation of the ``PreprocessingStage``
contract (Strategy pattern), analogous to ``ExtractionEngine`` in the image
batch processor.
"""

from .base import PreprocessingStage
from .page_split import PageSplitStage


__all__ = [
    "PreprocessingStage",
    "PageSplitStage",
]
