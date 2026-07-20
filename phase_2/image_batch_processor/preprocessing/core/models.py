"""Data models for the image preprocessing pipeline."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class WorkingImage:
    """
    One image as it flows through the pipeline (in-memory, not persisted).

    Carries the in-memory image data along with provenance metadata used to
    derive deterministic, order-preserving output names. Only the final working
    set of a run is written to disk.

    Attributes:
        source_name: Original source filename this image descends from.
        lineage: Ordered suffix tokens describing how the image was produced
            (e.g. a left/right token from splitting), used to build
            deterministic, order-preserving output names.
        width: Current pixel width.
        height: Current pixel height.
        image: The in-memory image data (e.g. a Pillow ``Image`` instance).
        fallback_used: True when this image was produced by a page split whose
            content-aware gutter detection found no confident gutter and fell
            back to the fixed ``fallback_ratio`` midpoint (Requirements 4.4,
            11.4). Defaults to False so it is only set on the outputs of a
            fallback split; all other images (covers, fixed-midpoint splits,
            1->1 stage outputs) leave it False.
    """

    source_name: str
    image: Any
    width: int
    height: int
    lineage: List[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class ImageResult:
    """
    Outcome of processing one source image through the whole pipeline.

    Mirrors ``ProcessingResult`` from the image batch processor.
    """

    source_path: str
    success: bool
    output_paths: List[str] = field(default_factory=list)
    output_count: int = 0
    stages_applied: List[str] = field(default_factory=list)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class PipelineReport:
    """
    Summary of a pipeline run across all source images.

    Mirrors ``BatchReport`` from the image batch processor, including a
    ``success_rate()`` accessor.
    """

    total_sources: int
    successful: int
    failed: int
    total_output_files: int
    processing_time: float
    results: List[ImageResult] = field(default_factory=list)

    def success_rate(self) -> float:
        """
        Calculate the fraction of sources processed successfully.

        Returns:
            ``successful / total_sources`` as a float between 0.0 and 1.0,
            or 0.0 when there are no sources.
        """
        if self.total_sources == 0:
            return 0.0
        return self.successful / self.total_sources
