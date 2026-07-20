"""Standalone entry point for the Image Preprocessing Pipeline.

This is a **standalone step run before extraction**. It reads scanned cookbook
photos from a read-only source directory (default ``phase_1/cookbook_images``),
threads each through an ordered list of stages — page split (1->N) followed by
adaptive-threshold white balance/binarization (1->1) — and writes fully
processed single-page images to a separate output directory with
deterministic, reading-order-preserving names.

To feed the existing image batch processor, point its ``image_dir`` at this
tool's ``output_dir``: the batch processor's sorted image discovery over that
directory yields the pages in the source book's reading order (Requirement 14.4).

The Phase 1 source images are never modified (Requirement 14.5).

This module now lives as a subpackage of ``image_batch_processor`` and uses
relative imports, so run it as a module rather than a bare script::

    uv run python -m preprocessing.main
"""

import logging
import os
import sys
from pathlib import Path

# Environment variables allowing the source/output directories to be
# overridden without editing this file, e.g. when invoked as a subprocess step
# from another pipeline (image_batch_processor's preprocessing subflow).
_ENV_SOURCE_DIR = "PREPROCESS_SOURCE_DIR"
_ENV_OUTPUT_DIR = "PREPROCESS_OUTPUT_DIR"

from .config.settings import (
    AdjustmentConfig,
    AdjustmentOperation,
    DewarpConfig,
    PageSplitConfig,
    PageSplitMethod,
    PipelineConfig,
    StageSpec,
    StageType,
)
from .core.pipeline import PreprocessingPipeline
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Resolve default paths against the workspace root so the entry point runs
# regardless of the current working directory. This file lives at
# ``phase_2/image_batch_processor/preprocessing/main.py``; the workspace root
# is four parents up.
_MODULE_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _MODULE_DIR.parent.parent.parent

# Source photos (Requirement 14.1): the Phase 1 cookbook images, read-only.
DEFAULT_SOURCE_DIR = _WORKSPACE_ROOT / "phase_1" / "cookbook_images"
# Output directory: kept under this tool's own module directory so it never
# overlaps the source directory (Requirement 10.5).
DEFAULT_OUTPUT_DIR = _MODULE_DIR / "output"
# A supported single-page output extension (Requirement 14.4).
DEFAULT_OUTPUT_FORMAT = ".jpg"


def build_default_config() -> PipelineConfig:
    """Build the default :class:`PipelineConfig` for the cookbook dataset.

    Uses the source directory ``phase_1/cookbook_images``, a configured output
    directory that does not overlap the source, and an ordered stage list of a
    page-split stage followed by editor-style adjustment stages (saturation,
    brightness, contrast, temperature, sharpen). The first and last source
    images (front/back cover photos) are skipped.

    Returns:
        A validated :class:`PipelineConfig`.

    Raises:
        ConfigurationError: If the constructed configuration is invalid (empty
            directory, empty stage list, unknown stage type, mismatched stage
            config, or an out-of-range parameter). Raised before any processing
            begins (Requirement 14.2).
    """
    stages = [
        # Stage 1: split each open-book spread into left then right pages.
        # Using fixed_midpoint rather than gutter_detection: most scans in this
        # dataset are split evenly at the center, and content-aware detection
        # can mistake a dark line of text for the spine shadow on some pages
        # (e.g. pages-140.jpg), producing an off-center split. split_ratio is
        # nudged slightly right of true center (0.5) because the scans in this
        # dataset consistently show the spine sitting a bit right of the
        # image's pixel midpoint.
        StageSpec(
            stage_type=StageType.PAGE_SPLIT,
            stage_config=PageSplitConfig(
                method=PageSplitMethod.FIXED_MIDPOINT,
                split_ratio=0.515,
                gutter_margin=0.0,
            ),
        ),
        # Stages 2+: editor-style adjustments, one slider per step, applied in
        # order. Tuned via experiment.py against sample pages:
        #   desaturate to grayscale, slight brightness/contrast trim, cool the
        #   color temperature, then sharpen.
        StageSpec(
            stage_type=StageType.ADJUSTMENT,
            stage_config=AdjustmentConfig(
                operation=AdjustmentOperation.SATURATION, amount=-100
            ),
        ),
        StageSpec(
            stage_type=StageType.ADJUSTMENT,
            stage_config=AdjustmentConfig(
                operation=AdjustmentOperation.BRIGHTNESS, amount=-3
            ),
        ),
        StageSpec(
            stage_type=StageType.ADJUSTMENT,
            stage_config=AdjustmentConfig(
                operation=AdjustmentOperation.CONTRAST, amount=3
            ),
        ),
        StageSpec(
            stage_type=StageType.ADJUSTMENT,
            stage_config=AdjustmentConfig(
                operation=AdjustmentOperation.TEMPERATURE, amount=-33
            ),
        ),
        StageSpec(
            stage_type=StageType.ADJUSTMENT,
            stage_config=AdjustmentConfig(
                operation=AdjustmentOperation.SHARPEN, amount=76
            ),
        ),
        # Final stage: rectilinear alignment. Straightens curved/foreshortened
        # text lines on the fully-adjusted image (text-line grid remap). Pages
        # with too little text (photos/decorative) pass through unchanged and
        # are logged for review.
        # Skipped for this run.
        # StageSpec(
        #     stage_type=StageType.DEWARP,
        #     stage_config=DewarpConfig(),
        # ),
    ]

    source_dir = os.environ.get(_ENV_SOURCE_DIR, str(DEFAULT_SOURCE_DIR))
    output_dir = os.environ.get(_ENV_OUTPUT_DIR, str(DEFAULT_OUTPUT_DIR))

    return PipelineConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        stages=stages,
        supported_extensions=[".jpg", ".jpeg", ".png"],
        output_format=DEFAULT_OUTPUT_FORMAT,
        # Drop the first/last pages (front/back cover photos).
        skip_first_last=True,
        # Process sources in parallel. This work is embarrassingly parallel and
        # PIL/numpy release the GIL during their C routines, so threads scale
        # well. Capped to keep peak memory reasonable (each worker holds a
        # full-size page image plus transient copies).
        max_workers=min(8, os.cpu_count() or 1),
    )


def _configure_logging() -> None:
    """Configure root logging for the run (console handler, INFO level)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """Run the standalone preprocessing pipeline over the cookbook images.

    Builds and validates the default configuration, aborting before any
    processing with a clear configuration error when validation fails
    (Requirement 14.2); runs the pipeline; and logs a summary of the report —
    total sources, total output files, successful sources, failed sources
    (Requirement 14.3).

    Returns:
        A process exit code: ``0`` on success, ``1`` when configuration
        validation fails (no output is written in that case).
    """
    _configure_logging()
    logger.info(
        "Starting Image Preprocessing Pipeline (standalone, runs before extraction)"
    )

    # Build + validate the configuration. Any ConfigurationError here means the
    # run must abort BEFORE processing any source image and write no output
    # (Requirement 14.2).
    try:
        config = build_default_config()
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        logger.error("Aborting before processing; no output was written.")
        return 1

    logger.info("Source directory: %s", config.source_dir)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Output format:    %s", config.output_format)
    logger.info(
        "Stages (in order): %s",
        " -> ".join(spec.stage_type.value for spec in config.stages),
    )
    logger.info(
        "Point the image batch processor's image_dir at the output directory "
        "above to consume these pages in reading order."
    )

    # Run the pipeline. A run-level naming collision surfaces as a
    # ConfigurationError; treat it as an abort with a clear message.
    try:
        report = PreprocessingPipeline(config).run()
    except ConfigurationError as exc:
        logger.error("Configuration error during run: %s", exc)
        logger.error("Aborting; no further output was written.")
        return 1

    # Log the report summary (Requirement 14.3).
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total sources:      %d", report.total_sources)
    logger.info("Total output files: %d", report.total_output_files)
    logger.info("Successful sources: %d", report.successful)
    logger.info("Failed sources:     %d", report.failed)
    logger.info("Success rate:       %.1f%%", report.success_rate() * 100)
    logger.info("Processing time:    %.2fs", report.processing_time)
    logger.info("=" * 60)

    # Surface per-source failures so they can be reviewed/re-processed.
    for result in report.results:
        if not result.success:
            logger.warning("Failed source: %s (%s)", result.source_path, result.error)

    return 0


if __name__ == "__main__":
    sys.exit(main())
