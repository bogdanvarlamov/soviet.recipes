"""Optional preprocessing subflow integration.

Runs the ``preprocessing`` subpackage (page split, white balance, etc.)
in-process before extraction, and points the batch processor's ``image_dir``
at its output.

The preprocessing pipeline was originally a separate ``uv``-managed project
with its own virtual environment; it has since been merged into
``image_batch_processor`` as the ``preprocessing`` subpackage (its
dependencies — Pillow, numpy, OpenCV, page-dewarp — do not conflict with the
batch processor's own dependencies), so it now runs as a direct Python import
rather than a subprocess.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from exceptions import ConfigurationError
from utils.file_utils import discover_images

from preprocessing.core.pipeline import PreprocessingPipeline
from preprocessing.exceptions import ConfigurationError as PreprocessingConfigurationError
from preprocessing.main import build_default_config as _build_default_preprocessing_config

# Environment variables preprocessing.main.build_default_config() reads to
# override its default source/output directories. Set them here (rather than
# constructing a PipelineConfig directly) so the default pipeline's own
# validation and stage recipe stay the single source of truth.
_ENV_SOURCE_DIR = "PREPROCESS_SOURCE_DIR"
_ENV_OUTPUT_DIR = "PREPROCESS_OUTPUT_DIR"

# Extensions the preprocessing pipeline may write and that the batch
# processor can subsequently discover.
_PREPROCESSED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def _has_existing_output(output_dir: Path) -> bool:
    """Return True when ``output_dir`` already contains preprocessed images."""
    try:
        return len(discover_images(output_dir, _PREPROCESSED_EXTENSIONS)) > 0
    except ValueError:
        # Directory does not exist (or is not a directory): no prior output.
        return False


def run_preprocessing_if_needed(
    source_dir: str,
    output_dir: str,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Run the image preprocessing pipeline, skipping it if possible.

    If ``output_dir`` already contains preprocessed images and ``force`` is
    False, the pipeline run is skipped entirely and the existing output is
    reused. Otherwise the ``preprocessing`` subpackage's default page-split +
    adjustment pipeline is run in-process against ``source_dir``, writing to
    ``output_dir``.

    Args:
        source_dir: Directory of raw images to preprocess.
        output_dir: Directory the preprocessing pipeline should write to.
        force: When True, always re-run preprocessing even if output already
            exists.
        logger: Logger to use (a module logger is used when omitted).

    Returns:
        The resolved output directory, ready to be used as the batch
        processor's ``image_dir``.

    Raises:
        ConfigurationError: If the pipeline configuration is invalid or the
            run itself fails.
    """
    log = logger or logging.getLogger(__name__)
    output_path = Path(output_dir)

    if not force and _has_existing_output(output_path):
        log.info(
            "Preprocessing output already exists at %s; skipping preprocessing "
            "step (set force_preprocessing=True to re-run).",
            output_path,
        )
        return output_path

    log.info(
        "Running preprocessing subflow: source=%s output=%s (force=%s)",
        source_dir,
        output_path,
        force,
    )

    previous_source_env = os.environ.get(_ENV_SOURCE_DIR)
    previous_output_env = os.environ.get(_ENV_OUTPUT_DIR)
    os.environ[_ENV_SOURCE_DIR] = str(source_dir)
    os.environ[_ENV_OUTPUT_DIR] = str(output_path)
    try:
        config = _build_default_preprocessing_config()
        report = PreprocessingPipeline(config, logger=log).run()
    except PreprocessingConfigurationError as exc:
        raise ConfigurationError(f"Preprocessing pipeline configuration error: {exc}")
    finally:
        if previous_source_env is None:
            os.environ.pop(_ENV_SOURCE_DIR, None)
        else:
            os.environ[_ENV_SOURCE_DIR] = previous_source_env
        if previous_output_env is None:
            os.environ.pop(_ENV_OUTPUT_DIR, None)
        else:
            os.environ[_ENV_OUTPUT_DIR] = previous_output_env

    log.info(
        "Preprocessing subflow complete: %d/%d source(s) succeeded, "
        "%d output file(s) written to %s",
        report.successful,
        report.total_sources,
        report.total_output_files,
        output_path,
    )
    return output_path
