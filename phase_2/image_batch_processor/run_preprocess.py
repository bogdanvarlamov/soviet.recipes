"""Run only the image preprocessing subflow (page split, white balance, etc.).

This is a standalone entry point for the preprocessing step on its own,
without running any extraction engine afterward. Useful for pre-generating
(or re-generating) the preprocessed image directory that the ``*-preprocessed*``
poe tasks in ``pyproject.toml`` consume.

Configuration is via environment variables (same names used by ``main.py``):

    IMAGE_DIR                 Source images to preprocess
                               (default: ../../phase_1/cookbook_images)
    PREPROCESSING_OUTPUT_DIR  Where to write preprocessed images
                               (default: phase_2/preprocessed)
    FORCE_PREPROCESSING       Set to 1/true/yes to overwrite/re-run even if
                               PREPROCESSING_OUTPUT_DIR already has output

Usage:
    uv run python run_preprocess.py
    uv run poe preprocess
    uv run poe preprocess-force   # overwrite any existing preprocessed images
"""

import logging
import os
from pathlib import Path

from utils.logging import setup_logging
from utils.preprocessing import run_preprocessing_if_needed


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    image_dir_env = os.environ.get("IMAGE_DIR")
    image_dir = Path(image_dir_env) if image_dir_env else Path("../../phase_1/cookbook_images")

    # Default preprocessing output lives under phase_2/ (a build artifact of the
    # active development phase), not next to the raw Phase 1 source scans.
    # Anchor to this package's location so the default is independent of cwd.
    phase_2_dir = Path(__file__).resolve().parent.parent
    preprocessing_output_dir = os.environ.get("PREPROCESSING_OUTPUT_DIR") or str(
        phase_2_dir / "preprocessed"
    )

    force_preprocessing = os.environ.get("FORCE_PREPROCESSING", "").lower() in (
        "1", "true", "yes",
    )

    logger.info("Running preprocessing-only step")
    logger.info(f"Source directory: {image_dir}")
    logger.info(f"Output directory: {preprocessing_output_dir}")
    logger.info(f"Force (overwrite existing output): {force_preprocessing}")

    output_dir = run_preprocessing_if_needed(
        source_dir=str(image_dir),
        output_dir=preprocessing_output_dir,
        force=force_preprocessing,
        logger=logger,
    )

    logger.info(f"Preprocessing output ready at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
