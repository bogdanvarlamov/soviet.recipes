"""Experiment harness for tuning the preprocessing recipe on a few pages.

This is a **development/tuning tool**, separate from ``main.py``. It runs an
ordered recipe of stages against a small set of sample source pages and writes
the image *after every step* so you can visually inspect each modification's
contribution, then comment out / reorder / retune individual steps.

How to use
----------
1. Edit ``SAMPLE_SOURCES`` to pick which source pages to preview.
2. Edit ``build_recipe()`` below — it returns the ordered list of stages applied
   *after* the page split. Comment out a line to drop that step, change an
   ``amount`` to retune it, or reorder the lines to change the order.
3. Run::

       uv run python experiment.py

4. Look in ``experiments/<page>-<half>/`` — you'll find one PNG per step,
   numbered in application order (``00_source`` is the raw split page, then
   ``01_...``, ``02_...`` for each recipe step), so you can flip through them
   and see exactly what each adjustment did.

Nothing here writes to the Phase 1 source directory; it only reads from it.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from .config.settings import (
    AdjustmentConfig,
    AdjustmentOperation,
    DeskewConfig,
    DewarpConfig,
    PageSplitConfig,
    PageSplitMethod,
    StageSpec,
    StageType,
    WhiteBalanceConfig,
    WhiteBalanceMethod,
)
from .core.factory import StageFactory
from .core.models import WorkingImage
from .stages.page_split import PageSplitStage
from .utils.image_io import load_image, save_image

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _MODULE_DIR.parent.parent.parent
SOURCE_DIR = _WORKSPACE_ROOT / "phase_1" / "cookbook_images"
EXPERIMENTS_DIR = _MODULE_DIR / "experiments"
# All combined final results (whole recipe applied in one go) are collected
# here, one file per sample half-page, so they're easy to compare side by side.
FINAL_DIR = EXPERIMENTS_DIR / "_final"

# A small, representative sample. Add/remove filenames to taste.
SAMPLE_SOURCES = [
    "pages-12.jpg",
    "pages-30.jpg",
    "pages-100.jpg",
    "pages-150.jpg",
]

# Page split applied first (1 -> 2). Kept fixed for the experiments so the
# recipe below operates on real single pages.
PAGE_SPLIT = PageSplitConfig(
    method=PageSplitMethod.FIXED_MIDPOINT,
    split_ratio=0.53,
    gutter_margin=0.0,
)


def build_recipe() -> List[StageSpec]:
    """Return the ordered list of stages applied after the page split.

    EDIT THIS. Each entry is one step. Comment out a line to disable that step,
    change an ``amount`` to retune it, or reorder lines to change the order.

    The defaults mirror the reference editor's Adjustments panel
    (Brightness -53, Contrast -23, Saturation -100, Highlights -100,
    Shadows -100, Temperature -33, Sharpness 76), applied in panel order.
    """
    return [
        # -- editor "Adjustments" panel, one slider per step ----------------
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
        # StageSpec(
        #     stage_type=StageType.ADJUSTMENT,
        #     stage_config=AdjustmentConfig(
        #         operation=AdjustmentOperation.HIGHLIGHTS, amount=-100
        #     ),
        # ),
        # StageSpec(
        #     stage_type=StageType.ADJUSTMENT,
        #     stage_config=AdjustmentConfig(
        #         operation=AdjustmentOperation.SHADOWS, amount=-100
        #     ),
        # ),
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
        # -- rectilinear alignment LAST: straighten text lines on the final,
        #    already-adjusted/sharpened image (also cleaner to detect text on).
        #    Corrects curvature + per-page perspective foreshortening; image/
        #    decorative pages with too little text pass through unchanged.
        StageSpec(
            stage_type=StageType.DEWARP,
            stage_config=DewarpConfig(),
        ),
        # -- optional: binarize at the end for comparison -------------------
        # StageSpec(
        #     stage_type=StageType.WHITE_BALANCE,
        #     stage_config=WhiteBalanceConfig(
        #         method=WhiteBalanceMethod.ADAPTIVE_THRESHOLD
        #     ),
        # ),
    ]


def _step_label(spec: StageSpec, index: int) -> str:
    """Build a short, filename-safe label for a recipe step."""
    cfg = spec.stage_config
    if isinstance(cfg, AdjustmentConfig):
        detail = f"{cfg.operation.value}_{int(cfg.amount):+d}"
    elif isinstance(cfg, WhiteBalanceConfig):
        detail = f"whitebalance_{cfg.method.value}"
    else:
        detail = spec.stage_type.value
    return f"{index:02d}_{detail}"


def _split_pages(source_name: str) -> List[Tuple[str, WorkingImage]]:
    """Load a source page and split it into labeled ('a'/'b') half pages."""
    source_path = SOURCE_DIR / source_name
    image = load_image(source_path)
    seed = WorkingImage(
        source_name=source_name,
        image=image,
        width=image.width,
        height=image.height,
    )
    split_stage = PageSplitStage(PAGE_SPLIT)
    halves = split_stage.apply([seed])
    labels = ["a", "b"] if len(halves) == 2 else [str(i) for i in range(len(halves))]
    return list(zip(labels, halves))


def run_experiment() -> None:
    """Run the recipe over the sample pages, writing per-step previews."""
    recipe = build_recipe()
    # Validate/build the stages once (mirrors the real pipeline's factory use).
    stages = StageFactory.create_stages(recipe)

    if EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)

    logger.info("Recipe (%d step(s) after page split):", len(stages))
    for i, spec in enumerate(recipe, start=1):
        logger.info("  %s", _step_label(spec, i))

    total_previews = 0
    for source_name in SAMPLE_SOURCES:
        stem = Path(source_name).stem
        for half_label, page in _split_pages(source_name):
            out_dir = EXPERIMENTS_DIR / f"{stem}-{half_label}"

            # Step 0: the raw split page, before any recipe step.
            current = [page]
            save_image(current[0].image, out_dir / "00_source.png")
            total_previews += 1

            # Apply each recipe step, saving the cumulative result after it.
            for i, (spec, stage) in enumerate(zip(recipe, stages), start=1):
                current = stage.apply(current)
                label = _step_label(spec, i)
                save_image(current[0].image, out_dir / f"{label}.png")
                total_previews += 1

            # The combined final result (all steps applied in one go). Saved
            # both alongside the per-step previews and collected in _final/ for
            # easy side-by-side comparison across pages.
            final_image = current[0].image
            save_image(final_image, out_dir / "final.png")
            save_image(final_image, FINAL_DIR / f"{stem}-{half_label}.png")
            total_previews += 1

            logger.info("Wrote %s previews -> %s", stem + "-" + half_label, out_dir)

    logger.info(
        "Done: %d preview image(s) across %d sample page(s) in %s",
        total_previews,
        len(SAMPLE_SOURCES),
        EXPERIMENTS_DIR,
    )
    logger.info("Combined final outputs collected in %s", FINAL_DIR)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
