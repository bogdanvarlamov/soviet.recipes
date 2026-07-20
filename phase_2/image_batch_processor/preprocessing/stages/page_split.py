"""Page-split stage (1 to N).

Splits a photo of an open book into separate left and right single-page images.
This is the 1->N stage: an interior spread yields exactly two output images
(left before right), while a designated cover passes through as exactly one
output.

Two spine-location approaches are selected by config:

- ``fixed_midpoint`` — split at ``split_ratio`` x width measured from the left
  edge (implemented here).
- ``gutter_detection`` — locate the darker vertical fold band within a central
  search region via a per-column intensity profile (numpy), with automatic
  fallback to ``fallback_ratio`` (and a recorded fallback-used indicator) when
  no confident gutter is found (see
  :meth:`PageSplitStage._detect_gutter_column`).

Regardless of the spine-location approach, the left and right page regions are
derived the same way: a configurable ``gutter_margin`` is trimmed symmetrically
around the spine so the spine shadow is excluded from both pages, and the two
resulting regions are guaranteed to lie within the source image's pixel bounds
and to not horizontally overlap.

Output ordering follows the deterministic naming scheme: the left page carries
the lineage token ``"a"`` and the right page the token ``"b"``, so that within a
source the left (first-read) page sorts before the right (second-read) page.
"""

from typing import List, Tuple

import numpy as np

from ..config.settings import PageSplitConfig, PageSplitMethod
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage
from ..utils.image_io import crop_region, get_dimensions

STAGE_TYPE = "page_split"

# Lineage tokens for the two halves of a split. "a" sorts before "b" in
# byte-wise lexicographic order, keeping the left (first-read) page ahead of the
# right (second-read) page in the deterministic output-naming scheme.
_LEFT_TOKEN = "a"
_RIGHT_TOKEN = "b"

# Confidence thresholds for content-aware gutter detection. The spine casts a
# shadow, so it appears as a localized dip (dark column band) in the per-column
# intensity profile. The dip depth is measured against the band's *median*
# intensity, used as a robust background level: unlike the mean or standard
# deviation, the median is not dragged down by a narrow, deep spine dip, so it
# reflects the true page brightness. A gutter is only "confident" when the
# darkest column is meaningfully below that background by BOTH measures:
#   - ``_GUTTER_ABS_MARGIN`` — at least this many intensity levels (0-255 scale)
#     below the median. Rejects flat / uniform bands whose darkest column is
#     essentially the background.
#   - ``_GUTTER_REL_DEPTH`` — at least this fraction of the background level
#     below the median. Rejects gentle lighting gradients and mild paper-texture
#     noise (small relative dips), which have no true spine and instead trigger
#     the midpoint fallback.
_GUTTER_ABS_MARGIN = 8.0
_GUTTER_REL_DEPTH = 0.15


class PageSplitStage(PreprocessingStage):
    """Split open-book spreads into left and right pages (1 to N).

    Covers (source filenames listed in ``config.cover_filenames``) are carried
    through unsplit as exactly one output. Every other image is treated as an
    interior spread and split into a left page followed by a right page.
    """

    def __init__(self, config: PageSplitConfig):
        """Create the stage with its validated configuration.

        Args:
            config: The page-split configuration.
        """
        self._config = config

    @property
    def stage_type(self) -> str:
        """Return the stage type identifier ``"page_split"``."""
        return STAGE_TYPE

    @property
    def config(self) -> PageSplitConfig:
        """Return the stage's configuration."""
        return self._config

    def validate_config(self) -> bool:
        """Validate that the stage is properly configured.

        Field-level ranges are already enforced by :class:`PageSplitConfig` at
        construction time. Here we additionally enforce the cross-field rule the
        stage depends on: the central search band must be well-formed
        (``search_band_max`` strictly greater than ``search_band_min``).

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the search band is not well-formed.
        """
        if self._config.search_band_max <= self._config.search_band_min:
            raise ConfigurationError(
                "page_split.search_band_max = "
                f"{self._config.search_band_max!r} must be greater than "
                f"search_band_min = {self._config.search_band_min!r}"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """Split each interior spread into two pages; pass covers through as one.

        Processes each input image in order. Outputs derived from an earlier
        input sort before outputs derived from a later input, and the two halves
        of a single spread are emitted left page before right page (Requirement
        4.1). Input images are never mutated in place (Requirement 3.1).

        Args:
            working_set: The ordered input working set.

        Returns:
            A new ordered working set. An interior spread contributes exactly two
            images (left then right); a cover contributes exactly one.

        Raises:
            StageError: If a computed left or right region would be less than
                1 pixel wide (Requirement 4.8). The input images are left
                unmodified and no working set is returned.
        """
        output: List[WorkingImage] = []
        for image in working_set:
            if self._is_cover(image):
                output.append(self._passthrough_cover(image))
            else:
                left, right = self._split_spread(image)
                output.append(left)
                output.append(right)
        return output

    def _is_cover(self, image: WorkingImage) -> bool:
        """Return True when the image is a designated cover (single page).

        A cover is identified by its ``source_name`` appearing in the configured
        ``cover_filenames``. First/last cover designation
        (``treat_first_last_as_covers``) depends on global source ordering that a
        stage does not observe, so it is resolved by the orchestrator upstream;
        this stage only matches explicit ``cover_filenames``.
        """
        return image.source_name in self._config.cover_filenames

    def _passthrough_cover(self, image: WorkingImage) -> WorkingImage:
        """Produce a single unsplit output for a cover (Requirement 4.6).

        Returns a new :class:`WorkingImage` wrapping a copy of the input image so
        the input is not mutated in place and lineage is left unchanged (the 1->1
        naming keeps the source stem).
        """
        cover_image = image.image.copy()
        width, height = get_dimensions(cover_image)
        return WorkingImage(
            source_name=image.source_name,
            image=cover_image,
            width=width,
            height=height,
            lineage=list(image.lineage),
        )

    def _split_spread(
        self, image: WorkingImage
    ) -> Tuple[WorkingImage, WorkingImage]:
        """Split an interior spread into a left page and a right page.

        Computes the spine column, trims ``gutter_margin`` x width / 2 pixels from
        each side of the spine, and crops the two non-overlapping regions that lie
        within the image bounds (Requirements 4.2, 4.5, 4.7).

        Returns:
            A ``(left, right)`` tuple of new :class:`WorkingImage` instances.

        Raises:
            StageError: If either region would be less than 1 pixel wide
                (Requirement 4.8).
        """
        width, height = get_dimensions(image.image)

        spine, fallback_used = self._locate_spine_column(image, width)
        margin_px = int(round(self._config.gutter_margin * width / 2.0))

        # Region boundaries. Pillow crop boxes are (left, upper, right, lower)
        # with the right/lower edges exclusive.
        left_right_edge = spine - margin_px
        right_left_edge = spine + margin_px

        # Clamp to the image bounds so both regions lie entirely within the
        # source image (Requirement 4.7).
        left_right_edge = max(0, min(left_right_edge, width))
        right_left_edge = max(0, min(right_left_edge, width))

        left_width = left_right_edge - 0
        right_width = width - right_left_edge

        if left_width < 1 or right_width < 1:
            # Emit no output and leave the source unchanged (Requirement 4.8).
            raise StageError(
                "computed page region is less than 1 pixel wide "
                f"(image width {width}, spine {spine}, margin {margin_px}px, "
                f"left width {left_width}, right width {right_width})",
                stage_name=STAGE_TYPE,
            )

        left_image = crop_region(image.image, (0, 0, left_right_edge, height))
        right_image = crop_region(image.image, (right_left_edge, 0, width, height))

        left = self._make_output(image, left_image, _LEFT_TOKEN, fallback_used)
        right = self._make_output(image, right_image, _RIGHT_TOKEN, fallback_used)
        return left, right

    def _make_output(
        self,
        source: WorkingImage,
        cropped_image,
        token: str,
        fallback_used: bool = False,
    ) -> WorkingImage:
        """Wrap a cropped page image in a new :class:`WorkingImage`.

        Appends ``token`` to a copy of the source lineage so output names sort in
        reading order (left ``"a"`` before right ``"b"``). Propagates the
        ``fallback_used`` indicator onto the output so both halves of a
        fallback split record that the fixed-midpoint fallback was used
        (Requirement 4.4).
        """
        width, height = get_dimensions(cropped_image)
        return WorkingImage(
            source_name=source.source_name,
            image=cropped_image,
            width=width,
            height=height,
            lineage=list(source.lineage) + [token],
            fallback_used=fallback_used,
        )

    def _locate_spine_column(
        self, image: WorkingImage, width: int
    ) -> Tuple[int, bool]:
        """Locate the spine (split) pixel column for a spread.

        Selects the approach from ``config.method``:

        - ``fixed_midpoint`` — ``round(split_ratio x width)`` (Requirement 4.2).
        - ``gutter_detection`` — delegated to :meth:`_detect_gutter_column`.

        Args:
            image: The spread being split.
            width: The image width in pixels.

        Returns:
            A ``(spine_column, fallback_used)`` tuple. ``spine_column`` is
            measured from the left edge. ``fallback_used`` is True only when the
            ``gutter_detection`` method found no confident gutter and fell back
            to ``fallback_ratio`` (Requirement 4.4); the ``fixed_midpoint`` path
            always reports False.
        """
        if self._config.method == PageSplitMethod.GUTTER_DETECTION:
            return self._detect_gutter_column(image, width)
        return self._fixed_midpoint_column(width), False

    def _fixed_midpoint_column(self, width: int) -> int:
        """Return the fixed-midpoint spine column ``round(split_ratio x width)``."""
        return int(round(self._config.split_ratio * width))

    def _detect_gutter_column(
        self, image: WorkingImage, width: int
    ) -> Tuple[int, bool]:
        """Locate the spine via content-aware gutter detection.

        The open book's fold casts a shadow, so the spine appears as the darkest
        vertical column band. This method builds a per-column intensity profile
        (mean grayscale intensity of every column, via numpy) and searches
        **only** within the central band bounded by the pixel columns at
        ``search_band_min x width`` and ``search_band_max x width`` measured from
        the left edge (Requirement 4.3). The darkest column in that band is the
        gutter candidate, and — being taken from within the band — it is
        guaranteed to lie within the configured search band.

        A candidate is accepted only when it is *confident*: the darkest column
        must be meaningfully darker than the band's background level (its median
        intensity) by both an absolute margin and a relative fraction of that
        background (see :data:`_GUTTER_ABS_MARGIN` and :data:`_GUTTER_REL_DEPTH`).
        A flat or low-contrast band produces no confident dip, so detection
        falls back to ``round(fallback_ratio x width)`` and reports that the
        fallback was used (Requirements 4.4, 11.4).

        Args:
            image: The spread being split.
            width: The image width in pixels.

        Returns:
            A ``(spine_column, fallback_used)`` tuple. When a confident gutter is
            found, ``fallback_used`` is False and ``spine_column`` lies within
            the search band. Otherwise ``spine_column`` is the fixed
            ``fallback_ratio`` column and ``fallback_used`` is True.
        """
        # Central search band, as pixel columns from the left edge (Req 4.3).
        band_start = int(round(self._config.search_band_min * width))
        band_end = int(round(self._config.search_band_max * width))

        # Clamp to valid column indices so slicing stays within the image.
        band_start = max(0, min(band_start, width))
        band_end = max(0, min(band_end, width))

        # A degenerate band (e.g. rounded to zero width on a tiny image) offers
        # nothing to search; fall back to the fixed midpoint.
        if band_end - band_start < 1:
            return self._fallback_column(width), True

        # Per-column mean intensity profile over the whole image height. Convert
        # to single-channel grayscale so the profile is a 1-D array of one mean
        # per column regardless of the source color mode.
        gray = np.asarray(image.image.convert("L"), dtype=np.float64)
        column_profile = gray.mean(axis=0)

        band = column_profile[band_start:band_end]
        # Median as a robust background level: a narrow, deep spine dip does not
        # drag it down the way the mean would.
        background = float(np.median(band))
        min_index = int(np.argmin(band))  # first darkest column (deterministic)
        min_value = float(band[min_index])

        # How far the darkest column dips below the band's background level.
        depth = background - min_value

        confident = (
            depth >= _GUTTER_ABS_MARGIN
            and depth >= _GUTTER_REL_DEPTH * background
        )
        if not confident:
            return self._fallback_column(width), True

        spine = band_start + min_index
        return spine, False

    def _fallback_column(self, width: int) -> int:
        """Return the fallback spine column ``round(fallback_ratio x width)``."""
        return int(round(self._config.fallback_ratio * width))
