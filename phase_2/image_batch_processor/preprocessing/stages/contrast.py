"""Contrast enhancement stage (1 to 1).

Increases the contrast between text and page background so downstream OCR/VLM
extraction is more accurate. This is a pure per-image transform: each input
:class:`WorkingImage` maps to exactly one output image with identical pixel
dimensions (Requirements 5.1, 5.2).

Three methods are supported, selected by the stage's
:class:`ContrastEnhancementConfig`:

- ``linear`` — a global linear contrast stretch scaled by ``factor``, using
  Pillow's :class:`PIL.ImageEnhance.Contrast` (Requirement 5.3).
- ``histogram_equalization`` — global histogram equalization via
  :func:`PIL.ImageOps.equalize` (Requirement 5.5).
- ``adaptive`` — a CLAHE-style (Contrast Limited Adaptive Histogram
  Equalization) local-contrast transform implemented with NumPy and bounded by
  ``clip_limit`` (Requirement 5.4).

The stage never performs file I/O and never mutates its input images in place;
it produces new :class:`WorkingImage` instances carrying a copy of the input
lineage. If a method cannot process an input image's color mode, the stage
raises a :class:`StageError` identifying itself rather than returning an output
image (Requirement 5.7).
"""

from typing import List

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ..config.settings import ContrastEnhancementConfig, ContrastMethod
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage

_STAGE_TYPE = "contrast_enhancement"

# Color modes each method can process. Modes outside these sets cause the stage
# to raise a StageError (Requirement 5.7).
_LINEAR_SUPPORTED_MODES = frozenset({"L", "LA", "RGB", "RGBA"})
_EQUALIZE_SUPPORTED_MODES = frozenset({"L", "RGB"})
_ADAPTIVE_SUPPORTED_MODES = frozenset({"L", "RGB"})


class ContrastEnhancementStage(PreprocessingStage):
    """A 1 to 1 stage that increases text/background contrast.

    Each input image is mapped to exactly one output image of identical
    dimensions. The concrete transform is chosen by the configured ``method``.
    """

    def __init__(self, config: ContrastEnhancementConfig):
        """Create the stage from its validated configuration.

        Args:
            config: The contrast-enhancement configuration (method plus the
                ``factor`` and ``clip_limit`` parameters).
        """
        self._config = config

    @property
    def stage_type(self) -> str:
        """Return the stage type identifier ``"contrast_enhancement"``."""
        return _STAGE_TYPE

    def validate_config(self) -> bool:
        """Validate the stage is properly configured.

        The Pydantic ``ContrastEnhancementConfig`` already enforces the field
        ranges (``factor > 0``, ``clip_limit > 0``) on construction. This method
        additionally confirms the injected config is of the expected type and
        the method is a recognized enum value.

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the configuration is not a
                :class:`ContrastEnhancementConfig` or the method is unknown.
        """
        if not isinstance(self._config, ContrastEnhancementConfig):
            raise ConfigurationError(
                "contrast_enhancement stage requires a ContrastEnhancementConfig, "
                f"got {type(self._config).__name__}"
            )
        if not isinstance(self._config.method, ContrastMethod):
            raise ConfigurationError(
                f"contrast_enhancement.method = {self._config.method!r} is not a "
                "recognized ContrastMethod"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """Enhance the contrast of every image in the working set.

        Produces exactly one output image per input image, in the same order,
        each with dimensions identical to its corresponding input. Input images
        are not mutated in place.

        Args:
            working_set: The ordered input working set.

        Returns:
            A new working set with one enhanced ``WorkingImage`` per input.

        Raises:
            StageError: If an input image's color mode cannot be processed by
                the configured method, or the transform otherwise fails. The
                error identifies this stage and no working set is returned.
        """
        return [self._process_one(item) for item in working_set]

    # -- internal helpers -------------------------------------------------

    def _process_one(self, item: WorkingImage) -> WorkingImage:
        """Transform a single working image into a new working image."""
        image = item.image
        if not isinstance(image, Image.Image):
            raise StageError(
                f"expected an in-memory Pillow image, got {type(image).__name__}",
                stage_name=_STAGE_TYPE,
            )

        method = self._config.method
        self._check_mode_supported(method, image.mode)

        try:
            if method == ContrastMethod.LINEAR:
                result = ImageEnhance.Contrast(image).enhance(self._config.factor)
            elif method == ContrastMethod.HISTOGRAM_EQUALIZATION:
                result = ImageOps.equalize(image)
            elif method == ContrastMethod.ADAPTIVE:
                result = self._apply_adaptive(image)
            else:  # pragma: no cover - guarded by validate_config
                raise StageError(
                    f"unsupported contrast method {method!r}",
                    stage_name=_STAGE_TYPE,
                )
        except StageError:
            raise
        except Exception as exc:  # transform failed (e.g. incompatible mode)
            raise StageError(
                f"failed to apply {method.value} contrast to a "
                f"{image.mode} image: {exc}",
                stage_name=_STAGE_TYPE,
            ) from exc

        # Dimensions must be preserved (Requirement 5.2); guard defensively.
        if result.size != image.size:
            raise StageError(
                f"contrast transform changed dimensions from {image.size} to "
                f"{result.size}",
                stage_name=_STAGE_TYPE,
            )

        return WorkingImage(
            source_name=item.source_name,
            image=result,
            width=result.width,
            height=result.height,
            lineage=list(item.lineage),
        )

    @staticmethod
    def _check_mode_supported(method: ContrastMethod, mode: str) -> None:
        """Raise a StageError if ``method`` cannot process ``mode``."""
        if method == ContrastMethod.LINEAR:
            supported = _LINEAR_SUPPORTED_MODES
        elif method == ContrastMethod.HISTOGRAM_EQUALIZATION:
            supported = _EQUALIZE_SUPPORTED_MODES
        else:
            supported = _ADAPTIVE_SUPPORTED_MODES

        if mode not in supported:
            raise StageError(
                f"{method.value} contrast cannot process color mode {mode!r}; "
                f"supported modes are {sorted(supported)}",
                stage_name=_STAGE_TYPE,
            )

    def _apply_adaptive(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE-style adaptive contrast bounded by ``clip_limit``.

        For grayscale (``L``) images the transform is applied directly to the
        single channel. For ``RGB`` images it is applied to the luminance (``Y``)
        channel in ``YCbCr`` space so chrominance is preserved and no color cast
        is introduced. Output dimensions and color mode always match the input.
        """
        clip_limit = self._config.clip_limit

        if image.mode == "L":
            channel = np.asarray(image, dtype=np.uint8)
            equalized = _clahe_channel(channel, clip_limit)
            return Image.fromarray(equalized, mode="L")

        # RGB: operate on the luminance channel only.
        ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8).copy()
        ycbcr[:, :, 0] = _clahe_channel(ycbcr[:, :, 0], clip_limit)
        return Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")


def _clahe_channel(
    channel: np.ndarray,
    clip_limit: float,
    grid: int = 8,
    n_bins: int = 256,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on one 2D channel.

    The channel is divided into a grid of contextual tiles. Each tile's
    histogram is clipped at ``clip_limit`` times the average bin count (the
    clipped mass is redistributed uniformly), then converted to a mapping via
    its cumulative distribution. Per-pixel output is bilinearly interpolated
    between the mappings of the four neighbouring tile centres to avoid block
    artifacts. The clip limit bounds how much local contrast is amplified
    (Requirement 5.4).

    Args:
        channel: A 2D ``uint8`` array (a single image channel).
        clip_limit: Contrast clip limit (> 0). Higher values allow more
            amplification; values near 1 strongly limit it.
        grid: Target number of tiles per axis (clamped to the channel size).
        n_bins: Number of intensity bins (256 for 8-bit).

    Returns:
        A 2D ``uint8`` array of the same shape as ``channel``.
    """
    height, width = channel.shape
    grid_x = max(1, min(grid, width))
    grid_y = max(1, min(grid, height))

    # Integer tile boundaries covering the full channel with no gaps/overlaps.
    x_bounds = np.linspace(0, width, grid_x + 1).round().astype(int)
    y_bounds = np.linspace(0, height, grid_y + 1).round().astype(int)

    # Build a mapping (LUT) per tile.
    maps = np.empty((grid_y, grid_x, n_bins), dtype=np.float64)
    for ty in range(grid_y):
        for tx in range(grid_x):
            tile = channel[y_bounds[ty]:y_bounds[ty + 1], x_bounds[tx]:x_bounds[tx + 1]]
            maps[ty, tx] = _tile_mapping(tile, clip_limit, n_bins)

    # Tile centre coordinates (in pixel space).
    cx = (np.arange(grid_x) + 0.5) * (width / grid_x)
    cy = (np.arange(grid_y) + 0.5) * (height / grid_y)

    x0, x1, wx = _interp_indices(np.arange(width), cx)
    y0, y1, wy = _interp_indices(np.arange(height), cy)

    # Reshape for broadcasting to (height, width).
    y0 = y0[:, None]
    y1 = y1[:, None]
    wy = wy[:, None]
    x0 = x0[None, :]
    x1 = x1[None, :]
    wx = wx[None, :]

    values = channel  # (height, width) uint8 used as the bin index.
    m00 = maps[y0, x0, values]
    m01 = maps[y0, x1, values]
    m10 = maps[y1, x0, values]
    m11 = maps[y1, x1, values]

    top = m00 * (1.0 - wx) + m01 * wx
    bottom = m10 * (1.0 - wx) + m11 * wx
    out = top * (1.0 - wy) + bottom * wy

    return np.clip(np.round(out), 0, n_bins - 1).astype(np.uint8)


def _tile_mapping(tile: np.ndarray, clip_limit: float, n_bins: int) -> np.ndarray:
    """Compute the clipped-CDF intensity mapping for a single tile."""
    n_pixels = tile.size
    if n_pixels == 0:
        return np.arange(n_bins, dtype=np.float64)

    hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float64)

    # Clip the histogram and redistribute the excess uniformly across all bins.
    clip = max(1.0, clip_limit * n_pixels / n_bins)
    excess = np.clip(hist - clip, 0.0, None).sum()
    hist = np.minimum(hist, clip) + excess / n_bins

    cdf = np.cumsum(hist)
    positive = cdf[cdf > 0]
    cdf_min = positive[0] if positive.size else 0.0
    denom = cdf[-1] - cdf_min
    if denom <= 0:
        # Degenerate (single distinct value): identity mapping.
        return np.arange(n_bins, dtype=np.float64)

    return (cdf - cdf_min) / denom * (n_bins - 1)


def _interp_indices(coords: np.ndarray, centers: np.ndarray):
    """Return neighbouring tile indices and interpolation weights for coords.

    For each coordinate this finds the two nearest tile centres and the weight
    toward the higher-indexed centre. Coordinates outside the centre range clamp
    to the edge tile (weight 0 or 1), which yields plain per-tile equalization
    when there is a single tile on that axis.
    """
    n = centers.shape[0]
    idx = np.searchsorted(centers, coords) - 1
    idx0 = np.clip(idx, 0, n - 1)
    idx1 = np.clip(idx + 1, 0, n - 1)

    c0 = centers[idx0]
    c1 = centers[idx1]
    span = c1 - c0
    safe_span = np.where(span > 0, span, 1.0)
    weight = np.where(span > 0, (coords - c0) / safe_span, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    return idx0, idx1, weight
