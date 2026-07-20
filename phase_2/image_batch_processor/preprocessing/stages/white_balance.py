"""White-balance / contrast-maximization stage (1 to 1).

Many cookbook scans have a beige/off-white page background rather than true
white, and text that is faint or dark gray rather than true black. This stage
removes that color cast and maximizes text/background contrast, using one of
two methods selected by :class:`WhiteBalanceConfig.method`:

- ``percentile_stretch`` (default): stretches each channel's own pixel-value
  percentiles to the full 0-255 range (a percentile-based "auto levels" /
  white-balance transform). This is a *global* transform: one black/white
  point is computed for the whole image (or, with ``per_channel``, per whole
  channel), so it does not fully binarize faint text — text that is only
  slightly darker than the background can still end up mid-gray rather than
  pure black.
- ``adaptive_threshold``: a local Sauvola-style binarization. Every pixel is
  compared against a threshold computed from the mean and standard deviation
  of its own local neighborhood, then driven fully to pure black (text) or
  pure white (background). Because the threshold adapts per neighborhood
  rather than using one global cutoff, this handles faint text and uneven
  page lighting far better, converting the page to true black-on-white.

Some cookbook pages contain photographs rather than only text. This transform
is intentionally lossy for those pages (it is a text-oriented, not a
photo-quality, transform) — acceptable here because only text is extracted
downstream; image content is handled by a separate pipeline.

The stage never performs file I/O and never mutates its input images in
place; it produces new :class:`WorkingImage` instances carrying a copy of the
input lineage. If the color mode cannot be processed, the stage raises a
:class:`StageError` identifying itself rather than returning an output image.
"""

from typing import List, Tuple

import numpy as np
from PIL import Image

from ..config.settings import WhiteBalanceConfig, WhiteBalanceMethod
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage

_STAGE_TYPE = "white_balance"

# Color modes this stage can process. Modes outside this set cause the stage
# to raise a StageError.
_SUPPORTED_MODES = frozenset({"L", "RGB"})


class WhiteBalanceStage(PreprocessingStage):
    """A 1 to 1 stage that white-balances and contrast-stretches an image.

    Each input image is mapped to exactly one output image of identical
    dimensions. Images in a color mode other than ``L`` or ``RGB`` are
    converted to ``RGB`` before the stretch (e.g. ``RGBA`` has its alpha
    dropped, ``P`` is expanded to full color) so the transform's percentile
    statistics operate on plain intensity values.
    """

    def __init__(self, config: WhiteBalanceConfig):
        """Create the stage from its validated configuration.

        Args:
            config: The white-balance configuration (percentile black/white
                points and whether to stretch per-channel or on combined
                intensity).
        """
        self._config = config

    @property
    def stage_type(self) -> str:
        """Return the stage type identifier ``"white_balance"``."""
        return _STAGE_TYPE

    def validate_config(self) -> bool:
        """Validate the stage is properly configured.

        The Pydantic ``WhiteBalanceConfig`` already enforces the percentile
        field ranges on construction. This method additionally enforces the
        cross-field rule the stage depends on: the white point must be
        strictly greater than the black point, and confirms the injected
        config is of the expected type.

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the configuration is not a
                :class:`WhiteBalanceConfig` or the percentile points are not
                well-formed.
        """
        if not isinstance(self._config, WhiteBalanceConfig):
            raise ConfigurationError(
                "white_balance stage requires a WhiteBalanceConfig, "
                f"got {type(self._config).__name__}"
            )
        if self._config.white_point_percentile <= self._config.black_point_percentile:
            raise ConfigurationError(
                "white_balance.white_point_percentile = "
                f"{self._config.white_point_percentile!r} must be greater than "
                f"black_point_percentile = {self._config.black_point_percentile!r}"
            )
        if not isinstance(self._config.method, WhiteBalanceMethod):
            raise ConfigurationError(
                f"white_balance.method = {self._config.method!r} is not a "
                "recognized WhiteBalanceMethod"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """White-balance and contrast-stretch every image in the working set.

        Produces exactly one output image per input image, in the same order,
        each with dimensions identical to its corresponding input. Input
        images are not mutated in place.

        Args:
            working_set: The ordered input working set.

        Returns:
            A new working set with one stretched ``WorkingImage`` per input.

        Raises:
            StageError: If an input image's color mode cannot be processed, or
                the transform otherwise fails. The error identifies this stage
                and no working set is returned.
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

        try:
            source = image if image.mode in _SUPPORTED_MODES else image.convert("RGB")
            if self._config.method == WhiteBalanceMethod.ADAPTIVE_THRESHOLD:
                result = self._apply_adaptive_threshold(source)
            else:
                array = np.asarray(source, dtype=np.float64)
                if self._config.per_channel or array.ndim == 2:
                    stretched = self._stretch_channels_independently(array)
                else:
                    stretched = self._stretch_by_combined_intensity(array)
                result = Image.fromarray(stretched.astype(np.uint8), mode=source.mode)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                f"failed to white-balance a {image.mode} image: {exc}",
                stage_name=_STAGE_TYPE,
            ) from exc

        if result.size != image.size:
            raise StageError(
                f"white-balance transform changed dimensions from {image.size} "
                f"to {result.size}",
                stage_name=_STAGE_TYPE,
            )

        return WorkingImage(
            source_name=item.source_name,
            image=result,
            width=result.width,
            height=result.height,
            lineage=list(item.lineage),
        )

    def _apply_adaptive_threshold(self, image: Image.Image) -> Image.Image:
        """Binarize ``image`` via local (Sauvola-style) adaptive thresholding.

        Converts to grayscale (luminance) first: binarization is a per-pixel
        black/white decision, so it operates on intensity rather than color.
        Every pixel is compared against a threshold derived from its own local
        neighborhood; pixels darker than their local threshold become pure
        black (0), everything else becomes pure white (255). Output mode
        matches the input's original mode (``RGB`` inputs get a grayscale
        image re-expanded to three equal channels so dimensions/mode are
        preserved).

        Args:
            image: The source image (``L`` or ``RGB`` mode).

        Returns:
            A binarized image in the same mode as ``image``.
        """
        gray = np.asarray(
            image if image.mode == "L" else image.convert("L"), dtype=np.float64
        )
        binarized = _sauvola_binarize(
            gray,
            window_fraction=self._config.window_fraction,
            k=self._config.sensitivity_k,
            dynamic_range=self._config.dynamic_range,
        )
        if image.mode == "L":
            return Image.fromarray(binarized, mode="L")
        return Image.fromarray(np.stack([binarized] * 3, axis=2), mode="RGB")

    def _stretch_channels_independently(self, array: np.ndarray) -> np.ndarray:
        """Stretch each channel to its own black/white percentile points.

        This is what performs the actual white balance: a beige cast that
        attenuates channels unevenly (e.g. blue more than red) is corrected
        because each channel is independently mapped to the full 0-255 range.

        Args:
            array: A ``float64`` array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            A ``float64`` array of the same shape, values clipped to [0, 255].
        """
        low = self._config.black_point_percentile
        high = self._config.white_point_percentile

        if array.ndim == 2:
            channels = [array]
        else:
            channels = [array[:, :, c] for c in range(array.shape[2])]

        stretched_channels = [
            _percentile_stretch(channel, low, high) for channel in channels
        ]

        if array.ndim == 2:
            return stretched_channels[0]
        return np.stack(stretched_channels, axis=2)

    def _stretch_by_combined_intensity(self, array: np.ndarray) -> np.ndarray:
        """Stretch all channels together using combined-intensity percentiles.

        Preserves hue (no per-channel color correction) while still maximizing
        contrast: the same black/white point (derived from every channel's
        pixel values pooled together) is applied uniformly to every channel.

        Args:
            array: A ``float64`` array of shape ``(H, W, C)``.

        Returns:
            A ``float64`` array of the same shape, values clipped to [0, 255].
        """
        low = self._config.black_point_percentile
        high = self._config.white_point_percentile

        black_point, white_point = _percentile_points(array.ravel(), low, high)
        return _apply_stretch(array, black_point, white_point)


def _percentile_points(values: np.ndarray, low: float, high: float):
    """Compute the (black_point, white_point) percentile values of ``values``.

    Guards against a degenerate (flat) input where the low and high
    percentiles coincide, which would otherwise divide by zero: in that case
    the white point is nudged up by one intensity level so the stretch becomes
    a no-op identity mapping rather than raising.
    """
    black_point = float(np.percentile(values, low))
    white_point = float(np.percentile(values, high))
    if white_point <= black_point:
        white_point = black_point + 1.0
    return black_point, white_point


def _apply_stretch(
    array: np.ndarray, black_point: float, white_point: float
) -> np.ndarray:
    """Linearly rescale ``array`` so black_point -> 0 and white_point -> 255."""
    scaled = (array - black_point) * (255.0 / (white_point - black_point))
    return np.clip(scaled, 0.0, 255.0)


def _percentile_stretch(channel: np.ndarray, low: float, high: float) -> np.ndarray:
    """Percentile-stretch a single channel to the full 0-255 range."""
    black_point, white_point = _percentile_points(channel.ravel(), low, high)
    return _apply_stretch(channel, black_point, white_point)


def _sauvola_binarize(
    gray: np.ndarray,
    window_fraction: float,
    k: float,
    dynamic_range: float,
) -> np.ndarray:
    """Binarize a grayscale array via Sauvola's local adaptive threshold.

    For each pixel, computes the local mean ``m`` and standard deviation ``s``
    over a square window of side ``window`` (derived from ``window_fraction``
    x image width, clamped to at least 3px and odd) centered on that pixel,
    using an integral-image (summed-area table) approach so the whole image is
    processed with vectorized numpy operations rather than a per-pixel Python
    loop. The Sauvola threshold at that pixel is::

        t = m * (1 + k * (s / dynamic_range - 1))

    A pixel is set to black (0) when its intensity is below ``t``, otherwise
    white (255). Unlike a single global threshold, ``t`` adapts to each
    neighborhood's local brightness and contrast, so faint text and pages with
    an uneven lighting gradient both binarize correctly: a dim corner gets a
    lower local threshold rather than being globally crushed to black or left
    unreadably gray.

    Args:
        gray: A 2D ``float64`` grayscale array.
        window_fraction: Window side as a fraction of image width (0, 1).
        k: Sensitivity to local standard deviation (> 0); higher values make
            the threshold more permissive in high-contrast neighborhoods.
        dynamic_range: Normalizing constant for the local standard deviation
            (> 0), matching Sauvola's ``R`` parameter (typically the expected
            maximum standard deviation for the image's bit depth, e.g. 128 for
            8-bit grayscale).

    Returns:
        A 2D ``uint8`` array of the same shape as ``gray`` containing only the
        values 0 and 255.
    """
    height, width = gray.shape
    window = max(3, int(round(window_fraction * width)))
    if window % 2 == 0:
        window += 1
    half = window // 2

    # Pad by reflection so windows near the border are still full-sized and
    # use real (mirrored) local content rather than artificial zeros.
    padded = np.pad(gray, half, mode="reflect")
    padded_sq = padded * padded

    # Integral images (summed-area tables), zero-padded by one row/column so
    # the standard box-sum formula below needs no extra bounds handling.
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral_sq = np.zeros_like(integral)
    integral_sq[1:, 1:] = np.cumsum(np.cumsum(padded_sq, axis=0), axis=1)

    # For output pixel (r, c), the window in `padded` coordinates spans rows
    # [r, r + window) and cols [c, c + window) (since `padded` already
    # includes the `half`-sized border on every side).
    box_sum = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    box_sum_sq = (
        integral_sq[window:, window:]
        - integral_sq[:-window, window:]
        - integral_sq[window:, :-window]
        + integral_sq[:-window, :-window]
    )

    n_pixels = float(window * window)
    mean = box_sum / n_pixels
    variance = np.clip(box_sum_sq / n_pixels - mean * mean, 0.0, None)
    std = np.sqrt(variance)

    threshold = mean * (1.0 + k * (std / dynamic_range - 1.0))

    return np.where(gray < threshold, 0, 255).astype(np.uint8)
