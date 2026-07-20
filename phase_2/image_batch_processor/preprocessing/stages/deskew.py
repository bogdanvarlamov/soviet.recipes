"""Deskew stage (1 to 1) — rotational alignment.

Corrects in-plane rotational skew so text lines become horizontal. Pages shot
by hand are often slightly rotated; this stage estimates that rotation and
undoes it.

Approach — projection-profile deskew (a classic, dependency-free method):

1. Convert the image to grayscale and downscale it (angle estimation does not
   need full resolution and downscaling makes the search fast).
2. Build an ink mask via Otsu's threshold (dark pixels = ink).
3. For each candidate angle in ``[-max_angle, +max_angle]`` (a coarse pass then
   a refinement pass around the best coarse angle), rotate the mask and compute
   the variance of its horizontal projection profile (row sums). When text
   lines are horizontal, rows alternate between dense (text) and sparse (gaps),
   maximizing that variance.
4. Rotate the full-resolution image by the best angle, expanding the canvas so
   no content is clipped and filling exposed corners with ``fill_value``.

This corrects **rotation only**. Perspective (keystone) distortion and spine
curvature are separate, harder problems handled elsewhere. The stage never
performs file I/O and never mutates its input in place; it returns a new
:class:`WorkingImage`. Output dimensions may differ from the input because the
rotated canvas is expanded to avoid clipping.
"""

from typing import List, Tuple

import numpy as np
from PIL import Image

from ..config.settings import DeskewConfig
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage

_STAGE_TYPE = "deskew"

# Modes handled natively; others are converted to RGB before rotating.
_SUPPORTED_MODES = frozenset({"L", "RGB"})


class DeskewStage(PreprocessingStage):
    """A 1 to 1 stage that rotates each image to make text lines horizontal."""

    def __init__(self, config: DeskewConfig):
        """Create the stage from its validated configuration.

        Args:
            config: The deskew configuration (search range/steps, estimation
                downscale width, and corner fill value).
        """
        self._config = config

    @property
    def stage_type(self) -> str:
        """Return the stage type identifier ``"deskew"``."""
        return _STAGE_TYPE

    def validate_config(self) -> bool:
        """Validate the stage is properly configured.

        The Pydantic ``DeskewConfig`` already enforces field ranges and step
        ordering on construction. This confirms the injected config type.

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the config is not a :class:`DeskewConfig`.
        """
        if not isinstance(self._config, DeskewConfig):
            raise ConfigurationError(
                "deskew stage requires a DeskewConfig, "
                f"got {type(self._config).__name__}"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """Deskew every image in the working set.

        Produces exactly one output image per input, in the same order. Input
        images are not mutated in place. Output dimensions may differ from the
        input (the rotated canvas is expanded to avoid clipping).

        Args:
            working_set: The ordered input working set.

        Returns:
            A new working set with one deskewed ``WorkingImage`` per input.

        Raises:
            StageError: If an input is not an in-memory image or the transform
                fails. The error identifies this stage.
        """
        return [self._process_one(item) for item in working_set]

    # -- internal helpers -------------------------------------------------

    def _process_one(self, item: WorkingImage) -> WorkingImage:
        image = item.image
        if not isinstance(image, Image.Image):
            raise StageError(
                f"expected an in-memory Pillow image, got {type(image).__name__}",
                stage_name=_STAGE_TYPE,
            )

        try:
            source = image if image.mode in _SUPPORTED_MODES else image.convert("RGB")
            angle = self._estimate_angle(source)
            result = self._rotate(source, angle)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                f"failed to deskew a {image.mode} image: {exc}",
                stage_name=_STAGE_TYPE,
            ) from exc

        return WorkingImage(
            source_name=item.source_name,
            image=result,
            width=result.width,
            height=result.height,
            lineage=list(item.lineage),
        )

    def _estimate_angle(self, image: Image.Image) -> float:
        """Estimate the skew angle (degrees) for ``image``."""
        gray = image.convert("L")

        # Downscale for a fast search; angle estimation is scale-invariant.
        if gray.width > self._config.estimate_width:
            scale = self._config.estimate_width / gray.width
            new_size = (
                self._config.estimate_width,
                max(1, int(round(gray.height * scale))),
            )
            gray = gray.resize(new_size, Image.BILINEAR)

        mask = _ink_mask(np.asarray(gray, dtype=np.uint8))
        mask_image = Image.fromarray(mask, mode="L")

        return estimate_skew_angle(
            mask_image,
            max_angle=self._config.max_angle,
            coarse_step=self._config.coarse_step,
            refine_step=self._config.refine_step,
        )

    def _rotate(self, image: Image.Image, angle: float) -> Image.Image:
        """Rotate ``image`` by ``angle`` degrees, expanding to avoid clipping."""
        if angle == 0.0:
            return image.copy()

        fill = self._config.fill_value
        fillcolor = fill if image.mode == "L" else (fill, fill, fill)
        return image.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=fillcolor,
        )


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Return a 0/255 ink mask (255 = ink) from a grayscale array via Otsu."""
    threshold = _otsu_threshold(gray)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def _otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu's threshold for an 8-bit grayscale array."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 128.0

    sum_all = np.dot(np.arange(256), hist)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    # Avoid division by zero at the ends.
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return 128.0

    cumulative_mean = np.cumsum(np.arange(256) * hist)
    mean_bg = np.divide(
        cumulative_mean, weight_bg, out=np.zeros_like(cumulative_mean), where=weight_bg > 0
    )
    mean_fg = np.divide(
        sum_all - cumulative_mean,
        weight_fg,
        out=np.zeros_like(cumulative_mean),
        where=weight_fg > 0,
    )
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var[~valid] = -1.0
    return float(np.argmax(between_var))


def estimate_skew_angle(
    mask_image: Image.Image,
    max_angle: float,
    coarse_step: float,
    refine_step: float,
) -> float:
    """Estimate the skew angle of an ink mask via projection-profile variance.

    Rotates ``mask_image`` through candidate angles and returns the angle whose
    horizontal projection profile (row sums) has the highest variance — i.e. the
    rotation that best aligns text rows into sharp peaks. A coarse scan over the
    full range is followed by a fine scan around the coarse best.

    Args:
        mask_image: A single-channel (``L``) ink mask (255 = ink).
        max_angle: Search bound; angles in ``[-max_angle, +max_angle]`` are
            considered.
        coarse_step: Angle step (degrees) for the coarse scan.
        refine_step: Angle step (degrees) for the refinement scan.

    Returns:
        The estimated correction angle in degrees (positive = counter-clockwise,
        matching Pillow's ``rotate``). ``0.0`` when no candidate improves on a
        level profile.
    """
    coarse_angles = _angle_range(-max_angle, max_angle, coarse_step)
    best_angle = _best_angle(mask_image, coarse_angles)

    # Refine within +/- one coarse step of the best coarse angle.
    low = max(-max_angle, best_angle - coarse_step)
    high = min(max_angle, best_angle + coarse_step)
    fine_angles = _angle_range(low, high, refine_step)
    return _best_angle(mask_image, fine_angles)


def _angle_range(low: float, high: float, step: float) -> List[float]:
    """Inclusive list of angles from ``low`` to ``high`` in ``step`` increments."""
    count = int(round((high - low) / step))
    angles = [low + i * step for i in range(count + 1)]
    if not angles or abs(angles[-1] - high) > 1e-9:
        angles.append(high)
    return angles


def _best_angle(mask_image: Image.Image, angles: List[float]) -> float:
    """Return the angle whose rotated projection profile has the max variance."""
    best_angle = 0.0
    best_score = -1.0
    for angle in angles:
        score = _projection_variance(mask_image, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _projection_variance(mask_image: Image.Image, angle: float) -> float:
    """Variance of the horizontal projection profile after rotating by ``angle``."""
    rotated = mask_image.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    row_sums = np.asarray(rotated, dtype=np.float64).sum(axis=1)
    return float(np.var(row_sums))
