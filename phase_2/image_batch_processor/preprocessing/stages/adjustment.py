"""Adjustment stage (1 to 1).

A single, parameterized stage that performs one photo-editor-style adjustment
per instance, mirroring the sliders in an image editor's "Adjustments" panel
(Brightness, Contrast, Saturation, Highlights, Shadows, Temperature, Sharpness).
Composing several of these in an ordered pipeline reproduces an editor session
while keeping each modification an independent, tunable, reorderable, and
individually removable step.

Each adjustment is a pure per-image transform: one input :class:`WorkingImage`
maps to exactly one output image with identical pixel dimensions. The ``amount``
uses the editor's slider scale (see :class:`AdjustmentConfig`) so editor
settings translate directly.

The concrete math approximates common editor behavior (it is not a bit-exact
reimplementation of any specific proprietary editor); values are meant to be
tuned visually via the experiment harness. Operations:

- ``brightness``  — multiplicative brightness (``ImageEnhance.Brightness``).
- ``contrast``    — contrast around mid-gray (``ImageEnhance.Contrast``).
- ``saturation``  — color saturation (``ImageEnhance.Color``); ``-100`` fully
  desaturates to grayscale. No-op on single-channel ``L`` images.
- ``highlights``  — pushes bright tones up/down using a luminance highlight mask.
- ``shadows``     — pushes dark tones up/down using a luminance shadow mask.
- ``temperature`` — warms (positive) or cools (negative) by trading red vs blue
  gain. No-op on ``L`` images.
- ``sharpen``     — unsharp-mask sharpening (``ImageFilter.UnsharpMask``).

The stage never performs file I/O and never mutates its input images in place.
If an input is not an in-memory image, it raises a :class:`StageError`
identifying itself.
"""

from typing import List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config.settings import AdjustmentConfig, AdjustmentOperation
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage

_STAGE_TYPE = "adjustment"

# Modes handled natively. Others are converted to RGB first.
_SUPPORTED_MODES = frozenset({"L", "RGB"})

# Maps the editor's sharpness slider [0, 100] to an unsharp-mask ``percent``
# strength. 100 on the slider -> 300% unsharp, a punchy but usable maximum.
_SHARPEN_PERCENT_SCALE = 3.0

# How strongly temperature trades red vs blue gain at amount = +/-100.
_TEMPERATURE_GAIN = 0.3

# How strongly highlights/shadows shift their masked tones at amount = +/-100
# (as a fraction of full range).
_TONE_SHIFT_SCALE = 0.5


class AdjustmentStage(PreprocessingStage):
    """A 1 to 1 stage performing one editor-style adjustment.

    The specific adjustment and its strength come from the injected
    :class:`AdjustmentConfig`. Dimensions are always preserved.
    """

    def __init__(self, config: AdjustmentConfig):
        """Create the stage from its validated configuration.

        Args:
            config: The adjustment configuration (operation + amount, plus
                unsharp-mask tuning for the sharpen operation).
        """
        self._config = config

    @property
    def stage_type(self) -> str:
        """Return the stage type identifier ``"adjustment"``."""
        return _STAGE_TYPE

    def validate_config(self) -> bool:
        """Validate the stage is properly configured.

        The Pydantic ``AdjustmentConfig`` already enforces field ranges and the
        amount/operation scale on construction. This confirms the injected
        config type and a recognized operation.

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the config is not an :class:`AdjustmentConfig`
                or the operation is unknown.
        """
        if not isinstance(self._config, AdjustmentConfig):
            raise ConfigurationError(
                "adjustment stage requires an AdjustmentConfig, "
                f"got {type(self._config).__name__}"
            )
        if not isinstance(self._config.operation, AdjustmentOperation):
            raise ConfigurationError(
                f"adjustment.operation = {self._config.operation!r} is not a "
                "recognized AdjustmentOperation"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """Apply the configured adjustment to every image in the working set.

        Produces exactly one output image per input, in the same order, each
        with dimensions identical to its corresponding input. Input images are
        not mutated in place.

        Args:
            working_set: The ordered input working set.

        Returns:
            A new working set with one adjusted ``WorkingImage`` per input.

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
            result = self._dispatch(source)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                f"failed to apply {self._config.operation.value} adjustment to a "
                f"{image.mode} image: {exc}",
                stage_name=_STAGE_TYPE,
            ) from exc

        if result.size != image.size:
            raise StageError(
                f"{self._config.operation.value} adjustment changed dimensions "
                f"from {image.size} to {result.size}",
                stage_name=_STAGE_TYPE,
            )

        return WorkingImage(
            source_name=item.source_name,
            image=result,
            width=result.width,
            height=result.height,
            lineage=list(item.lineage),
        )

    def _dispatch(self, image: Image.Image) -> Image.Image:
        """Route to the concrete transform for the configured operation."""
        op = self._config.operation
        amount = self._config.amount

        if op == AdjustmentOperation.BRIGHTNESS:
            return ImageEnhance.Brightness(image).enhance(max(0.0, 1.0 + amount / 100.0))
        if op == AdjustmentOperation.CONTRAST:
            return ImageEnhance.Contrast(image).enhance(max(0.0, 1.0 + amount / 100.0))
        if op == AdjustmentOperation.SATURATION:
            if image.mode == "L":
                return image.copy()  # grayscale has no saturation to adjust
            return ImageEnhance.Color(image).enhance(max(0.0, 1.0 + amount / 100.0))
        if op == AdjustmentOperation.SHARPEN:
            return self._sharpen(image)
        if op == AdjustmentOperation.HIGHLIGHTS:
            return self._tone_shift(image, target="highlights")
        if op == AdjustmentOperation.SHADOWS:
            return self._tone_shift(image, target="shadows")
        if op == AdjustmentOperation.TEMPERATURE:
            return self._temperature(image)
        # Guarded by validate_config / Pydantic enum.
        raise StageError(
            f"unsupported adjustment operation {op!r}", stage_name=_STAGE_TYPE
        )

    def _sharpen(self, image: Image.Image) -> Image.Image:
        """Unsharp-mask sharpening scaled from the editor's sharpness slider."""
        percent = int(round(self._config.amount * _SHARPEN_PERCENT_SCALE))
        if percent <= 0:
            return image.copy()
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=self._config.sharpen_radius,
                percent=percent,
                threshold=self._config.sharpen_threshold,
            )
        )

    def _tone_shift(self, image: Image.Image, target: str) -> Image.Image:
        """Shift highlight or shadow tones using a luminance-derived mask.

        A per-pixel luminance mask isolates the bright end (highlights) or dark
        end (shadows) of the tonal range; the masked pixels are shifted by
        ``amount`` scaled by :data:`_TONE_SHIFT_SCALE`. ``amount < 0`` darkens
        the targeted tones (pulls highlights down / deepens shadows), ``> 0``
        lifts them. The same shift is applied to every color channel using the
        shared luminance mask so hue is preserved.
        """
        amount = self._config.amount
        if amount == 0.0:
            return image.copy()

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            luminance = array
        else:
            # Rec. 601 luma for the mask.
            luminance = (
                0.299 * array[:, :, 0]
                + 0.587 * array[:, :, 1]
                + 0.114 * array[:, :, 2]
            )
        norm = luminance / 255.0

        if target == "highlights":
            # 0 below mid-gray, ramps to 1 at white.
            mask = np.clip((norm - 0.5) * 2.0, 0.0, 1.0)
        else:  # shadows
            # 0 above mid-gray, ramps to 1 at black.
            mask = np.clip((0.5 - norm) * 2.0, 0.0, 1.0)

        shift = np.float32((amount / 100.0) * _TONE_SHIFT_SCALE * 255.0)
        delta = shift * mask  # (H, W)

        if array.ndim == 2:
            adjusted = array + delta
        else:
            adjusted = array + delta[:, :, None]

        adjusted = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(adjusted, mode=image.mode)

    def _temperature(self, image: Image.Image) -> Image.Image:
        """Warm (positive) or cool (negative) by trading red vs blue gain."""
        amount = self._config.amount
        if image.mode == "L" or amount == 0.0:
            return image.copy()

        warmth = amount / 100.0
        red_gain = np.float32(1.0 + warmth * _TEMPERATURE_GAIN)
        blue_gain = np.float32(1.0 - warmth * _TEMPERATURE_GAIN)

        array = np.asarray(image, dtype=np.float32)
        array[:, :, 0] *= red_gain
        array[:, :, 2] *= blue_gain
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(array, mode=image.mode)
