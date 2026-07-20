"""Configuration models for the image preprocessing pipeline.

Defines the ``StageConfig`` hierarchy (parallel to ``EngineConfig`` in the
image batch processor): a base ``StageConfig`` plus one Pydantic subclass per
shipped stage. Each subclass performs field-level range validation and raises
``ConfigurationError`` (rather than the default Pydantic ``ValidationError``)
when a parameter falls outside its declared allowed range, so misconfiguration
is surfaced with the offending stage, parameter, value, and allowed range.

Cross-field validation (``search_band_max > search_band_min``) and the
``PipelineConfig`` / ``StageSpec`` models are implemented separately (task 3.2).
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..exceptions import ConfigurationError


class PageSplitMethod(str, Enum):
    """Spine-location approach for the page-split stage."""

    FIXED_MIDPOINT = "fixed_midpoint"
    GUTTER_DETECTION = "gutter_detection"


class ContrastMethod(str, Enum):
    """Contrast-enhancement approach for the contrast stage."""

    LINEAR = "linear"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    ADAPTIVE = "adaptive"


def _require(stage: str, field: str, value: float, allowed: str, ok: bool) -> None:
    """Raise a ``ConfigurationError`` when ``ok`` is False.

    The message identifies the offending stage, parameter name, provided value,
    and allowed range (Requirement 7.3).
    """
    if not ok:
        raise ConfigurationError(
            f"{stage}.{field} = {value!r} is out of range; allowed range is {allowed}"
        )


class StageConfig(BaseModel):
    """Base configuration for preprocessing stages.

    Parallel to ``EngineConfig`` in the image batch processor; concrete stages
    declare their own configuration subclass.
    """
    pass


class PageSplitConfig(StageConfig):
    """Configuration for the page-split stage (1 to N).

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``split_ratio`` in (0, 1)
      - ``gutter_margin`` in [0, 0.5)
      - ``search_band_min`` in (0, 1)
      - ``search_band_max`` in (0, 1)
      - ``fallback_ratio`` in (0, 1)
    """

    method: PageSplitMethod = PageSplitMethod.FIXED_MIDPOINT
    split_ratio: float = 0.5
    gutter_margin: float = 0.0
    search_band_min: float = 0.35
    search_band_max: float = 0.65
    fallback_ratio: float = 0.5
    cover_filenames: List[str] = []
    treat_first_last_as_covers: bool = False

    @field_validator("split_ratio")
    @classmethod
    def _validate_split_ratio(cls, v: float) -> float:
        _require("page_split", "split_ratio", v, "(0, 1)", 0.0 < v < 1.0)
        return v

    @field_validator("gutter_margin")
    @classmethod
    def _validate_gutter_margin(cls, v: float) -> float:
        _require("page_split", "gutter_margin", v, "[0, 0.5)", 0.0 <= v < 0.5)
        return v

    @field_validator("search_band_min")
    @classmethod
    def _validate_search_band_min(cls, v: float) -> float:
        _require("page_split", "search_band_min", v, "(0, 1)", 0.0 < v < 1.0)
        return v

    @field_validator("search_band_max")
    @classmethod
    def _validate_search_band_max(cls, v: float) -> float:
        _require("page_split", "search_band_max", v, "(0, 1)", 0.0 < v < 1.0)
        return v

    @field_validator("fallback_ratio")
    @classmethod
    def _validate_fallback_ratio(cls, v: float) -> float:
        _require("page_split", "fallback_ratio", v, "(0, 1)", 0.0 < v < 1.0)
        return v

    @model_validator(mode="after")
    def _validate_search_band_ordering(self) -> "PageSplitConfig":
        """Cross-field check: the search band must be non-degenerate.

        ``search_band_max`` must be strictly greater than ``search_band_min``;
        otherwise the central detection band is empty or inverted (Requirement
        7.4). Raises ``ConfigurationError`` identifying both configured values.
        """
        if self.search_band_max <= self.search_band_min:
            raise ConfigurationError(
                "page_split.search_band_max must be greater than "
                "page_split.search_band_min; got "
                f"search_band_min = {self.search_band_min!r}, "
                f"search_band_max = {self.search_band_max!r}"
            )
        return self


class ContrastEnhancementConfig(StageConfig):
    """Configuration for the contrast-enhancement stage (1 to 1).

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``factor`` > 0
      - ``clip_limit`` > 0
    """

    method: ContrastMethod = ContrastMethod.LINEAR
    factor: float = 1.5
    clip_limit: float = 2.0

    @field_validator("factor")
    @classmethod
    def _validate_factor(cls, v: float) -> float:
        _require("contrast_enhancement", "factor", v, "(0, inf)", v > 0.0)
        return v

    @field_validator("clip_limit")
    @classmethod
    def _validate_clip_limit(cls, v: float) -> float:
        _require("contrast_enhancement", "clip_limit", v, "(0, inf)", v > 0.0)
        return v


class WhiteBalanceMethod(str, Enum):
    """Approach used by the white-balance stage."""

    # Global per-channel percentile stretch (removes color cast, moderate
    # contrast boost). Struggles with uneven/gradient illumination across a
    # page, since the black/white points are computed once for the whole
    # image.
    PERCENTILE_STRETCH = "percentile_stretch"
    # Local adaptive binarization (Sauvola-style): every pixel is compared
    # against a threshold computed from its own neighborhood's mean and
    # standard deviation, then driven fully to black (text) or white
    # (background). This is far more extreme than a global stretch and
    # correctly handles pages with lighting gradients/vignetting, since the
    # threshold adapts locally rather than using one global cutoff.
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


class WhiteBalanceConfig(StageConfig):
    """Configuration for the white-balance / contrast-maximization stage (1 to 1).

    Two methods are available, selected by ``method``:

    - ``percentile_stretch`` (default): estimates a black point and a white
      point from the image's own pixel-value percentiles and linearly
      stretches them to 0 and 255 respectively. On a beige page with dark
      text, the low percentile falls on the text and the high percentile
      falls on the page background, so the stretch simultaneously removes the
      beige color cast (background -> white) and maximizes contrast
      (text -> black). This is a *global* transform: one black/white point is
      computed for the whole image (or, with ``per_channel``, per whole
      channel).
    - ``adaptive_threshold``: a *local* Sauvola-style binarization. For every
      pixel, a threshold is computed from the mean and standard deviation of
      an ``window_fraction`` x width neighborhood around it, and the pixel is
      driven fully to black or white depending on which side of that local
      threshold it falls on. Because the threshold adapts per neighborhood,
      this handles pages with an uneven lighting gradient (a corner or edge
      darker/lighter than the rest) far better than a single global stretch,
      and produces true black text on a true white background even where the
      original text was faint. Output is single-channel (grayscale).

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``black_point_percentile`` in [0, 50)
      - ``white_point_percentile`` in (50, 100]
      - ``window_fraction`` in (0, 1)
      - ``sensitivity_k`` > 0
      - ``dynamic_range`` > 0
    """

    method: WhiteBalanceMethod = WhiteBalanceMethod.PERCENTILE_STRETCH

    # -- percentile_stretch parameters --
    black_point_percentile: float = 1.0
    white_point_percentile: float = 99.0
    per_channel: bool = True

    # -- adaptive_threshold (Sauvola) parameters --
    window_fraction: float = 0.04
    sensitivity_k: float = 0.2
    dynamic_range: float = 128.0

    @field_validator("black_point_percentile")
    @classmethod
    def _validate_black_point_percentile(cls, v: float) -> float:
        _require(
            "white_balance", "black_point_percentile", v, "[0, 50)", 0.0 <= v < 50.0
        )
        return v

    @field_validator("white_point_percentile")
    @classmethod
    def _validate_white_point_percentile(cls, v: float) -> float:
        _require(
            "white_balance", "white_point_percentile", v, "(50, 100]", 50.0 < v <= 100.0
        )
        return v

    @field_validator("window_fraction")
    @classmethod
    def _validate_window_fraction(cls, v: float) -> float:
        _require("white_balance", "window_fraction", v, "(0, 1)", 0.0 < v < 1.0)
        return v

    @field_validator("sensitivity_k")
    @classmethod
    def _validate_sensitivity_k(cls, v: float) -> float:
        _require("white_balance", "sensitivity_k", v, "(0, inf)", v > 0.0)
        return v

    @field_validator("dynamic_range")
    @classmethod
    def _validate_dynamic_range(cls, v: float) -> float:
        _require("white_balance", "dynamic_range", v, "(0, inf)", v > 0.0)
        return v


class AdjustmentOperation(str, Enum):
    """A single tonal/color/sharpen adjustment, mirroring a photo-editor slider.

    Each value corresponds to one slider in a typical image editor's
    "Adjustments" panel. One :class:`AdjustmentConfig` (and therefore one
    pipeline step) performs exactly one of these, so a workflow is composed by
    listing several ``adjustment`` stages in order — each individually tunable,
    reorderable, or commented out during experimentation.
    """

    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HIGHLIGHTS = "highlights"
    SHADOWS = "shadows"
    TEMPERATURE = "temperature"
    SHARPEN = "sharpen"


# Operations whose ``amount`` uses the editor's bidirectional slider scale,
# i.e. an integer-ish value in [-100, 100] with 0 = no change.
_BIDIRECTIONAL_ADJUSTMENTS = frozenset(
    {
        AdjustmentOperation.BRIGHTNESS,
        AdjustmentOperation.CONTRAST,
        AdjustmentOperation.SATURATION,
        AdjustmentOperation.HIGHLIGHTS,
        AdjustmentOperation.SHADOWS,
        AdjustmentOperation.TEMPERATURE,
    }
)


class AdjustmentConfig(StageConfig):
    """Configuration for one adjustment step (1 to 1).

    Mirrors a single slider in a photo editor's Adjustments panel. The
    ``amount`` uses the editor's slider scale so the settings map directly:

    - For ``brightness``, ``contrast``, ``saturation``, ``highlights``,
      ``shadows``, ``temperature``: ``amount`` is in [-100, 100] where 0 means
      "no change" (a no-op step). Negative/positive mirror the editor:
      ``saturation = -100`` fully desaturates (grayscale), ``temperature < 0``
      is cooler (more blue), ``highlights = -100`` pulls bright tones down,
      ``shadows = -100`` deepens dark tones, and so on.
    - For ``sharpen``: ``amount`` is in [0, 100] (the editor's sharpness
      slider), internally mapped to an unsharp-mask strength; 0 means no
      sharpening. ``sharpen_radius`` and ``sharpen_threshold`` tune the
      unsharp mask.

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``amount`` in [-100, 100] for bidirectional ops, [0, 100] for sharpen
      - ``sharpen_radius`` > 0
      - ``sharpen_threshold`` >= 0
    """

    operation: AdjustmentOperation
    amount: float = 0.0
    sharpen_radius: float = 2.0
    sharpen_threshold: int = 3

    @field_validator("sharpen_radius")
    @classmethod
    def _validate_sharpen_radius(cls, v: float) -> float:
        _require("adjustment", "sharpen_radius", v, "(0, inf)", v > 0.0)
        return v

    @field_validator("sharpen_threshold")
    @classmethod
    def _validate_sharpen_threshold(cls, v: int) -> int:
        _require("adjustment", "sharpen_threshold", v, "[0, inf)", v >= 0)
        return v

    @model_validator(mode="after")
    def _validate_amount_range(self) -> "AdjustmentConfig":
        """Range-check ``amount`` against the operation's slider scale."""
        if self.operation == AdjustmentOperation.SHARPEN:
            _require("adjustment", "amount", self.amount, "[0, 100]", 0.0 <= self.amount <= 100.0)
        else:
            _require(
                "adjustment",
                "amount",
                self.amount,
                "[-100, 100]",
                -100.0 <= self.amount <= 100.0,
            )
        return self


class DeskewConfig(StageConfig):
    """Configuration for the deskew (rotation-alignment) stage (1 to 1).

    Corrects in-plane rotational skew so text lines become horizontal. The
    skew angle is estimated by rotating a downscaled ink mask through a range
    of candidate angles and choosing the one that maximizes the variance of the
    horizontal projection profile (text rows align into sharp peaks), then the
    full-resolution image is rotated by that angle.

    This stage corrects *rotation only*. Perspective (keystone) distortion and
    spine curvature are separate, harder corrections not handled here.

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``max_angle`` in (0, 45]
      - ``coarse_step`` in (0, max_angle]
      - ``refine_step`` in (0, coarse_step]
      - ``estimate_width`` >= 50
      - ``fill_value`` in [0, 255]
    """

    max_angle: float = 8.0
    coarse_step: float = 1.0
    refine_step: float = 0.2
    estimate_width: int = 800
    fill_value: int = 255

    @field_validator("max_angle")
    @classmethod
    def _validate_max_angle(cls, v: float) -> float:
        _require("deskew", "max_angle", v, "(0, 45]", 0.0 < v <= 45.0)
        return v

    @field_validator("coarse_step")
    @classmethod
    def _validate_coarse_step(cls, v: float) -> float:
        _require("deskew", "coarse_step", v, "(0, inf)", v > 0.0)
        return v

    @field_validator("refine_step")
    @classmethod
    def _validate_refine_step(cls, v: float) -> float:
        _require("deskew", "refine_step", v, "(0, inf)", v > 0.0)
        return v

    @field_validator("estimate_width")
    @classmethod
    def _validate_estimate_width(cls, v: int) -> int:
        _require("deskew", "estimate_width", v, "[50, inf)", v >= 50)
        return v

    @field_validator("fill_value")
    @classmethod
    def _validate_fill_value(cls, v: int) -> int:
        _require("deskew", "fill_value", v, "[0, 255]", 0 <= v <= 255)
        return v

    @model_validator(mode="after")
    def _validate_step_ordering(self) -> "DeskewConfig":
        """Cross-field checks: steps must be no larger than the range they scan."""
        if self.coarse_step > self.max_angle:
            raise ConfigurationError(
                "deskew.coarse_step must be <= max_angle; got "
                f"coarse_step = {self.coarse_step!r}, max_angle = {self.max_angle!r}"
            )
        if self.refine_step > self.coarse_step:
            raise ConfigurationError(
                "deskew.refine_step must be <= coarse_step; got "
                f"refine_step = {self.refine_step!r}, coarse_step = {self.coarse_step!r}"
            )
        return self


class DewarpConfig(StageConfig):
    """Configuration for the dewarp (text-line geometric flattening) stage (1 to 1).

    Corrects both text-line curvature and per-page perspective foreshortening
    using a text-line grid remap (Option A): detect text lines, fit how each
    bows, then remap so the lines become straight and evenly spaced. Estimated
    from each page's own text, so left and right half-pages get opposite
    corrections. When too few text lines are detected to model the geometry
    confidently, the stage passes the image through unchanged (safe fallback).

    Field ranges (validated below, raising ``ConfigurationError``):
      - ``max_detect_width`` >= 50
      - ``min_text_lines`` >= 2
      - ``poly_order`` in [1, 4]
      - ``min_line_width_ratio`` in (0, 1]
    """

    # Downscale width used for text-line detection (speed).
    max_detect_width: int = 1200
    # Minimum number of detected text lines required to model the curvature;
    # below this, fall back to a pass-through.
    min_text_lines: int = 4
    # Order of the polynomial fit to each text line's vertical curve across x.
    poly_order: int = 2
    # A detected text-line span must cover at least this fraction of the page
    # width to be treated as a line (rejects short fragments/margins). Kept
    # modest so partial lines still count; genuine image/decorative pages are
    # excluded by the blob-height filter, not this.
    min_line_width_ratio: float = 0.15

    @field_validator("max_detect_width")
    @classmethod
    def _validate_max_detect_width(cls, v: int) -> int:
        _require("dewarp", "max_detect_width", v, "[50, inf)", v >= 50)
        return v

    @field_validator("min_text_lines")
    @classmethod
    def _validate_min_text_lines(cls, v: int) -> int:
        _require("dewarp", "min_text_lines", v, "[2, inf)", v >= 2)
        return v

    @field_validator("poly_order")
    @classmethod
    def _validate_poly_order(cls, v: int) -> int:
        _require("dewarp", "poly_order", v, "[1, 4]", 1 <= v <= 4)
        return v

    @field_validator("min_line_width_ratio")
    @classmethod
    def _validate_min_line_width_ratio(cls, v: float) -> float:
        _require("dewarp", "min_line_width_ratio", v, "(0, 1]", 0.0 < v <= 1.0)
        return v


class StageType(str, Enum):
    """Supported pipeline stage types.

    Each value maps to exactly one ``StageConfig`` subclass (enforced by
    ``StageSpec``) and, later, one ``PreprocessingStage`` implementation built
    by the factory. The enum is intentionally extensible: new stages (e.g.
    sharpening, deskew, denoise) are added here as new members.
    """

    PAGE_SPLIT = "page_split"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    WHITE_BALANCE = "white_balance"
    ADJUSTMENT = "adjustment"
    DESKEW = "deskew"
    DEWARP = "dewarp"


# Maps each stage type to the concrete ``StageConfig`` subclass it requires.
# Used for cross-field validation in ``StageSpec`` (parallel to the batch
# processor's ``engine_config_map`` in ``validate_engine_config_matches_type``).
_STAGE_CONFIG_MAP = {
    StageType.PAGE_SPLIT: PageSplitConfig,
    StageType.CONTRAST_ENHANCEMENT: ContrastEnhancementConfig,
    StageType.WHITE_BALANCE: WhiteBalanceConfig,
    StageType.ADJUSTMENT: AdjustmentConfig,
    StageType.DESKEW: DeskewConfig,
    StageType.DEWARP: DewarpConfig,
}


class StageSpec(BaseModel):
    """One entry in the ordered pipeline: a stage type plus its config.

    The ``stage_config`` must be an instance of the concrete ``StageConfig``
    subclass declared for ``stage_type`` (page_split -> ``PageSplitConfig``,
    contrast_enhancement -> ``ContrastEnhancementConfig``). This mirrors
    ``BatchProcessorConfig.validate_engine_config_matches_type`` in the image
    batch processor.
    """

    stage_type: StageType
    stage_config: StageConfig

    @model_validator(mode="after")
    def _validate_config_matches_type(self) -> "StageSpec":
        """Ensure the concrete ``stage_config`` matches the declared type.

        Raises ``ConfigurationError`` (Requirement 6.3 / Error-Handling
        Scenario 5) identifying the offending stage type, the expected config
        subclass, and the supplied config subclass.
        """
        expected = _STAGE_CONFIG_MAP.get(self.stage_type)
        if expected is None:  # pragma: no cover - guarded by the enum
            raise ConfigurationError(
                f"Unsupported stage_type '{self.stage_type}'; supported types "
                f"are {[t.value for t in StageType]}"
            )
        if not isinstance(self.stage_config, expected):
            raise ConfigurationError(
                f"stage_type '{self.stage_type.value}' requires "
                f"{expected.__name__}, got "
                f"{type(self.stage_config).__name__}"
            )
        return self


def _normalize_extension(fmt: str) -> str:
    """Normalize an image format/extension to a canonical ``.ext`` form.

    Lower-cases and ensures a single leading dot so ``"JPEG"``, ``"jpeg"``,
    and ``".jpeg"`` all resolve to ``".jpeg"``.
    """
    normalized = fmt.strip().lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


class PipelineConfig(BaseModel):
    """Top-level, validated configuration for a preprocessing run.

    Mirrors ``BatchProcessorConfig`` in the image batch processor: non-empty
    path validators plus an ``@model_validator`` cross-field check. Raises
    ``ConfigurationError`` (rather than Pydantic's ``ValidationError``) so
    misconfiguration surfaces a domain error before any processing begins
    (Requirement 7).
    """

    source_dir: str = "phase_1/cookbook_images"
    output_dir: str
    stages: List[StageSpec]
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )
    output_format: Optional[str] = None
    max_workers: int = Field(default=1, ge=1)
    # When True, the first and last discovered source images (in reading order)
    # are excluded from the run. Useful for scans whose first/last pages are the
    # front/back cover photos rather than recipe content.
    skip_first_last: bool = False

    @field_validator("source_dir")
    @classmethod
    def _validate_source_dir(cls, v: str) -> str:
        """Reject a null/empty/whitespace source directory (Requirement 7.1)."""
        if v is None or not v.strip():
            raise ConfigurationError(
                "source_dir cannot be empty or whitespace"
            )
        return v

    @field_validator("output_dir")
    @classmethod
    def _validate_output_dir(cls, v: str) -> str:
        """Reject a null/empty/whitespace output directory (Requirement 7.1)."""
        if v is None or not v.strip():
            raise ConfigurationError(
                "output_dir cannot be empty or whitespace"
            )
        return v

    @field_validator("stages")
    @classmethod
    def _validate_stages_nonempty(cls, v: List[StageSpec]) -> List[StageSpec]:
        """Reject an empty stage list (Requirement 7.2)."""
        if not v:
            raise ConfigurationError(
                "stages cannot be empty; at least one stage is required"
            )
        return v

    @model_validator(mode="after")
    def _validate_output_not_within_source(self) -> "PipelineConfig":
        """Reject an output dir equal to or nested within the source dir.

        Uses resolved (absolute, symlink-free) paths so relative and
        ``..``-containing paths are compared correctly (Requirement 10.5).
        """
        source_resolved = Path(self.source_dir).resolve()
        output_resolved = Path(self.output_dir).resolve()
        if output_resolved.is_relative_to(source_resolved):
            raise ConfigurationError(
                "output_dir must not overlap the source directory: "
                f"output_dir {str(output_resolved)!r} is equal to or nested "
                f"within source_dir {str(source_resolved)!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_output_format(self) -> "PipelineConfig":
        """Resolve/validate the configured output format (Requirement 7.5/7.6).

        When no output format is configured (``None``), the format is resolved
        later from each source image's own format at processing time, so no
        action is taken here. When an output format is configured, it must be
        among the supported extensions; otherwise a ``ConfigurationError`` is
        raised.
        """
        if self.output_format is None:
            return self
        if not self.output_format.strip():
            raise ConfigurationError(
                "output_format cannot be empty or whitespace when provided"
            )
        normalized = _normalize_extension(self.output_format)
        supported = {_normalize_extension(e) for e in self.supported_extensions}
        if normalized not in supported:
            raise ConfigurationError(
                f"output_format {self.output_format!r} is not among the "
                f"supported formats {sorted(supported)}"
            )
        return self
