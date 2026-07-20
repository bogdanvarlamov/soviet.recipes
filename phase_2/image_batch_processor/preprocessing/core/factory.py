"""Factory for creating preprocessing stage instances.

Mirrors ``EngineFactory`` in the image batch processor: it maps a stage type
(the :class:`StageType` enum, or its string value) to a concrete
:class:`PreprocessingStage` implementation, validates that the supplied config
subclass matches the requested stage type, and builds the ordered stage list for
a run.

The factory is the single place that turns validated configuration
(:class:`StageSpec` entries from a :class:`PipelineConfig`) into runnable stage
objects. It performs *all* stage-type and config validation up front — before
constructing any stage — so a misconfigured pipeline is rejected with a
:class:`ConfigurationError` and no stage ever begins processing (Requirement
6.6). New stages are added by registering them in ``_STAGE_REGISTRY``.
"""

from typing import Dict, List, Tuple, Type, Union

from ..config.settings import (
    AdjustmentConfig,
    ContrastEnhancementConfig,
    DeskewConfig,
    DewarpConfig,
    PageSplitConfig,
    StageConfig,
    StageSpec,
    StageType,
    WhiteBalanceConfig,
)
from ..exceptions import ConfigurationError
from ..stages.adjustment import AdjustmentStage
from ..stages.base import PreprocessingStage
from ..stages.contrast import ContrastEnhancementStage
from ..stages.deskew import DeskewStage
from ..stages.dewarp import DewarpStage
from ..stages.page_split import PageSplitStage
from ..stages.white_balance import WhiteBalanceStage


class StageFactory:
    """Creates configured :class:`PreprocessingStage` instances from stage specs.

    Parallel to ``EngineFactory``: a stage is selected by its type and built from
    a validated config. The factory maps each supported :class:`StageType` to a
    ``(config subclass, stage class)`` pair, validates the supplied config
    against the declared subclass, and constructs the implementation.
    """

    # Maps each supported stage type to the concrete ``StageConfig`` subclass it
    # requires and the ``PreprocessingStage`` implementation it builds. Adding a
    # new stage is a single registry entry (open for extension).
    _STAGE_REGISTRY: Dict[
        StageType, Tuple[Type[StageConfig], Type[PreprocessingStage]]
    ] = {
        StageType.PAGE_SPLIT: (PageSplitConfig, PageSplitStage),
        StageType.CONTRAST_ENHANCEMENT: (
            ContrastEnhancementConfig,
            ContrastEnhancementStage,
        ),
        StageType.WHITE_BALANCE: (WhiteBalanceConfig, WhiteBalanceStage),
        StageType.ADJUSTMENT: (AdjustmentConfig, AdjustmentStage),
        StageType.DESKEW: (DeskewConfig, DeskewStage),
        StageType.DEWARP: (DewarpConfig, DewarpStage),
    }

    @classmethod
    def get_supported_stage_types(cls) -> List[str]:
        """Return the complete set of supported stage type strings.

        Parallel to ``EngineFactory.get_supported_engines`` (Requirement 6.7).

        Returns:
            The supported stage type string values, in registration order.
        """
        return [stage_type.value for stage_type in cls._STAGE_REGISTRY]

    @classmethod
    def create_stage(cls, spec: StageSpec) -> PreprocessingStage:
        """Create exactly one stage instance from a single stage spec.

        Validates the stage type is supported and the supplied config subclass
        matches that type, then constructs and returns exactly one corresponding
        :class:`PreprocessingStage` implementation (Requirement 6.1).

        Args:
            spec: The stage spec (a stage type plus its matching config).

        Returns:
            A single configured :class:`PreprocessingStage` instance.

        Raises:
            ConfigurationError: If the stage type is not supported (Requirement
                6.2) or the config subclass does not match the declared stage
                type (Requirement 6.3). No stage instance is created in either
                case.
        """
        _, stage_cls = cls._validate_spec(spec)
        return stage_cls(spec.stage_config)

    @classmethod
    def create_stages(cls, specs: List[StageSpec]) -> List[PreprocessingStage]:
        """Build the ordered stage list from a list of stage specs.

        Produces one :class:`PreprocessingStage` per entry, in the identical
        index order as ``specs`` (Requirement 6.4). An empty spec list yields an
        empty stage list without error (Requirement 6.5).

        All stage-type and config validation for *every* entry completes before
        any stage is constructed, so a single invalid entry raises a
        :class:`ConfigurationError` and no stage is created (Requirement 6.6).

        Args:
            specs: The ordered list of stage specs (``PipelineConfig.stages``).

        Returns:
            An ordered list of configured stages, one per spec, in the same
            order. Empty when ``specs`` is empty.

        Raises:
            ConfigurationError: If any entry has an unsupported stage type
                (Requirement 6.2) or a config subclass that does not match its
                declared type (Requirement 6.3). No stages are created.
        """
        # Validate every entry first (Requirement 6.6): all-or-nothing, so no
        # stage is constructed if any entry is invalid.
        resolved: List[Tuple[StageSpec, Type[PreprocessingStage]]] = []
        for spec in specs:
            _, stage_cls = cls._validate_spec(spec)
            resolved.append((spec, stage_cls))

        # Only after full validation do we construct, preserving order.
        return [stage_cls(spec.stage_config) for spec, stage_cls in resolved]

    @classmethod
    def _validate_spec(
        cls, spec: StageSpec
    ) -> Tuple[Type[StageConfig], Type[PreprocessingStage]]:
        """Validate one spec and return its ``(config class, stage class)`` pair.

        Args:
            spec: The stage spec to validate.

        Returns:
            The registered ``(config subclass, stage class)`` pair for the spec's
            stage type.

        Raises:
            ConfigurationError: If the stage type is unsupported (Requirement
                6.2) or the supplied config subclass does not match the declared
                stage type (Requirement 6.3).
        """
        stage_type = cls._resolve_stage_type(spec.stage_type)
        config_cls, stage_cls = cls._STAGE_REGISTRY[stage_type]

        if not isinstance(spec.stage_config, config_cls):
            raise ConfigurationError(
                f"stage_type '{stage_type.value}' requires "
                f"{config_cls.__name__}, got "
                f"{type(spec.stage_config).__name__}"
            )

        return config_cls, stage_cls

    @classmethod
    def _resolve_stage_type(cls, stage_type: Union[StageType, str]) -> StageType:
        """Resolve a stage type (enum or string) to a supported :class:`StageType`.

        Args:
            stage_type: The requested stage type, as a :class:`StageType` enum
                member or its string value.

        Returns:
            The resolved :class:`StageType` present in the registry.

        Raises:
            ConfigurationError: If the stage type is not among the supported
                types. The message names the unsupported type and enumerates the
                complete set of supported types (Requirement 6.2).
        """
        resolved: Union[StageType, None] = None
        if isinstance(stage_type, StageType):
            resolved = stage_type if stage_type in cls._STAGE_REGISTRY else None
        else:
            # Accept the raw string value (e.g. "page_split").
            for candidate in cls._STAGE_REGISTRY:
                if candidate.value == stage_type:
                    resolved = candidate
                    break

        if resolved is None:
            raise ConfigurationError(
                f"Unsupported stage type: {stage_type!r}. Supported types are: "
                f"{cls.get_supported_stage_types()}"
            )
        return resolved
