"""Base preprocessing stage interface.

Defines the abstract contract every pipeline stage implements. A stage is an
interchangeable, pluggable transform step (Strategy pattern), analogous to
``ExtractionEngine`` in the image batch processor: the orchestrator depends only
on this small, stable interface, so stages are independently testable and
swappable.

The core abstraction is a **working set** of in-memory images. A stage consumes
the current working set and produces the next one. A 1->1 stage (e.g. contrast
enhancement) maps each input image to exactly one output image; a 1->N stage
(e.g. page splitting) maps each input image to one or more output images.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core.models import WorkingImage


class PreprocessingStage(ABC):
    """Abstract contract for a pluggable preprocessing stage.

    Every concrete stage transforms a working set of in-memory images into a new
    working set. The orchestrator threads a source image's working set through
    each configured stage in order, passing the full output set of stage *k* as
    the sole input set of stage *k+1*.

    Contract guarantees every implementation MUST uphold:

    - **No in-place mutation** (Requirement 3.1): a stage never mutates the pixel
      data, dimensions, or metadata of its input images. It produces new
      ``WorkingImage`` instances (referential transparency), which keeps re-runs
      deterministic.
    - **At least one output per input** (Requirement 3.2): for each input image a
      stage produces N output images where N is an integer and N >= 1. No input
      image is silently dropped. A stage that wishes to reject an image raises a
      :class:`StageError` rather than returning an empty set.
    - **Relative-order preservation** (Requirement 3.3): outputs derived from an
      earlier input image sort before outputs derived from a later input image,
      and the outputs of a single 1->N expansion are emitted in reading order
      (e.g. the left page before the right page).
    - **In-memory only** (Requirement 3.4): a stage operates solely on in-memory
      images.
    - **No file I/O** (Requirement 3.5): a stage performs no file input or
      output. Loading source images and saving the final set are the
      responsibility of the orchestrator and the Image I/O utility, which keeps
      stages deterministic and unit-testable.
    - **Failure signalling** (Requirement 3.6): if a stage cannot process an
      image it raises a :class:`StageError` identifying the failing stage, leaves
      the input images unmodified, and returns no working set.
    """

    @abstractmethod
    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        """Transform a working set of images into a new working set.

        Args:
            working_set: The ordered input working set (one or more in-memory
                :class:`WorkingImage` instances). For the first stage applied to
                a source image this contains exactly one image; later stages may
                receive several.

        Returns:
            A new ordered working set of one or more ``WorkingImage`` instances.
            A 1->1 stage returns the same count it received; a 1->N stage may
            return more. The input images are not mutated in place, relative
            order is preserved, and every input yields at least one output.

        Raises:
            StageError: If the stage cannot process an image. The error
                identifies the failing stage; the input images are left
                unmodified and no working set is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the stage is properly configured.

        Surfaces configuration errors early, before any processing begins,
        mirroring ``ExtractionEngine.validate_config``.

        Returns:
            True if the configuration is valid.

        Raises:
            ConfigurationError: If the stage configuration is invalid.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def stage_type(self) -> str:
        """The stage type string identifying this implementation.

        Matches the type string the :class:`StageFactory` maps to this
        implementation (e.g. ``"page_split"`` or ``"contrast_enhancement"``) and
        is recorded in the per-source ``ImageResult`` as a stage-applied name.

        Returns:
            The stage type identifier.
        """
        raise NotImplementedError
