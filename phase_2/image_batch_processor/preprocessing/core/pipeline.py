"""PreprocessingPipeline orchestrator (template-method).

Coordinates an end-to-end preprocessing run, analogous to ``BatchProcessor`` in
the image batch processor. It builds the ordered stage list **once** per run and
reuses it for every source (Requirement 2.5); discovers source images in
deterministic order; for each source seeds a single-image working set and threads
it through every stage in order, passing stage *k*'s full output set as the sole
input of stage *k+1* (Requirement 2.2); treats an empty stage list as a
pass-through (Requirement 2.3); assigns deterministic, collision-free,
reading-order-preserving names across the whole run; and writes exactly the final
working set of each source to the output directory (Requirement 2.4).

Every write target is resolved and checked to lie strictly inside the resolved
output directory before writing, so no write can escape the permitted output
directory (Requirements 10.1, 10.2). For any readable source processed under a
validated configuration, at least one output file is written (Requirement 11.2).

Per-source failure isolation (task 12.2)
----------------------------------------
Each source is processed independently. If loading a source or applying a stage
raises :class:`ImageLoadError` / :class:`StageError`, or writing its outputs
raises :class:`ImageSaveError`, the pipeline records exactly one failed
:class:`ImageResult` for that source (source id, error message, the failing stage
identity for a :class:`StageError`, and the count of output images actually
written) and continues with the remaining sources without aborting the batch
(Requirements 12.3-12.7). Successful sources record their output paths, output
count, and the ordered names of stages applied (Requirement 13.3). The run only
ever reads sources and writes under the output directory, so source files are
left byte-for-byte unchanged (Requirements 10.3, 10.4). A run-wide naming
collision (:class:`ConfigurationError` from :func:`assign_output_names`) is a
run-level error and is allowed to surface rather than being isolated per source.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type

from ..config.settings import PipelineConfig
from .factory import StageFactory
from .models import ImageResult, PipelineReport, WorkingImage
from ..exceptions import ImageLoadError, ImageSaveError, StageError
from ..stages.base import PreprocessingStage
from ..utils.image_io import (
    discover_source_images,
    get_dimensions,
    load_image,
    save_image,
)
from ..utils.naming import assign_output_names


@dataclass
class _SourceOutcome:
    """Intermediate result of threading one source through all stages.

    Carries the source path, its final working set (the images to be written),
    and the wall-clock time spent processing it. Naming and writing happen after
    every source has been processed so global output order can be assigned across
    the whole run.
    """

    source_path: Path
    working_set: List[WorkingImage]
    processing_time: float


@dataclass
class _SourceProcessing:
    """Outcome of the *load + stage* phase for a single source.

    Isolates a per-source failure so one bad photo (or one failing stage) does
    not abort the batch (Requirement 12.6). Exactly one of ``outcome`` /
    ``error`` is set:

    - ``outcome`` is present when the source loaded and threaded through every
      stage successfully; its final working set is eligible for naming/writing.
    - ``error`` is present (with ``outcome`` ``None``) when loading the source
      raised :class:`ImageLoadError` or a stage raised :class:`StageError`. For a
      :class:`StageError` the message already embeds the failing stage identity
      (Requirement 12.5).
    """

    source_path: Path
    outcome: Optional[_SourceOutcome] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class PreprocessingPipeline:
    """Orchestrates a preprocessing run over a directory of source images.

    Template-method style, mirroring ``BatchProcessor``: :meth:`run` defines the
    algorithm (build stages once, discover sources, thread each source's working
    set through the ordered stages, name the final set globally, write it, and
    aggregate a report) and delegates the per-source transform to the configured
    stages built by the :class:`StageFactory`. Per-source failures are isolated
    and recorded so the batch always runs to completion.
    """

    def __init__(
        self,
        config: PipelineConfig,
        factory: Optional[Type[StageFactory]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Create the pipeline from a validated configuration.

        Args:
            config: The validated :class:`PipelineConfig` for the run.
            factory: The stage factory used to build the ordered stage list.
                Defaults to :class:`StageFactory`. Injectable for testing.
            logger: Logger instance (a module logger is used when omitted).
        """
        self.config = config
        self.factory = factory or StageFactory
        self.logger = logger or logging.getLogger(__name__)
        # Mirror BatchProcessor: never below 1 (1 = sequential, deterministic).
        self.max_workers = max(1, config.max_workers)

    def run(self) -> PipelineReport:
        """Execute the full preprocessing run and return a report.

        Builds the ordered stage list exactly once and reuses it for every source
        (Requirement 2.5), discovers sources in deterministic order, threads each
        through the stages, assigns global deterministic names, writes every final
        image under the output directory, and aggregates a :class:`PipelineReport`.

        Per-source failures (:class:`ImageLoadError`, :class:`StageError`,
        :class:`ImageSaveError`) are isolated: each yields exactly one failed
        :class:`ImageResult` and the batch continues (Requirements 12.3-12.7). The
        report's successes plus failures equal the total source count, and
        ``total_output_files`` is the sum of output counts across successful
        sources (Requirement 13.1).

        Returns:
            A :class:`PipelineReport` summarizing the run.

        Raises:
            ConfigurationError: If output naming collides across the run
                (surfaced by :func:`assign_output_names`); this is a run-level
                error, not isolated per source.
        """
        run_start = time.time()

        # Build the ordered stage list ONCE and reuse it for every source
        # (Requirement 2.5). Validate configuration up front so misconfiguration
        # surfaces before any source is processed.
        stages = self.factory.create_stages(self.config.stages)
        for stage in stages:
            stage.validate_config()

        sources = discover_source_images(
            self.config.source_dir, self.config.supported_extensions
        )

        # Optionally drop the first and last discovered images (front/back
        # cover photos). Slicing [1:-1] naturally yields an empty set when there
        # are two or fewer sources.
        if self.config.skip_first_last and sources:
            skipped = [sources[0]] + ([sources[-1]] if len(sources) > 1 else [])
            sources = sources[1:-1]
            self.logger.info(
                "Skipping first/last source(s) (covers): %s",
                ", ".join(p.name for p in skipped),
            )

        total_sources = len(sources)
        self.logger.info(
            "Starting preprocessing run: %d source(s) discovered in %s "
            "(stages: %d, workers: %d)",
            total_sources,
            self.config.source_dir,
            len(stages),
            self.max_workers,
        )

        # Phase 1: load + thread each source through the ordered stages, isolating
        # per-source load/stage failures. The returned list preserves source
        # discovery order regardless of completion order under concurrency.
        processed = self._process_sources(stages, sources)

        # Phase 2: assign deterministic, collision-free, reading-order-preserving
        # names across the WHOLE run (Requirements 8.3-8.5), for the sources that
        # threaded through all stages successfully. Failed sources contribute no
        # working images, so naming spans only the successful ones while global
        # source order is preserved (they are supplied in source processing order,
        # and within each source in reading order). A collision here is a
        # run-level error and is allowed to propagate.
        all_images: List[WorkingImage] = []
        for item in processed:
            if item.outcome is not None:
                all_images.extend(item.outcome.working_set)
        assignments = assign_output_names(all_images, self.config.output_format)

        # Phase 3: write each successful source's final working set, isolating
        # per-source write failures, and aggregate one ImageResult per source.
        output_dir = Path(self.config.output_dir)
        output_dir_resolved = output_dir.resolve()
        stage_names = [stage.stage_type for stage in stages]

        results: List[ImageResult] = []
        total_output_files = 0
        cursor = 0  # position within the run-wide `assignments` list

        for item in processed:
            if item.outcome is None:
                # Failed during load or a stage: record one failed result with no
                # output written (Requirements 12.3, 12.4, 12.5, 12.7, 13.4).
                results.append(
                    ImageResult(
                        source_path=str(item.source_path),
                        success=False,
                        output_paths=[],
                        output_count=0,
                        stages_applied=[],
                        error=item.error,
                        processing_time=item.processing_time,
                    )
                )
                self.logger.warning(
                    "\u2717 %s failed: %s", item.source_path.name, item.error
                )
                continue

            # Slice this source's own (WorkingImage, name) assignments so writing
            # stays aligned even if a prior/next source needs different counts.
            count = len(item.outcome.working_set)
            source_assignments = assignments[cursor : cursor + count]
            cursor += count

            write_start = time.time()
            output_paths: List[str] = []
            try:
                self._write_source_outputs(
                    source_assignments,
                    output_dir,
                    output_dir_resolved,
                    output_paths,
                )
            except ImageSaveError as exc:
                # A write failed part-way through this source. Record a failed
                # result carrying the count of outputs ACTUALLY written for that
                # source (Requirement 12.7) and continue the batch (12.6).
                results.append(
                    ImageResult(
                        source_path=str(item.outcome.source_path),
                        success=False,
                        output_paths=output_paths,
                        output_count=len(output_paths),
                        stages_applied=[],
                        error=str(exc),
                        processing_time=item.outcome.processing_time
                        + (time.time() - write_start),
                    )
                )
                self.logger.warning(
                    "\u2717 %s failed while writing after %d output(s): %s",
                    item.outcome.source_path.name,
                    len(output_paths),
                    exc,
                )
                continue

            total_output_files += len(output_paths)
            results.append(
                ImageResult(
                    source_path=str(item.outcome.source_path),
                    success=True,
                    output_paths=output_paths,
                    output_count=len(output_paths),
                    stages_applied=list(stage_names),
                    error=None,
                    processing_time=item.outcome.processing_time
                    + (time.time() - write_start),
                )
            )
            self.logger.info(
                "OK %s -> %d output(s)",
                item.outcome.source_path.name,
                len(output_paths),
            )

        successful = sum(1 for result in results if result.success)
        failed = total_sources - successful
        run_time = time.time() - run_start
        self.logger.info(
            "Preprocessing run complete: %d/%d source(s) succeeded, "
            "%d failed, %d output file(s) written to %s (%.2fs)",
            successful,
            total_sources,
            failed,
            total_output_files,
            output_dir_resolved,
            run_time,
        )

        return PipelineReport(
            total_sources=total_sources,
            successful=successful,
            failed=failed,
            total_output_files=total_output_files,
            processing_time=run_time,
            results=results,
        )

    # -- per-source processing -------------------------------------------

    def _process_sources(
        self, stages: List[PreprocessingStage], sources: List[Path]
    ) -> List[_SourceProcessing]:
        """Load + thread every source through the stages, preserving source order.

        Sources are processed sequentially when ``max_workers <= 1`` and via a
        thread pool otherwise (mirroring ``BatchProcessor._run_batch``). The
        returned list is always in source discovery order regardless of the order
        in which concurrent work completes, so downstream global naming is
        deterministic. Per-source load/stage failures are captured (not raised)
        so the batch is never aborted (Requirement 12.6).

        Args:
            stages: The ordered stage list, built once and shared across sources.
            sources: The discovered source paths, in deterministic order.

        Returns:
            One :class:`_SourceProcessing` per source, in source discovery order.
        """
        processed: List[Optional[_SourceProcessing]] = [None] * len(sources)

        def handle(indexed: Tuple[int, Path]) -> Tuple[int, _SourceProcessing]:
            index, source_path = indexed
            return index, self._process_one_source(stages, source_path)

        indexed_sources = list(enumerate(sources))

        if self.max_workers <= 1:
            for item in indexed_sources:
                index, record = handle(item)
                processed[index] = record
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(handle, item) for item in indexed_sources]
                for future in as_completed(futures):
                    index, record = future.result()
                    processed[index] = record

        return processed  # type: ignore[return-value]

    def _process_one_source(
        self, stages: List[PreprocessingStage], source_path: Path
    ) -> _SourceProcessing:
        """Seed and thread a single source's working set through all stages.

        Seeds a working set of exactly one :class:`WorkingImage` loaded from the
        source, carrying the source's file name and an empty lineage (Requirement
        2.1). Applies each stage in order, passing stage *k*'s complete output set
        as the sole input of stage *k+1* (Requirement 2.2). When the stage list is
        empty the seeded working set is returned unchanged (Requirement 2.3).

        Load and stage failures are caught here and returned as a failed
        :class:`_SourceProcessing` so the caller can record the failure and
        continue with the remaining sources (Requirements 12.3-12.6). A
        :class:`StageError`'s message embeds the failing stage's identity
        (Requirement 12.5).

        Args:
            stages: The ordered stage list.
            source_path: The source image path to process.

        Returns:
            A :class:`_SourceProcessing`: on success it carries the final working
            set; on failure it carries the error message and ``outcome=None``.
        """
        start = time.time()

        try:
            image = load_image(source_path)
            width, height = get_dimensions(image)
            # Seed: exactly one image, source name = the file name, empty lineage.
            working_set: List[WorkingImage] = [
                WorkingImage(
                    source_name=source_path.name,
                    image=image,
                    width=width,
                    height=height,
                    lineage=[],
                )
            ]

            # Apply each stage in order; stage k's full output is stage k+1's sole
            # input. An empty stage list leaves the seeded set unchanged.
            for stage in stages:
                working_set = stage.apply(working_set)
        except (ImageLoadError, StageError) as exc:
            return _SourceProcessing(
                source_path=source_path,
                outcome=None,
                error=str(exc),
                processing_time=time.time() - start,
            )

        return _SourceProcessing(
            source_path=source_path,
            outcome=_SourceOutcome(
                source_path=source_path,
                working_set=working_set,
                processing_time=time.time() - start,
            ),
        )

    def _write_source_outputs(
        self,
        source_assignments: List[Tuple[WorkingImage, str]],
        output_dir: Path,
        output_dir_resolved: Path,
        output_paths: List[str],
    ) -> None:
        """Write one source's final working set, appending each written path.

        Writes exactly the source's own ``(WorkingImage, name)`` assignments, so
        each source writes exactly its own final images and no others (Requirement
        2.4). Each target is resolved and verified to lie strictly inside the
        resolved output directory before writing (Requirements 10.1, 10.2).

        Paths are appended to the caller-owned ``output_paths`` list *as each
        image is written*, so if an :class:`ImageSaveError` is raised part-way
        through the caller can observe the count of outputs actually written for
        the source (Requirement 12.7).

        Args:
            source_assignments: This source's ``(WorkingImage, name)`` pairs, in
                reading order.
            output_dir: The (unresolved) output directory.
            output_dir_resolved: The resolved output directory used for the
                containment check.
            output_paths: Caller-owned accumulator; each successfully written
                path is appended in order.

        Raises:
            ImageSaveError: If a resolved target escapes the output directory, or
                an image cannot be written. Paths written before the failure
                remain recorded in ``output_paths``.
        """
        for working_image, name in source_assignments:
            destination = self._resolve_write_target(
                output_dir, output_dir_resolved, name
            )
            saved = save_image(working_image.image, destination)
            output_paths.append(str(saved))

    @staticmethod
    def _resolve_write_target(
        output_dir: Path, output_dir_resolved: Path, name: str
    ) -> Path:
        """Resolve an output name to a path strictly inside the output directory.

        Rejects any target whose resolved path is the output directory itself or
        lies outside it (defends against absolute names or ``..`` traversal),
        leaving the target unchanged (Requirements 10.1, 10.2).

        Args:
            output_dir: The (unresolved) output directory.
            output_dir_resolved: The resolved output directory.
            name: The assigned output file name.

        Returns:
            The resolved, containment-checked destination path.

        Raises:
            ImageSaveError: If the resolved target is not strictly within the
                output directory.
        """
        candidate = (output_dir / name).resolve()
        if candidate == output_dir_resolved or not candidate.is_relative_to(
            output_dir_resolved
        ):
            raise ImageSaveError(
                str(candidate),
                "resolved path is outside the permitted output directory "
                f"{str(output_dir_resolved)!r}",
            )
        return candidate
