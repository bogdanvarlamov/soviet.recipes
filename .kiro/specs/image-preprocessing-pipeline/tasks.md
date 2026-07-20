# Implementation Plan: Image Preprocessing Pipeline

## Overview

This plan implements the standalone, pluggable Image Preprocessing Pipeline under
`phase_2/image_preprocessing_pipeline/`, mirroring the architecture of the existing
`image_batch_processor` (Strategy for stages, Factory for stage creation, Template-Method
orchestrator, Pydantic config hierarchy, lightweight dataclass results). Implementation is in
Python (>=3.11) managed with `uv`, adding Pillow and numpy as new dependencies. Tests use pytest
with Hypothesis property-based tests (`@settings(max_examples=100)`), one test per correctness
property, each tagged with `# Feature: image-preprocessing-pipeline, Property {N}: {description}`
and annotated with the requirement clauses it validates.

Tasks build incrementally: exceptions and config first, then models and I/O, then the stage
contract and the two shipped stages, then the factory, then the orchestrator, and finally the
standalone entry point and end-to-end integration. Every step wires into the previous one so no
code is left orphaned.

## Tasks

- [x] 1. Set up module structure, dependencies, and test scaffolding
  - Create the package directory structure under `phase_2/image_preprocessing_pipeline/`:
    `stages/`, `core/`, `config/`, `utils/`, plus `tests/unit/`, `tests/property/`, `tests/integration/`
  - Add `pyproject.toml` and install new dependencies with `uv add pillow numpy` (pydantic, pytest,
    hypothesis already available)
  - Create `tests/property/strategies.py` with Hypothesis strategies that generate synthetic
    in-memory images of varying dimensions/color modes and synthetic gutter spreads for detection
  - Add `__init__.py` files for each package
  - _Requirements: 14.1_

- [x] 2. Implement the exception hierarchy
  - Create `exceptions.py` with base `PreprocessingError` and subclasses `ImageLoadError`,
    `ImageSaveError`, `StageError`, and `ConfigurationError` (parallel to `exceptions.py` in
    image_batch_processor)
  - Ensure `StageError` can carry the identity of the failing stage; `ImageLoadError`/`ImageSaveError`
    carry the affected file path and failure cause
  - _Requirements: 3.6, 12.1, 12.2_

- [x] 3. Implement configuration models and validation
  - [x] 3.1 Implement the StageConfig hierarchy (Pydantic)
    - Create `config/settings.py` with base `StageConfig`, `PageSplitConfig` (method, split_ratio,
      gutter_margin, search_band_min, search_band_max, fallback_ratio, cover_filenames,
      treat_first_last_as_covers), and `ContrastEnhancementConfig` (method, factor, clip_limit)
    - Add field-level range validation (split_ratio in (0,1), gutter_margin in [0,0.5), bands in
      (0,1), factor > 0, clip_limit > 0) raising `ConfigurationError` on violation
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 5.3, 5.4, 5.5, 7.3_

  - [x] 3.2 Implement PipelineConfig and StageSpec with cross-field validation
    - Add `StageSpec` (stage_type enum + matching stage_config) and `PipelineConfig` (source_dir,
      output_dir, stages, supported_extensions, output_format, max_workers)
    - Validate non-empty/non-whitespace source and output dirs; non-empty stage list; each
      `stage_config` subclass matches its declared `stage_type`; `search_band_max > search_band_min`;
      reject output_dir equal to or nested within source_dir; resolve/validate output_format
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 7.7, 10.5_

  - [x] 3.3 Write unit tests for configuration validation
    - Test each range violation, empty-path rejection, empty stage list, type/config mismatch,
      band ordering, output/source overlap, and output-format resolution
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.5_

- [x] 4. Implement data models
  - [x] 4.1 Implement WorkingImage, ImageResult, and PipelineReport
    - Create `core/models.py` with `WorkingImage` (source_name, lineage, width, height, in-memory
      image), `ImageResult` dataclass (source_path, success, output_paths, output_count,
      stages_applied, error, processing_time), and `PipelineReport` dataclass (totals, successful,
      failed, total_output_files, processing_time, results) with a `success_rate()` accessor
      returning 0.0 when total is 0
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x] 4.2 Write unit tests for data models
    - Test `success_rate()` including the zero-source case, and that successes + failures equals
      total sources
    - _Requirements: 13.1, 13.2_

- [x] 5. Implement the Image I/O utility, discovery, and output naming
  - [x] 5.1 Implement image load/save/crop/dimensions and source discovery
    - Create `utils/image_io.py` with functions to load an image into memory (raising
      `ImageLoadError` on decode failure), report width/height, crop a region, save an in-memory
      image to a destination path (creating the output dir, raising `ImageSaveError` on write
      failure), and discover eligible source files non-recursively in deterministic sorted order
      with case-insensitive extension matching; raise a domain error for an inaccessible source dir
      and return an empty set when nothing matches
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 10.1, 12.1, 12.2_

  - [x] 5.2 Implement deterministic output naming
    - Add a pure naming function deriving each output name from source name + ordered lineage,
      emitting the left page name before the right page name, preserving global source order across
      the dataset, guaranteeing pairwise-distinct names, and detecting collisions
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 5.3 Write property test for deterministic naming
    - **Property 8: Deterministic naming**
    - **Validates: Requirements 8.1, 8.2, 8.4, 8.5**

- [x] 6. Implement the PreprocessingStage interface
  - Create `stages/base.py` with the abstract `PreprocessingStage` contract (apply over a working
    set, config validation, no in-place mutation, no file I/O, at-least-one output, order
    preservation), analogous to `ExtractionEngine`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7. Implement the ContrastEnhancementStage (1→1)
  - [x] 7.1 Implement ContrastEnhancementStage
    - Create `stages/contrast.py` producing exactly one output per input with identical dimensions,
      supporting `linear` (factor), `adaptive` (clip_limit), and `histogram_equalization` methods,
      producing a valid encodable image, and raising `StageError` when the method cannot process the
      input color mode
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 7.2 Write property test for the 1→1 contrast stage
    - **Property 3: 1→1 stage preserves count and dimensions**
    - **Validates: Requirements 5.1, 5.2**

  - [x] 7.3 Write unit tests for contrast methods and error handling
    - Test each method preserves dimensions and that an unsupported color mode raises `StageError`
    - _Requirements: 5.3, 5.4, 5.5, 5.7_

- [x] 8. Implement the PageSplitStage (1→N)
  - [x] 8.1 Implement fixed-midpoint splitting, gutter margins, and cover handling
    - Create `stages/page_split.py` producing exactly two ordered outputs (left before right) for an
      interior spread, splitting at `split_ratio` × width, trimming `gutter_margin` × width / 2 from
      each side of the spine, and passing covers through as exactly one output; ensure regions lie
      within bounds and do not overlap; raise `StageError` when a computed region is < 1 px wide
    - _Requirements: 4.1, 4.2, 4.5, 4.6, 4.7, 4.8_

  - [x] 8.2 Implement content-aware gutter detection with midpoint fallback
    - Add the `gutter_detection` method locating the spine within the central search band via a
      column intensity profile (numpy), falling back to `fallback_ratio` and setting a
      fallback-used indicator when no confident gutter is found
    - _Requirements: 4.3, 4.4_

  - [x] 8.3 Write property test for split page count
    - **Property 4: Split produces the expected page count**
    - **Validates: Requirements 4.1, 4.6**

  - [x] 8.4 Write property test for split region bounds and non-overlap
    - **Property 5: Split regions within bounds and non-overlapping**
    - **Validates: Requirements 4.7**

  - [x] 8.5 Write property test for gutter detection band and fallback
    - **Property 7: Spine within search bounds or flagged fallback**
    - **Validates: Requirements 4.3, 4.4**

  - [x] 8.6 Write unit tests for degenerate-region handling
    - Test that a computed region under 1 px wide raises `StageError` and leaves input unchanged
    - _Requirements: 4.8_

- [x] 9. Write stage-contract property tests spanning both stages
  - [x] 9.1 Write property test for order preservation across a stage
    - **Property 2: No silent drops (at-least-one output)**
    - **Validates: Requirements 3.2**

  - [x] 9.2 Write property test for non-degenerate outputs
    - **Property 6: Non-degenerate outputs**
    - **Validates: Requirements 5.6, 11.1**

- [x] 10. Implement the StageFactory
  - [x] 10.1 Implement StageFactory
    - Create `core/factory.py` mapping stage types (`page_split`, `contrast_enhancement`) to
      implementations, validating config/type match, raising `ConfigurationError` (listing supported
      types) for unknown types, building the ordered stage list in configured order, returning an
      empty list for an empty stage list, completing all validation before any processing, and
      exposing supported stage types
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 10.2 Write unit tests for StageFactory
    - Test correct stage per type, unknown-type and mismatched-config errors, ordered construction,
      empty-list handling, and supported-type enumeration
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.7_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement the PreprocessingPipeline orchestrator
  - [x] 12.1 Implement pipeline core threading and output writing
    - Create `core/pipeline.py`: build the ordered stage list once per run and reuse it; for each
      discovered source seed a single-image working set (source name + empty lineage), apply each
      stage in order passing stage k's full output as stage k+1's sole input, treat an empty stage
      list as a passthrough, write exactly the final working set to the output directory with
      deterministic names, and reject any write targeting a path outside the output directory
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.1, 10.2, 11.2_

  - [x] 12.2 Implement per-source failure isolation and reporting
    - Wrap per-source processing to catch `ImageLoadError`/`StageError`/`ImageSaveError`, record one
      failed `ImageResult` (source id, error, failing stage identity, output count) and continue
      with remaining sources; record successful results (paths, count, ordered stages applied);
      aggregate a `PipelineReport` and leave source files byte-for-byte unchanged
    - _Requirements: 10.3, 10.4, 12.3, 12.4, 12.5, 12.6, 12.7, 13.1, 13.3, 13.4_

  - [x] 12.3 Write property test for global order preservation
    - **Property 1: Order preservation through the pipeline**
    - **Validates: Requirements 3.3, 8.3**

  - [x] 12.4 Write property test for determinism of results
    - **Property 9: Determinism of results**
    - **Validates: Requirements 9.1, 9.2, 9.3**

  - [x] 12.5 Write property test for source immutability
    - **Property 10: Source immutability**
    - **Validates: Requirements 10.3, 10.4**

  - [x] 12.6 Write property test for guaranteed valid output on readable input
    - **Property 11: Guaranteed result for readable input**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

  - [x] 12.7 Write property test for per-source failure isolation
    - **Property 12: Per-source failure isolation**
    - **Validates: Requirements 12.3, 12.6**

- [x] 13. Implement the standalone entry point and wire everything together
  - [x] 13.1 Implement the CLI/main entry point
    - Create `main.py` building a default `PipelineConfig` (source `phase_1/cookbook_images`,
      ordered page-split then contrast stages, configured output dir), aborting with a clear
      configuration error before processing when validation fails, running the pipeline, logging the
      report summary (total sources, total outputs, successes, failures), and writing outputs only
      under the output directory with reading-order-preserving names consumable by the batch
      processor's discovery
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 13.2 Write end-to-end integration test
    - Run the full pipeline (page split then contrast) against a small fixture directory, assert
      output counts, ordered collision-free names, readable outputs with expected dimensions,
      compatibility with the batch processor's `discover_images` order, and that the source
      directory is unchanged after the run
    - _Requirements: 14.4, 14.5, 10.3, 10.4_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core
  implementation tasks are never optional.
- Each task references specific requirement clauses for traceability.
- Each of the 12 correctness properties maps to exactly one property-based test task, annotated with
  its property number and the requirement clauses it validates.
- Property tests use Hypothesis with `@settings(max_examples=100)` and the required
  `# Feature: image-preprocessing-pipeline, Property {N}: {description}` comment tag.
- The pipeline is standalone: it runs before extraction, and the user points the existing image
  batch processor's `image_dir` at this tool's `output_dir`. `phase_1/cookbook_images/` stays
  read-only.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "3.1", "4.1", "5.1", "6"] },
    { "id": 2, "tasks": ["3.2", "4.2", "5.2", "7.1", "8.1"] },
    { "id": 3, "tasks": ["3.3", "5.3", "7.2", "7.3", "8.2", "10.1"] },
    { "id": 4, "tasks": ["8.3", "8.4", "8.5", "8.6", "9.1", "9.2", "10.2", "12.1"] },
    { "id": 5, "tasks": ["12.2"] },
    { "id": 6, "tasks": ["12.3", "12.4", "12.5", "12.6", "12.7", "13.1"] },
    { "id": 7, "tasks": ["13.2"] }
  ]
}
```
