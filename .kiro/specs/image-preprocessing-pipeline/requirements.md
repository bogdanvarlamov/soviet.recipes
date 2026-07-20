# Requirements Document

## Introduction

The Image Preprocessing Pipeline is a standalone, pluggable tool that prepares scanned cookbook
photos before the Phase 2 extraction pipeline runs. It reads source photos from a read-only input
directory (default `phase_1/cookbook_images/`), threads each image through an ordered list of
configurable stages, and writes fully processed single-page images to a separate output directory.
The user then points the existing image batch processor's `image_dir` at that output directory.

The pipeline is built around a working set of in-memory images: each stage consumes the current set
and produces the next set, so image counts can grow (page splitting, 1→N) or stay the same (contrast
enhancement, 1→1). Two stages ship in the first version — page splitting and contrast enhancement —
but the architecture is explicitly open-ended so additional stages can be added later as plugins
without changing the orchestrator. The tool is deterministic, offline, and never modifies the Phase 1
source images.

These requirements are derived from the approved design document and are organized to trace back to
the design's components, data models, correctness properties, and error-handling scenarios.

## Glossary

- **Preprocessing_Pipeline**: The orchestrator that discovers source images, builds the ordered stage
  list, threads each source image's working set through every stage in order, writes final images,
  and produces a report. (Design Component 1.)
- **Preprocessing_Stage**: The abstract contract that every pluggable transform step implements.
  (Design Component 2.)
- **Page_Split_Stage**: The 1→N stage that splits a photo of an open book into left and right pages.
  (Design Component 3.)
- **Contrast_Enhancement_Stage**: The 1→1 stage that increases contrast between text and page
  background. (Design Component 4.)
- **Stage_Factory**: The component that maps a stage type string to a stage implementation, validates
  its config, and builds the ordered stage list. (Design Component 5.)
- **Image_IO**: The utility layer that loads images into memory, reports dimensions, crops regions,
  saves images, and discovers source files. (Design Component 6.)
- **Entry_Point**: The standalone CLI/main that builds a validated configuration, runs the pipeline,
  and logs a summary. (Design Component 7.)
- **Config_Validator**: The Pydantic validation layer for `PipelineConfig`, `StageSpec`, and
  `StageConfig` subclasses.
- **Working_Set**: The set of one or more in-memory images flowing through the pipeline for a single
  source image.
- **Working_Image**: One in-memory image in the working set, carrying its source name, lineage, and
  dimensions.
- **Lineage**: The ordered list of tokens describing how a Working_Image was produced, used to build
  deterministic output names.
- **Source_Image**: A photo discovered in the source directory that seeds one working set.
- **Pipeline_Report**: The batch summary of a run (totals, successes, failures, per-source detail).
- **Image_Result**: The record of processing one source image through the whole pipeline.
- **Gutter**: The vertical spine/fold band of an open book, near the center of a two-page spread.
- **Cover**: A source image designated as a single logical page that is carried through without
  splitting.

## Requirements

### Requirement 1: Source image discovery

**User Story:** As a user, I want the pipeline to find all eligible source photos in a directory in a
stable order, so that the processed output preserves the book's reading order.

#### Acceptance Criteria

1. WHEN a preprocessing run begins, THE Preprocessing_Pipeline SHALL discover every file located
   directly within the configured source directory (non-recursive, excluding subdirectories) whose
   file extension matches one of the configured supported extensions, comparing extensions
   case-insensitively.
2. WHEN discovery completes, THE Image_IO SHALL return the discovered source files in a deterministic
   total order such that repeating discovery over identical source-directory contents produces an
   identical sequence, and that sequence reproduces the source files' reading order.
3. WHERE the source directory contains files whose extension does not match any configured supported
   extension, THE Image_IO SHALL exclude those files from the discovered set.
4. IF the configured source directory does not exist or cannot be read, THEN THE Image_IO SHALL raise
   a domain error indicating the source directory is inaccessible and SHALL return no discovered set.
5. IF no file in the configured source directory matches a configured supported extension, THEN THE
   Image_IO SHALL return an empty discovered set without raising an error.

### Requirement 2: Working-set orchestration through ordered stages

**User Story:** As a user, I want each source photo threaded through the configured stages in order,
so that stages compose predictably regardless of whether they add images or keep the count the same.

#### Acceptance Criteria

1. WHEN processing a source image, THE Preprocessing_Pipeline SHALL seed a Working_Set containing
   exactly one Working_Image loaded from that source, with that Working_Image carrying the source name
   and an empty Lineage.
2. THE Preprocessing_Pipeline SHALL apply each configured stage to the Working_Set in the configured
   order, passing the complete output set of stage k as the sole input set of stage k+1, so that stage
   k+1 receives no images other than those produced by stage k.
3. WHEN the configured stage list is empty, THE Preprocessing_Pipeline SHALL treat the seeded
   Working_Set as the final Working_Set without modification.
4. WHEN all stages have been applied to a source image, THE Preprocessing_Pipeline SHALL write to the
   output directory exactly the set of Working_Images present in the final Working_Set, writing one
   output file per Working_Image and no additional files.
5. THE Preprocessing_Pipeline SHALL construct the ordered stage list exactly once per run and reuse
   that same ordered list, unchanged, for every source image processed in that run.

### Requirement 3: Stage contract guarantees

**User Story:** As a developer, I want every stage to obey a common contract, so that stages are
interchangeable and the orchestrator does not need to special-case them.

#### Acceptance Criteria

1. WHEN a stage is applied to a Working_Set, THE Preprocessing_Stage SHALL produce a new Working_Set
   without in-place mutation of the input images' pixel data, dimensions, or metadata.
2. WHEN a stage is applied to a Working_Set, THE Preprocessing_Stage SHALL produce, for each input
   image, N output images where N is an integer and N ≥ 1, with no input image silently dropped.
3. WHEN a stage is applied to a Working_Set, THE Preprocessing_Stage SHALL preserve relative order
   such that outputs derived from an earlier input image sort before outputs derived from a later
   input image, and the outputs of a single 1→N expansion are emitted in reading order where the
   first-read page precedes the second-read page.
4. THE Preprocessing_Stage SHALL operate only on in-memory images.
5. THE Preprocessing_Stage SHALL perform no file input or output.
6. IF a stage cannot process an image, THEN THE Preprocessing_Stage SHALL raise a StageError
   identifying the failing stage, leave the input images unmodified, and return no Working_Set.

### Requirement 4: Page splitting stage

**User Story:** As a user, I want photos of open books split into separate left and right pages, so
that downstream extraction processes one page at a time.

#### Acceptance Criteria

1. WHEN the Page_Split_Stage processes an interior spread image, where an interior spread is any
   source image not designated as a Cover, THE Page_Split_Stage SHALL produce exactly two output
   images, emitting the left page before the right page.
2. WHERE the split method is `fixed_midpoint`, THE Page_Split_Stage SHALL split the image at the pixel
   column located at `split_ratio` × image width measured from the left edge.
3. WHERE the split method is `gutter_detection`, THE Page_Split_Stage SHALL locate the spine column
   within a central search band bounded by the pixel column at `search_band_min` × image width and the
   pixel column at `search_band_max` × image width, both measured from the left edge.
4. IF the `gutter_detection` method finds no confident gutter within the search band, THEN THE
   Page_Split_Stage SHALL split at the pixel column located at `fallback_ratio` × image width measured
   from the left edge AND set a fallback-used indicator to true in the output record.
5. WHERE a `gutter_margin` is configured, THE Page_Split_Stage SHALL trim `gutter_margin` × image width
   divided by 2 pixels from each side of the spine, so an equal width is excluded from both the left
   and right pages and the spine shadow is excluded from both pages.
6. WHERE a source image is designated as a Cover, THE Page_Split_Stage SHALL produce exactly one
   output image without splitting.
7. THE Page_Split_Stage SHALL produce left and right page regions that lie entirely within the source
   image pixel bounds, with the left region left of the spine, the right region right of the spine,
   and the two regions not horizontally overlapping.
8. IF the computed split produces a left or right page region with a width of less than 1 pixel, THEN
   THE Page_Split_Stage SHALL emit no output image, leave the source image unchanged, and flag a
   StageError for review.

### Requirement 5: Contrast enhancement stage

**User Story:** As a user, I want contrast between text and page background increased, so that
downstream OCR/VLM extraction is more accurate.

#### Acceptance Criteria

1. WHEN the Contrast_Enhancement_Stage processes an input image, THE Contrast_Enhancement_Stage SHALL
   produce exactly one output image for that input image.
2. THE Contrast_Enhancement_Stage SHALL produce an output image whose pixel width and height are
   identical to the pixel width and height of its input image.
3. WHERE the configured `method` is `linear`, THE Contrast_Enhancement_Stage SHALL apply the linear
   contrast transform scaled by the configured `factor`.
4. WHERE the configured `method` is `adaptive`, THE Contrast_Enhancement_Stage SHALL apply the
   adaptive contrast transform bounded by the configured `clip_limit`.
5. WHERE the configured `method` is `histogram_equalization`, THE Contrast_Enhancement_Stage SHALL
   apply the histogram-equalization contrast transform.
6. THE Contrast_Enhancement_Stage SHALL produce an output image with strictly positive width and
   height that can be encoded in the configured output format without error.
7. IF the configured `method` cannot process the input image's color mode, THEN THE
   Contrast_Enhancement_Stage SHALL raise a StageError identifying the stage rather than returning an
   output image.

### Requirement 6: Stage creation and validation

**User Story:** As a developer, I want stages selected by type string and validated at build time, so
that misconfiguration is caught before any processing begins.

#### Acceptance Criteria

1. WHEN the Stage_Factory receives a supported stage type accompanied by a config whose subclass
   matches that stage type, THE Stage_Factory SHALL create exactly one corresponding
   Preprocessing_Stage implementation instance for that entry.
2. IF a requested stage type is not among the supported stage types, THEN THE Stage_Factory SHALL raise
   a ConfigurationError whose message identifies the unsupported stage type and enumerates the
   complete set of supported stage types, and SHALL create no Preprocessing_Stage instances.
3. IF a supplied stage config subclass does not match the config subclass declared for its stage type,
   THEN THE Stage_Factory SHALL raise a ConfigurationError whose message identifies the offending stage
   type, the declared config subclass, and the supplied config subclass, and SHALL create no
   Preprocessing_Stage instances.
4. WHEN the Stage_Factory builds the stage list from a configured stage list of one or more entries,
   THE Stage_Factory SHALL produce an ordered stage list containing one Preprocessing_Stage per
   configured entry, in the identical index order as the configured stage list.
5. WHEN the Stage_Factory receives an empty configured stage list, THE Stage_Factory SHALL return an
   empty ordered stage list without raising an error.
6. THE Stage_Factory SHALL complete all stage type and config validation for every entry in the
   configured stage list, and SHALL raise any ConfigurationError, before any Preprocessing_Stage begins
   processing.
7. WHEN queried, THE Stage_Factory SHALL return the complete set of supported stage type strings it
   accepts.

### Requirement 7: Configuration validation

**User Story:** As a user, I want invalid configuration rejected before a run starts, so that I never
produce partial or incorrect output from a bad setup.

#### Acceptance Criteria

1. IF the configured source directory path or output directory path is null, empty, or contains only
   whitespace, THEN THE Config_Validator SHALL raise a ConfigurationError that identifies the offending
   path field, before creating any output file or directory.
2. IF the configured stage list contains zero stages, THEN THE Config_Validator SHALL raise a
   ConfigurationError, before creating any output file or directory.
3. IF any stage configuration parameter falls outside its declared allowed range, THEN THE
   Config_Validator SHALL raise a ConfigurationError that identifies the offending stage, the parameter
   name, the provided value, and the allowed range, before creating any output file or directory.
4. IF `search_band_max` is less than or equal to `search_band_min`, THEN THE Config_Validator SHALL
   raise a ConfigurationError that identifies both configured values, before creating any output file
   or directory.
5. WHERE no output format is configured, THE Config_Validator SHALL set the output format to match the
   detected source image format.
6. IF no output format is configured AND the source image format cannot be determined or is not among
   the supported formats, THEN THE Config_Validator SHALL raise a ConfigurationError that indicates the
   source format could not be resolved, before creating any output file or directory.
7. WHEN configuration validation completes with no violations, THE Config_Validator SHALL return the
   validated configuration and allow processing to begin.

### Requirement 8: Deterministic output naming

**User Story:** As a user, I want output filenames derived deterministically from source and lineage,
so that the downstream batch processor reads pages in correct global order without collisions.

#### Acceptance Criteria

1. THE Preprocessing_Pipeline SHALL derive each output name solely from the Working_Image's source
   name and Lineage, and SHALL produce the identical output name on every run whenever the same source
   name and Lineage are supplied.
2. WHEN a source image is split into a left page and a right page, THE Preprocessing_Pipeline SHALL
   assign output names such that the left page's name precedes the right page's name in ascending
   byte-wise lexicographic order.
3. THE Preprocessing_Pipeline SHALL assign output names such that, for any two source images ordered
   by the pipeline's source processing order, every output name derived from the earlier source
   precedes, in ascending byte-wise lexicographic order, every output name derived from the later
   source.
4. WHEN a run completes, THE Preprocessing_Pipeline SHALL have produced output names that are pairwise
   distinct across every Working_Image written in that run.
5. IF two Working_Images in the same run would resolve to the same output name, THEN THE
   Preprocessing_Pipeline SHALL halt naming for that run and report an error indicating the naming
   collision, and SHALL NOT overwrite or write either colliding output.

### Requirement 9: Determinism of processing

**User Story:** As a developer, I want repeated runs to produce identical results, so that the
pipeline is reproducible and testable.

#### Acceptance Criteria

1. WHEN the same source image is processed twice through the same ordered stage configuration, THE
   Preprocessing_Pipeline SHALL produce the same number of output images in the second run as in the
   first run.
2. WHEN the same source image is processed twice through the same ordered stage configuration, THE
   Preprocessing_Pipeline SHALL produce, for each output name present in the first run, a byte-for-byte
   identical output image under the same output name in the second run.
3. WHEN the same source image is processed twice through the same ordered stage configuration, THE
   Preprocessing_Pipeline SHALL produce Image_Result metadata in the second run that is identical to
   the first run, including identical output paths, identical output count, and identical ordered names
   of stages applied.

### Requirement 10: Source immutability

**User Story:** As a user, I want my original scans protected, so that the Phase 1 source of truth is
never altered.

#### Acceptance Criteria

1. THE Preprocessing_Pipeline SHALL write all output files only within the configured output directory
   or its subdirectories.
2. IF a write, move, or delete operation would target a path outside the configured output directory,
   THEN THE Preprocessing_Pipeline SHALL reject the operation, leave the target path unchanged, and
   report an error indicating the path is outside the permitted output directory.
3. WHILE a run is in progress, THE Preprocessing_Pipeline SHALL leave every file under the source
   directory unmodified, uncreated, and undeleted, such that each source file's content and count are
   byte-for-byte identical to their pre-run state.
4. WHEN a run completes, whether successfully or with errors, THE Preprocessing_Pipeline SHALL leave
   every file under the source directory byte-for-byte identical to its pre-run state, verifiable by
   comparing per-file content hashes taken before and after the run.
5. IF the configured output directory is identical to, or nested within, the configured source
   directory, THEN THE Preprocessing_Pipeline SHALL reject the configuration before processing any file
   and report an error indicating that the output directory must not overlap the source directory.

### Requirement 11: Guaranteed valid output for readable input

**User Story:** As a user, I want any readable photo to yield a usable result, so that a difficult
image does not silently fail.

#### Acceptance Criteria

1. WHEN a source image that can be decoded into an in-memory image is processed with a configuration
   that has passed all validation rules, THE Preprocessing_Pipeline SHALL produce a final Working_Set
   in which every output image has an integer width of at least 1 pixel and an integer height of at
   least 1 pixel.
2. WHEN a source image that can be decoded into an in-memory image is processed with a configuration
   that has passed all validation rules, THE Preprocessing_Pipeline SHALL write at least one output
   image file for that source to the configured output directory.
3. WHEN a source image that can be decoded into an in-memory image is processed with a configuration
   that has passed all validation rules, THE Preprocessing_Pipeline SHALL produce a final Working_Set
   in which every output image is a valid image that can be encoded in the configured output format.
4. IF content-aware gutter detection finds no confident gutter within the configured central search
   band, THEN THE Preprocessing_Pipeline SHALL apply the fixed-midpoint split, record that a fallback
   was used, and produce a valid output set for that source rather than failing it.

### Requirement 12: Per-source failure isolation and error handling

**User Story:** As a user, I want a single bad photo to be recorded and skipped, so that the rest of
the batch still completes.

#### Acceptance Criteria

1. IF a source file matches a supported extension but cannot be decoded, THEN THE Image_IO SHALL raise
   an ImageLoadError that identifies the affected source file path and the decode failure cause.
2. IF an output image cannot be written, THEN THE Image_IO SHALL raise an ImageSaveError that
   identifies the affected destination file path and the write failure cause.
3. IF processing a source image raises an ImageLoadError, StageError, or ImageSaveError, THEN THE
   Preprocessing_Pipeline SHALL record for that source exactly one Image_Result with a failed status
   that is distinct from the success status.
4. WHEN a source image fails, THE Preprocessing_Pipeline SHALL record in that source's Image_Result the
   source identifier and the error message.
5. WHERE the failure originates in a stage, THE Preprocessing_Pipeline SHALL record in the failed
   source's Image_Result the identity of the stage that failed.
6. WHEN a source image fails, THE Preprocessing_Pipeline SHALL continue processing all remaining
   discovered source images without aborting the batch.
7. WHEN a source image fails, THE Preprocessing_Pipeline SHALL record in that source's Image_Result the
   count of output images actually written for that source.

### Requirement 13: Run reporting

**User Story:** As a user, I want a summary of each run, so that I can see how many images were
processed and review any failures.

#### Acceptance Criteria

1. WHEN a run completes, THE Preprocessing_Pipeline SHALL produce a Pipeline_Report containing the
   total number of source images, the number of successes, the number of failures, the total number of
   output files written, and one per-source Image_Result entry for every source image, where each
   count is a non-negative integer and the sum of successes and failures equals the total number of
   source images.
2. THE Pipeline_Report SHALL report a success rate computed as successful sources divided by total
   sources, expressed as a value between 0.0 and 1.0 inclusive, and SHALL return 0.0 when the total
   number of source images is 0.
3. WHEN a source image is processed successfully, THE Preprocessing_Pipeline SHALL record in its
   Image_Result a success status, the output paths, an output count equal to the number of output paths
   recorded, and the names of stages applied in the order they were executed.
4. IF a source image fails to process, THEN THE Preprocessing_Pipeline SHALL record in its Image_Result
   a failure status, an output count of 0, and an error indication identifying the cause of the
   failure, and SHALL count that source among the failures in the Pipeline_Report.

### Requirement 14: Standalone entry point and decoupled integration

**User Story:** As a user, I want a standalone command that runs preprocessing before extraction, so
that I can produce an output directory to feed the existing batch processor.

#### Acceptance Criteria

1. WHEN the Entry_Point is invoked, THE Entry_Point SHALL build a PipelineConfig with the source
   directory defaulting to `phase_1/cookbook_images`, the output directory set to the configured value,
   and an ordered stage list defaulting to a page-split stage followed by a contrast-enhancement stage.
2. IF the constructed PipelineConfig fails validation (empty source or output directory, empty stage
   list, an unknown stage type, a stage config that does not match its declared stage type, or an
   out-of-range parameter), THEN THE Entry_Point SHALL abort before processing any source image, report
   a configuration error indicating the invalid field, and write no output.
3. WHEN the pipeline run finishes, THE Entry_Point SHALL log a summary of the Pipeline_Report including
   total sources, total output files, successful sources, and failed sources.
4. WHEN the pipeline run finishes, THE Entry_Point SHALL write every final processed image to the
   configured output directory as single-page images using a supported extension (`.jpg`, `.jpeg`, or
   `.png`) with deterministic, reading-order-preserving names, such that the existing image batch
   processor's image discovery over that directory yields the images in the source book's reading
   order.
5. THE Entry_Point SHALL write all outputs only under the configured output directory and SHALL NOT
   modify or delete any file under the source directory.
