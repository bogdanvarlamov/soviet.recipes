# Requirements Document

## Introduction

This document specifies requirements for adding first-class support for the two
DocTags-native vision-language models, **SmolDocling** (`ds4sd/SmolDocling-256M-preview`)
and **Granite-Docling** (`ibm-granite/granite-docling-258M`), to the existing
`DoclingEngine` in the Image Batch Processor.

Today the engine's VLM path only supports a remote, OpenAI-compatible endpoint
(a local llama.cpp server running Qwen3-VL). Because Qwen3-VL is a general
vision-language model that is not trained on Docling's DocTags coordinate grid,
its DocTags output contains no valid location tokens (every element is emitted
as `<loc_0><loc_0><loc_0><loc_0>`). As a result, the engine's debug
visualizations (reading-order overlay, bounding boxes) are empty for VLM runs,
even though they work for the EasyOCR path, which derives geometry from a
separate layout model.

SmolDocling and Granite-Docling are purpose-built to emit DocTags **with** real
`<loc_x>` location tokens and reading order. Adding support for them restores
bounding-box / reading-order debugging parity for VLM runs and gives higher
structural fidelity for the cookbook digitization task.

This feature supports **both** ways these models can be run:

- **Local (in-process) inference** via Docling's `vlm_model_specs` presets using
  the Hugging Face Transformers framework.
- **Remote inference** via an OpenAI-compatible GGUF endpoint (llama.cpp),
  reusing the existing `ApiVlmOptions` remote path but with `response_format` set
  to DocTags.

It also covers the project's **run entrypoints** — the environment-variable
wiring in `main.py` and the `poethepoet` (`poe`) task shortcuts in
`pyproject.toml` documented in the README (e.g. `poe pipeline`,
`poe pipeline-preprocessed-easyocr`, `poe pipeline-small`). New `poe` tasks and
env vars are added so a SmolDocling or Granite-Docling run can be kicked off the
same way the existing EasyOCR / Docling-VLM / LLM runs are, including sample and
preprocessed variants. This is what the request means by "poe pipeline options."

## Glossary

- **DoclingEngine**: The existing extraction engine (`engines/docling.py`) that
  wraps Docling's `DocumentConverter`.
- **VLM Pipeline**: Docling's `VlmPipeline`, which converts a page end-to-end
  with a vision-language model instead of the layout + OCR pipeline.
- **DocTags**: Docling's structured markup output. When emitted by a
  DocTags-native model, it includes `<loc_x>` location tokens encoding element
  bounding boxes on a normalized grid.
- **DocTags-native model**: A VLM trained to emit DocTags with valid location
  tokens (SmolDocling, Granite-Docling).
- **Local backend**: In-process inference using `vlm_model_specs` presets
  (`InlineVlmOptions`) via Transformers/MLX.
- **Remote backend**: Inference offloaded to an OpenAI-compatible server
  (llama.cpp) via `ApiVlmOptions`.
- **Inference framework**: The runtime used for local inference (Transformers or
  MLX).
- **Pipeline options**: `VlmPipelineOptions` — the object configuring the VLM
  pipeline (selected `vlm_options`, image scale, page-image generation, etc.).
- **Run entrypoint**: `main.py` (and `run_pipeline.py`), which translate
  environment variables (`ENGINE`, `USE_VLM`, `VLM_SCALE`, `LLAMA_HF_REPO`, etc.)
  into a `BatchProcessorState` / `engine_config`.
- **poe task**: A `poethepoet` task defined under `[tool.poe.tasks]` in
  `pyproject.toml`, run via `uv run poe <task>`, that sets env vars to kick off a
  specific run variant (documented in the README run table).

## Requirements

### Requirement 1: Model selection

**User Story:** As a developer, I want to select SmolDocling or Granite-Docling
as the VLM backend through configuration, so that I can produce DocTags with
real layout geometry without editing engine code.

#### Acceptance Criteria

1. THE DoclingConfig SHALL provide a configuration field that selects the VLM
   model among at least: SmolDocling, Granite-Docling, and the existing custom
   remote model (e.g. Qwen3-VL).
2. WHEN a DocTags-native model (SmolDocling or Granite-Docling) is selected THEN
   the engine SHALL configure the VLM pipeline to use that model.
3. WHERE an unknown or unsupported model identifier is provided THEN the system
   SHALL raise a ConfigurationError before processing begins.
4. WHEN no model is explicitly selected THEN the engine SHALL retain its current
   default behavior for backward compatibility.

### Requirement 2: Local in-process inference backend

**User Story:** As a developer, I want to run SmolDocling and Granite-Docling
locally in-process using Docling's built-in model presets, so that I do not need
a separate model server.

#### Acceptance Criteria

1. THE DoclingConfig SHALL allow selecting a local (in-process) VLM backend.
2. WHEN the local backend is selected with SmolDocling THEN the engine SHALL
   configure the VLM pipeline using `vlm_model_specs.SMOLDOCLING_TRANSFORMERS`.
3. WHEN the local backend is selected with Granite-Docling THEN the engine SHALL
   configure the VLM pipeline using `vlm_model_specs.GRANITEDOCLING_TRANSFORMERS`.
4. THE local backend SHALL allow selecting the accelerator device (CPU or CUDA)
   consistent with the engine's existing device-selection behavior.
5. WHERE the selected local model requires a dependency or model download that
   is unavailable THEN the system SHALL surface a clear ConfigurationError or
   ExtractionError describing the problem.

### Requirement 3: Remote (GGUF / llama.cpp) inference for DocTags-native models

**User Story:** As a developer on a machine without a supported local inference
GPU, I want to run SmolDocling or Granite-Docling as a GGUF model behind my
existing llama.cpp server, so that I can reuse my current remote setup.

#### Acceptance Criteria

1. THE DoclingConfig SHALL allow selecting a remote backend for a DocTags-native
   model.
2. WHEN a DocTags-native model is run via the remote backend THEN the engine
   SHALL configure `ApiVlmOptions` with the DocTags response format.
3. THE remote backend SHALL continue to support the existing custom model
   (Qwen3-VL) configuration without change.
4. WHEN the remote backend is selected THEN the engine SHALL verify the server
   is reachable before processing the batch, consistent with existing behavior.

### Requirement 4: Exposed pipeline options

**User Story:** As a developer, I want to configure the VLM pipeline options
through DoclingConfig, so that I can tune image scale and artifact generation
per run.

#### Acceptance Criteria

1. THE DoclingConfig SHALL expose configuration for the VLM pipeline image
   scale (`images_scale`) and the per-request scale used by the model.
2. THE DoclingConfig SHALL expose configuration for page-image and
   picture-image generation used by downstream artifact saving.
3. WHEN pipeline options are provided THEN the engine SHALL apply them to the
   constructed `VlmPipelineOptions`.
4. WHERE pipeline option values are invalid (e.g. non-positive scale) THEN the
   configuration SHALL be rejected with a validation error.

### Requirement 5: Layout debugging parity

**User Story:** As a developer, I want reading-order and bounding-box debug
visualizations to work for DocTags-native VLM runs, so that I can inspect layout
quality the same way I can for the EasyOCR path.

#### Acceptance Criteria

1. WHEN a DocTags-native model produces output THEN the saved DocTags SHALL
   contain non-degenerate location tokens (i.e. not all `<loc_0>`).
2. WHEN a DocTags-native model produces output THEN the engine SHALL generate
   reading-order debug visualization images where geometry is available.
3. IF the produced document contains no usable geometry THEN the engine SHALL
   skip visualization without failing the extraction.

### Requirement 6: Configuration validation

**User Story:** As a developer, I want invalid VLM configurations caught early,
so that a bad combination of options does not fail on every image.

#### Acceptance Criteria

1. THE DoclingConfig SHALL validate that the selected backend and model
   combination is coherent (e.g. a local backend is only used with a supported
   local model).
2. WHERE the configuration requests the remote backend THEN the required remote
   fields (URL, model name) SHALL be present and validated.
3. WHEN configuration validation fails THEN the engine's `validate_config`
   SHALL raise a ConfigurationError before any image is processed.

### Requirement 7: Backward compatibility and artifact separation

**User Story:** As a developer, I want existing Qwen3-VL runs to keep working
and I want each model's artifacts kept separate, so that I can compare engines
and models without overwriting results.

#### Acceptance Criteria

1. WHEN the existing Qwen3-VL remote configuration is used THEN the engine SHALL
   behave as it does today.
2. WHEN outputs are written THEN artifact folders SHALL be namespaced so a
   SmolDocling run does not overwrite a Granite-Docling or Qwen3-VL run.
3. THE existing EasyOCR path SHALL remain unaffected by these changes.

### Requirement 8: Run entrypoints and poe tasks

**User Story:** As a developer, I want `poe` task shortcuts to kick off
SmolDocling and Granite-Docling runs, so that I can launch them the same way I
launch the existing EasyOCR, Docling-VLM, and LLM runs.

#### Acceptance Criteria

1. THE `main.py` run entrypoint SHALL read environment variables that select the
   VLM model (`vlm_model_id`) and backend (`vlm_backend`) and map them into the
   Docling `engine_config`.
2. THE `pyproject.toml` SHALL define `poe` tasks that run SmolDocling and
   Granite-Docling via the local (in-process) backend without requiring a
   llama-server.
3. THE `pyproject.toml` SHALL define `poe` tasks that run SmolDocling and
   Granite-Docling via the remote GGUF backend, launching `llama-server` with the
   appropriate GGUF repo through `run_pipeline.py`.
4. THE new tasks SHALL provide sample (few-image) variants consistent with the
   existing `sample`/`sample-llm` pattern (via `MAX_IMAGES`).
5. THE new tasks SHALL provide preprocessed-image variants consistent with the
   existing `pipeline-preprocessed-*` pattern (via `ENABLE_PREPROCESSING`).
6. WHEN a local-backend task runs THEN it SHALL invoke `main.py` directly (no
   llama-server), consistent with how `pipeline-preprocessed-easyocr` runs.
7. WHEN a new env var is unset THEN the run entrypoint SHALL fall back to the
   current default behavior (backward compatible).

### Requirement 9: Documentation and examples

**User Story:** As a developer, I want documented example configurations and run
commands, so that I can run each supported model without reading the source.

#### Acceptance Criteria

1. THE README SHALL document how to select SmolDocling and Granite-Docling for
   both local and remote backends, including the new `poe` tasks and env vars in
   the run table.
2. THE documentation SHALL note that DocTags-native models restore
   bounding-box / reading-order debugging.
3. THE documentation SHALL note the local vs remote tradeoffs relevant to the
   project's Windows environment (no torch CUDA on AMD/Vulkan GPUs).
