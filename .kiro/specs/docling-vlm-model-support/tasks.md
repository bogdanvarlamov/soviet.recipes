# Implementation Plan

## Overview

This plan adds support for the DocTags-native VLM models SmolDocling and
Granite-Docling to the existing `DoclingEngine`, covering both local
(in-process, Transformers) and remote (llama.cpp GGUF) backends, exposes the
`VlmPipelineOptions` through `DoclingConfig`, and wires the new models into the
run entrypoint (`main.py` env vars) and `poe` task shortcuts (`pyproject.toml`).
Work is scoped to `config/settings.py`, `engines/docling.py`, `main.py`,
`pyproject.toml`, tests, and documentation. Tasks marked with `*` are optional
property/unit tests.

## Task Dependency Graph

Tasks are grouped into waves; tasks in the same wave can be worked in parallel,
and each wave depends on the previous one. Tasks 4 and 5 both depend on 2/3;
task 6 depends on both. Tasks 7, 8, 9, and 10 depend on 6. Task 11 (poe tasks)
depends on 10 (env wiring). Docs (12) come after the run plumbing, and the
checkpoint (13) is last.

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "dependsOn": [] },
    { "wave": 2, "tasks": ["2"], "dependsOn": ["1"] },
    { "wave": 3, "tasks": ["3", "3.1", "3.2"], "dependsOn": ["2"] },
    { "wave": 4, "tasks": ["4", "4.1", "4.2", "5", "5.1"], "dependsOn": ["3"] },
    { "wave": 5, "tasks": ["6", "6.1", "6.2"], "dependsOn": ["4", "5"] },
    { "wave": 6, "tasks": ["7", "7.1", "8", "9", "9.1", "10", "10.1"], "dependsOn": ["6"] },
    { "wave": 7, "tasks": ["11"], "dependsOn": ["10"] },
    { "wave": 8, "tasks": ["12"], "dependsOn": ["7", "8", "9", "11"] },
    { "wave": 9, "tasks": ["13"], "dependsOn": ["12"] }
  ]
}
```

## Tasks

- [ ] 1. Verify and add dependencies for local VLM inference
  - Confirm Docling's Transformers-based VLM extras are installed in the
    `cookbook-processing` environment (transformers, torch, model deps)
  - Add any missing dependencies to `phase_2/image_batch_processor/pyproject.toml`
    and run `uv sync`
  - Confirm `docling.datamodel.vlm_model_specs` exposes
    `SMOLDOCLING_TRANSFORMERS` and `GRANITEDOCLING_TRANSFORMERS`
  - _Requirements: 2.2, 2.3, 2.5_

- [ ] 2. Extend DoclingConfig with model/backend/pipeline fields
  - Add `vlm_model_id` (Literal: qwen3-vl, smoldocling, granite-docling, custom)
    defaulting to `qwen3-vl`
  - Add `vlm_backend` (Literal: local, remote) defaulting to `remote`
  - Add pipeline option fields: `vlm_images_scale` (gt=0), `vlm_generate_page_images`,
    `vlm_generate_picture_images`
  - Retain all existing VLM fields for backward compatibility
  - _Requirements: 1.1, 1.4, 4.1, 4.2_

- [ ] 3. Add DoclingConfig validation
  - Add `model_validator(mode="after")` enforcing backend/model coherence
    (local ⇒ smoldocling or granite-docling; remote ⇒ vlm_url and vlm_model present)
  - Validate pipeline option values (image scale > 0 already enforced by Field)
  - _Requirements: 6.1, 6.2, 4.4_

- [ ]* 3.1 Write property test for backend/model coherence
  - **Property 6: Backend/model coherence**
  - **Validates: Requirements 6.1, 6.2**

- [ ]* 3.2 Write property test for invalid pipeline option rejection
  - **Property 5: Invalid pipeline option values are rejected**
  - **Validates: Requirements 4.4**

- [ ] 4. Add local VLM preset resolution to DoclingEngine
  - Add module-level `_LOCAL_VLM_PRESETS` mapping model ids to
    `vlm_model_specs` Transformers presets
  - Implement `_build_local_vlm_options()` returning the matching preset,
    raising ConfigurationError for unsupported local models
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 1.3_

- [ ]* 4.1 Write property test for local preset resolution
  - **Property 1: Model selection resolves to the correct pipeline configuration**
  - **Validates: Requirements 1.1, 1.2, 2.2, 2.3**

- [ ]* 4.2 Write property test for unsupported local model rejection
  - **Property 2: Unsupported local model is rejected**
  - **Validates: Requirements 1.3, 2.5, 6.1**

- [ ] 5. Refactor remote VLM options into its own method
  - Extract current `ApiVlmOptions` construction into `_build_remote_vlm_options()`
  - When `vlm_model_id` is a DocTags-native model, force
    `response_format=ResponseFormat.DOCTAGS` (log a warning if config conflicts)
  - _Requirements: 3.1, 3.2, 3.3_

- [ ]* 5.1 Write property test for remote DocTags format forcing
  - **Property 3: Remote DocTags-native selection forces DocTags response format**
  - **Validates: Requirements 3.2**

- [ ] 6. Refactor `_build_vlm_converter` to select backend and apply pipeline options
  - Branch on `vlm_backend` to choose local vs remote `vlm_options`
  - Build `VlmPipelineOptions` from the config pipeline fields
    (images_scale, generate_page_images, generate_picture_images,
    enable_remote_services only for remote)
  - Construct the `DocumentConverter` with `VlmPipeline`
  - _Requirements: 2.1, 3.1, 4.3_

- [ ]* 6.1 Write property test for pipeline option application
  - **Property 4: Pipeline options are applied**
  - **Validates: Requirements 4.1, 4.2, 4.3**

- [ ]* 6.2 Write property test for Qwen3-VL backward compatibility
  - **Property 7: Backward compatibility of remote Qwen3-VL**
  - **Validates: Requirements 1.4, 3.3, 7.1**

- [ ] 7. Update artifact namespacing by model id
  - Change the artifact folder prefix to include `vlm_model_id` for VLM runs
  - Keep the EasyOCR (`docling`) prefix unchanged
  - _Requirements: 7.2, 7.3_

- [ ]* 7.1 Write property test for artifact namespace separation
  - **Property 8: Artifact namespace separation**
  - **Validates: Requirements 7.2**

- [ ] 8. Gate server reachability check to the remote backend
  - Update `validate_config` so `_check_vlm_server_reachable` runs only for the
    remote backend
  - For the local backend, rely on converter initialization to surface model
    resolution errors
  - _Requirements: 3.4, 2.5, 6.3_

- [ ] 9. Confirm debug visualization parity
  - Verify the existing `get_visualization('reading_order')` and
    `save_as_doctags` calls produce non-degenerate geometry for a DocTags-native
    model, and remain resilient (skip without failing) when geometry is absent
  - _Requirements: 5.1, 5.2, 5.3_

- [ ]* 9.1 Write property test for visualization resilience
  - **Property 9: Visualization resilience**
  - **Validates: Requirements 5.3**

- [ ] 10. Wire new env vars into the run entrypoint (`main.py`)
  - Read `VLM_MODEL_ID` (default `qwen3-vl`) and `VLM_BACKEND` (default `remote`)
    on the Docling VLM branch and map them into `engine_config`
  - Only populate remote server fields (vlm_url, vlm_model, vlm_timeout, vlm_scale)
    when `vlm_backend == "remote"`
  - Preserve current defaults when the new env vars are unset
  - _Requirements: 8.1, 8.6, 8.7_

- [ ]* 10.1 Write property test for run-entrypoint env mapping
  - **Property 10: Run-entrypoint env mapping**
  - **Validates: Requirements 8.1, 8.7**

- [ ] 11. Add poe tasks for the new models (`pyproject.toml`)
  - Add local-backend tasks `pipeline-smoldocling` and `pipeline-granite-docling`
    (invoke `main.py` directly; no llama-server)
  - Add `sample-*` few-image variants (`MAX_IMAGES=3`)
  - Add `pipeline-preprocessed-*` variants (`ENABLE_PREPROCESSING=1`)
  - Add at least one remote GGUF variant via `run_pipeline.py` with the
    confirmed `LLAMA_HF_REPO` for a DocTags-native GGUF
  - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 12. Update documentation and example configurations
  - Document SmolDocling and Granite-Docling selection for local and remote
    backends in `phase_2/image_batch_processor/README.md`
  - Add the new `poe` tasks and `VLM_MODEL_ID` / `VLM_BACKEND` env vars to the
    README run table
  - Note that DocTags-native models restore bounding-box / reading-order debugging
  - Note local vs remote tradeoffs for the Windows environment
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 13. Checkpoint - verify build and tests
  - Run `pytest` for the image_batch_processor package and ensure all tests pass
  - Optionally run `uv run poe sample-smoldocling` (or `sample-granite-docling`)
    and confirm the saved doctags contain non-`loc_0` coordinates and that a
    reading-order visualization image is produced
  - Ask the user if questions arise
  - _Requirements: 5.1, 5.2, 8.2_

## Notes

- MLX presets (`SMOLDOCLING_MLX`, `GRANITEDOCLING_MLX`) are intentionally out of
  scope: they target Apple MPS, and this project runs on Windows.
- Local Transformers inference falls back to CPU on AMD/Vulkan GPUs; the ~256M
  models are feasible on CPU but slower, which is why the remote GGUF backend is
  also supported.
- Open decisions are tracked in `design.md` ("Open decisions"): DocTags format
  coercion vs rejection, artifact folder naming for `qwen3-vl`, and per-run preset
  overrides.
- Optional tasks (`*`) follow the project's property-test conventions: minimum
  100 iterations and a `# Feature: docling-vlm-model-support, Property {N}` tag.
