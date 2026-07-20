# Design Document: Docling VLM Model Support (SmolDocling & Granite-Docling)

## Overview

This feature extends the existing `DoclingEngine` so its VLM path can run the two
DocTags-native models, SmolDocling and Granite-Docling, in addition to the
current remote Qwen3-VL setup. It also exposes the `VlmPipelineOptions` knobs
through `DoclingConfig`, and adds the **run entrypoint** wiring (env vars in
`main.py`) plus **poe task** shortcuts in `pyproject.toml` so these models can be
launched the same way the existing runs are.

The change touches `config/settings.py` (`DoclingConfig`), `engines/docling.py`
(converter construction and validation), `main.py` (env-var → config mapping),
`pyproject.toml` (new `poe` tasks), and `README.md` (docs). It does not change
the `ExtractionEngine` interface, the `BatchProcessor`, the factory, or the
CrewAI flow. The engine's output side
(saving markdown, doctags, page/element images, confidence reports, and debug
visualizations) already exists and is reused unchanged; it simply starts
producing meaningful geometry once a DocTags-native model is in use.

### Why this works

Docling's `VlmPipeline` is end-to-end: element bounding boxes only exist in the
resulting `DoclingDocument` if the model emits `<loc_x>` tokens in its DocTags.
SmolDocling and Granite-Docling are trained to do this; Qwen3-VL is not. The
engine already calls `result.document.get_visualization(viz_mode='reading_order')`
and `save_as_doctags(...)`. Switching the underlying model to a DocTags-native
one is what makes those existing calls produce useful output.

## Architecture

### Where this fits

```
DoclingConfig  ──►  DoclingEngine._initialize_converter()
                        │
                        ├─ use_vlm == False ─► _build_ocr_converter()      (unchanged)
                        │
                        └─ use_vlm == True  ─► _build_vlm_converter()
                                                   │
                                                   ├─ vlm_backend == "local"  ─► _build_local_vlm_options()
                                                   │        (vlm_model_specs preset -> InlineVlmOptions)
                                                   │
                                                   └─ vlm_backend == "remote" ─► _build_remote_vlm_options()
                                                            (ApiVlmOptions, existing path)
                        │
                        ▼
                 VlmPipelineOptions(vlm_options=..., images_scale=..., ...)
                        │
                        ▼
        DocumentConverter(format_options={IMAGE: ImageFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=...)})
```

### Design patterns

- **Strategy (backend selection)**: The VLM backend (`local` vs `remote`) and the
  model selection resolve to a single `vlm_options` object that the pipeline
  consumes. The rest of the converter construction is identical for both.
- **Preset lookup table**: A small mapping resolves a model enum to the correct
  `vlm_model_specs` preset for the local backend, keeping the branching flat and
  testable.

## Components and Interfaces

### 1. Configuration model changes (`config/settings.py`)

Extend `DoclingConfig` with backend and model selection plus pipeline options.
New fields (names indicative):

```python
from typing import Literal, Optional
from pydantic import Field, model_validator

# New enums expressed as Literals to match existing style
VlmBackend = Literal["local", "remote"]
VlmModel = Literal["qwen3-vl", "smoldocling", "granite-docling", "custom"]

class DoclingConfig(EngineConfig):
    # ... existing fields ...

    # --- VLM model / backend selection ---
    # Which model family to run through the VLM pipeline.
    vlm_model_id: VlmModel = "qwen3-vl"
    # Where inference runs: in-process ("local") or via an OpenAI-compatible
    # server ("remote", e.g. llama.cpp).
    vlm_backend: VlmBackend = "remote"

    # --- Pipeline options (VlmPipelineOptions) ---
    # Image scale applied by the pipeline when rasterizing pages/elements.
    vlm_images_scale: float = Field(default=2.0, gt=0)
    # Whether to generate page/picture images for downstream artifact saving.
    vlm_generate_page_images: bool = True
    vlm_generate_picture_images: bool = True

    # Existing remote fields (vlm_url, vlm_model, vlm_api_key, vlm_timeout,
    # vlm_scale, vlm_response_format, vlm_prompt) are retained.
```

Notes:

- `vlm_model` (the free-form model-name string sent to a remote server) is
  retained and used only by the remote backend. `vlm_model_id` is the new
  high-level selector.
- The existing `vlm_scale` maps to the per-request `scale` in `ApiVlmOptions`.
  `vlm_images_scale` maps to `VlmPipelineOptions.images_scale`. Keeping both
  mirrors the two distinct scales Docling already exposes.

#### Config validation

Add a `model_validator(mode="after")` to `DoclingConfig`:

1. If `vlm_backend == "local"`, then `vlm_model_id` must be a locally supported
   model (`smoldocling` or `granite-docling`). Local `qwen3-vl`/`custom` is not
   supported in scope and must raise.
2. If `vlm_backend == "remote"`, then `vlm_url` and `vlm_model` must be
   non-empty.
3. When `vlm_model_id` is a DocTags-native model, `vlm_response_format` should be
   `doctags` (the validator may coerce with a warning, or reject a conflicting
   explicit value — see open decision below).

### 2. Engine changes (`engines/docling.py`)

#### Preset resolution

```python
from docling.datamodel import vlm_model_specs

_LOCAL_VLM_PRESETS = {
    "smoldocling": vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
    "granite-docling": vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
}
```

Transformers presets are chosen (not MLX) because MLX targets Apple MPS; the
project runs on Windows. Device selection reuses the engine's existing
`AcceleratorDevice.CUDA if self.config.use_gpu else AcceleratorDevice.CPU`.

#### `_build_vlm_converter` refactor

```python
def _build_vlm_converter(self) -> DocumentConverter:
    if self.config.vlm_backend == "local":
        vlm_options = self._build_local_vlm_options()
    else:
        vlm_options = self._build_remote_vlm_options()  # existing ApiVlmOptions logic

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=(self.config.vlm_backend == "remote"),
        vlm_options=vlm_options,
        generate_page_images=self.config.vlm_generate_page_images,
        generate_picture_images=self.config.vlm_generate_picture_images,
        images_scale=self.config.vlm_images_scale,
    )
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )
```

#### `_build_local_vlm_options`

```python
def _build_local_vlm_options(self):
    preset = _LOCAL_VLM_PRESETS.get(self.config.vlm_model_id)
    if preset is None:
        raise ConfigurationError(
            f"Local VLM backend does not support model "
            f"'{self.config.vlm_model_id}'. Supported: "
            f"{sorted(_LOCAL_VLM_PRESETS)}"
        )
    # Presets are InlineVlmOptions; return as-is. Optional per-run overrides
    # (prompt, scale, device) may be applied via model_copy(update=...).
    return preset
```

For the local backend, device availability is handled by Docling/Transformers;
the engine surfaces failures as `ConfigurationError`/`ExtractionError`.

#### `_build_remote_vlm_options`

This is the current `ApiVlmOptions` construction, extracted into its own method.
When `vlm_model_id` is a DocTags-native model, `response_format` is forced to
`ResponseFormat.DOCTAGS` regardless of `vlm_response_format`, since that is the
only format these GGUF models produce meaningfully.

#### Artifact namespacing

The current prefix logic is:

```python
prefix = "docling_vlm" if self.config.use_vlm else "docling"
```

Extend to include the model id so runs do not collide:

```python
if self.config.use_vlm:
    prefix = f"docling_vlm_{self.config.vlm_model_id}"
else:
    prefix = "docling"
```

(Backward-compatibility note: this changes existing VLM output folder names.
Alternative: keep `docling_vlm` for `qwen3-vl` to preserve current paths. This is
an open decision below.)

#### Server reachability

`_check_vlm_server_reachable` is only invoked for the remote backend. For the
local backend, `validate_config` instead attempts converter initialization
(which triggers model resolution/download) and surfaces errors.

### 3. Validation flow (`validate_config`)

```
validate_config()
  ├─ _initialize_converter()          # builds converter; local backend resolves model
  ├─ if remote backend: _check_vlm_server_reachable()
  └─ return True  (or raise ConfigurationError)
```

### 4. Run entrypoint env-var wiring (`main.py`)

`main.py` already maps env vars (`ENGINE`, `USE_VLM`, `VLM_SCALE`,
`LLAMA_HF_REPO`) into `engine_config`. Add two new env vars, read only on the
Docling branch:

- `VLM_MODEL_ID` → `engine_config["vlm_model_id"]` (default `qwen3-vl`)
- `VLM_BACKEND` → `engine_config["vlm_backend"]` (default `remote`)

Sketch (added inside the existing `else:` Docling branch, when `use_vlm` is true):

```python
vlm_model_id = os.environ.get("VLM_MODEL_ID", "qwen3-vl")
vlm_backend = os.environ.get("VLM_BACKEND", "remote")
engine_config.update({
    "vlm_model_id": vlm_model_id,
    "vlm_backend": vlm_backend,
})
if vlm_backend == "remote":
    engine_config.update({
        "vlm_url": "http://localhost:8080/v1/chat/completions",
        "vlm_model": model_id,          # from LLAMA_HF_REPO, existing behavior
        "vlm_timeout": 1800,
        "vlm_scale": float(os.environ.get("VLM_SCALE", "2.0")),
    })
# local backend needs no server fields; the preset carries the model.
```

Backward compatibility: when `VLM_MODEL_ID`/`VLM_BACKEND` are unset, the values
default to `qwen3-vl`/`remote`, reproducing today's behavior exactly.

Local-backend runs need no llama-server, so their `poe` tasks invoke `main.py`
directly (like `pipeline-preprocessed-easyocr`) rather than `run_pipeline.py`.

### 5. poe tasks (`pyproject.toml`)

New tasks under `[tool.poe.tasks]`, following the existing naming and env
patterns. Local tasks call `python main.py`; remote tasks call
`python run_pipeline.py` with a DocTags-native GGUF repo.

```toml
# --- Local (in-process) DocTags-native models: no llama-server needed ---
[tool.poe.tasks.pipeline-smoldocling]
cmd = "python main.py"
env = { ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "smoldocling" }

[tool.poe.tasks.pipeline-granite-docling]
cmd = "python main.py"
env = { ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "granite-docling" }

# Sample (few-image) variants
[tool.poe.tasks.sample-smoldocling]
cmd = "python main.py"
env = { ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "smoldocling", MAX_IMAGES = "3" }

[tool.poe.tasks.sample-granite-docling]
cmd = "python main.py"
env = { ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "granite-docling", MAX_IMAGES = "3" }

# Preprocessed-image variants
[tool.poe.tasks.pipeline-preprocessed-smoldocling]
cmd = "python main.py"
env = { ENABLE_PREPROCESSING = "1", ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "smoldocling" }

[tool.poe.tasks.pipeline-preprocessed-granite-docling]
cmd = "python main.py"
env = { ENABLE_PREPROCESSING = "1", ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "local", VLM_MODEL_ID = "granite-docling" }

# --- Remote GGUF (llama.cpp) variants: launch a server with a DocTags-native repo ---
[tool.poe.tasks.pipeline-granite-docling-gguf]
cmd = "python run_pipeline.py"
env = { ENGINE = "docling", USE_VLM = "1", VLM_BACKEND = "remote", VLM_MODEL_ID = "granite-docling", LLAMA_HF_REPO = "ibm-granite/granite-docling-258M-GGUF" }
```

Notes:

- The exact GGUF repo id / quant tag for the remote tasks is confirmed during
  task 1; a community GGUF (e.g. `ibm-granite/granite-docling-258M-GGUF` or a
  SmolDocling GGUF) is used. Only `granite-docling` is shown above; a SmolDocling
  GGUF task mirrors it.
- Remote tasks reuse the shared `[tool.poe.env]` llama-server defaults; only
  `LLAMA_HF_REPO` and the VLM selection are overridden per task.

## Data Models

No new runtime data models. The feature adds fields to the existing
`DoclingConfig` Pydantic model and a module-level preset lookup dict in the
engine. `DoclingDocument`, `ProcessingResult`, and `BatchReport` are unchanged.

## Correctness Properties

### Property 1: Model selection resolves to the correct pipeline configuration
*For any* supported `vlm_model_id` with the local backend, the engine constructs
`VlmPipelineOptions` whose `vlm_options` is the matching `vlm_model_specs` preset.
**Validates: Requirements 1.1, 1.2, 2.2, 2.3**

### Property 2: Unsupported local model is rejected
*For any* `vlm_model_id` that is not a supported local model, selecting the local
backend raises a ConfigurationError before any image is processed.
**Validates: Requirements 1.3, 2.5, 6.1**

### Property 3: Remote DocTags-native selection forces DocTags response format
*For any* DocTags-native model on the remote backend, the constructed
`ApiVlmOptions.response_format` is DocTags regardless of the configured
`vlm_response_format`.
**Validates: Requirements 3.2**

### Property 4: Pipeline options are applied
*For any* valid `vlm_images_scale`, `vlm_generate_page_images`, and
`vlm_generate_picture_images`, the constructed `VlmPipelineOptions` reflects those
exact values.
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 5: Invalid pipeline option values are rejected
*For any* non-positive image scale, `DoclingConfig` construction fails validation.
**Validates: Requirements 4.4**

### Property 6: Backend/model coherence
*For any* configuration, `DoclingConfig` validation accepts it only if the
backend and model combination is coherent (local backend ⇒ locally supported
model; remote backend ⇒ URL and model name present).
**Validates: Requirements 6.1, 6.2**

### Property 7: Backward compatibility of remote Qwen3-VL
*For any* configuration equal to the current default remote Qwen3-VL setup, the
constructed converter is functionally equivalent to the pre-change behavior.
**Validates: Requirements 1.4, 3.3, 7.1**

### Property 8: Artifact namespace separation
*For any* two runs with different `vlm_model_id` values, the computed artifact
folder prefixes differ.
**Validates: Requirements 7.2**

### Property 9: Visualization resilience
*For any* converted document that lacks usable geometry, visualization generation
is skipped without raising, and extraction still returns text.
**Validates: Requirements 5.3**

### Property 10: Run-entrypoint env mapping
*For any* combination of `VLM_MODEL_ID` and `VLM_BACKEND` environment values, the
`engine_config` built by `main.py` carries those exact values; when they are
unset, it defaults to `qwen3-vl` / `remote`.
**Validates: Requirements 8.1, 8.7**

## Error Handling

Reuses the existing exception types (`ConfigurationError`, `ExtractionError`).

1. **Configuration errors (fail fast)**: unsupported local model, incoherent
   backend/model combination, missing remote fields, invalid pipeline option
   values.
2. **Model resolution errors (local backend)**: missing dependency or failed
   model download surfaced as `ConfigurationError` during `validate_config` /
   converter init, or `ExtractionError` during conversion.
3. **Remote errors**: unreachable server surfaced by `_check_vlm_server_reachable`
   as `ConfigurationError` (unchanged).
4. **Visualization errors**: caught and logged as warnings, never fatal
   (unchanged).

## Testing Strategy

### Unit tests

- Preset resolution: each supported `vlm_model_id` maps to the expected
  `vlm_model_specs` preset.
- Config validation: coherent and incoherent backend/model combinations;
  missing remote fields; invalid scale.
- Remote DocTags-native forces `ResponseFormat.DOCTAGS`.
- Artifact prefix computation per model id.

Docling model loading and network calls are mocked so tests do not download
models or require a running server.

### Property-based tests (Hypothesis)

Each property test runs a minimum of 100 iterations and is tagged:
`# Feature: docling-vlm-model-support, Property {N}: {description}`

Map one property test per correctness property above. Generators produce
combinations of backend, model id, scales, and boolean pipeline flags; assertions
check the resolved `VlmPipelineOptions` / validation outcome, using mocked Docling
constructors where needed.

### Integration test (optional / manual)

Run one cookbook page through the local SmolDocling backend (CPU acceptable for a
256M model) and assert that the saved `.doctags.json` contains at least one
non-`loc_0` location token and that at least one reading-order visualization image
is produced.

## Implementation Notes

### Dependencies

Local inference requires the Transformers-based extras for Docling
(`transformers`, `torch`, and model-specific deps). These may already be present
via the current environment; task 1 verifies and adds them to `pyproject.toml` if
missing. GGUF/remote usage adds no new Python dependency (reuses `httpx` +
llama.cpp server, already in use).

### Environment considerations

- The project runs on Windows in the `cookbook-processing` conda env. torch-based
  local inference uses CUDA only on NVIDIA GPUs; on CPU-only or AMD/Vulkan setups
  it falls back to CPU. SmolDocling/Granite-Docling are ~256–258M params, so CPU
  inference is feasible but slower. This is why the remote GGUF backend is also
  supported.
- MLX presets are intentionally excluded (Apple-only).

### Open decisions (resolve during review)

1. **DocTags response format handling on remote**: coerce silently vs reject a
   conflicting `vlm_response_format`. Proposed: coerce to DocTags with a logged
   warning.
2. **Artifact folder naming**: include `vlm_model_id` in the prefix for all VLM
   runs (cleaner separation) vs preserve `docling_vlm` for `qwen3-vl` (keeps
   existing paths stable). Proposed: include model id for all; document the change.
3. **Per-run overrides for local presets** (custom prompt/scale via
   `InlineVlmOptions.model_copy`): include now vs defer. Proposed: defer unless
   needed.

### Reference

- Docling Vision Models guide:
  https://docling-project.github.io/docling/usage/vision_models/
  (documents `vlm_model_specs.SMOLDOCLING_TRANSFORMERS`,
  `GRANITEDOCLING_TRANSFORMERS`, and `VlmPipelineOptions(vlm_options=...)`).
  Content was rephrased for compliance with licensing restrictions.
