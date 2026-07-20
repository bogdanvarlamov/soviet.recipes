# soviet.recipes

The GitHub repo for the soviet.recipes project — a cookbook digitization pipeline that converts scanned cookbook pages into machine-readable text.

## Quick Start

All commands run from the batch processor directory using `uv run poe`:

```bash
cd phase_2/image_batch_processor
uv sync                 # install dependencies (first time only)
```

Tasks are defined in `phase_2/image_batch_processor/pyproject.toml` under `[tool.poe.tasks]`.

### Common runs

```bash
uv run poe sample       # quick 3-image sanity check (starts VLM server, runs, stops)
uv run poe pipeline     # full batch: start llama-server, extract all pages, shut down
uv run poe serve        # start only the local llama.cpp server and keep it running
uv run poe extract      # run the batch against an already-running server
uv run poe test         # run the test suite
```

### Engine / model variants

```bash
uv run poe sample-llm             # 3-image check using the direct LLM engine
uv run poe pipeline-llm           # full batch via the LLM engine (parallel slots)
uv run poe pipeline-small         # full batch with the smaller Qwen3-VL 4B model
uv run poe pipeline-small-lowscale # 4B model at lower image scale (faster, less VRAM)
```

### Preprocessing

The 224 raw scans split into ~444 page images (page split + white balance).

```bash
uv run poe preprocess         # run preprocessing only
uv run poe preprocess-force   # re-run and overwrite existing preprocessed output
```

### Preprocessed-image pipelines

Run extraction against the page-split/white-balanced images (preprocessing is skipped automatically if its output already exists):

```bash
uv run poe pipeline-preprocessed-easyocr    # Docling + EasyOCR (no VLM server needed)
uv run poe pipeline-preprocessed-small      # Docling (VLM backend) + Qwen3-VL 4B
uv run poe pipeline-preprocessed-llm-small  # LLM engine + Qwen3-VL 4B
```

## Notes

- VLM tasks download and cache the model + mmproj projector on first run via llama.cpp.
- Defaults (server binary path, model repo, GPU flags) live in `[tool.poe.env]` and can be overridden with environment variables of the same name.
