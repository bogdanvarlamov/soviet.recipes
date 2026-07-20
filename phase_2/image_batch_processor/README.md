# Image Batch Processor

A flexible, workflow-based system for extracting text from directories of images using pluggable extraction engines.

## Overview

The Image Batch Processor orchestrates batch processing of images to extract text content, with support for multiple extraction technologies. Built on CrewAI Flows for workflow management, it provides a clean architecture with separation of concerns between workflow logic, engine abstraction, and concrete implementations.

### Key Features

- **Pluggable Engine Architecture**: Switch between different extraction technologies without changing workflow code
- **One-to-One Mapping**: Each input image produces exactly one output text file
- **Robust Error Handling**: Retry logic with exponential backoff for transient failures
- **Workflow Orchestration**: CrewAI Flows manages state transitions and processing stages
- **Comprehensive Validation**: Input validation and configuration checks before processing begins
- **Optional Preprocessing Subflow**: Run page splitting/white balance/etc. (via the bundled `preprocessing` subpackage) before extraction, with automatic skip-if-already-processed behavior

## Architecture

```
┌─────────────────────────────────────────┐
│      CrewAI Flow Layer                  │
│  (ImageBatchProcessorFlow)              │
│                                          │
│  preprocess_images (optional, @start)   │
│         │                               │
│         ▼                               │
│  initialize_workflow -> create_engine   │
│  -> discover_images -> process_images   │
│  -> generate_report                     │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Batch Processor Layer              │
│  (BatchProcessor)                       │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Engine Interface                   │
│  (ExtractionEngine ABC)                 │
└─────────────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
  Docling       LLM         API
  Engine       Engine      Engine
```

### Optional Preprocessing Subflow

`ImageBatchProcessorFlow` starts with a `preprocess_images` step. When
`state.enable_preprocessing` is `True`, it runs the bundled `preprocessing`
subpackage (page split, white balance, etc.) in-process over
`state.image_dir`, then rewrites `state.image_dir` to that pipeline's output
directory before extraction begins.

The preprocessing pipeline was originally a separate `uv`-managed project but
has since been merged into `image_batch_processor` as the `preprocessing`
subpackage. Its dependencies (Pillow/numpy/OpenCV/page-dewarp) do not conflict
with the batch processor's own, so it now runs as a direct Python import
rather than a subprocess.

If `preprocessing_output_dir` already contains preprocessed images, the
subflow is **skipped** and the existing output is reused — set
`force_preprocessing=True` (or `FORCE_PREPROCESSING=1`) to force a re-run.

```python
from flow.batch_flow import ImageBatchProcessorFlow
from flow.state import BatchProcessorState

state = BatchProcessorState(
    image_dir="../../phase_1/cookbook_images",
    output_dir="./output/text",
    engine_type="docling",
    engine_config={},
    enable_preprocessing=True,             # run the preprocessing subflow first
    preprocessing_output_dir="./preprocessed",  # defaults to phase_2/preprocessed
    force_preprocessing=False,             # re-run even if output already exists
)

flow = ImageBatchProcessorFlow()
flow._state = state
result = flow.kickoff()
```

Or via `main.py` environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_PREPROCESSING` | unset (disabled) | Set to `1`/`true`/`yes` to run the preprocessing subflow before extraction |
| `PREPROCESSING_OUTPUT_DIR` | `phase_2/preprocessed` | Where the preprocessing subflow writes its output (and where extraction reads from) |
| `FORCE_PREPROCESSING` | unset (disabled) | Set to `1`/`true`/`yes` to re-run preprocessing even if output already exists |

## Engine Interface

All extraction engines implement the `ExtractionEngine` abstract base class:

```python
class ExtractionEngine(ABC):
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image file."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the engine is properly configured."""
        pass
```

### Available Engines

1. **DoclingEngine**: Uses the Docling library for document processing
2. **LLMEngine**: Uses local or remote LLMs for image-to-text extraction
3. **APIEngine**: Calls external API services for text extraction

## Configuration

The system uses Pydantic models for type-safe configuration with validation.

### Base Configuration Structure

```python
from config.settings import BatchProcessorConfig, DoclingConfig

config = BatchProcessorConfig(
    image_dir="/path/to/images",
    output_dir="/path/to/output",
    engine_type="docling",
    engine_config=DoclingConfig(),
    max_retries=3
)
```

### Engine-Specific Configurations

#### Docling Engine

The Docling engine supports two text backends:

- **EasyOCR** (traditional OCR) when `use_vlm=False`
- **Vision LLM** via Docling's VLM pipeline when `use_vlm=True` (default). This
  targets an OpenAI-compatible endpoint such as a local llama.cpp server
  running a Qwen3-VL model.

```python
from config.settings import DoclingConfig

# EasyOCR backend
docling_config = DoclingConfig(
    use_vlm=False,                # Use EasyOCR instead of a vision LLM
    model_path="/path/to/model",  # Optional
    use_gpu=False,                # CUDA-only; ignored on CPU-only/AMD setups
    batch_size=1,                 # Batch size for processing
    ocr_enabled=True              # Enable OCR
)

# Vision LLM backend (local llama.cpp / Qwen3-VL)
docling_config = DoclingConfig(
    use_vlm=True,                                             # Use a vision LLM
    vlm_url="http://localhost:8080/v1/chat/completions",      # llama-server endpoint
    vlm_model="qwen3-vl",                                     # Model alias (see /v1/models)
    vlm_api_key=None,                                         # Optional bearer token
    vlm_timeout=300,                                          # Per-request timeout (s)
    vlm_scale=2.0,                                            # Image upscaling factor
    vlm_response_format="markdown",                           # "markdown" | "doctags" | "html"
    vlm_prompt="Transcribe ALL text..."                       # Transcription instruction
)
```

#### LLM Engine

```python
from config.settings import LLMConfig

llm_config = LLMConfig(
    model_name="qwen3-vl",        # Required
    temperature=0.0,              # 0.0 to 2.0
    max_tokens=4096,              # Maximum tokens
    api_key="sk-no-key-required", # Optional (llama.cpp needs no real key)
    base_url="http://localhost:8080/v1",  # Optional; OpenAI-compatible endpoint
    timeout=300,                  # Per-request timeout in seconds
    prompt="Transcribe ALL text..."  # Transcription instruction
)
```

The `LLMEngine` is a standalone alternative to the Docling VLM backend: it
sends each image directly to an OpenAI-compatible vision endpoint (such as a
local llama.cpp server) and returns the model's transcription, without
Docling's layout analysis or debug artifacts.

#### API Engine

```python
from config.settings import APIConfig

api_config = APIConfig(
    api_url="https://api.example.com",  # Required
    api_key="your-api-key",             # Required
    timeout=30,                         # Request timeout in seconds
    max_retries=3,                      # Retry attempts
    verify_ssl=True                     # SSL verification
)
```

### Batch Processor Configuration

```python
from config.settings import BatchProcessorConfig

config = BatchProcessorConfig(
    image_dir="/path/to/images",              # Required: Input directory
    output_dir="/path/to/output",             # Required: Output directory
    engine_type="docling",                    # Required: "docling", "llm", or "api"
    engine_config=engine_config,              # Required: Engine-specific config
    max_retries=3,                            # Optional: Retry attempts per image
    supported_extensions=[                    # Optional: Image file extensions
        ".jpg", ".jpeg", ".png", ".tiff", ".bmp"
    ]
)
```

### Configuration Validation

The configuration system automatically validates:

- **Non-empty paths**: `image_dir` and `output_dir` cannot be empty
- **Valid engine type**: Must be one of "docling", "llm", or "api"
- **Type matching**: `engine_config` must match the specified `engine_type`
- **Value ranges**: Fields like `max_retries` must be >= 0, `temperature` must be 0.0-2.0

## Project Structure

```
image_batch_processor/
├── engines/              # Extraction engine implementations
│   ├── base.py          # ExtractionEngine ABC
│   ├── docling.py       # Docling engine
│   ├── llm.py           # LLM engine
│   └── api.py           # API engine
├── core/                # Core business logic
│   ├── processor.py     # BatchProcessor
│   ├── factory.py       # EngineFactory
│   └── models.py        # Data models
├── flow/                # CrewAI Flow orchestration
│   └── batch_flow.py    # ImageBatchProcessorFlow
├── config/              # Configuration management
│   └── settings.py      # Pydantic config models
├── utils/               # Utilities
│   └── logging.py       # Logging setup
├── exceptions.py        # Custom exceptions
├── main.py             # Entry point
└── run_pipeline.py     # Orchestrates llama-server + the batch run
```

## Usage

### Basic Usage

```python
from config.settings import BatchProcessorConfig, DoclingConfig
from core.processor import BatchProcessor
from engines.docling import DoclingEngine

# Configure the processor
config = BatchProcessorConfig(
    image_dir="./input_images",
    output_dir="./output_text",
    engine_type="docling",
    engine_config=DoclingConfig()
)

# Create engine and processor
engine = DoclingEngine(config.engine_config)
processor = BatchProcessor(
    engine=engine,
    output_dir=config.output_dir,
    max_retries=config.max_retries
)

# Process the batch
results = processor.process_batch(config.image_dir)
```

### Using CrewAI Flow

```python
from flow.batch_flow import ImageBatchProcessorFlow, BatchProcessorState

# Initialize state
state = BatchProcessorState(
    image_dir="./input_images",
    output_dir="./output_text",
    engine_type="docling",
    engine_config={"use_gpu": False}
)

# Run the flow
flow = ImageBatchProcessorFlow()
result = flow.kickoff(initial_state=state)
```

## Data Models

### ProcessingResult

Represents the outcome of processing a single image:

```python
@dataclass
class ProcessingResult:
    image_path: str              # Path to source image
    success: bool                # Whether processing succeeded
    output_path: Optional[str]   # Path to output text file
    error: Optional[str]         # Error message if failed
    attempts: int                # Number of attempts made
    processing_time: float       # Time taken in seconds
```

### BatchReport

Summary of batch processing results:

```python
@dataclass
class BatchReport:
    total_images: int            # Total images in batch
    successful: int              # Successfully processed
    failed: int                  # Failed to process
    processing_time: float       # Total processing time
    results: List[ProcessingResult]  # Individual results
    
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
```

## Error Handling

The system defines custom exceptions for different error scenarios:

- `BatchProcessorError`: Base exception for all processor errors
- `ExtractionError`: Raised when text extraction fails
- `ConfigurationError`: Raised when configuration is invalid
- `ValidationError`: Raised when input validation fails

### Retry Logic

Failed extractions are automatically retried with exponential backoff:
- Configurable maximum retry attempts
- Exponential backoff: 1s, 2s, 4s, 8s...
- Individual image failures don't halt the entire batch
- Detailed error logging for debugging

## Development

### Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=image_batch_processor

# Run specific test suite
pytest tests/unit/test_config.py -v
```

### Running the Application

```bash
# Always use uv to run
uv run python phase_2/image_batch_processor/main.py
```

## Running with a Local VLM (llama.cpp + Qwen3-VL)

By default the Docling engine uses a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server running a Qwen3-VL model as its text backend. Task shortcuts are defined
with [poethepoet](https://poethepoet.natn.io/) in `pyproject.toml` and run via
`uv run poe <task>`:

| Command | Description |
| --- | --- |
| `uv run poe pipeline` | Start llama-server (if not already running), wait until ready, run the batch (Docling engine), then shut the server down |
| `uv run poe sample` | Same as `pipeline` but only processes the first few images (`MAX_IMAGES`, default 3) — a quick sanity check before a full run |
| `uv run poe pipeline-llm` | Full run using the direct **LLM engine** (parallelized across server slots) instead of Docling |
| `uv run poe sample-llm` | 3-image sample using the LLM engine |
| `uv run poe pipeline-small` | Same as `pipeline` but with the smaller dense **Qwen3-VL 4B** model (`LLAMA_HF_REPO`) — lower VRAM, faster startup |
| `uv run poe pipeline-small-lowscale` | Like `pipeline-small` but also at a lower image scale (`VLM_SCALE=1.0`) — fewer image tokens, faster encoding |
| `uv run poe serve` | Start only the llama.cpp server |
| `uv run poe extract` | Run only `main.py` (assumes a server is already running) |
| `uv run poe test` | Run the test suite |

Relevant environment variables:
- `MAX_IMAGES` — cap the number of images processed (unset = whole directory).
- `ENGINE` — `docling` (default) or `llm`. The `*-llm` tasks set this to `llm`.
- `LLM_CONCURRENCY` — worker count for the LLM engine (default 4). Match it to
  the number of parallel slots your `llama-server` was launched with.
- `IMAGE_MAX_SIZE` — LLM engine only; downscale each image so its longest edge
  is at most this many pixels before sending (default 1600). Fewer pixels means
  far fewer image tokens and much faster prompt-eval; lower it for speed, raise
  it for fine-text fidelity.
- `VLM_SCALE` — Docling engine only; image scale sent to the VLM (default 2.0).
  Lower it toward 1.0 for fewer image tokens and faster encoding at some cost to
  fine-text fidelity. The `pipeline-small-lowscale` task sets this to `1.0`.

### Docling vs. LLM engine

- **Docling** (`pipeline`/`sample`): runs the full Docling pipeline and emits
  Markdown, DocTags, confidence reports, and debug visualizations. Processed
  sequentially (one shared converter, not thread-safe), so it's slower.
- **LLM** (`pipeline-llm`/`sample-llm`): sends each image straight to the VLM
  and writes the model's Markdown response as the page's `.txt`. No DocTags or
  debug artifacts, but it's stateless and parallelized across the server's
  slots, so it's substantially faster on a large batch.

On the first run, llama-server automatically downloads and caches the model +
mmproj (via the `-hf` flag) before it starts serving; subsequent runs reuse the
cached files. By default the runner waits indefinitely for the server to become
ready, so a large first-run download won't be cut short. Set
`LLAMA_STARTUP_TIMEOUT` (seconds) if you'd rather cap the wait.

### Model Choice

The default `unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:Q8_0` is a mixture-of-experts
model (~30B total params, only ~3B active per token), chosen for throughput
across a large batch. For maximum transcription accuracy at the cost of speed,
use the dense `unsloth/Qwen3-VL-32B-Instruct-GGUF:Q8_0` instead. Switch by
overriding `LLAMA_HF_REPO`.

> **Vulkan note:** Qwen3-VL's vision encoder uses an `UPSCALE` op the Vulkan
> backend doesn't yet support, so image encoding falls back to the CPU and each
> page is slower than pure text generation would suggest. Lowering the image
> scale from 2.0 toward 1.0 (set `VLM_SCALE`, or use the `pipeline-small-lowscale`
> task) shrinks the image and speeds up encoding at some cost to fine-text
> fidelity. `vlm_timeout` is set to 1800s to accommodate the slower per-page time.

### Output Layout

Each run writes to its own timestamped directory under `output/`, so results
from different models or backends never overwrite each other and are easy to
compare:

```
output/
└── run_20260715_143022/
    ├── text/                     # one .txt per page (the extracted text)
    ├── docling_vlm_markdown/     # per-page Markdown (+ referenced images)
    ├── docling_vlm_doctags/      # DocTags JSON (structure + metadata)
    ├── docling_vlm_reports/      # per-page confidence reports
    └── docling_vlm_debug/        # reading-order / key-value visualizations
```

The `docling_vlm_*` prefix is used for the VLM backend; the EasyOCR backend
uses `docling_*`. To compare runs, just diff the corresponding folders across
two `run_<timestamp>` directories.

### Monitoring a Run

Every run mirrors all log output to a `run.log` file inside its own run
directory (`output/run_<timestamp>/run.log`), in addition to the console. This
lets you monitor an unattended or overnight run from any terminal — even after
closing the one that launched it.

**Tail the newest run's log live** (PowerShell):

```powershell
$log = Get-ChildItem output\run_*\run.log | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $log.FullName -Wait -Tail 50
```

This streams the per-image progress (`✓ Processed pages-N`, `Progress: X/N
complete`) and the final summary as they happen.

**Check how many pages are done** (progress = number of `.txt` files written,
since each page is saved as it completes):

```powershell
(Get-ChildItem output\run_*\text\*.txt | Measure-Object).Count
```

Other signals:
- The `🌊 Flow: ImageBatchProcessorFlow` tree in the launching console is
  CrewAI's live step status (console-only; not captured in `run.log`).
- The `llama-server` console prints `prompt eval time` / `eval time` /
  `tokens per second` after each request — the best gauge of per-page speed.

### One-Command Workflow

`run_pipeline.py` handles the full lifecycle: it reuses an already-running
server if one is reachable, otherwise it launches `llama-server`, polls the
`/health` endpoint until the model is fully loaded and ready, runs the batch,
and tears the server down afterward (even on failure).

Configuration is via environment variables. Sensible defaults for a local
**Vulkan** build of llama.cpp are set in `[tool.poe.env]` in `pyproject.toml`;
any real environment variable of the same name overrides the default.

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_SERVER_BIN` | `...\llama-b7248-bin-win-vulkan-x64\llama-server.exe` | Path to the (Vulkan) llama-server binary |
| `LLAMA_HF_REPO` | `unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:Q8_0` | Hugging Face GGUF repo (optionally `:quant`); auto-downloads the model + mmproj |
| `LLAMA_MODEL` | *(none)* | Path to a local `.gguf` model (used only if `LLAMA_HF_REPO` is unset) |
| `LLAMA_MMPROJ` | *(none)* | Path to a local vision projector (mmproj) `.gguf` file |
| `LLAMA_HOST` | `127.0.0.1` | Host to bind/probe |
| `LLAMA_PORT` | `8080` | Port to bind/probe |
| `LLAMA_EXTRA_ARGS` | `-c 32768 -ngl 99 -sm row -t 16 -b 512 -ub 512 --flash-attn 1` | Server + GPU flags appended to the server command (context size, GPU offload, etc.) |
| `LLAMA_STARTUP_TIMEOUT` | *(none — waits forever)* | Optional cap (seconds) on how long to wait for the server to become ready |

> **Model download:** the default uses llama.cpp's `-hf` flag to pull
> `unsloth/Qwen3-VL-32B-Instruct-GGUF:Q8_0` (model + vision projector, ~34 GB)
> into the local llama.cpp cache. Append a different quant tag (e.g.
> `...-GGUF:Q6_K`) to trade quality for size, or unset `LLAMA_HF_REPO` and set
> `LLAMA_MODEL` (and `LLAMA_MMPROJ` for vision) to use local files.

> **GPU / Vulkan:** the defaults point `LLAMA_SERVER_BIN` at the Vulkan build
> and pass `-c 32768` (context window), `-ngl 99` (offload all layers to the
> GPU), plus row-split and flash attention. A full page image at `vlm_scale=2.0`
> tokenizes to ~4K tokens, which exceeds llama-server's default 4096 context —
> so a generous `-c` is required. Raise it further if you increase `vlm_scale`
> or hit `exceed_context_size_error`; adjust the rest for your hardware.

Example (Windows `cmd`):

```cmd
set LLAMA_MODEL=C:\models\Qwen3-VL-7B.gguf
set LLAMA_MMPROJ=C:\models\mmproj-Qwen3-VL-7B.gguf
uv run poe pipeline
```

Example (PowerShell):

```powershell
$env:LLAMA_MODEL="C:\models\Qwen3-VL-7B.gguf"
$env:LLAMA_MMPROJ="C:\models\mmproj-Qwen3-VL-7B.gguf"
uv run poe pipeline
```

> Note: `poe serve` and `run_pipeline.py` assume `llama-server` is on your
> `PATH`. If it isn't, set `LLAMA_SERVER_BIN` to the full path of the binary.
> The `vlm_model` in `main.py` must match the alias reported at
> `http://localhost:8080/v1/models`.

## Requirements

- Python >= 3.11
- crewai >= 1.6.1
- docling >= 2.0.0
- easyocr >= 1.7.0
- pydantic >= 2.12.5
- pytest >= 9.0.1
- hypothesis >= 6.148.5
- poethepoet >= 0.30.0 (dev, task runner)
- A local [llama.cpp](https://github.com/ggml-org/llama.cpp) server with a Qwen3-VL model (for the VLM backend)

## License

See LICENSE file in the repository root.
