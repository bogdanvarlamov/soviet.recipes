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

## Architecture

```
┌─────────────────────────────────────────┐
│      CrewAI Flow Layer                  │
│  (ImageBatchProcessorFlow)              │
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

```python
from config.settings import DoclingConfig

docling_config = DoclingConfig(
    model_path="/path/to/model",  # Optional
    use_gpu=False,                # Use GPU acceleration
    batch_size=1,                 # Batch size for processing
    ocr_enabled=True              # Enable OCR
)
```

#### LLM Engine

```python
from config.settings import LLMConfig

llm_config = LLMConfig(
    model_name="gpt-4",           # Required
    temperature=0.0,              # 0.0 to 2.0
    max_tokens=4096,              # Maximum tokens
    api_key="your-api-key",       # Optional
    base_url="https://api..."     # Optional
)
```

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
└── main.py             # Entry point
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

## Requirements

- Python >= 3.11
- crewai >= 1.6.1
- pydantic >= 2.12.5
- pytest >= 9.0.1
- hypothesis >= 6.148.5

## License

See LICENSE file in the repository root.
