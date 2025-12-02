# Design Document: Image Batch Processor

## Overview

The Image Batch Processor is a workflow system that extracts text from directories of images using pluggable extraction engines. The system is built on CrewAI Flows for workflow orchestration and follows a clean architecture with separation of concerns between workflow logic, engine abstraction, and concrete engine implementations.

The core design principle is the Engine Interface pattern, which allows different text extraction technologies (Docling, LLMs, APIs) to be used interchangeably without modifying workflow code. Each image in a batch is processed sequentially, with error handling and retry logic ensuring robustness.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrewAI Flow Layer                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         ImageBatchProcessorFlow                       │  │
│  │  - Orchestrates batch processing                      │  │
│  │  - Manages state transitions                          │  │
│  │  - Coordinates error handling                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Batch Processor Layer                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         BatchProcessor                                │  │
│  │  - Iterates through images                            │  │
│  │  - Invokes engine via interface                       │  │
│  │  - Handles retries and failures                       │  │
│  │  - Saves output files                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Engine Interface Layer                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         ExtractionEngine (Abstract)                   │  │
│  │  + extract_text(image_path: str) -> str               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Docling    │ │     LLM      │ │     API      │
    │   Engine     │ │   Engine     │ │   Engine     │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### Design Patterns

1. **Strategy Pattern**: The ExtractionEngine interface with multiple implementations allows runtime selection of extraction strategy
2. **Template Method**: BatchProcessor defines the processing algorithm while delegating extraction to engines
3. **Dependency Injection**: Engines are injected into the processor, enabling testability and flexibility

## Components and Interfaces

### 1. ExtractionEngine (Abstract Base Class)

The core abstraction that all extraction engines must implement.

```python
from abc import ABC, abstractmethod
from typing import Optional

class ExtractionEngine(ABC):
    """Abstract interface for text extraction engines."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
```

### 2. Concrete Engine Implementations

#### DoclingEngine

```python
class DoclingEngine(ExtractionEngine):
    """Text extraction using Docling library."""
    
    def __init__(self, config: DoclingConfig):
        self.config = config
        
    def extract_text(self, image_path: str) -> str:
        # Use Docling API to extract text
        pass
        
    def validate_config(self) -> bool:
        # Validate Docling configuration
        pass
```

#### LLMEngine

```python
class LLMEngine(ExtractionEngine):
    """Text extraction using local or remote LLM."""
    
    def __init__(self, model_name: str, config: LLMConfig):
        self.model_name = model_name
        self.config = config
        
    def extract_text(self, image_path: str) -> str:
        # Use LLM to extract text from image
        pass
        
    def validate_config(self) -> bool:
        # Validate LLM configuration and availability
        pass
```

#### APIEngine

```python
class APIEngine(ExtractionEngine):
    """Text extraction using external API service."""
    
    def __init__(self, api_url: str, api_key: str, config: APIConfig):
        self.api_url = api_url
        self.api_key = api_key
        self.config = config
        
    def extract_text(self, image_path: str) -> str:
        # Call external API to extract text
        pass
        
    def validate_config(self) -> bool:
        # Validate API configuration and connectivity
        pass
```

### 3. BatchProcessor

Handles the core batch processing logic.

```python
from pathlib import Path
from typing import List, Dict
import logging

class ProcessingResult:
    """Result of processing a single image."""
    def __init__(self, image_path: str, success: bool, 
                 output_path: Optional[str] = None, 
                 error: Optional[str] = None):
        self.image_path = image_path
        self.success = success
        self.output_path = output_path
        self.error = error

class BatchProcessor:
    """Processes batches of images using an extraction engine."""
    
    def __init__(self, 
                 engine: ExtractionEngine,
                 output_dir: Path,
                 max_retries: int = 3,
                 logger: Optional[logging.Logger] = None):
        self.engine = engine
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        
    def process_batch(self, image_dir: Path) -> Dict[str, List[ProcessingResult]]:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            Dictionary with 'success' and 'failed' lists of ProcessingResult
        """
        pass
        
    def process_single_image(self, image_path: Path) -> ProcessingResult:
        """
        Process a single image with retry logic.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ProcessingResult indicating success or failure
        """
        pass
        
    def _save_text(self, text: str, image_path: Path) -> Path:
        """
        Save extracted text to output file.
        
        Args:
            text: Extracted text
            image_path: Original image path (for naming)
            
        Returns:
            Path to saved text file
        """
        pass
```

### 4. ImageBatchProcessorFlow (CrewAI Flow)

Orchestrates the workflow using CrewAI Flows.

```python
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

class BatchProcessorState(BaseModel):
    """State object for the batch processing flow."""
    image_dir: str
    output_dir: str
    engine_type: str
    engine_config: dict
    total_images: int = 0
    processed_images: int = 0
    successful: int = 0
    failed: int = 0
    results: list = []

class ImageBatchProcessorFlow(Flow[BatchProcessorState]):
    """CrewAI Flow for orchestrating image batch processing."""
    
    @start()
    def initialize_workflow(self):
        """Initialize and validate the workflow."""
        pass
        
    @listen(initialize_workflow)
    def create_engine(self):
        """Create and configure the extraction engine."""
        pass
        
    @listen(create_engine)
    def discover_images(self):
        """Discover all images in the input directory."""
        pass
        
    @listen(discover_images)
    def process_images(self):
        """Process all discovered images."""
        pass
        
    @listen(process_images)
    def generate_report(self):
        """Generate final processing report."""
        pass
```

### 5. Configuration System

```python
from pydantic import BaseModel, Field
from typing import Literal

class EngineConfig(BaseModel):
    """Base configuration for engines."""
    pass

class DoclingConfig(EngineConfig):
    """Configuration for Docling engine."""
    model_path: Optional[str] = None
    # Docling-specific settings

class LLMConfig(EngineConfig):
    """Configuration for LLM engine."""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    # LLM-specific settings

class APIConfig(EngineConfig):
    """Configuration for API engine."""
    api_url: str
    api_key: str
    timeout: int = 30
    # API-specific settings

class BatchProcessorConfig(BaseModel):
    """Main configuration for batch processor."""
    image_dir: str
    output_dir: str
    engine_type: Literal["docling", "llm", "api"]
    engine_config: EngineConfig
    max_retries: int = Field(default=3, ge=0)
    supported_extensions: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
```

### 6. Factory Pattern for Engine Creation

```python
class EngineFactory:
    """Factory for creating extraction engines."""
    
    @staticmethod
    def create_engine(engine_type: str, config: EngineConfig) -> ExtractionEngine:
        """
        Create an extraction engine based on type.
        
        Args:
            engine_type: Type of engine ("docling", "llm", "api")
            config: Engine-specific configuration
            
        Returns:
            Configured ExtractionEngine instance
            
        Raises:
            ValueError: If engine_type is not supported
        """
        if engine_type == "docling":
            return DoclingEngine(config)
        elif engine_type == "llm":
            return LLMEngine(config)
        elif engine_type == "api":
            return APIEngine(config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
```

## Data Models

### ProcessingResult

Represents the outcome of processing a single image.

```python
@dataclass
class ProcessingResult:
    image_path: str          # Path to source image
    success: bool            # Whether processing succeeded
    output_path: Optional[str]  # Path to output text file (if successful)
    error: Optional[str]     # Error message (if failed)
    attempts: int            # Number of attempts made
    processing_time: float   # Time taken in seconds
```

### BatchReport

Summary of batch processing results.

```python
@dataclass
class BatchReport:
    total_images: int
    successful: int
    failed: int
    processing_time: float
    results: List[ProcessingResult]
    
    def success_rate(self) -> float:
        return self.successful / self.total_images if self.total_images > 0 else 0.0
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Image discovery completeness
*For any* directory containing image files, the system should identify all and only files with supported image extensions, excluding non-image files.
**Validates: Requirements 1.1**

### Property 2: Processing completeness
*For any* set of discovered images, the batch processor should attempt to process each image exactly once.
**Validates: Requirements 1.2**

### Property 3: One-to-one output mapping
*For any* batch of images, the number of successfully created output files should equal the number of successfully processed images, maintaining a one-to-one correspondence.
**Validates: Requirements 1.3**

### Property 4: Filename transformation consistency
*For any* image file with path `/dir/name.ext`, the output file should be located at `{output_dir}/name.txt`, preserving the base filename while replacing the extension.
**Validates: Requirements 1.4, 6.2, 6.3**

### Property 5: Engine consistency within batch
*For any* batch processing session with a selected engine, all images in that batch should be processed using the same engine instance.
**Validates: Requirements 2.5, 3.4**

### Property 6: Engine factory correctness
*For any* valid engine type string ("docling", "llm", "api"), the factory should instantiate an engine of the corresponding concrete class.
**Validates: Requirements 3.5**

### Property 7: Error isolation
*For any* image that fails after exhausting retries, the batch processor should continue processing remaining images without terminating the entire batch.
**Validates: Requirements 4.3**

### Property 8: Retry behavior
*For any* image that fails extraction, the system should retry the operation up to the configured maximum retry count before marking it as failed.
**Validates: Requirements 4.2, 4.5**

### Property 9: Result reporting accuracy
*For any* completed batch, the reported success and failure counts should exactly match the actual number of images that succeeded and failed processing.
**Validates: Requirements 1.5, 4.4**

### Property 10: Output file creation
*For any* image that is successfully processed, a corresponding text file should exist in the output directory containing the extracted text.
**Validates: Requirements 6.1**

### Property 11: Output directory adherence
*For any* configured output directory path, all output files should be written to that directory and no other location.
**Validates: Requirements 6.4**

### Property 12: File overwrite idempotence
*For any* image processed multiple times, the output file should contain the results from the most recent processing, with earlier results being overwritten.
**Validates: Requirements 6.5**

### Property 13: Input validation completeness
*For any* invalid input (non-existent directory, empty directory, invalid engine type), the system should raise an appropriate error before beginning batch processing.
**Validates: Requirements 7.1, 7.2, 7.3**

### Property 14: Output directory creation
*For any* non-existent output directory path, the system should create the directory (including parent directories) before processing begins.
**Validates: Requirements 7.4**

### Property 15: Engine validation enforcement
*For any* selected engine, the system should call the engine's validate_config method and raise an error if validation fails, before processing any images.
**Validates: Requirements 7.5**

### Property 16: Error logging completeness
*For any* image that fails processing, the log should contain an entry with both the error message and the image filename that caused the error.
**Validates: Requirements 8.3**

## Error Handling

### Error Types

```python
class BatchProcessorError(Exception):
    """Base exception for batch processor errors."""
    pass

class ExtractionError(BatchProcessorError):
    """Raised when text extraction fails."""
    pass

class ConfigurationError(BatchProcessorError):
    """Raised when configuration is invalid."""
    pass

class ValidationError(BatchProcessorError):
    """Raised when input validation fails."""
    pass
```

### Error Handling Strategy

1. **Configuration Errors**: Fail fast before processing begins
   - Invalid engine type
   - Missing required configuration
   - Engine validation failure

2. **Validation Errors**: Fail fast with clear messages
   - Non-existent input directory
   - Empty input directory
   - Invalid file paths

3. **Extraction Errors**: Retry with exponential backoff
   - Network failures (for API engine)
   - Temporary resource issues
   - Malformed images
   - After max retries, log and continue with next image

4. **File I/O Errors**: Retry once, then fail
   - Permission errors
   - Disk full
   - Invalid output path

### Retry Logic

```python
def process_with_retry(self, image_path: Path, max_retries: int) -> ProcessingResult:
    """Process image with exponential backoff retry."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            text = self.engine.extract_text(str(image_path))
            output_path = self._save_text(text, image_path)
            return ProcessingResult(
                image_path=str(image_path),
                success=True,
                output_path=str(output_path),
                attempts=attempt + 1
            )
        except ExtractionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                self.logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {image_path}: {e}"
                )
    
    return ProcessingResult(
        image_path=str(image_path),
        success=False,
        error=str(last_error),
        attempts=max_retries
    )
```

## Testing Strategy

### Unit Testing

Unit tests will verify specific behaviors and edge cases:

1. **Engine Interface Compliance**: Test that each engine implementation correctly implements the ExtractionEngine interface
2. **Factory Pattern**: Test that EngineFactory creates correct engine types
3. **Filename Transformation**: Test edge cases like files with multiple dots, no extension, special characters
4. **Configuration Validation**: Test various invalid configurations
5. **Error Handling**: Test specific error scenarios with mocked engines
6. **File I/O**: Test output file creation, overwriting, directory creation

### Property-Based Testing

Property-based testing will verify universal properties across many inputs using **Hypothesis** (Python's property-based testing library). Each property test should run a minimum of 100 iterations.

**Test Configuration**:
```python
from hypothesis import given, settings
import hypothesis.strategies as st

# Configure for minimum 100 iterations
@settings(max_examples=100)
```

**Property Test Requirements**:
- Each property-based test MUST be tagged with a comment referencing the design document property
- Tag format: `# Feature: image-batch-processor, Property {number}: {property_text}`
- Each correctness property MUST be implemented by a SINGLE property-based test
- Tests should use Hypothesis strategies to generate diverse inputs

**Example Property Test Structure**:
```python
@settings(max_examples=100)
@given(
    image_files=st.lists(st.text(min_size=1), min_size=1, max_size=20),
    extensions=st.sampled_from(['.jpg', '.png', '.jpeg'])
)
def test_one_to_one_mapping(image_files, extensions):
    """
    Feature: image-batch-processor, Property 3: One-to-one output mapping
    For any batch of images, output files should equal successfully processed images.
    """
    # Test implementation
    pass
```

### Integration Testing

Integration tests will verify end-to-end workflows:

1. **Full Batch Processing**: Process a small batch of real images through each engine
2. **CrewAI Flow Integration**: Test that the Flow correctly orchestrates all stages
3. **Mixed Success/Failure**: Test batches with some images that succeed and some that fail
4. **Configuration Loading**: Test loading configuration from files

### Test Data Strategy

- **Unit Tests**: Use mocked engines and synthetic data
- **Property Tests**: Use Hypothesis to generate random but valid test data
- **Integration Tests**: Use a small set of real cookbook images from the phase_1 directory

## Implementation Notes

### Dependencies

```
crewai>=0.1.0
pydantic>=2.0.0
hypothesis>=6.0.0  # For property-based testing
pytest>=7.0.0
python>=3.10
```

### Project Structure

```
image_batch_processor/
├── __init__.py
├── engines/
│   ├── __init__.py
│   ├── base.py           # ExtractionEngine ABC
│   ├── docling.py        # DoclingEngine
│   ├── llm.py            # LLMEngine
│   └── api.py            # APIEngine
├── core/
│   ├── __init__.py
│   ├── processor.py      # BatchProcessor
│   ├── factory.py        # EngineFactory
│   └── models.py         # Data models
├── flow/
│   ├── __init__.py
│   └── batch_flow.py     # ImageBatchProcessorFlow
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration models
├── exceptions.py         # Custom exceptions
└── utils/
    ├── __init__.py
    ├── logging.py        # Logging utilities
    └── file_utils.py     # File handling utilities

tests/
├── unit/
│   ├── test_engines.py
│   ├── test_processor.py
│   ├── test_factory.py
│   └── test_models.py
├── property/
│   ├── test_properties.py      # Property-based tests
│   └── strategies.py           # Hypothesis strategies
└── integration/
    ├── test_end_to_end.py
    └── test_flow.py
```

### Performance Considerations

1. **Sequential Processing**: Current design processes images sequentially for simplicity. Future enhancement could add parallel processing.
2. **Memory Management**: Extract text from one image at a time to avoid loading all images into memory.
3. **Streaming**: For very large images, consider streaming approaches in engine implementations.
4. **Caching**: Engine implementations may cache models/connections for efficiency.

### Extensibility Points

1. **New Engines**: Add new extraction technologies by implementing ExtractionEngine
2. **Custom Workflows**: Extend ImageBatchProcessorFlow for custom orchestration
3. **Output Formats**: Currently outputs plain text; could be extended to support structured formats (JSON, Markdown)
4. **Parallel Processing**: Add parallel processing capability while maintaining the same interface
5. **Progress Callbacks**: Add callback hooks for progress monitoring in UI applications
