# Project Structure

## Repository Layout

```
soviet.recipes/
├── phase_1/                          # Source data
│   └── cookbook_images/              # 224 scanned cookbook pages (JPG)
├── phase_2/                          # Active development
│   ├── docling/                      # Docling-specific tooling
│   └── image_batch_processor/        # Main application
└── .kiro/                            # Kiro configuration
    ├── specs/                        # Feature specifications
    └── steering/                     # AI assistant guidance
```

## Image Batch Processor Structure

```
phase_2/image_batch_processor/
├── engines/                          # Extraction engine implementations
│   ├── base.py                       # ExtractionEngine ABC
│   ├── docling.py                    # Docling engine (planned)
│   ├── llm.py                        # LLM engine (planned)
│   └── api.py                        # API engine (planned)
├── core/                             # Core business logic
│   ├── processor.py                  # BatchProcessor (planned)
│   ├── factory.py                    # EngineFactory (planned)
│   └── models.py                     # Data models (planned)
├── flow/                             # CrewAI Flow orchestration
│   └── batch_flow.py                 # ImageBatchProcessorFlow (planned)
├── config/                           # Configuration management
│   └── settings.py                   # Config models (planned)
├── utils/                            # Utilities
│   └── logging.py                    # Logging setup
├── exceptions.py                     # Custom exceptions
├── main.py                           # Entry point
└── pyproject.toml                    # Project metadata & dependencies
```

## Module Organization

### engines/
Contains all extraction engine implementations. Each engine must implement the `ExtractionEngine` abstract base class defined in `engines/base.py`.

### core/
Business logic for batch processing, including the main `BatchProcessor` class and supporting models.

### flow/
CrewAI Flow definitions for workflow orchestration. The main flow is `ImageBatchProcessorFlow`.

### config/
Pydantic models for configuration validation and management.

### utils/
Shared utilities like logging setup and file handling helpers.

## Naming Conventions

- **Files**: snake_case (e.g., `batch_flow.py`)
- **Classes**: PascalCase (e.g., `ExtractionEngine`, `BatchProcessor`)
- **Functions/Methods**: snake_case (e.g., `extract_text`, `process_batch`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- **Private members**: Leading underscore (e.g., `_save_text`)

## Import Organization

Follow this order:
1. Standard library imports
2. Third-party imports (crewai, pydantic, etc.)
3. Local application imports

Example:
```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from crewai.flow.flow import Flow

from image_batch_processor.exceptions import ExtractionError
```

## Testing Structure

```
tests/
├── unit/                             # Unit tests for individual components
├── property/                         # Property-based tests using Hypothesis
│   ├── test_properties.py            # Main property tests
│   └── strategies.py                 # Hypothesis strategies
└── integration/                      # End-to-end integration tests
```

Each property-based test must:
- Run minimum 100 iterations (`@settings(max_examples=100)`)
- Include a comment tag: `# Feature: image-batch-processor, Property {N}: {description}`
- Map to exactly one correctness property from the design document
