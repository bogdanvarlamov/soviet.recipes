# Technology Stack

## Language & Version

- **Python**: >=3.11 (specified in pyproject.toml)
- **Package Manager**: uv (modern Python package manager)
- **System Environment**: conda "cookbook-processing" environment

## Core Dependencies

- **crewai** (>=1.6.1): Workflow orchestration framework using Flows
- **pydantic** (>=2.12.5): Data validation and settings management
- **pytest** (>=9.0.1): Testing framework
- **hypothesis** (>=6.148.5): Property-based testing library

## Project Management

- **Build System**: pyproject.toml with uv
- **Virtual Environment**: .venv directories per project
- **Lock File**: uv.lock for dependency resolution

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Running the Application
```ps
# ALWAYS with uv
uv run python phase_2/image_batch_processor/main.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=image_batch_processor

# Run property-based tests specifically
pytest tests/property/

# Run with verbose output
pytest -v
```

## Architecture Patterns

- **Strategy Pattern**: ExtractionEngine interface with multiple implementations
- **Factory Pattern**: EngineFactory for creating engine instances
- **Template Method**: BatchProcessor defines algorithm, delegates to engines
- **Dependency Injection**: Engines injected into processor for testability
