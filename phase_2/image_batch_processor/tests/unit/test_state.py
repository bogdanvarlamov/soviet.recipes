"""Unit tests for BatchProcessorState model."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from flow.state import BatchProcessorState


def test_batch_processor_state_initialization():
    """Test that BatchProcessorState can be initialized with required fields."""
    state = BatchProcessorState(
        image_dir="/path/to/images",
        output_dir="/path/to/output",
        engine_type="docling",
        engine_config={"model_path": "/path/to/model"}
    )
    
    assert state.image_dir == "/path/to/images"
    assert state.output_dir == "/path/to/output"
    assert state.engine_type == "docling"
    assert state.engine_config == {"model_path": "/path/to/model"}
    assert state.total_images == 0
    assert state.processed_images == 0
    assert state.successful == 0
    assert state.failed == 0
    assert state.results == []


def test_batch_processor_state_with_tracking_fields():
    """Test that BatchProcessorState tracking fields can be set."""
    state = BatchProcessorState(
        image_dir="/path/to/images",
        output_dir="/path/to/output",
        engine_type="llm",
        engine_config={"model_name": "gpt-4"},
        total_images=10,
        processed_images=5,
        successful=4,
        failed=1,
        results=[
            {"image_path": "img1.jpg", "success": True},
            {"image_path": "img2.jpg", "success": False}
        ]
    )
    
    assert state.total_images == 10
    assert state.processed_images == 5
    assert state.successful == 4
    assert state.failed == 1
    assert len(state.results) == 2


def test_batch_processor_state_default_values():
    """Test that BatchProcessorState has correct default values."""
    state = BatchProcessorState(
        image_dir="/path/to/images",
        output_dir="/path/to/output",
        engine_type="api",
        engine_config={"api_url": "https://api.example.com"}
    )
    
    # Verify default values
    assert state.total_images == 0
    assert state.processed_images == 0
    assert state.successful == 0
    assert state.failed == 0
    assert isinstance(state.results, list)
    assert len(state.results) == 0


def test_batch_processor_state_validation():
    """Test that BatchProcessorState validates field types."""
    # Valid state
    state = BatchProcessorState(
        image_dir="/path/to/images",
        output_dir="/path/to/output",
        engine_type="docling",
        engine_config={}
    )
    assert state is not None
    
    # Test that negative values are rejected for count fields
    with pytest.raises(ValueError):
        BatchProcessorState(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="docling",
            engine_config={},
            total_images=-1
        )


def test_batch_processor_state_update_tracking():
    """Test that tracking fields can be updated after initialization."""
    state = BatchProcessorState(
        image_dir="/path/to/images",
        output_dir="/path/to/output",
        engine_type="docling",
        engine_config={}
    )
    
    # Update tracking fields
    state.total_images = 5
    state.processed_images = 3
    state.successful = 2
    state.failed = 1
    state.results.append({"image_path": "test.jpg", "success": True})
    
    assert state.total_images == 5
    assert state.processed_images == 3
    assert state.successful == 2
    assert state.failed == 1
    assert len(state.results) == 1
