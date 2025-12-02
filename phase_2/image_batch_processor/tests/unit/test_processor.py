"""Unit tests for BatchProcessor."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import time

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from core.processor import BatchProcessor
from core.models import ProcessingResult, BatchReport
from engines.base import ExtractionEngine
from exceptions import ExtractionError


class MockEngine(ExtractionEngine):
    """Mock extraction engine for testing."""
    
    def __init__(self, should_fail=False, fail_count=0):
        """
        Initialize mock engine.
        
        Args:
            should_fail: If True, always fails
            fail_count: Number of times to fail before succeeding
        """
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0
        self.extracted_texts = {}
        
    def extract_text(self, image_path: str) -> str:
        """Mock text extraction."""
        self.call_count += 1
        
        if self.should_fail:
            raise ExtractionError(f"Mock extraction failed for {image_path}")
        
        if self.fail_count > 0:
            self.fail_count -= 1
            raise ExtractionError(f"Mock temporary failure for {image_path}")
        
        # Return mock extracted text
        text = f"Extracted text from {Path(image_path).name}"
        self.extracted_texts[image_path] = text
        return text
    
    def validate_config(self) -> bool:
        """Mock config validation."""
        return True


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_initialization(self, tmp_path):
        """Test that BatchProcessor initializes correctly."""
        engine = MockEngine()
        output_dir = tmp_path / "output"
        
        processor = BatchProcessor(
            engine=engine,
            output_dir=output_dir,
            max_retries=3
        )
        
        assert processor.engine == engine
        assert processor.output_dir == output_dir
        assert processor.max_retries == 3
        assert processor.supported_extensions == ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    def test_process_single_image_success(self, tmp_path):
        """Test successful processing of a single image."""
        engine = MockEngine()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir)
        
        # Create a test image
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        
        result = processor.process_single_image(image_path)
        
        assert result.success is True
        assert result.image_path == str(image_path)
        assert result.output_path == str(output_dir / "test.txt")
        assert result.error is None
        assert result.attempts == 1
        assert result.processing_time >= 0  # Can be 0 for very fast operations
        
        # Verify output file was created
        output_file = Path(result.output_path)
        assert output_file.exists()
        assert "Extracted text from test.jpg" in output_file.read_text()
    
    def test_process_single_image_with_retry_success(self, tmp_path):
        """Test that retry logic works when extraction initially fails."""
        # Engine fails twice, then succeeds
        engine = MockEngine(fail_count=2)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir, max_retries=3)
        
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        
        result = processor.process_single_image(image_path)
        
        assert result.success is True
        assert result.attempts == 3  # Failed twice, succeeded on third attempt
        assert engine.call_count == 3
    
    def test_process_single_image_exhausts_retries(self, tmp_path):
        """Test that processing fails after exhausting all retries."""
        engine = MockEngine(should_fail=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir, max_retries=3)
        
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        
        result = processor.process_single_image(image_path)
        
        assert result.success is False
        assert result.attempts == 3
        assert result.output_path is None
        assert "Mock extraction failed" in result.error
        assert engine.call_count == 3
    
    def test_process_batch_success(self, tmp_path):
        """Test successful batch processing of multiple images."""
        engine = MockEngine()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create test images
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.png").touch()
        (input_dir / "image3.jpeg").touch()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir)
        
        report = processor.process_batch(input_dir)
        
        assert report.total_images == 3
        assert report.successful == 3
        assert report.failed == 0
        assert len(report.results) == 3
        assert report.success_rate() == 1.0
        
        # Verify all output files were created
        assert (output_dir / "image1.txt").exists()
        assert (output_dir / "image2.txt").exists()
        assert (output_dir / "image3.txt").exists()
    
    def test_process_batch_with_failures(self, tmp_path):
        """Test batch processing with some failures."""
        engine = MockEngine(should_fail=True)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create test images
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.jpg").touch()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir, max_retries=2)
        
        report = processor.process_batch(input_dir)
        
        assert report.total_images == 2
        assert report.successful == 0
        assert report.failed == 2
        assert report.success_rate() == 0.0
    
    def test_process_batch_mixed_results(self, tmp_path):
        """Test batch processing with mixed success and failure."""
        # Create an engine that fails for specific images
        engine = MockEngine()
        
        # Override extract_text to fail for image2
        original_extract = engine.extract_text
        def selective_extract(image_path: str) -> str:
            if "image2" in image_path:
                raise ExtractionError("Selective failure")
            return original_extract(image_path)
        
        engine.extract_text = selective_extract
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.jpg").touch()
        (input_dir / "image3.jpg").touch()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir, max_retries=1)
        
        report = processor.process_batch(input_dir)
        
        assert report.total_images == 3
        assert report.successful == 2
        assert report.failed == 1
        assert report.success_rate() == pytest.approx(2/3, rel=0.01)
    
    def test_process_batch_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        engine = MockEngine()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"  # Does not exist yet
        
        (input_dir / "image1.jpg").touch()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir)
        
        assert not output_dir.exists()
        
        report = processor.process_batch(input_dir)
        
        assert output_dir.exists()
        assert report.successful == 1
    
    def test_process_batch_empty_directory(self, tmp_path):
        """Test batch processing with empty directory."""
        engine = MockEngine()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir)
        
        report = processor.process_batch(input_dir)
        
        assert report.total_images == 0
        assert report.successful == 0
        assert report.failed == 0
        assert report.success_rate() == 0.0
    
    def test_save_text_creates_correct_output(self, tmp_path):
        """Test that _save_text creates output file with correct name."""
        engine = MockEngine()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        processor = BatchProcessor(engine=engine, output_dir=output_dir)
        
        image_path = Path("/some/path/photo.jpg")
        text = "Extracted text content"
        
        output_path = processor._save_text(text, image_path)
        
        assert output_path == output_dir / "photo.txt"
        assert output_path.exists()
        assert output_path.read_text() == text
    
    def test_custom_supported_extensions(self, tmp_path):
        """Test that custom supported extensions are respected."""
        engine = MockEngine()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create various image files
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.png").touch()
        (input_dir / "image3.gif").touch()
        
        # Only process .gif files
        processor = BatchProcessor(
            engine=engine,
            output_dir=output_dir,
            supported_extensions=['.gif']
        )
        
        report = processor.process_batch(input_dir)
        
        assert report.total_images == 1
        assert (output_dir / "image3.txt").exists()
        assert not (output_dir / "image1.txt").exists()
        assert not (output_dir / "image2.txt").exists()
