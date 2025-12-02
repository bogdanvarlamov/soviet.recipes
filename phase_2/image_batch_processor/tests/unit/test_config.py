"""Unit tests for configuration models."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pydantic import ValidationError

from config.settings import (
    EngineConfig,
    DoclingConfig,
    LLMConfig,
    APIConfig,
    BatchProcessorConfig,
)


class TestEngineConfig:
    """Tests for base EngineConfig."""
    
    def test_base_engine_config_instantiation(self):
        """Test that base EngineConfig can be instantiated."""
        config = EngineConfig()
        assert isinstance(config, EngineConfig)


class TestDoclingConfig:
    """Tests for DoclingConfig."""
    
    def test_docling_config_defaults(self):
        """Test DoclingConfig with default values."""
        config = DoclingConfig()
        assert config.model_path is None
        assert config.use_gpu is False
        assert config.batch_size == 1
        assert config.ocr_enabled is True
    
    def test_docling_config_custom_values(self):
        """Test DoclingConfig with custom values."""
        config = DoclingConfig(
            model_path="/path/to/model",
            use_gpu=True,
            batch_size=4,
            ocr_enabled=False
        )
        assert config.model_path == "/path/to/model"
        assert config.use_gpu is True
        assert config.batch_size == 4
        assert config.ocr_enabled is False
    
    def test_docling_config_invalid_batch_size(self):
        """Test that invalid batch_size raises ValidationError."""
        with pytest.raises(ValidationError):
            DoclingConfig(batch_size=0)


class TestLLMConfig:
    """Tests for LLMConfig."""
    
    def test_llm_config_required_fields(self):
        """Test that model_name is required."""
        with pytest.raises(ValidationError):
            LLMConfig()
    
    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(model_name="gpt-4")
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.api_key is None
        assert config.base_url is None
    
    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2048,
            api_key="test-key",
            base_url="https://api.example.com"
        )
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
    
    def test_llm_config_invalid_temperature(self):
        """Test that invalid temperature raises ValidationError."""
        with pytest.raises(ValidationError):
            LLMConfig(model_name="gpt-4", temperature=3.0)


class TestAPIConfig:
    """Tests for APIConfig."""
    
    def test_api_config_required_fields(self):
        """Test that api_url and api_key are required."""
        with pytest.raises(ValidationError):
            APIConfig()
    
    def test_api_config_defaults(self):
        """Test APIConfig with default values."""
        config = APIConfig(
            api_url="https://api.example.com",
            api_key="test-key"
        )
        assert config.api_url == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.verify_ssl is True
    
    def test_api_config_custom_values(self):
        """Test APIConfig with custom values."""
        config = APIConfig(
            api_url="https://api.example.com",
            api_key="test-key",
            timeout=60,
            max_retries=5,
            verify_ssl=False
        )
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.verify_ssl is False


class TestBatchProcessorConfig:
    """Tests for BatchProcessorConfig."""
    
    def test_batch_processor_config_required_fields(self):
        """Test that required fields must be provided."""
        with pytest.raises(ValidationError):
            BatchProcessorConfig()
    
    def test_batch_processor_config_valid_docling(self):
        """Test valid BatchProcessorConfig with Docling engine."""
        config = BatchProcessorConfig(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="docling",
            engine_config=DoclingConfig()
        )
        assert config.image_dir == "/path/to/images"
        assert config.output_dir == "/path/to/output"
        assert config.engine_type == "docling"
        assert isinstance(config.engine_config, DoclingConfig)
        assert config.max_retries == 3
    
    def test_batch_processor_config_valid_llm(self):
        """Test valid BatchProcessorConfig with LLM engine."""
        config = BatchProcessorConfig(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="llm",
            engine_config=LLMConfig(model_name="gpt-4")
        )
        assert config.engine_type == "llm"
        assert isinstance(config.engine_config, LLMConfig)
    
    def test_batch_processor_config_valid_api(self):
        """Test valid BatchProcessorConfig with API engine."""
        config = BatchProcessorConfig(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="api",
            engine_config=APIConfig(
                api_url="https://api.example.com",
                api_key="test-key"
            )
        )
        assert config.engine_type == "api"
        assert isinstance(config.engine_config, APIConfig)
    
    def test_batch_processor_config_empty_image_dir(self):
        """Test that empty image_dir raises ValidationError."""
        with pytest.raises(ValidationError, match="image_dir cannot be empty"):
            BatchProcessorConfig(
                image_dir="",
                output_dir="/path/to/output",
                engine_type="docling",
                engine_config=DoclingConfig()
            )
    
    def test_batch_processor_config_empty_output_dir(self):
        """Test that empty output_dir raises ValidationError."""
        with pytest.raises(ValidationError, match="output_dir cannot be empty"):
            BatchProcessorConfig(
                image_dir="/path/to/images",
                output_dir="",
                engine_type="docling",
                engine_config=DoclingConfig()
            )
    
    def test_batch_processor_config_invalid_engine_type(self):
        """Test that invalid engine_type raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchProcessorConfig(
                image_dir="/path/to/images",
                output_dir="/path/to/output",
                engine_type="invalid",
                engine_config=DoclingConfig()
            )
    
    def test_batch_processor_config_mismatched_engine_config(self):
        """Test that mismatched engine_config raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="engine_type 'docling' requires DoclingConfig"
        ):
            BatchProcessorConfig(
                image_dir="/path/to/images",
                output_dir="/path/to/output",
                engine_type="docling",
                engine_config=LLMConfig(model_name="gpt-4")
            )
    
    def test_batch_processor_config_custom_max_retries(self):
        """Test BatchProcessorConfig with custom max_retries."""
        config = BatchProcessorConfig(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="docling",
            engine_config=DoclingConfig(),
            max_retries=5
        )
        assert config.max_retries == 5
    
    def test_batch_processor_config_invalid_max_retries(self):
        """Test that negative max_retries raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchProcessorConfig(
                image_dir="/path/to/images",
                output_dir="/path/to/output",
                engine_type="docling",
                engine_config=DoclingConfig(),
                max_retries=-1
            )
    
    def test_batch_processor_config_custom_extensions(self):
        """Test BatchProcessorConfig with custom supported_extensions."""
        config = BatchProcessorConfig(
            image_dir="/path/to/images",
            output_dir="/path/to/output",
            engine_type="docling",
            engine_config=DoclingConfig(),
            supported_extensions=[".jpg", ".png"]
        )
        assert config.supported_extensions == [".jpg", ".png"]
