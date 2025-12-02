"""Unit tests for EngineFactory."""

import pytest

from image_batch_processor.core.factory import EngineFactory
from image_batch_processor.config.settings import (
    DoclingConfig,
    LLMConfig,
    APIConfig,
    EngineConfig,
)
from image_batch_processor.engines.docling import DoclingEngine
from image_batch_processor.engines.llm import LLMEngine
from image_batch_processor.engines.api import APIEngine


class TestEngineFactory:
    """Test cases for EngineFactory."""
    
    def test_create_docling_engine(self):
        """Test creating a Docling engine."""
        config = DoclingConfig(model_path="/path/to/model")
        engine = EngineFactory.create_engine("docling", config)
        
        assert isinstance(engine, DoclingEngine)
        assert engine.config == config
    
    def test_create_llm_engine(self):
        """Test creating an LLM engine."""
        config = LLMConfig(model_name="gpt-4", temperature=0.5)
        engine = EngineFactory.create_engine("llm", config)
        
        assert isinstance(engine, LLMEngine)
        assert engine.config == config
        assert engine.model_name == "gpt-4"
    
    def test_create_api_engine(self):
        """Test creating an API engine."""
        config = APIConfig(api_url="https://api.example.com", api_key="test-key")
        engine = EngineFactory.create_engine("api", config)
        
        assert isinstance(engine, APIEngine)
        assert engine.config == config
        assert engine.api_url == "https://api.example.com"
        assert engine.api_key == "test-key"
    
    def test_unsupported_engine_type(self):
        """Test that unsupported engine types raise ValueError."""
        config = DoclingConfig()
        
        with pytest.raises(ValueError) as exc_info:
            EngineFactory.create_engine("unsupported", config)
        
        assert "Unsupported engine type: 'unsupported'" in str(exc_info.value)
        assert "docling" in str(exc_info.value)
        assert "llm" in str(exc_info.value)
        assert "api" in str(exc_info.value)
    
    def test_wrong_config_type_for_docling(self):
        """Test that wrong config type for docling raises ValueError."""
        config = LLMConfig(model_name="gpt-4")
        
        with pytest.raises(ValueError) as exc_info:
            EngineFactory.create_engine("docling", config)
        
        assert "docling" in str(exc_info.value).lower()
        assert "DoclingConfig" in str(exc_info.value)
    
    def test_wrong_config_type_for_llm(self):
        """Test that wrong config type for llm raises ValueError."""
        config = DoclingConfig()
        
        with pytest.raises(ValueError) as exc_info:
            EngineFactory.create_engine("llm", config)
        
        assert "llm" in str(exc_info.value).lower()
        assert "LLMConfig" in str(exc_info.value)
    
    def test_wrong_config_type_for_api(self):
        """Test that wrong config type for api raises ValueError."""
        config = DoclingConfig()
        
        with pytest.raises(ValueError) as exc_info:
            EngineFactory.create_engine("api", config)
        
        assert "api" in str(exc_info.value).lower()
        assert "APIConfig" in str(exc_info.value)
    
    def test_get_supported_engines(self):
        """Test getting list of supported engine types."""
        supported = EngineFactory.get_supported_engines()
        
        assert isinstance(supported, list)
        assert "docling" in supported
        assert "llm" in supported
        assert "api" in supported
        assert "passthrough" in supported
        assert len(supported) == 4
