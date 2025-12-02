"""Configuration models for the image batch processor."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


class EngineConfig(BaseModel):
    """Base configuration for extraction engines."""
    pass


class DoclingConfig(EngineConfig):
    """Configuration for Docling-based extraction engine."""
    model_path: Optional[str] = None
    use_gpu: bool = False
    batch_size: int = Field(default=1, ge=1)
    ocr_enabled: bool = True


class LLMConfig(EngineConfig):
    """Configuration for LLM-based extraction engine."""
    model_name: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class APIConfig(EngineConfig):
    """Configuration for API-based extraction engine."""
    api_url: str
    api_key: str
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    verify_ssl: bool = True


class PassthroughConfig(EngineConfig):
    """Configuration for Passthrough engine (for testing)."""
    pass


class BatchProcessorConfig(BaseModel):
    """Main configuration for the batch processor."""
    image_dir: str
    output_dir: str
    engine_type: Literal["docling", "llm", "api", "passthrough"]
    engine_config: EngineConfig
    max_retries: int = Field(default=3, ge=0)
    supported_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    )
    
    @field_validator("image_dir")
    @classmethod
    def validate_image_dir(cls, v: str) -> str:
        """Validate that image directory path is provided."""
        if not v or not v.strip():
            raise ValueError("image_dir cannot be empty")
        return v
    
    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate that output directory path is provided."""
        if not v or not v.strip():
            raise ValueError("output_dir cannot be empty")
        return v
    
    @field_validator("engine_type")
    @classmethod
    def validate_engine_type(cls, v: str) -> str:
        """Validate that engine type is one of the supported types."""
        valid_types = ["docling", "llm", "api", "passthrough"]
        if v not in valid_types:
            raise ValueError(
                f"engine_type must be one of {valid_types}, got '{v}'"
            )
        return v
    
    @model_validator(mode="after")
    def validate_engine_config_matches_type(self) -> "BatchProcessorConfig":
        """Validate that engine_config matches the engine_type."""
        engine_config_map = {
            "docling": DoclingConfig,
            "llm": LLMConfig,
            "api": APIConfig,
            "passthrough": PassthroughConfig,
        }
        
        expected_config_type = engine_config_map.get(self.engine_type)
        if expected_config_type and not isinstance(
            self.engine_config, expected_config_type
        ):
            raise ValueError(
                f"engine_type '{self.engine_type}' requires "
                f"{expected_config_type.__name__}, "
                f"got {type(self.engine_config).__name__}"
            )
        
        return self
