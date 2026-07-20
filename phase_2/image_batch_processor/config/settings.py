"""Configuration models for the image batch processor."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


class EngineConfig(BaseModel):
    """Base configuration for extraction engines."""
    pass


class DoclingConfig(EngineConfig):
    """Configuration for Docling-based extraction engine.

    Supports two text backends:
      - EasyOCR (traditional OCR) when ``use_vlm`` is False
      - A remote vision LLM via Docling's VLM pipeline when ``use_vlm`` is True.
        This targets an OpenAI-compatible endpoint such as a local llama.cpp
        server (llama-server) running a Qwen3-VL model.
    """
    model_path: Optional[str] = None
    # Request GPU (CUDA) for the EasyOCR/torch pipeline. Only effective on
    # machines with an NVIDIA CUDA GPU and a CUDA-enabled torch build; on CPU-
    # only or AMD/Vulkan setups EasyOCR falls back to CPU regardless.
    use_gpu: bool = False
    batch_size: int = Field(default=1, ge=1)
    ocr_enabled: bool = True

    # Base directory for the engine's artifacts (markdown, doctags, reports,
    # debug images). When set, artifact folders are created underneath it;
    # otherwise the engine falls back to a repo-relative ./output directory.
    output_dir: Optional[str] = None

    # VLM backend (local llama.cpp / OpenAI-compatible server)
    use_vlm: bool = True
    vlm_url: str = "http://localhost:8080/v1/chat/completions"
    vlm_model: str = "qwen3-vl"
    vlm_api_key: Optional[str] = None
    vlm_timeout: int = Field(default=300, ge=1)
    vlm_scale: float = Field(default=2.0, gt=0)
    vlm_response_format: Literal["markdown", "doctags", "html"] = "markdown"
    vlm_prompt: str = Field(
        default=(
            "Transcribe ALL text from this scanned Soviet-era cookbook page "
            "exactly as it appears, preserving the original language (Russian "
            "and/or English), reading order, headings, ingredient lists, and "
            "step numbering. Render tabular data as Markdown tables. Output "
            "only the transcribed content as clean Markdown."
        )
    )


class LLMConfig(EngineConfig):
    """Configuration for LLM-based extraction engine.

    Designed to work with any OpenAI-compatible endpoint, including a local
    llama.cpp server (llama-server) running a vision model such as Qwen3-VL.
    """
    model_name: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = Field(default=300, ge=1)
    # If set, downscale images so their longest edge is at most this many
    # pixels before sending. Fewer pixels -> far fewer image tokens -> much
    # faster prompt-eval. Lower for speed, raise for fine-text fidelity.
    max_image_size: Optional[int] = Field(default=None, ge=1)
    # When True, the engine exposes a ``skip_page`` tool the model can call for
    # pages with no transcribable text (e.g. a full-page photograph). This
    # avoids wasting retries on pages that will never yield text.
    allow_skip: bool = True
    prompt: str = Field(
        default=(
            "You are an expert transcriptionist digitizing a scanned page from "
            "a Soviet-era cookbook. Transcribe ALL text from this image exactly "
            "as it appears, preserving the original language (Russian and/or "
            "English). Maintain the reading order, headings, ingredient lists, "
            "and step numbering. Render tabular data as Markdown tables. Output "
            "only the transcribed content as clean Markdown, with no commentary, "
            "explanations, or added text of your own. If the page contains no "
            "transcribable text at all (for example a full-page photograph, "
            "illustration, or a blank page), do NOT invent or guess text: call "
            "the skip_page tool instead."
        )
    )


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
