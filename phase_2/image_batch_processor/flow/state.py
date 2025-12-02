"""State model for CrewAI Flow batch processing."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BatchProcessorState(BaseModel):
    """
    State object for the batch processing CrewAI Flow.
    
    This model tracks the complete state of a batch processing workflow,
    including configuration, progress tracking, and results.
    
    Attributes:
        image_dir: Path to directory containing input images
        output_dir: Path to directory for output text files
        engine_type: Type of extraction engine to use ("docling", "llm", "api")
        engine_config: Configuration dictionary for the selected engine
        total_images: Total number of images discovered for processing
        processed_images: Number of images that have been processed (success or failure)
        successful: Number of images successfully processed
        failed: Number of images that failed processing
        results: List of processing results for individual images
    """
    
    image_dir: str = ""
    output_dir: str = ""
    engine_type: str = "docling"
    engine_config: Dict[str, Any] = Field(default_factory=dict)
    total_images: int = Field(default=0, ge=0)
    processed_images: int = Field(default=0, ge=0)
    successful: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        # Allow arbitrary types for flexibility with engine configs
        arbitrary_types_allowed=True,
        # Validate on assignment to catch errors early
        validate_assignment=True
    )
