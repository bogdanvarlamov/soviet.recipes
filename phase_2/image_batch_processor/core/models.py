"""Data models for batch processing."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    
    image_path: str
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1
    processing_time: float = 0.0


@dataclass
class BatchReport:
    """Summary of batch processing results."""
    
    total_images: int
    successful: int
    failed: int
    processing_time: float
    results: List[ProcessingResult]
    
    def success_rate(self) -> float:
        """
        Calculate the success rate of the batch processing.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if self.total_images == 0:
            return 0.0
        return self.successful / self.total_images
