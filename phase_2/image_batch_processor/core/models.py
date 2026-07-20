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
    # True when the engine intentionally skipped the page because it had no
    # transcribable text (e.g. a full-page photograph). Skipped pages are a
    # form of success: they were handled correctly and are not retried.
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BatchReport:
    """Summary of batch processing results."""
    
    total_images: int
    successful: int
    failed: int
    processing_time: float
    results: List[ProcessingResult]
    # Pages the engine intentionally skipped (no transcribable text). These are
    # counted separately from ``successful`` (text extracted) and ``failed``.
    skipped: int = 0
    
    def success_rate(self) -> float:
        """
        Calculate the rate of pages handled without failure.
        
        Skipped pages (no transcribable text) count as handled, since they were
        correctly identified rather than failed. Equivalent to
        ``(total - failed) / total``.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if self.total_images == 0:
            return 0.0
        return (self.successful + self.skipped) / self.total_images
