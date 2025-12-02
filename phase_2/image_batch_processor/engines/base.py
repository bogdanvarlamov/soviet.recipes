"""Base extraction engine interface."""

from abc import ABC, abstractmethod


class ExtractionEngine(ABC):
    """Abstract interface for text extraction engines."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
