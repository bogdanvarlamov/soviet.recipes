"""Docling-based extraction engine implementation."""

from engines.base import ExtractionEngine
from config.settings import DoclingConfig
from exceptions import ExtractionError, ConfigurationError


class DoclingEngine(ExtractionEngine):
    """Text extraction using Docling library."""
    
    def __init__(self, config: DoclingConfig):
        """
        Initialize Docling engine.
        
        Args:
            config: Docling-specific configuration
        """
        self.config = config
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file using Docling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        # TODO: Implement in task 5
        raise NotImplementedError("DoclingEngine.extract_text not yet implemented")
    
    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # TODO: Implement in task 5
        raise NotImplementedError("DoclingEngine.validate_config not yet implemented")
