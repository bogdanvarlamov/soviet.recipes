"""API-based extraction engine implementation."""

from engines.base import ExtractionEngine
from config.settings import APIConfig
from exceptions import ExtractionError, ConfigurationError


class APIEngine(ExtractionEngine):
    """Text extraction using external API service."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize API engine.
        
        Args:
            config: API-specific configuration
        """
        self.config = config
        self.api_url = config.api_url
        self.api_key = config.api_key
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file using external API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        # TODO: Implement in task 7
        raise NotImplementedError("APIEngine.extract_text not yet implemented")
    
    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # TODO: Implement in task 7
        raise NotImplementedError("APIEngine.validate_config not yet implemented")
