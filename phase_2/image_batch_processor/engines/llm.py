"""LLM-based extraction engine implementation."""

from engines.base import ExtractionEngine
from config.settings import LLMConfig
from exceptions import ExtractionError, ConfigurationError


class LLMEngine(ExtractionEngine):
    """Text extraction using local or remote LLM."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM engine.
        
        Args:
            config: LLM-specific configuration
        """
        self.config = config
        self.model_name = config.model_name
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file using LLM.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        # TODO: Implement in task 6
        raise NotImplementedError("LLMEngine.extract_text not yet implemented")
    
    def validate_config(self) -> bool:
        """
        Validate that the engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # TODO: Implement in task 6
        raise NotImplementedError("LLMEngine.validate_config not yet implemented")
