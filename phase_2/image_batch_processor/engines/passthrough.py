"""Passthrough engine for testing - logs but doesn't actually extract text."""

import logging
from pathlib import Path

from engines.base import ExtractionEngine
from config.settings import EngineConfig


class PassthroughConfig(EngineConfig):
    """Configuration for Passthrough engine (no special config needed)."""
    pass


class PassthroughEngine(ExtractionEngine):
    """
    A simple passthrough engine for testing.
    
    This engine doesn't perform actual text extraction. Instead, it:
    - Logs the image path being processed
    - Returns a dummy text string
    
    Useful for testing the workflow without requiring actual extraction engines.
    """
    
    def __init__(self, config: PassthroughConfig = None):
        """
        Initialize the passthrough engine.
        
        Args:
            config: Optional configuration (not used)
        """
        self.config = config or PassthroughConfig()
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, image_path: str) -> str:
        """
        Simulate text extraction by logging and returning dummy text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dummy extracted text string
        """
        image_name = Path(image_path).name
        self.logger.info(f"[PassthroughEngine] Processing image: {image_name}")
        
        # Return dummy text that includes the image filename
        dummy_text = f"Extracted text from {image_name}\n\nThis is dummy content for testing purposes."
        
        self.logger.info(f"[PassthroughEngine] Successfully 'extracted' {len(dummy_text)} characters")
        
        return dummy_text
    
    def validate_config(self) -> bool:
        """
        Validate the engine configuration.
        
        Returns:
            Always True (passthrough engine has no special requirements)
        """
        self.logger.info("[PassthroughEngine] Configuration validated (passthrough mode)")
        return True
