"""Factory for creating extraction engine instances."""

from typing import Type

from engines.base import ExtractionEngine
from config.settings import (
    EngineConfig,
    DoclingConfig,
    LLMConfig,
    APIConfig,
    PassthroughConfig,
)


class EngineFactory:
    """Factory for creating extraction engines based on type."""
    
    # Engine type to class mapping
    # Note: Engine implementations will be imported when they're available
    _ENGINE_REGISTRY = {
        "docling": "DoclingEngine",
        "llm": "LLMEngine",
        "api": "APIEngine",
        "passthrough": "PassthroughEngine",
    }
    
    @staticmethod
    def create_engine(engine_type: str, config: EngineConfig) -> ExtractionEngine:
        """
        Create an extraction engine based on type.
        
        Args:
            engine_type: Type of engine ("docling", "llm", "api")
            config: Engine-specific configuration
            
        Returns:
            Configured ExtractionEngine instance
            
        Raises:
            ValueError: If engine_type is not supported
        """
        if engine_type not in EngineFactory._ENGINE_REGISTRY:
            raise ValueError(
                f"Unsupported engine type: '{engine_type}'. "
                f"Supported types are: {list(EngineFactory._ENGINE_REGISTRY.keys())}"
            )
        
        # Import engines dynamically to avoid circular imports
        # and to allow factory to exist before all engines are implemented
        if engine_type == "docling":
            from engines.docling import DoclingEngine
            if type(config).__name__ != "DoclingConfig":
                raise ValueError(
                    f"Engine type 'docling' requires DoclingConfig, "
                    f"got {type(config).__name__}"
                )
            return DoclingEngine(config)
        
        elif engine_type == "llm":
            from engines.llm import LLMEngine
            if type(config).__name__ != "LLMConfig":
                raise ValueError(
                    f"Engine type 'llm' requires LLMConfig, "
                    f"got {type(config).__name__}"
                )
            return LLMEngine(config)
        
        elif engine_type == "api":
            from engines.api import APIEngine
            if type(config).__name__ != "APIConfig":
                raise ValueError(
                    f"Engine type 'api' requires APIConfig, "
                    f"got {type(config).__name__}"
                )
            return APIEngine(config)
        
        elif engine_type == "passthrough":
            from engines.passthrough import PassthroughEngine
            if type(config).__name__ != "PassthroughConfig":
                raise ValueError(
                    f"Engine type 'passthrough' requires PassthroughConfig, "
                    f"got {type(config).__name__}"
                )
            return PassthroughEngine(config)
        
        # This should never be reached due to the check above
        raise ValueError(f"Unsupported engine type: '{engine_type}'")
    
    @staticmethod
    def get_supported_engines() -> list[str]:
        """
        Get list of supported engine types.
        
        Returns:
            List of supported engine type strings
        """
        return list(EngineFactory._ENGINE_REGISTRY.keys())
