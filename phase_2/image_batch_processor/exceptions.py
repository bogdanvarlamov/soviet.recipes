"""Custom exception classes for the Image Batch Processor."""


class BatchProcessorError(Exception):
    """Base exception for batch processor errors."""
    pass


class ExtractionError(BatchProcessorError):
    """Raised when text extraction fails."""
    pass


class ConfigurationError(BatchProcessorError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(BatchProcessorError):
    """Raised when input validation fails."""
    pass
