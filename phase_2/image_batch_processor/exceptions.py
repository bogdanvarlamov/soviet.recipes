"""Custom exception classes for the Image Batch Processor."""


class BatchProcessorError(Exception):
    """Base exception for batch processor errors."""
    pass


class ExtractionError(BatchProcessorError):
    """Raised when text extraction fails."""
    pass


class PageSkipped(BatchProcessorError):
    """Raised when the engine intentionally skips a page (no transcribable text).

    This is not an error condition: the page was correctly identified as having
    nothing to extract (e.g. a full-page photograph or blank page). The
    processor treats it as a terminal, non-retryable, non-failure outcome.
    """

    def __init__(self, reason: str = "No transcribable text on page"):
        self.reason = reason
        super().__init__(reason)


class ConfigurationError(BatchProcessorError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(BatchProcessorError):
    """Raised when input validation fails."""
    pass
