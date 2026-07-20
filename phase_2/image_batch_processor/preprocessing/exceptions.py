"""Custom exception classes for the Image Preprocessing Pipeline.

Mirrors the exception conventions of ``image_batch_processor/exceptions.py``:
a single base error with focused subclasses. The base ``PreprocessingError``
lets callers catch any pipeline-domain failure, while the subclasses identify
the specific failure mode (load, save, stage transform, or configuration).
"""

from typing import Optional


class PreprocessingError(Exception):
    """Base exception for all image preprocessing pipeline errors."""
    pass


class ImageLoadError(PreprocessingError):
    """Raised when a source image (or source directory) cannot be read.

    Carries the affected file/directory path and the underlying cause so the
    orchestrator can record an informative failed result.
    """

    def __init__(self, path: str, cause: Optional[str] = None):
        self.path = path
        self.cause = cause
        message = f"Failed to load image: {path}"
        if cause:
            message = f"{message} ({cause})"
        super().__init__(message)


class ImageSaveError(PreprocessingError):
    """Raised when an output image cannot be written.

    Carries the affected destination path and the underlying write-failure
    cause.
    """

    def __init__(self, path: str, cause: Optional[str] = None):
        self.path = path
        self.cause = cause
        message = f"Failed to save image: {path}"
        if cause:
            message = f"{message} ({cause})"
        super().__init__(message)


class StageError(PreprocessingError):
    """Raised when a stage cannot process an image.

    Carries the identity of the failing stage so the orchestrator can record
    which stage failed for a given source.
    """

    def __init__(self, message: str, stage_name: Optional[str] = None):
        self.stage_name = stage_name
        if stage_name:
            message = f"[{stage_name}] {message}"
        super().__init__(message)


class ConfigurationError(PreprocessingError):
    """Raised when pipeline or stage configuration is invalid."""
    pass
