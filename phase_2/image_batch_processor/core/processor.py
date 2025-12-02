"""Batch processor for extracting text from images."""

import logging
import time
from pathlib import Path
from typing import List, Optional

from core.models import ProcessingResult, BatchReport
from engines.base import ExtractionEngine
from exceptions import ExtractionError
from utils.file_utils import (
    discover_images,
    generate_output_filename,
    ensure_output_directory,
    save_text_to_file
)


class BatchProcessor:
    """Processes batches of images using an extraction engine."""
    
    def __init__(
        self,
        engine: ExtractionEngine,
        output_dir: Path,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        supported_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            engine: Extraction engine to use for text extraction
            output_dir: Directory where output text files will be saved
            max_retries: Maximum number of retry attempts per image (default: 3)
            logger: Logger instance (creates default if None)
            supported_extensions: List of supported image extensions (default: common formats)
        """
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        self.supported_extensions = supported_extensions or [
            '.jpg', '.jpeg', '.png', '.tiff', '.bmp'
        ]
        
    def process_batch(self, image_dir: Path) -> BatchReport:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images to process
            
        Returns:
            BatchReport with summary statistics and detailed results
        """
        batch_start_time = time.time()
        image_dir = Path(image_dir)
        
        # Ensure output directory exists
        ensure_output_directory(self.output_dir)
        
        # Discover all images in the directory
        image_files = discover_images(image_dir, self.supported_extensions)
        total_images = len(image_files)
        
        self.logger.info(f"Starting batch processing: {total_images} images found in {image_dir}")
        
        # Process each image
        results: List[ProcessingResult] = []
        successful = 0
        failed = 0
        
        for idx, image_path in enumerate(image_files, 1):
            self.logger.info(f"Processing image {idx}/{total_images}: {image_path.name}")
            
            result = self.process_single_image(image_path)
            results.append(result)
            
            if result.success:
                successful += 1
                self.logger.info(
                    f"✓ Successfully processed {image_path.name} "
                    f"(attempts: {result.attempts}, time: {result.processing_time:.2f}s)"
                )
            else:
                failed += 1
                self.logger.error(
                    f"✗ Failed to process {image_path.name} after {result.attempts} attempts: "
                    f"{result.error}"
                )
        
        batch_processing_time = time.time() - batch_start_time
        
        # Log completion summary
        self.logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed "
            f"(total time: {batch_processing_time:.2f}s)"
        )
        
        return BatchReport(
            total_images=total_images,
            successful=successful,
            failed=failed,
            processing_time=batch_processing_time,
            results=results
        )
    
    def process_single_image(self, image_path: Path) -> ProcessingResult:
        """
        Process a single image with retry logic and exponential backoff.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ProcessingResult indicating success or failure
        """
        image_path = Path(image_path)
        last_error: Optional[Exception] = None
        start_time = time.time()
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Extract text using the engine
                text = self.engine.extract_text(str(image_path))
                
                # Save the extracted text
                output_path = self._save_text(text, image_path)
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    image_path=str(image_path),
                    success=True,
                    output_path=str(output_path),
                    error=None,
                    attempts=attempt,
                    processing_time=processing_time
                )
                
            except ExtractionError as e:
                last_error = e
                
                if attempt < self.max_retries:
                    # Exponential backoff: 2^(attempt-1) seconds
                    wait_time = 2 ** (attempt - 1)
                    self.logger.warning(
                        f"Retry {attempt}/{self.max_retries} for {image_path.name}: {e}. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts exhausted for {image_path.name}"
                    )
            
            except Exception as e:
                # Catch unexpected errors and wrap them
                last_error = e
                self.logger.error(
                    f"Unexpected error on attempt {attempt}/{self.max_retries} "
                    f"for {image_path.name}: {type(e).__name__}: {e}"
                )
                
                if attempt < self.max_retries:
                    wait_time = 2 ** (attempt - 1)
                    time.sleep(wait_time)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            image_path=str(image_path),
            success=False,
            output_path=None,
            error=str(last_error) if last_error else "Unknown error",
            attempts=self.max_retries,
            processing_time=processing_time
        )
    
    def _save_text(self, text: str, image_path: Path) -> Path:
        """
        Save extracted text to output file.
        
        Args:
            text: Extracted text content
            image_path: Original image path (used for naming)
            
        Returns:
            Path to the saved text file
        """
        # Generate output filename
        output_filename = generate_output_filename(image_path)
        output_path = self.output_dir / output_filename
        
        # Save text to file
        save_text_to_file(text, output_path)
        
        return output_path
