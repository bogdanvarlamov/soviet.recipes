"""Batch processor for extracting text from images."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from core.models import ProcessingResult, BatchReport
from engines.base import ExtractionEngine
from exceptions import ExtractionError, PageSkipped
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
        supported_extensions: Optional[List[str]] = None,
        max_workers: int = 1
    ):
        """
        Initialize the batch processor.
        
        Args:
            engine: Extraction engine to use for text extraction
            output_dir: Directory where output text files will be saved
            max_retries: Maximum number of retry attempts per image (default: 3)
            logger: Logger instance (creates default if None)
            supported_extensions: List of supported image extensions (default: common formats)
            max_workers: Number of images to process concurrently (default: 1,
                i.e. sequential). Only safe to raise above 1 with a thread-safe,
                stateless engine such as LLMEngine.
        """
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        self.supported_extensions = supported_extensions or [
            '.jpg', '.jpeg', '.png', '.tiff', '.bmp'
        ]
        self.max_workers = max(1, max_workers)
        
    def process_batch(
        self, image_dir: Path, max_images: Optional[int] = None
    ) -> BatchReport:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images to process
            max_images: If set, only process the first N images (useful for
                quick sample runs). Processes all images when None.
            
        Returns:
            BatchReport with summary statistics and detailed results
        """
        batch_start_time = time.time()
        image_dir = Path(image_dir)
        
        # Ensure output directory exists
        ensure_output_directory(self.output_dir)
        
        # Discover all images in the directory
        image_files = discover_images(image_dir, self.supported_extensions)
        if max_images is not None:
            image_files = image_files[:max_images]
        total_images = len(image_files)
        
        self.logger.info(
            f"Starting batch processing: {total_images} images found in {image_dir} "
            f"(workers: {self.max_workers})"
        )
        
        # Process images (sequentially or concurrently), preserving input order
        # in the results list.
        results: List[ProcessingResult] = self._run_batch(image_files, total_images)
        
        skipped = sum(1 for r in results if r.skipped)
        # "successful" counts pages that actually produced text (excludes skips).
        successful = sum(1 for r in results if r.success and not r.skipped)
        failed = sum(1 for r in results if not r.success)
        
        batch_processing_time = time.time() - batch_start_time
        
        # Log completion summary
        self.logger.info(
            f"Batch processing complete: {successful} successful, {skipped} skipped, "
            f"{failed} failed (total time: {batch_processing_time:.2f}s)"
        )
        
        return BatchReport(
            total_images=total_images,
            successful=successful,
            failed=failed,
            processing_time=batch_processing_time,
            results=results,
            skipped=skipped,
        )
    
    def _run_batch(
        self, image_files: List[Path], total_images: int
    ) -> List[ProcessingResult]:
        """
        Process the discovered images, sequentially or via a thread pool.
        
        Results are returned in the same order as ``image_files`` regardless of
        completion order.
        """
        results: List[Optional[ProcessingResult]] = [None] * total_images
        
        def handle(indexed_image):
            index, image_path = indexed_image
            result = self.process_single_image(image_path)
            if result.success:
                self.logger.info(
                    f"✓ Processed {image_path.name} "
                    f"(attempts: {result.attempts}, time: {result.processing_time:.2f}s)"
                )
            else:
                self.logger.error(
                    f"✗ Failed {image_path.name} after {result.attempts} attempts: "
                    f"{result.error}"
                )
            return index, result
        
        indexed = list(enumerate(image_files))
        
        if self.max_workers <= 1:
            for index, image_path in indexed:
                self.logger.info(
                    f"Processing image {index + 1}/{total_images}: {image_path.name}"
                )
                _, result = handle((index, image_path))
                results[index] = result
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(handle, item) for item in indexed]
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{total_images} complete")
        
        return results  # type: ignore[return-value]
    
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
            
            except PageSkipped as e:
                # The engine intentionally skipped this page (no transcribable
                # text). This is terminal: do not retry, and treat it as a
                # handled (non-failure) outcome. Write an empty output file so
                # the one-to-one image->text mapping is preserved.
                output_path = self._save_text("", image_path)
                processing_time = time.time() - start_time
                self.logger.info(
                    f"Skipped {image_path.name}: {e.reason}"
                )
                return ProcessingResult(
                    image_path=str(image_path),
                    success=True,
                    output_path=str(output_path),
                    error=None,
                    attempts=attempt,
                    processing_time=processing_time,
                    skipped=True,
                    skip_reason=e.reason,
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
