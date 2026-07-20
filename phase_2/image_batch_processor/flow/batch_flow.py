"""CrewAI Flow for orchestrating image batch processing."""

import logging
from pathlib import Path
from typing import Dict, Any

from crewai.flow.flow import Flow, listen, start

from flow.state import BatchProcessorState
from core.factory import EngineFactory
from core.processor import BatchProcessor
from core.models import BatchReport
from config.settings import (
    EngineConfig,
    DoclingConfig,
    LLMConfig,
    APIConfig,
    PassthroughConfig,
)
from exceptions import ValidationError, ConfigurationError
from utils.file_utils import discover_images
from utils.preprocessing import run_preprocessing_if_needed


class ImageBatchProcessorFlow(Flow[BatchProcessorState]):
    """
    CrewAI Flow for orchestrating image batch processing.
    
    This flow manages the complete workflow of batch processing images:
    1. Initialize and validate inputs
    2. Create the extraction engine
    3. Discover images in the input directory
    4. Process all images
    5. Generate final report
    
    Usage:
        flow = ImageBatchProcessorFlow()
        result = flow.kickoff()
    """
    
    def __init__(self):
        """
        Initialize the flow.
        
        The state will be set when the flow is kicked off with initial state.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.processor = None
    
    @start()
    def preprocess_images(self):
        """
        Optionally run the image preprocessing subflow before extraction.

        When ``state.enable_preprocessing`` is True, runs the bundled
        ``preprocessing`` subpackage (page split, white balance,
        etc.) over ``state.image_dir`` and rewrites ``state.image_dir`` to
        the preprocessing output directory so extraction runs against the
        preprocessed images. If that output directory already contains
        preprocessed images, the subflow is skipped and the existing output
        is reused, unless ``state.force_preprocessing`` is True.

        When ``state.enable_preprocessing`` is False (default), this step is
        a no-op and ``state.image_dir`` is left untouched.

        Raises:
            ConfigurationError: If the preprocessing pipeline fails to run.
        """
        if not self.state.enable_preprocessing:
            self.logger.info("Preprocessing subflow disabled; using image_dir as-is")
            return

        # Default preprocessing output lives under phase_2/ (a build artifact
        # of the active development phase), not next to the raw Phase 1 source
        # scans. Anchor to this package's location so the default is
        # independent of cwd.
        phase_2_dir = Path(__file__).resolve().parent.parent.parent
        preprocessing_output_dir = (
            self.state.preprocessing_output_dir
            or str(phase_2_dir / "preprocessed")
        )

        output_dir = run_preprocessing_if_needed(
            source_dir=self.state.image_dir,
            output_dir=preprocessing_output_dir,
            force=self.state.force_preprocessing,
            logger=self.logger,
        )

        # Point extraction at the preprocessed images.
        self.state.image_dir = str(output_dir)
        self.state.preprocessing_output_dir = str(output_dir)

    @listen(preprocess_images)
    def initialize_workflow(self):
        """
        Initialize and validate the workflow.
        
        Validates:
        - Image directory exists
        - Image directory contains at least one image
        - Engine type is valid
        - Output directory can be created
        
        Raises:
            ValidationError: If validation fails
        """
        self.logger.info("Initializing batch processing workflow")
        
        # Validate image directory exists
        image_dir = Path(self.state.image_dir)
        if not image_dir.exists():
            raise ValidationError(f"Image directory does not exist: {image_dir}")
        
        if not image_dir.is_dir():
            raise ValidationError(f"Image path is not a directory: {image_dir}")
        
        # Validate engine type
        supported_engines = EngineFactory.get_supported_engines()
        if self.state.engine_type not in supported_engines:
            raise ValidationError(
                f"Invalid engine type '{self.state.engine_type}'. "
                f"Supported types: {supported_engines}"
            )
        
        # Validate that directory contains at least one image
        # Use common image extensions for validation
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        try:
            images = discover_images(image_dir, supported_extensions)
            if len(images) == 0:
                raise ValidationError(
                    f"No images found in directory: {image_dir}. "
                    f"Supported extensions: {supported_extensions}"
                )
        except ValueError as e:
            raise ValidationError(f"Error discovering images: {e}")
        
        # Validate output directory (will be created if needed)
        output_dir = Path(self.state.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValidationError(
                f"Output path exists but is not a directory: {output_dir}"
            )
        
        self.logger.info(
            f"Validation complete: {len(images)} images found, "
            f"engine type: {self.state.engine_type}"
        )
    
    @listen(initialize_workflow)
    def create_engine(self):
        """
        Create and configure the extraction engine via factory.
        
        Instantiates the appropriate engine based on engine_type and validates
        its configuration.
        
        Raises:
            ConfigurationError: If engine configuration is invalid
        """
        self.logger.info(f"Creating {self.state.engine_type} extraction engine")
        
        # Convert engine_config dict to appropriate config model
        config = self._create_engine_config(
            self.state.engine_type,
            self.state.engine_config
        )
        
        # Create engine via factory
        try:
            self.engine = EngineFactory.create_engine(
                self.state.engine_type,
                config
            )
        except ValueError as e:
            raise ConfigurationError(f"Failed to create engine: {e}")
        
        # Validate engine configuration
        try:
            self.engine.validate_config()
            self.logger.info("Engine configuration validated successfully")
        except Exception as e:
            raise ConfigurationError(
                f"Engine validation failed for {self.state.engine_type}: {e}"
            )
    
    @listen(create_engine)
    def discover_images(self):
        """
        Discover all images in the input directory.
        
        Updates state with the total number of images found.
        """
        self.logger.info(f"Discovering images in {self.state.image_dir}")
        
        image_dir = Path(self.state.image_dir)
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        images = discover_images(image_dir, supported_extensions)
        
        # Update state with total images, capped by max_images if set
        discovered = len(images)
        if self.state.max_images is not None:
            self.state.total_images = min(discovered, self.state.max_images)
            self.logger.info(
                f"Discovered {discovered} images; limiting to "
                f"{self.state.total_images} (max_images={self.state.max_images})"
            )
        else:
            self.state.total_images = discovered
            self.logger.info(f"Discovered {self.state.total_images} images to process")
    
    @listen(discover_images)
    def process_images(self):
        """
        Process all discovered images using BatchProcessor.
        
        Updates state with processing results and statistics.
        """
        self.logger.info("Starting batch image processing")
        
        # Create BatchProcessor with the engine
        output_dir = Path(self.state.output_dir)
        self.processor = BatchProcessor(
            engine=self.engine,
            output_dir=output_dir,
            max_retries=3,  # Could be made configurable
            logger=self.logger,
            max_workers=self.state.max_workers
        )
        
        # Process the batch (optionally limited to the first N images)
        image_dir = Path(self.state.image_dir)
        batch_report = self.processor.process_batch(
            image_dir, max_images=self.state.max_images
        )
        
        # Update state with results
        self.state.processed_images = batch_report.total_images
        self.state.successful = batch_report.successful
        self.state.failed = batch_report.failed
        self.state.skipped = batch_report.skipped
        # Wall-clock batch time (accurate for parallel runs, unlike summing
        # per-image times).
        self.state.processing_time = batch_report.processing_time
        
        # Convert ProcessingResult dataclasses to dicts for state storage
        self.state.results = [
            {
                'image_path': result.image_path,
                'success': result.success,
                'output_path': result.output_path,
                'error': result.error,
                'attempts': result.attempts,
                'processing_time': result.processing_time,
                'skipped': result.skipped,
                'skip_reason': result.skip_reason,
            }
            for result in batch_report.results
        ]
        
        self.logger.info(
            f"Batch processing complete: {self.state.successful} successful, "
            f"{self.state.skipped} skipped, {self.state.failed} failed"
        )
    
    @listen(process_images)
    def generate_report(self) -> BatchReport:
        """
        Generate final processing report.
        
        Returns:
            BatchReport with complete processing statistics
        """
        self.logger.info("Generating final batch report")
        
        # Reconstruct ProcessingResult objects from state
        from core.models import ProcessingResult
        
        results = [
            ProcessingResult(
                image_path=r['image_path'],
                success=r['success'],
                output_path=r['output_path'],
                error=r['error'],
                attempts=r['attempts'],
                processing_time=r['processing_time'],
                skipped=r.get('skipped', False),
                skip_reason=r.get('skip_reason'),
            )
            for r in self.state.results
        ]
        
        # Use the wall-clock batch time (accurate for parallel runs) rather
        # than summing per-image times, which would over-count concurrency.
        report = BatchReport(
            total_images=self.state.total_images,
            successful=self.state.successful,
            failed=self.state.failed,
            processing_time=self.state.processing_time,
            results=results,
            skipped=self.state.skipped,
        )
        
        self.logger.info(
            f"Final report: {report.total_images} images, "
            f"{report.successful} successful ({report.success_rate():.1%}), "
            f"{report.skipped} skipped, {report.failed} failed, "
            f"total time: {report.processing_time:.2f}s"
        )
        
        return report
    
    def _create_engine_config(
        self,
        engine_type: str,
        config_dict: Dict[str, Any]
    ) -> EngineConfig:
        """
        Create appropriate EngineConfig instance from dictionary.
        
        Args:
            engine_type: Type of engine
            config_dict: Configuration dictionary
            
        Returns:
            Appropriate EngineConfig subclass instance
            
        Raises:
            ConfigurationError: If config creation fails
        """
        try:
            if engine_type == "docling":
                return DoclingConfig(**config_dict)
            elif engine_type == "llm":
                return LLMConfig(**config_dict)
            elif engine_type == "api":
                return APIConfig(**config_dict)
            elif engine_type == "passthrough":
                return PassthroughConfig(**config_dict)
            else:
                raise ConfigurationError(f"Unknown engine type: {engine_type}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create {engine_type} config: {e}"
            )
