"""Main entry point for image batch processor."""

import logging
from pathlib import Path

from flow.batch_flow import ImageBatchProcessorFlow
from flow.state import BatchProcessorState
from utils.logging import setup_logging


def main():
    """Execute the batch processing flow with passthrough engine."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting image batch processor with passthrough engine")
    
    # Define paths (relative to workspace root)
    image_dir = Path("../../phase_1/cookbook_images")
    output_dir = Path("./output/passthrough_test")
    
    # Create initial state
    initial_state = BatchProcessorState(
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        engine_type="passthrough",
        engine_config={}  # Passthrough engine needs no config
    )
    
    # Create flow and set initial state
    flow = ImageBatchProcessorFlow()
    flow._state = initial_state
    
    try:
        logger.info(f"Processing images from: {image_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Kickoff the flow
        result = flow.kickoff()
        
        # Log final results
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total images: {result.total_images}")
        logger.info(f"Successful: {result.successful}")
        logger.info(f"Failed: {result.failed}")
        logger.info(f"Success rate: {result.success_rate():.1%}")
        logger.info(f"Total processing time: {result.processing_time:.2f}s")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
