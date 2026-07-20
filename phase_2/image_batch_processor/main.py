"""Main entry point for image batch processor."""

import logging
import os
from datetime import datetime
from pathlib import Path

from flow.batch_flow import ImageBatchProcessorFlow
from flow.state import BatchProcessorState
from utils.logging import setup_logging, add_file_logging


def main():
    """Execute the batch processing flow with a local llama.cpp Qwen3-VL engine."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting image batch processor (local llama.cpp Qwen3-VL backend)")
    
    # Define paths (relative to workspace root). By default we read the raw
    # Phase 1 scans, but IMAGE_DIR can point at a preprocessed image directory
    # (e.g. the output of the preprocessing subpackage) instead.
    image_dir_env = os.environ.get("IMAGE_DIR")
    image_dir = Path(image_dir_env) if image_dir_env else Path("../../phase_1/cookbook_images")

    # Each run gets its own timestamped directory so results can be compared
    # across runs (e.g. different models or backends) without overwriting.
    # Resolved to an absolute path: docling_core's save_as_markdown/doctags
    # re-joins a *relative* multi-segment artifacts_dir with its own parent a
    # second time (a bug in docling_core's _get_output_paths), which doubles
    # the path and fails with "No such file or directory" once the run
    # directory has more than one path segment (e.g. "./output/run_...").
    # Passing an absolute path sidesteps that code path entirely.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"./output/run_{timestamp}").resolve()
    output_dir = run_dir / "text"

    # Mirror all logs to a per-run log file so the run can be monitored/tailed.
    log_file = add_file_logging(run_dir / "run.log")
    logger.info(f"Run output directory: {run_dir}")
    logger.info(f"Logging to: {log_file}")

    # Optional cap on the number of images to process (set MAX_IMAGES for a
    # quick sample run; unset to process the whole directory).
    max_images_env = os.environ.get("MAX_IMAGES")
    max_images = int(max_images_env) if max_images_env else None
    if max_images:
        logger.info(f"MAX_IMAGES set: processing only the first {max_images} images")

    # Optional preprocessing subflow (page split, white balance, etc.) run
    # before extraction. Disabled by default to preserve existing behavior.
    enable_preprocessing = os.environ.get("ENABLE_PREPROCESSING", "").lower() in (
        "1", "true", "yes",
    )
    force_preprocessing = os.environ.get("FORCE_PREPROCESSING", "").lower() in (
        "1", "true", "yes",
    )
    preprocessing_output_dir = os.environ.get("PREPROCESSING_OUTPUT_DIR", "")
    if enable_preprocessing:
        logger.info(
            f"Preprocessing subflow enabled (force={force_preprocessing}); "
            f"output_dir={preprocessing_output_dir or '(default)'}"
        )

    # Choose the extraction engine. ENGINE=llm uses the direct VLM engine
    # (simple, stateless, safe to parallelize); anything else uses Docling.
    # The served model id is shared via LLAMA_HF_REPO so it matches whatever
    # llama-server has loaded.
    engine_choice = os.environ.get("ENGINE", "docling").lower()
    model_id = os.environ.get(
        "LLAMA_HF_REPO", "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:Q8_0"
    )

    if engine_choice == "llm":
        # Parallelize across the server's slots (LLMEngine is thread-safe).
        max_workers = int(os.environ.get("LLM_CONCURRENCY", "4"))
        engine_type = "llm"
        image_max_size = int(os.environ.get("IMAGE_MAX_SIZE", "1600"))
        engine_config = {
            "model_name": model_id,
            "base_url": "http://localhost:8080/v1",
            "api_key": "sk-no-key-required",
            "temperature": 0.0,
            "max_tokens": 8192,
            "timeout": 1800,
            "max_image_size": image_max_size,
        }
        logger.info(
            f"Engine: LLM (direct VLM) | model={model_id} | workers={max_workers} "
            f"| max_image_size={image_max_size}"
        )
    else:
        # Docling shares one converter and isn't safe to parallelize, so keep
        # it sequential.
        max_workers = 1
        engine_type = "docling"
        # Text backend: a remote VLM (default) or traditional EasyOCR. EasyOCR
        # needs no llama-server at all, so USE_VLM=0 runs fully offline.
        use_vlm = os.environ.get("USE_VLM", "1").lower() in ("1", "true", "yes")
        engine_config = {
            "use_vlm": use_vlm,
            "output_dir": str(run_dir),
        }
        if use_vlm:
            # Image scale sent to the VLM. Lower (toward 1.0) means fewer
            # image tokens -> faster encoding and less memory, at some cost to
            # fine-text fidelity. Override with VLM_SCALE for a lower-scale run.
            vlm_scale = float(os.environ.get("VLM_SCALE", "2.0"))
            engine_config.update({
                "vlm_url": "http://localhost:8080/v1/chat/completions",
                "vlm_model": model_id,
                "vlm_response_format": "markdown",
                "vlm_timeout": 1800,
                "vlm_scale": vlm_scale,
            })
            logger.info(
                f"Engine: Docling (VLM backend) | model={model_id} | workers=1 "
                f"| vlm_scale={vlm_scale}"
            )
        else:
            logger.info("Engine: Docling (EasyOCR backend) | workers=1")

    initial_state = BatchProcessorState(
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        engine_type=engine_type,
        max_images=max_images,
        max_workers=max_workers,
        engine_config=engine_config,
        enable_preprocessing=enable_preprocessing,
        preprocessing_output_dir=preprocessing_output_dir,
        force_preprocessing=force_preprocessing,
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
        logger.info(f"Skipped (no text): {result.skipped}")
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
