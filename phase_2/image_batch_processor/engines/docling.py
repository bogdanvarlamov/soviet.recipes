"""Docling-based extraction engine for text extraction from images."""

import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from engines.base import ExtractionEngine
from config.settings import DoclingConfig
from exceptions import ExtractionError, ConfigurationError

import json

class DoclingEngine(ExtractionEngine):
    """
    Text extraction engine using Docling library with OCR support.
    
    This engine uses Docling's DocumentConverter with EasyOCR for extracting
    text from images. It supports Russian and English language detection.
    """
    
    def __init__(self, config: DoclingConfig):
        """
        Initialize the Docling extraction engine.
        
        Args:
            config: DoclingConfig with engine-specific settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._converter: Optional[DocumentConverter] = None
    
    def _initialize_converter(self) -> DocumentConverter:
        """
        Initialize the Docling DocumentConverter with OCR configuration.
        
        Returns:
            Configured DocumentConverter instance
        """
        if self._converter is not None:
            return self._converter
        
        try:
            if self.config.use_vlm:
                self._converter = self._build_vlm_converter()
                self.logger.info(
                    "Docling DocumentConverter initialized with VLM backend "
                    f"(url={self.config.vlm_url}, model={self.config.vlm_model})"
                )
            else:
                self._converter = self._build_ocr_converter()
                self.logger.info(
                    "Docling DocumentConverter initialized with EasyOCR backend"
                )
            return self._converter
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling converter: {e}")
            raise ConfigurationError(
                f"Failed to initialize Docling converter: {e}"
            ) from e
    
    def _build_ocr_converter(self) -> DocumentConverter:
        """Build a DocumentConverter using the traditional EasyOCR pipeline."""
        # Configure OCR with EasyOCR for Russian language.
        #
        # Device selection is controlled via accelerator_options (the modern
        # Docling API); EasyOcrOptions.use_gpu is deprecated. When use_gpu is
        # True we request CUDA, otherwise CPU. Note: torch-based EasyOCR only
        # supports CUDA (NVIDIA) or MPS (Apple) GPUs - it cannot use AMD/Vulkan
        # GPUs on Windows, and will silently fall back to CPU if CUDA is
        # unavailable.
        device = AcceleratorDevice.CUDA if self.config.use_gpu else AcceleratorDevice.CPU
        ocr_options = EasyOcrOptions(lang=["ru", "en"])
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=device),
            do_ocr=self.config.ocr_enabled,
            ocr_options=ocr_options,
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=2.0,  # Higher resolution for better quality
            do_table_structure=True,  # Detect table structures
        )
        return DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )
    
    def _build_vlm_converter(self) -> DocumentConverter:
        """
        Build a DocumentConverter that uses a remote vision LLM as its text
        backend via Docling's VLM pipeline.

        Targets an OpenAI-compatible /v1/chat/completions endpoint, such as a
        local llama.cpp server running Qwen3-VL.
        """
        response_format_map = {
            "markdown": ResponseFormat.MARKDOWN,
            "doctags": ResponseFormat.DOCTAGS,
            "html": ResponseFormat.HTML,
        }
        response_format = response_format_map[self.config.vlm_response_format]

        # Optional bearer auth header (llama.cpp usually needs none)
        headers = {}
        if self.config.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"

        vlm_options = ApiVlmOptions(
            url=self.config.vlm_url,
            params={"model": self.config.vlm_model},
            headers=headers,
            prompt=self.config.vlm_prompt,
            timeout=self.config.vlm_timeout,
            scale=self.config.vlm_scale,
            response_format=response_format,
        )

        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True,  # required to call an external endpoint
            vlm_options=vlm_options,
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=2.0,
        )

        return DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file using Docling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
            
        Raises:
            ExtractionError: If extraction fails
        """
        image_file = Path(image_path)
        
        # Validate that the file exists
        if not image_file.exists():
            raise ExtractionError(f"Image file not found: {image_path}")
        
        if not image_file.is_file():
            raise ExtractionError(f"Path is not a file: {image_path}")
        
        self.logger.info(f"Extracting text from: {image_file.name}")
        
        try:
            # Initialize converter if not already done
            converter = self._initialize_converter()
            
            # Convert the document
            result = converter.convert(str(image_file))
            
            # Determine the base output directory for engine artifacts.
            if self.config.output_dir:
                # Use the run-specific directory supplied via config (e.g. a
                # timestamped folder), so runs are kept separate for comparison.
                base_output_dir = Path(self.config.output_dir)
            else:
                # Fall back to a repo-relative ./output directory.
                current_path = Path.cwd()
                if "image_batch_processor" in str(current_path):
                    base_output_dir = current_path / "output"
                else:
                    base_output_dir = Path("phase_2/image_batch_processor/output")
            
            # Namespace artifact folders by backend so a VLM run does not
            # overwrite artifacts from a previous EasyOCR run (and vice versa).
            prefix = "docling_vlm" if self.config.use_vlm else "docling"

            # Create output directories
            debug_dir = base_output_dir / f"{prefix}_debug"
            markdown_dir = base_output_dir / f"{prefix}_markdown"
            doctags_dir = base_output_dir / f"{prefix}_doctags"
            
            debug_dir.mkdir(parents=True, exist_ok=True)
            markdown_dir.mkdir(parents=True, exist_ok=True)
            doctags_dir.mkdir(parents=True, exist_ok=True)
            
            # Create images directory for this page
            images_dir = markdown_dir / "images" / image_file.stem
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Save page images
            for page_no, page in result.document.pages.items():
                if page.image and page.image.pil_image:
                    page_image_filename = images_dir / f"page-{page_no}.png"
                    with page_image_filename.open("wb") as fp:
                        page.image.pil_image.save(fp, format="PNG")
                    self.logger.info(f"Saved page image: {page_image_filename}")
            
            # Save images of figures and tables.
            # Guarded per-element: the VLM pipeline may emit tables/pictures
            # without rasterized crops, in which case get_image() returns None.
            table_counter = 0
            picture_counter = 0
            for element, _level in result.document.iterate_items():
                try:
                    if isinstance(element, TableItem):
                        element_image = element.get_image(result.document)
                        if element_image is None:
                            continue
                        table_counter += 1
                        element_image_filename = images_dir / f"table-{table_counter}.png"
                        with element_image_filename.open("wb") as fp:
                            element_image.save(fp, "PNG")
                        self.logger.info(f"Saved table image: {element_image_filename}")
                    
                    if isinstance(element, PictureItem):
                        element_image = element.get_image(result.document)
                        if element_image is None:
                            continue
                        picture_counter += 1
                        element_image_filename = images_dir / f"picture-{picture_counter}.png"
                        with element_image_filename.open("wb") as fp:
                            element_image.save(fp, "PNG")
                        self.logger.info(f"Saved picture image: {element_image_filename}")
                except Exception as e:
                    self.logger.warning(f"Could not save element image: {e}")
            
            # Save markdown with externally referenced images (not base64 embedded)
            md_output_file = markdown_dir / f"{image_file.stem}.md"
            result.document.save_as_markdown(
                md_output_file,
                image_mode=ImageRefMode.REFERENCED
            )
            self.logger.info(f"Saved markdown to: {md_output_file}")
            self.logger.info(f"Total images saved: {len(result.document.pages)} pages, {table_counter} tables, {picture_counter} pictures")
            
            # Save doctags output (preserves metadata and structure)
            doctags_output_file = doctags_dir / f"{image_file.stem}.doctags.json"
            result.document.save_as_doctags(doctags_output_file)
            self.logger.info(f"Saved doctags to: {doctags_output_file}")
            
            # Save conversion confidence report
            reports_dir = base_output_dir / f"{prefix}_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = reports_dir / f"{image_file.stem}_report.txt"
            with report_file.open("w", encoding="utf-8") as fp:
                fp.write(f"Confidence Report for: {image_file.name}\n")
                fp.write("=" * 60 + "\n\n")
                # Use string representation instead of JSON for complex objects
                fp.write(str(result.confidence))
            self.logger.info(f"Saved confidence report to: {report_file}")
            
            # Generate debug visualization images showing how Docling processed the document
            try:
                # Get visualization images with reading order and labels
                viz_images = result.document.get_visualization(
                    show_label=True,
                    show_branch_numbering=True,
                    viz_mode='reading_order'
                )

                self.logger.info(f"visualization images length: {len(viz_images)}")
                
                # Save each page visualization
                for page_num, viz_image in viz_images.items():
                    if viz_image is not None:
                        debug_image_path = debug_dir / f"{image_file.stem}_page{page_num}_reading_order.png"
                        viz_image.save(str(debug_image_path))
                        self.logger.info(f"Saved reading order visualization to: {debug_image_path}")
                
                # Also get key-value visualization
                kv_viz_images = result.document.get_visualization(
                    show_label=True,
                    show_cell_id=True,
                    viz_mode='key_value'
                )
                
                for page_num, viz_image in kv_viz_images.items():
                    if viz_image is not None:
                        debug_image_path = debug_dir / f"{image_file.stem}_page{page_num}_key_value.png"
                        viz_image.save(str(debug_image_path))
                        self.logger.info(f"Saved key-value visualization to: {debug_image_path}")
                        
            except Exception as e:
                self.logger.warning(f"Could not generate visualization images: {e}")
            
            # Extract text from the document
            # Using export_to_markdown() to get clean text representation
            extracted_text = result.document.export_to_markdown()
            
            self.logger.info(
                f"Successfully extracted {len(extracted_text)} characters "
                f"from {image_file.name}"
            )
            
            return extracted_text
            
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            error_msg = f"Failed to extract text from {image_file.name}: {e}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e
    
    def validate_config(self) -> bool:
        """
        Validate that the Docling engine is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.logger.info("Validating Docling engine configuration")
        
        try:
            # Attempt to initialize the converter to validate configuration
            self._initialize_converter()
            
            # When using the VLM backend, verify the server is reachable so a
            # full batch fails fast instead of erroring on every image.
            if self.config.use_vlm:
                self._check_vlm_server_reachable()
            
            self.logger.info("Docling engine configuration is valid")
            return True
            
        except ConfigurationError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            error_msg = f"Docling configuration validation failed: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _check_vlm_server_reachable(self) -> None:
        """
        Probe the configured VLM server's OpenAI-compatible /models endpoint.

        Raises:
            ConfigurationError: If the server cannot be reached.
        """
        import httpx

        # Derive the /models endpoint from the chat/completions URL.
        base = self.config.vlm_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        models_url = f"{base}/models"

        headers = {}
        if self.config.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"

        try:
            response = httpx.get(models_url, headers=headers, timeout=10.0)
            response.raise_for_status()
            self.logger.info(f"VLM server reachable at {models_url}")
        except Exception as e:
            raise ConfigurationError(
                f"Could not reach VLM server at {models_url}: {e}. "
                f"Is llama-server running?"
            ) from e
