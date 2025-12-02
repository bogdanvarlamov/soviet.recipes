"""Docling-based extraction engine for text extraction from images."""

import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
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
            # Configure OCR with EasyOCR for Russian language
            ocr_options = EasyOcrOptions(lang=["ru", "en"])
            pipeline_options = PdfPipelineOptions(
                do_ocr=self.config.ocr_enabled,
                ocr_options=ocr_options,
                generate_page_images=True,
                generate_picture_images=True,
                images_scale=2.0,  # Higher resolution for better quality
                do_table_structure=True,  # Detect table structures
            )
            
            # Initialize the document converter with OCR configuration
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=pipeline_options
                    ),
                }
            )
            
            self.logger.info("Docling DocumentConverter initialized successfully (debug mode enabled)")
            return self._converter
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling converter: {e}")
            raise ConfigurationError(
                f"Failed to initialize Docling converter: {e}"
            ) from e
    
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
            
            # Determine output directory - use phase_2/image_batch_processor/output
            # Find the phase_2/image_batch_processor directory
            current_path = Path.cwd()
            if "image_batch_processor" in str(current_path):
                # We're already in the image_batch_processor directory
                base_output_dir = current_path / "output"
            else:
                # Try to find it relative to current location
                base_output_dir = Path("phase_2/image_batch_processor/output")
            
            # Create output directories
            debug_dir = base_output_dir / "docling_debug"
            markdown_dir = base_output_dir / "docling_markdown"
            doctags_dir = base_output_dir / "docling_doctags"
            
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
            
            # Save images of figures and tables
            table_counter = 0
            picture_counter = 0
            for element, _level in result.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = images_dir / f"table-{table_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    self.logger.info(f"Saved table image: {element_image_filename}")
                
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = images_dir / f"picture-{picture_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    self.logger.info(f"Saved picture image: {element_image_filename}")
            
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
            reports_dir = base_output_dir / "docling_reports"
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
            
            self.logger.info("Docling engine configuration is valid")
            return True
            
        except ConfigurationError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            error_msg = f"Docling configuration validation failed: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
