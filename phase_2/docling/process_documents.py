"""
Docling Document Processing Script
Processes documents using the Docling platform.
"""

from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)

# Configuration: List your input files here
INPUT_FILES = [
    # "../../phase_1/cookbook_images/pages-1.jpg", # this is the cover, not really much to convert here
    # Add more files as needed:
    "../../phase_1/cookbook_images/pages-2.jpg",
    "../../phase_1/cookbook_images/pages-3.jpg",

    "../../phase_1/cookbook_images/pages-12.jpg",
    "../../phase_1/cookbook_images/pages-13.jpg",
]

# Output directory for processed results
OUTPUT_DIR = Path("output")


def process_documents(input_files: list[str], output_dir: Path):
    """
    Process a list of documents using Docling.
    
    Args:
        input_files: List of file paths to process
        output_dir: Directory to save processed output
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Configure OCR with EasyOCR for Russian language
    ocr_options = EasyOcrOptions(lang=["ru", "en"])
    pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
    
    # Initialize the document converter with OCR configuration for both PDF and images
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )
    
    print(f"Processing {len(input_files)} file(s)...")
    
    for file_path in input_files:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            # Convert the document
            result = converter.convert(str(file_path))
            
            # Save the output as markdown
            md_output_file = output_dir / f"{file_path.stem}.md"
            result.document.save_as_markdown(md_output_file)
            
            # Save the Docling document in native DocTags format (preserves metadata)
            doctags_output_file = output_dir / f"{file_path.stem}.doctags.json"
            result.document.save_as_doctags(doctags_output_file)
            
            print(f"✅ Saved markdown to: {md_output_file}")
            print(f"✅ Saved DocTags to: {doctags_output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    print(f"\n✨ Processing complete! Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    process_documents(INPUT_FILES, OUTPUT_DIR)
